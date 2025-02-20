import streamlit as st
import pandas as pd
import joblib
import requests
import random
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta

# Load environment variables (API keys)
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Check if key is available
if not TMDB_API_KEY:
    raise ValueError("🚨 TMDB_API_KEY not found! Please set it in Hugging Face Secrets.")

# Cache settings
CACHE_FILE = "movies_cache.pkl"
CACHE_EXPIRATION_DAYS = 7  # Refresh once per week

# Load trained model and vectorizer
model = joblib.load("models/xgb_mood_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Define mood labels and corresponding emotion icons in the desired order
mood_mapping = {
    "love": ("love", "❤️"),
    "joy": ("joy", "😃"),
    "surprise": ("surprise", "😲"),
    "sadness": ("sadness", "😢"),
    "fear": ("fear", "😨"),
    "anger": ("anger", "😡"),
}

# Hugging Face original order to custom order mapping
huggingface_to_custom = {
    "anger": "anger",
    "fear": "fear",
    "joy": "joy",
    "love": "love",
    "sadness": "sadness",
    "surprise": "surprise"
}

# TMDb API endpoint and image URL
WEEK_ENDPOINT = "https://api.themoviedb.org/3/trending/movie/week"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# Get the first day of the current week (Monday)
first_day_of_current_week = datetime.now() - timedelta(days=datetime.now().weekday())
current_week = datetime.now().isocalendar()[1]  # ISO week number

# 🕰 Cache movie fetching for one week
@st.cache_data(ttl=60 * 60 * 24 * 7, hash_funcs={int: str})
def fetch_trending_movies(week=current_week):
    """Fetch trending movies from TMDb and classify them once per week."""
    movies_cache = []
    page = 1

    while len(movies_cache) < 150:  # Fetch enough movies for all moods
        try:
            response = requests.get(WEEK_ENDPOINT, params={"api_key": TMDB_API_KEY, "page": page})
            response.raise_for_status()
            results = response.json().get("results", [])

            for movie in results:
                title = movie.get("title")
                overview = movie.get("overview")
                poster = TMDB_IMAGE_URL + movie["poster_path"] if movie.get("poster_path") else None
                release_date = movie.get("release_date")

                if title and overview and release_date:
                    release_date_obj = datetime.strptime(release_date, "%Y-%m-%d")
                    if release_date_obj < first_day_of_current_week:  # Ensure the movie was released before this week
                        hf_mood = classify_mood(overview)
                        custom_mood = huggingface_to_custom.get(hf_mood, "unknown")
                        movies_cache.append({
                            "title": title,
                            "overview": overview,
                            "poster": poster,
                            "mood": custom_mood,
                            "release_date": release_date
                        })

            page += 1
            if not results:
                break
        except Exception as e:
            st.error(f"Failed to fetch trending movies (Page {page}): {e}")
            break

    # Sort by release date (newest first)
    movies_cache.sort(key=lambda x: x["release_date"], reverse=True)
    return movies_cache

def classify_mood(movie_overview):
    """Predict movie mood using XGBoost model and map to custom order."""
    X = vectorizer.transform([movie_overview])
    mood_label = model.predict(X)[0]
    hf_mood = ["anger", "fear", "joy", "love", "sadness", "surprise"][mood_label]
    return hf_mood

def fetch_recommendations(user_mood):
    """Fetch 3 recommendations per mood from cached trending movies. Get more if fewer than 3."""
    mood_movies = []
    page = 1

    while len(mood_movies) < 3:
        trending_movies = fetch_trending_movies(current_week)

        # Filter movies by user mood
        for movie in trending_movies:
            if movie["mood"] == user_mood and movie["title"] not in [m["title"] for m in mood_movies]:
                mood_movies.append(movie)
                if len(mood_movies) >= 3:
                    break

        # If fewer than 3, fetch more pages
        if len(mood_movies) < 3:
            try:
                response = requests.get(WEEK_ENDPOINT, params={"api_key": TMDB_API_KEY, "page": page})
                response.raise_for_status()
                results = response.json().get("results", [])

                for movie in results:
                    title = movie.get("title")
                    overview = movie.get("overview")
                    poster = TMDB_IMAGE_URL + movie["poster_path"] if movie.get("poster_path") else None
                    release_date = movie.get("release_date")

                    if title and overview and release_date:
                        release_date_obj = datetime.strptime(release_date, "%Y-%m-%d")
                        if release_date_obj < first_day_of_current_week:
                            hf_mood = classify_mood(overview)
                            custom_mood = huggingface_to_custom.get(hf_mood, "unknown")
                            if custom_mood == user_mood and title not in [m["title"] for m in mood_movies]:
                                mood_movies.append({
                                    "title": title,
                                    "overview": overview,
                                    "poster": poster,
                                    "mood": custom_mood,
                                    "release_date": release_date
                                })

                page += 1
                if not results:
                    break
            except Exception as e:
                st.error(f"Failed to fetch additional trending movies: {e}")
                break

    return mood_movies[:3]

# Streamlit UI
st.title("🎬 CineMood: Get Your Mood-Based Trending Movies! ⚡")

# User selects their mood
user_mood, mood_icon = st.selectbox(
    "Select your mood:",
    [(mood, emoji) for mood, (mood, emoji) in mood_mapping.items()],
    format_func=lambda x: f"{x[1]} {x[0]}"
)

# Fetch recommendations based on user mood
recommended_movies = fetch_recommendations(user_mood)

# Display recommendations
st.subheader(f"{mood_icon} Recommended Trending Movies for Your Mood: {user_mood.capitalize()}")

if recommended_movies:
    for movie in recommended_movies:
        st.markdown(f"### 🎬 {movie['title']} ({movie['release_date']})")
        st.write(f"📖 {movie['overview']}")
        if movie['poster']:
            st.image(movie['poster'], width=200)
        st.write("---")
else:
    st.write("❌ No matching movies found. Try again later!")

# Footer Section
st.markdown("**Made by [Thanh Tung Vu](https://thanhtungvudata.github.io/)**")
