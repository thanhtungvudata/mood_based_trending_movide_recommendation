import os
import requests
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from tqdm import tqdm

# Load API keys from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Hugging Face Emotion Classification Model
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# TMDb API Endpoints
TMDB_ENDPOINTS = [
    "https://api.themoviedb.org/3/trending/movie/week",
    "https://api.themoviedb.org/3/movie/top_rated",
    "https://api.themoviedb.org/3/movie/popular"
]

# Define target samples per mood
TARGET_SAMPLES_PER_MOOD = 200

# Dictionary to store movies per mood
movie_moods = {
    "joy": [], "sadness": [], "love": [], "anger": [], "fear": [], "surprise": []
}

# Set to track unique movie titles
unique_movie_titles = set()

def get_movies_from_tmdb(endpoint, page=1):
    """Fetch movies from TMDb API based on the given endpoint and page number."""
    try:
        response = requests.get(endpoint, params={"api_key": TMDB_API_KEY, "page": page}, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching movies from {endpoint}: {e}")
        return []

def classify_mood(movie_overview):
    """Classify movie mood using the Hugging Face emotion classifier."""
    if not movie_overview or len(movie_overview) < 10:
        return None
    try:
        result = classifier(movie_overview)
        mood = result[0]["label"]
        return mood if mood in movie_moods else None
    except Exception as e:
        print(f"❌ Error during mood classification: {e}")
        return None

def collect_movie_data():
    """Fetch movies, classify moods, and ensure 200 samples per mood."""
    for endpoint in TMDB_ENDPOINTS:
        print(f"📥 Fetching movies from {endpoint}...")
        page = 1

        while not all(len(movies) >= TARGET_SAMPLES_PER_MOOD for movies in movie_moods.values()):
            movies = get_movies_from_tmdb(endpoint, page)
            if not movies:
                break

            for movie in tqdm(movies, desc=f"Processing page {page}"):
                title, overview = movie.get("title"), movie.get("overview")
                if not title or not overview or title in unique_movie_titles:
                    continue

                mood = classify_mood(overview)
                if mood and len(movie_moods[mood]) < TARGET_SAMPLES_PER_MOOD:
                    movie_moods[mood].append({"Movie_Title": title, "Overview": overview, "Mood": mood})
                    unique_movie_titles.add(title)

            page += 1

            # Stop when each mood reaches its target
            if all(len(movies) >= TARGET_SAMPLES_PER_MOOD for movies in movie_moods.values()):
                break

def save_dataset():
    """Save the collected movie data into a CSV file."""
    all_movies = []
    for mood, movies in movie_moods.items():
        all_movies.extend(movies)

    df = pd.DataFrame(all_movies)
    df.to_csv("data/movie_mood_dataset.csv", index=False)
    print("✅ Movie mood dataset saved as movie_mood_dataset.csv")

if __name__ == "__main__":
    print("🚀 Collecting movies and ensuring 200 per mood...")
    collect_movie_data()
    save_dataset()
    print("🎬 Dataset generation complete!")
