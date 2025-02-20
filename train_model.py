import pandas as pd
import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

print("🚀 Starting XGBoost model training with hyperparameter tuning ...")

# Load dataset
df = pd.read_csv("data/movie_mood_dataset.csv")
print(f"📂 Dataset loaded successfully. Total samples: {df.shape[0]}")

# Encode moods into numerical labels
label_encoder = LabelEncoder()
df["Mood_Label"] = label_encoder.fit_transform(df["Mood"])
print(f"🔢 Mood labels encoded. Unique moods: {len(label_encoder.classes_)}")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=2000,  # Limit features to avoid overfitting
    stop_words="english"
)
X = vectorizer.fit_transform(df["Overview"])
y = df["Mood_Label"]
print(f"📊 TF-IDF vectorization complete. Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📉 Data split into training ({X_train.shape[0]}) and test ({X_test.shape[0]}) sets.")


sample_weights = compute_sample_weight('balanced', y_train)

# Define base XGBoost classifier with class imbalance handling
base_xgb = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    random_state=42
)

# Define hyperparameter grid for tuning
param_grid = {
    'max_depth': [6, 8],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 500],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Perform GridSearchCV for hyperparameter tuning
print("🔍 Performing hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train, sample_weight=sample_weights)

# Best model after tuning
best_xgb_model = grid_search.best_estimator_
print(f"🏆 Best hyperparameters found: {grid_search.best_params_}")

# Calculate sample weights
sample_weights = compute_sample_weight("balanced", y_train)

# Train the best model on full training set
print("⏳ Training best model with optimized parameters...")
best_xgb_model.fit(X_train, y_train, verbose=True)
print("✅ Model training complete.")

# Save model and vectorizer
joblib.dump(best_xgb_model, "models/xgb_mood_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("💾 Model and vectorizer saved successfully.")

# Predictions on the test set
print("🔍 Generating predictions on test set...")
y_pred = best_xgb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 XGBoost Model Accuracy: {accuracy:.2%}")

# Classification report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("🚀 Training pipeline completed successfully!")
