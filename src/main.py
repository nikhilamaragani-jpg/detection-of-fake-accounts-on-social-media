"""
Detection of Fake Accounts on Social Media
ML pipeline with model comparison + SQLite prediction logs
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import get_dataset, preprocess_data
from model import train_models, evaluate_models, predict_account
from database import init_db, log_prediction


def main():
    print("=" * 55)
    print("Fake Account Detection - Prototype")
    print("=" * 55)

    init_db()

    df = get_dataset(prefer_csv=True)
    print(f"Dataset size: {len(df)} records")

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    print("Data preprocessed and split.")

    models = train_models(X_train, y_train)
    best_model = evaluate_models(models, X_test, y_test)

    sample_features = {
        "account_age_days": 12,
        "followers": 5,
        "following": 800,
        "posts_count": 3,
        "has_profile_pic": 0,
        "has_bio": 0,
        "follower_following_ratio": 5 / 801,
    }

    vector = [sample_features[name] for name in feature_names]
    label, prob = predict_account(best_model, vector)
    log_prediction(sample_features, label, prob)

    print("\n--- Sample Prediction ---")
    print(f"Features: {sample_features}")
    print(f"Predicted Label: {'Fake' if label == 1 else 'Genuine'} (probability={prob:.2f})")
    print("Prediction saved to SQLite database (data/predictions.db)")


if __name__ == "__main__":
    main()
