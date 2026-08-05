"""
Detection of Fake Accounts on Social Media
End-to-end ML pipeline with multi-model comparison + SQLite audit logs
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import get_dataset, preprocess_data
from model import train_models, evaluate_models, predict_account
from database import init_db, log_prediction


def main():
    print("=" * 60)
    print("  Fake Account Detection  |  B.Tech ML Prototype")
    print("  Preprocess · Train · Compare · Predict · Audit log")
    print("=" * 60)

    init_db()

    df = get_dataset(prefer_csv=True)
    print(f"Dataset size: {len(df)} records")

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    print(f"Features used: {', '.join(feature_names)}")
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

    print("\n--- Sample Prediction (suspicious profile pattern) ---")
    print(f"Features: {sample_features}")
    print(f"Predicted Label: {'Fake' if label == 1 else 'Genuine'} (p_fake={prob:.2f})")
    print("Prediction saved to SQLite (data/predictions.db)")
    print("\nDone. Review metrics above for interview discussion.")


if __name__ == "__main__":
    main()
