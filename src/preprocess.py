"""
Data preprocessing utilities
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "account_age_days",
    "followers",
    "following",
    "posts_count",
    "has_profile_pic",
    "has_bio",
    "follower_following_ratio",
]


def create_sample_data(n_samples: int = 200) -> pd.DataFrame:
    np.random.seed(42)

    data = {
        "account_age_days": np.random.randint(1, 2000, n_samples),
        "followers": np.random.randint(0, 10000, n_samples),
        "following": np.random.randint(0, 5000, n_samples),
        "posts_count": np.random.randint(0, 3000, n_samples),
        "has_profile_pic": np.random.randint(0, 2, n_samples),
        "has_bio": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)
    df["follower_following_ratio"] = df["followers"] / (df["following"] + 1)

    df["is_fake"] = (
        (df["account_age_days"] < 30)
        | ((df["followers"] < 10) & (df["following"] > 500))
        | (df["has_profile_pic"] == 0)
    ).astype(int)

    return df


def load_csv_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = FEATURE_NAMES + ["is_fake"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[required].copy()


def get_dataset(prefer_csv: bool = True) -> pd.DataFrame:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "sample_social_accounts.csv")

    if prefer_csv and os.path.exists(csv_path):
        print(f"Loading dataset from: {csv_path}")
        return load_csv_data(csv_path)

    print("CSV not found. Generating synthetic dataset...")
    return create_sample_data()


def preprocess_data(df: pd.DataFrame):
    X = df[FEATURE_NAMES].values
    y = df["is_fake"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, FEATURE_NAMES
