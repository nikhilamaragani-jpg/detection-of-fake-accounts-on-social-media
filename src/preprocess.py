"""
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_sample_data(n_samples: int = 200) -> pd.DataFrame:
    """
    Creates a small synthetic dataset for demonstration.
    Replace this with a real dataset later.
    """
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

    # Simple rule to create labels for demo purposes
    df["is_fake"] = (
        (df["account_age_days"] < 30)
        | ((df["followers"] < 10) & (df["following"] > 500))
        | (df["has_profile_pic"] == 0)
    ).astype(int)

    return df


def preprocess_data(df: pd.DataFrame):
    features = [
        "account_age_days",
        "followers",
        "following",
        "posts_count",
        "has_profile_pic",
        "has_bio",
        "follower_following_ratio",
    ]

    X = df[features]
    y = df["is_fake"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
