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


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the required dataset columns as numeric values after validation."""
    if df.columns.duplicated().any():
        raise ValueError("Dataset must not contain duplicate column names.")
    required = FEATURE_NAMES + ["is_fake"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset must contain at least one row.")

    validated = df[required].copy()
    for column in required:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        values = validated[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Column '{column}' contains missing or non-finite values.")

    count_columns = ["account_age_days", "followers", "following", "posts_count"]
    for column in count_columns:
        if (validated[column] < 0).any():
            raise ValueError(f"Column '{column}' must contain non-negative values.")
        if not np.equal(validated[column] % 1, 0).all():
            raise ValueError(f"Column '{column}' must contain whole-number counts.")
    for column in ["has_profile_pic", "has_bio", "is_fake"]:
        if not validated[column].isin([0, 1]).all():
            raise ValueError(f"Column '{column}' must contain only 0 or 1.")
    if (validated["follower_following_ratio"] < 0).any():
        raise ValueError("Column 'follower_following_ratio' must be non-negative.")

    return validated


def create_sample_data(n_samples: int = 200) -> pd.DataFrame:
    if n_samples < 1:
        raise ValueError("n_samples must be greater than zero.")
    rng = np.random.default_rng(42)

    data = {
        "account_age_days": rng.integers(1, 2000, n_samples),
        "followers": rng.integers(0, 10000, n_samples),
        "following": rng.integers(0, 5000, n_samples),
        "posts_count": rng.integers(0, 3000, n_samples),
        "has_profile_pic": rng.integers(0, 2, n_samples),
        "has_bio": rng.integers(0, 2, n_samples),
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
    return validate_dataset(df)


def get_dataset(prefer_csv: bool = True) -> pd.DataFrame:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "sample_social_accounts.csv")

    if prefer_csv and os.path.exists(csv_path):
        print(f"Loading dataset from: {csv_path}")
        return load_csv_data(csv_path)

    print("CSV not found. Generating synthetic dataset...")
    return create_sample_data()


def preprocess_data(df: pd.DataFrame):
    df = validate_dataset(df)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["is_fake"].to_numpy(dtype=int)
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) != 2 or class_counts.min() < 3:
        raise ValueError(
            "Model selection requires at least three rows for each label."
        )

    test_size = max(2, int(np.ceil(0.2 * len(y))))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, FEATURE_NAMES
