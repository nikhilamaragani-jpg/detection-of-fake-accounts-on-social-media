"""
SQLite helper for logging model predictions
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_age_days INTEGER,
            followers INTEGER,
            following INTEGER,
            posts_count INTEGER,
            predicted_label INTEGER,
            predicted_probability REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(features: dict, label: int, probability: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            account_age_days, followers, following, posts_count,
            predicted_label, predicted_probability, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            features.get("account_age_days"),
            features.get("followers"),
            features.get("following"),
            features.get("posts_count"),
            label,
            probability,
            datetime.utcnow().isoformat() + "Z",
        ),
    )
    conn.commit()
    conn.close()
