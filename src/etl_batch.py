"""
ETL-style batch scoring script (Data Engineering angle).

Extract sample accounts → transform features → load predictions to SQLite / CSV.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import get_dataset, FEATURE_NAMES, preprocess_data
from model import train_models, evaluate_models, predict_account
from database import init_db, log_prediction


def run_batch() -> None:
    print("=== ETL batch: extract → train/transform → load predictions ===")
    init_db()
    df = get_dataset(prefer_csv=True)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    models = train_models(X_train, y_train)
    best = evaluate_models(models, X_test, y_test)

    # Score all rows as a simple batch load
    rows = []
    for _, row in df.iterrows():
        vector = [float(row[c]) for c in feature_names]
        label, prob = predict_account(best, vector)
        payload = {c: float(row[c]) for c in feature_names}
        log_prediction(payload, label, prob)
        rows.append(
            {
                **payload,
                "predicted_label": int(label),
                "p_fake": float(prob),
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "batch_predictions.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Loaded {len(rows)} predictions → SQLite + {out_csv}")


if __name__ == "__main__":
    run_batch()
