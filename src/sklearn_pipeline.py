"""Reusable scikit-learn Pipeline wrapper for train/inference."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_rf_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
