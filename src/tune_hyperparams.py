"""
Hyperparameter tuning for RandomForest (ML engineering demo).
Uses GridSearchCV with stratified CV; writes best params + score.
"""

from __future__ import annotations

import json
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import get_dataset, preprocess_data


def main() -> None:
    print("=== Hyperparameter tuning (RandomForest / GridSearchCV) ===")
    df = get_dataset(prefer_csv=True)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    param_grid = {
        "n_estimators": [50, 120],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "holdout_f1": float(
            __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                y_test, search.predict(X_test), zero_division=0
            )
        ),
        "features": feature_names,
    }
    path = os.path.join(out_dir, "hyperparam_search.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
