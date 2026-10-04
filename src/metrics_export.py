"""Export evaluation artifacts for analysis / dashboards."""

from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd

from model import calculate_metrics


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_metrics(
    models: Dict,
    X_test,
    y_test,
    feature_names,
    selection_scores: Dict,
    output_dir: str | None = None,
) -> str:
    if not models or set(models) != set(selection_scores):
        raise ValueError("Model and cross-validation score names must match.")
    if output_dir is None:
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
        )
    ensure_dir(output_dir)
    summary = {}
    best_name, best_cv_f1 = None, -1.0

    for name, model in models.items():
        metrics = calculate_metrics(model, X_test, y_test)
        metrics["training_cv_f1"] = float(selection_scores[name])
        summary[name] = metrics
        cv_f1 = metrics["training_cv_f1"]
        if cv_f1 > best_cv_f1:
            best_cv_f1, best_name = cv_f1, name

        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["actual_genuine", "actual_fake"],
            columns=["predicted_genuine", "predicted_fake"],
        ).to_csv(
            os.path.join(output_dir, f"confusion_matrix_{name}.csv"),
            index=True,
            index_label="actual_label",
        )

        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            fi.to_csv(
                os.path.join(output_dir, f"feature_importance_{name}.csv"),
                index=False,
            )

    summary["best_model_by_training_cv_f1"] = best_name
    summary["positive_label"] = 1
    summary["positive_label_name"] = "fake"
    summary["holdout_accounts"] = int(len(y_test))
    summary["evaluation_note"] = (
        "Holdout metrics are exploratory estimates from a small demo dataset, "
        "not production performance claims."
    )
    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics exported to {out_path}")
    return out_path
