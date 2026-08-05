"""Export evaluation artifacts for analysis / dashboards."""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_metrics(
    models: Dict,
    X_test,
    y_test,
    feature_names,
    output_dir: str = "data/outputs",
) -> str:
    ensure_dir(output_dir)
    summary = {}
    best_name, best_f1 = None, -1.0

    for name, model in models.items():
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds).tolist()
        summary[name] = {"accuracy": acc, "f1": f1, "confusion_matrix": cm}
        if f1 > best_f1:
            best_f1, best_name = f1, name

        # confusion matrix CSV per model
        pd.DataFrame(cm).to_csv(
            os.path.join(output_dir, f"confusion_matrix_{name}.csv"), index=False
        )

        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            fi.to_csv(os.path.join(output_dir, f"feature_importance_{name}.csv"), index=False)

    summary["best_model_by_f1"] = best_name
    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics exported to {out_path}")
    return out_path
