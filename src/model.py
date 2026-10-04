"""
Model training and evaluation for Fake Account Detection.

Includes Random Forest, Logistic Regression, and Gradient Boosting
(aligned with project-report discussion of boosting-style robustness).
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def calculate_metrics(model, X_test, y_test):
    """Calculate class-aware binary metrics using fake (1) as the positive class."""
    predictions = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        metrics["average_precision"] = float(average_precision_score(y_test, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced",
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def cross_validate_models(models: dict, X_train, y_train) -> dict:
    """Estimate model-selection F1 on training folds, keeping the holdout untouched."""
    class_counts = np.unique(y_train, return_counts=True)[1]
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Cross-validation requires at least two training rows per label.")
    cv = StratifiedKFold(
        n_splits=min(3, int(class_counts.min())),
        shuffle=True,
        random_state=42,
    )
    return {
        name: float(
            cross_val_score(model, X_train, y_train, scoring="f1", cv=cv).mean()
        )
        for name, model in models.items()
    }


def evaluate_models(models: dict, X_test, y_test, selection_scores: dict):
    if not models or set(models) != set(selection_scores):
        raise ValueError("Model and cross-validation score names must match.")
    print(f"\n--- Model Evaluation (holdout: {len(y_test)} accounts) ---")
    print("Interpret results cautiously: small samples produce unstable metrics.")
    best_name = max(selection_scores, key=selection_scores.get)
    best_model = None

    for name, model in models.items():
        preds = model.predict(X_test)
        metrics = calculate_metrics(model, X_test, y_test)
        print(f"\n{name}")
        print(f"  Training CV F1: {selection_scores[name]:.3f}")
        print(
            f"  Accuracy: {metrics['accuracy']:.3f} | "
            f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}"
        )
        print(
            f"  Fake precision: {metrics['precision']:.3f} | "
            f"Fake recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}"
        )
        print(
            "  ROC AUC: "
            f"{metrics['roc_auc']:.3f}"
            if metrics["roc_auc"] is not None
            else "  ROC AUC: unavailable (evaluation split contains one class)"
        )
        print(classification_report(y_test, preds, zero_division=0))

        if name == best_name:
            best_model = model

    print(
        f"Selected model by training cross-validation F1: "
        f"{best_name} ({selection_scores[best_name]:.3f})"
    )
    return best_model


def predict_account(model, feature_vector):
    x = np.array(feature_vector, dtype=float).reshape(1, -1)
    label = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
    else:
        probability = float(label)
    return label, probability
