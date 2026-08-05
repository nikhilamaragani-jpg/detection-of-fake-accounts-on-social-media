"""
Model training and evaluation for Fake Account Detection.

Includes Random Forest, Logistic Regression, and Gradient Boosting
(aligned with project-report discussion of boosting-style robustness).
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


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


def evaluate_models(models: dict, X_test, y_test):
    print("\n--- Model Evaluation ---")
    best_name = None
    best_score = -1.0
    best_model = None

    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"\n{name}")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  F1-score: {f1:.3f}")
        print(classification_report(y_test, preds, zero_division=0))

        # Prefer F1 for imbalanced trust-and-safety style tasks
        if f1 > best_score:
            best_score = f1
            best_name = name
            best_model = model

    print(f"Best model by F1: {best_name} (F1={best_score:.3f})")
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
