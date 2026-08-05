"""
Model training and evaluation for Fake Account Detection
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(models: dict, X_test, y_test):
    print("\n--- Model Evaluation ---")
    best_name = None
    best_acc = -1
    best_model = None

    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n{name} Accuracy: {acc:.2f}")
        print(classification_report(y_test, preds, zero_division=0))

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print(f"Best model: {best_name} ({best_acc:.2f})")
    return best_model


def predict_account(model, feature_vector):
    label = int(model.predict([feature_vector])[0])
    probability = float(model.predict_proba([feature_vector])[0][1])
    return label, probability
