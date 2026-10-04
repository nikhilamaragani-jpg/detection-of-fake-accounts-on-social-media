import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics_export import export_metrics
from model import (
    calculate_metrics,
    cross_validate_models,
    evaluate_models,
    train_models,
)
from preprocess import (
    FEATURE_NAMES,
    create_sample_data,
    preprocess_data,
    validate_dataset,
)
from eda_report import build_report


def test_sample_data_is_deterministic_without_changing_global_random_state():
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)

    first = create_sample_data(20)
    observed = np.random.random()
    second = create_sample_data(20)

    pd.testing.assert_frame_equal(first, second)
    assert observed == expected


def test_validate_dataset_rejects_invalid_values():
    df = create_sample_data(20)
    df.loc[0, "followers"] = -1

    with pytest.raises(ValueError, match="non-negative"):
        validate_dataset(df)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("has_bio", 2, "only 0 or 1"),
        ("is_fake", np.nan, "missing or non-finite"),
        ("posts_count", 1.5, "whole-number"),
        ("follower_following_ratio", -0.1, "non-negative"),
        ("follower_following_ratio", 99, "must equal followers"),
    ],
)
def test_validate_dataset_rejects_invalid_feature_values(column, value, message):
    df = create_sample_data(20)
    if column == "posts_count":
        df[column] = df[column].astype(float)
    df.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_dataset(df)


def test_preprocessing_stratifies_both_labels():
    df = create_sample_data(100)

    _, _, y_train, y_test, names = preprocess_data(df)

    assert names == FEATURE_NAMES
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}
    full_balance = pd.Series(df["is_fake"]).value_counts(normalize=True)
    train_balance = pd.Series(y_train).value_counts(normalize=True)
    assert all(
        abs(full_balance[label] - train_balance[label]) <= 1 / len(y_train)
        for label in [0, 1]
    )


def test_preprocessing_requires_two_examples_per_class():
    df = create_sample_data(20)
    df["is_fake"] = 0
    df.loc[0, "is_fake"] = 1

    with pytest.raises(ValueError, match="at least three rows for each label"):
        preprocess_data(df)


def test_repository_csv_passes_derived_ratio_validation():
    df = pd.read_csv(ROOT / "data" / "sample_social_accounts.csv")

    validated = validate_dataset(df)

    assert len(validated) == len(df)


def test_model_metrics_and_exports_are_analyst_friendly(tmp_path):
    df = create_sample_data(100)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    models = train_models(X_train, y_train)
    selection_scores = cross_validate_models(models, X_train, y_train)
    selected_model = evaluate_models(models, X_test, y_test, selection_scores)
    assert selected_model is models[max(selection_scores, key=selection_scores.get)]
    metrics = calculate_metrics(models["RandomForest"], X_test, y_test)

    assert {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
    } <= metrics.keys()
    assert len(metrics["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in metrics["confusion_matrix"])

    output_file = export_metrics(
        models,
        X_test,
        y_test,
        feature_names,
        selection_scores,
        output_dir=str(tmp_path),
    )
    exported = json.loads(Path(output_file).read_text(encoding="utf-8"))
    assert exported["positive_label_name"] == "fake"
    assert exported["best_model_by_training_cv_f1"] in models
    assert all("training_cv_f1" in exported[name] for name in models)
    assert (tmp_path / "confusion_matrix_RandomForest.csv").is_file()


def test_eda_report_computes_class_mix_and_documents_limits():
    report = build_report(create_sample_data(20))

    assert "Rows: 20" in report
    assert "genuine (0)" in report
    assert "fake (1)" in report
    assert "do not establish causal relationships" in report
