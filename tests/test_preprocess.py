import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import create_sample_data, preprocess_data, FEATURE_NAMES


def test_sample_data_shape():
    df = create_sample_data(50)
    assert len(df) == 50
    for col in FEATURE_NAMES + ["is_fake"]:
        assert col in df.columns


def test_split_sizes():
    df = create_sample_data(100)
    X_train, X_test, y_train, y_test, names = preprocess_data(df)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert names == FEATURE_NAMES
