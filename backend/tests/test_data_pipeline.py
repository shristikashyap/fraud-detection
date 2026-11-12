import pandas as pd
import pytest

from fraud_detection.data.features import engineer_features
from fraud_detection.data.preprocessing import prepare_splits


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [5, 1, 4, 2, 3, 6, 7, 8, 9, 10],
            "Amount": [10, 20, None, 40, 50, 60, 70, 80, 90, 100],
            "V1": [0.1, 0.2, 0.3, None, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "Class": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        }
    )


def test_feature_engineering_keeps_labels_separate_and_adds_numeric_features():
    features, labels = engineer_features(sample_transactions())

    assert "is_fraud" not in features
    assert labels.tolist() == [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    assert {"amount_signed_log", "elapsed_day_sin", "elapsed_day_cos", "v1"} <= set(features)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)


def test_split_is_chronological_and_imputation_uses_training_only():
    raw = sample_transactions()
    raw.loc[9, "Amount"] = 1_000_000
    splits = prepare_splits(raw, train_fraction=0.6, validation_fraction=0.2)

    assert splits.train["time_seconds"].is_monotonic_increasing
    assert splits.validation["time_seconds"].is_monotonic_increasing
    assert splits.test["time_seconds"].is_monotonic_increasing
    assert splits.imputation_values["amount"] == 40
    assert splits.validation.notna().all().all()
    assert splits.test.notna().all().all()
    assert splits.imputation_values["amount"] != splits.validation["amount"].median()


def test_labels_outside_binary_domain_are_rejected():
    raw = sample_transactions()
    raw.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="encoded as 0 or 1"):
        engineer_features(raw)


def test_missing_required_column_is_rejected():
    raw = sample_transactions().drop(columns="Amount")

    with pytest.raises(ValueError, match="amount"):
        engineer_features(raw)
