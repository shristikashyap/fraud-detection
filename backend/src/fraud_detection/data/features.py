"""Feature engineering for the OpenML credit-card transaction schema."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET_NAME = "is_fraud"
FEATURE_COLUMNS = (
    "time_seconds",
    "amount",
    "amount_signed_log",
    "elapsed_day_sin",
    "elapsed_day_cos",
)


def engineer_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric model features and binary labels from labeled transactions."""
    frame = _normalize_columns(raw)
    target_column = _find_column(frame.columns, ("class", "is_fraud", "label"))

    labels = pd.to_numeric(frame[target_column], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("Fraud labels must be present and encoded as 0 or 1.")

    features = engineer_transaction_features(frame)
    labels = labels.astype("int8").rename(TARGET_NAME)
    return features, labels


def engineer_transaction_features(
    raw: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Engineer model inputs from unlabeled transaction records.

    OpenML's `Time` is elapsed seconds in the source sample, not a calendar
    timestamp. Its within-day cyclic features are therefore only dataset
    representations, not an inference about a real user's normal schedule.
    """
    if raw.empty:
        raise ValueError("Cannot engineer features from an empty transaction.")

    frame = _normalize_columns(raw)
    time_column = _find_column(frame.columns, ("time", "time_seconds"))
    amount_column = _find_column(frame.columns, ("amount",))
    time_seconds = pd.to_numeric(frame[time_column], errors="coerce")
    amount = pd.to_numeric(frame[amount_column], errors="coerce")
    hour_angle = (time_seconds.mod(86_400) / 86_400) * (2 * np.pi)

    features = pd.DataFrame(index=frame.index)
    features["time_seconds"] = time_seconds
    features["amount"] = amount
    features["amount_signed_log"] = np.sign(amount) * np.log1p(np.abs(amount))
    features["elapsed_day_sin"] = np.sin(hour_angle)
    features["elapsed_day_cos"] = np.cos(hour_angle)

    component_columns = sorted(
        (column for column in frame.columns if re.fullmatch(r"v\d+", column)),
        key=lambda column: int(column[1:]),
    )
    for column in component_columns:
        features[column] = pd.to_numeric(frame[column], errors="coerce")

    if feature_columns is not None:
        for column in feature_columns:
            if column not in features:
                features[column] = np.nan
        features = features.loc[:, list(feature_columns)]

    return features.replace([np.inf, -np.inf], np.nan).astype("float64")


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    normalized_columns = [str(column).strip().lower() for column in frame.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("Dataset contains duplicate columns after normalization.")
    frame.columns = normalized_columns
    return frame


def _find_column(columns: pd.Index, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    expected = ", ".join(candidates)
    raise ValueError(f"Dataset is missing a required column (one of: {expected}).")
