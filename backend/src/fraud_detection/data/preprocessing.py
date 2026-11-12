"""Leakage-conscious chronological split and missing-value handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.data.features import TARGET_NAME, engineer_features


@dataclass(frozen=True)
class PreparedSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    imputation_values: dict[str, float]


def prepare_splits(
    raw: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PreparedSplits:
    """Engineer features, split in time order, and impute from train only."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    features, labels = engineer_features(raw)
    ordered = features.assign(**{TARGET_NAME: labels}).sort_values(
        "time_seconds", kind="mergesort", na_position="last"
    )
    row_count = len(ordered)
    train_end = int(row_count * train_fraction)
    validation_end = train_end + int(row_count * validation_fraction)
    if train_end == 0 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Dataset is too small for the requested train/validation/test split.")

    pieces = [
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    ]
    feature_columns = [column for column in ordered.columns if column != TARGET_NAME]

    train_medians = pieces[0][feature_columns].median(axis=0).fillna(0.0)
    imputation_values = {column: float(train_medians[column]) for column in feature_columns}
    for piece in pieces:
        piece.loc[:, feature_columns] = (
            piece[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(train_medians)
        )
        piece[TARGET_NAME] = piece[TARGET_NAME].astype("int8")

    return PreparedSplits(
        train=pieces[0],
        validation=pieces[1],
        test=pieces[2],
        imputation_values=imputation_values,
    )
