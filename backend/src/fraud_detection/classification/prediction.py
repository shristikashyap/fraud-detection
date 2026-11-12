"""Cold-start prediction wrapper around a fitted classifier pipeline."""

from __future__ import annotations

import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_detection.data.features import engineer_transaction_features


@dataclass
class FraudPredictor:
    """Model and validation-tuned operating threshold for transaction scoring."""

    estimator: Pipeline
    feature_columns: list[str]
    threshold: float
    medium_risk_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("Fraud threshold must be between 0 and 1.")
        if not 0 < self.medium_risk_fraction <= 1:
            raise ValueError("medium_risk_fraction must be in (0, 1].")
        if not self.feature_columns:
            raise ValueError("At least one model feature is required.")

    def predict(self, transaction: Mapping[str, Any]) -> dict[str, object]:
        """Return a measured fraud probability, category, and thresholded result."""
        if not transaction:
            raise ValueError("Transaction data must not be empty.")
        raw = pd.DataFrame([dict(transaction)])
        features = engineer_transaction_features(raw, self.feature_columns)
        probability = float(self.estimator.predict_proba(features)[0, 1])
        low_risk_ceiling = self.threshold * self.medium_risk_fraction

        if probability >= self.threshold:
            category = "HIGH RISK"
        elif probability >= low_risk_ceiling:
            category = "MEDIUM RISK"
        else:
            category = "LOW RISK"

        return {
            "fraud_probability": probability,
            "risk_category": category,
            "prediction": "FRAUD" if probability >= self.threshold else "NOT FRAUD",
            "decision_threshold": self.threshold,
        }


def load_predictor(model_path: Path) -> FraudPredictor:
    """Load a persisted Phase 2 predictor and validate its artifact type."""
    artifact = joblib.load(model_path)
    if not isinstance(artifact, FraudPredictor):
        raise TypeError(f"Model artifact has an unexpected type: {type(artifact).__name__}")
    return artifact
