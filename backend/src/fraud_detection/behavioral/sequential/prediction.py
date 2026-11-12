"""Reusable LSTM inference for a user's chronological behavioral sequence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from fraud_detection.behavioral.sequential.preprocessing import SequencePreprocessor
from fraud_detection.behavioral.sequential.sequences import (
    EventFeatures,
    build_event_features,
    build_sequence_windows,
)


class SequenceModel(Protocol):
    def predict(self, values: np.ndarray, *, verbose: int = 0) -> np.ndarray: ...


@dataclass
class LSTMSequencePredictor:
    model: SequenceModel
    preprocessor: SequencePreprocessor
    threshold: float
    minimum_context_events: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("LSTM anomaly threshold must be between 0 and 1.")
        if self.minimum_context_events < 1:
            raise ValueError("minimum_context_events must be positive.")

    def predict(self, activities: Sequence[Mapping[str, Any]]) -> dict[str, object]:
        """Return an advisory sequence anomaly score for the final supplied event."""
        if not activities:
            raise ValueError("At least one activity is required for sequence inference.")
        frame = pd.DataFrame([dict(activity) for activity in activities])
        if "user_id" not in frame or frame["user_id"].nunique(dropna=False) != 1:
            raise ValueError("All activities in an LSTM sequence must belong to one user.")

        events = build_event_features(frame, require_labels=False)
        if len(events) < self.minimum_context_events:
            return {
                "status": "insufficient_context",
                "event_count": len(events),
                "minimum_context_events": self.minimum_context_events,
                "anomaly_score": None,
                "behavioral_anomaly": None,
                "is_fraud_decision": False,
            }

        windows = build_sequence_windows(
            events[-self.preprocessor.sequence_length :],
            sequence_length=self.preprocessor.sequence_length,
        )
        if not windows:
            raise RuntimeError("Sequence window construction returned no inference window.")
        transformed = self.preprocessor.transform_windows(windows[-1:])
        output = np.asarray(self.model.predict(transformed, verbose=0))
        if output.shape not in {(1,), (1, 1)}:
            raise ValueError(f"LSTM model returned unexpected prediction shape {output.shape}.")
        score = float(output.reshape(-1)[0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("LSTM model returned a non-finite or out-of-range anomaly score.")
        return {
            "status": "available",
            "event_count": len(events),
            "anomaly_score": score,
            "threshold": self.threshold,
            "behavioral_anomaly": score >= self.threshold,
            "signal_type": "synthetic-supervised-sequence-anomaly",
            "is_fraud_decision": False,
        }


def load_sequence_predictor(
    model_path: Path, preprocessing_path: Path, *, minimum_context_events: int = 3
) -> LSTMSequencePredictor:
    """Load a Keras artifact and its separately persisted preprocessing config."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to load the Phase 4 LSTM model. "
            "Install the backend 'tensorflow' extra."
        ) from exc

    payload = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
    preprocessor = SequencePreprocessor.from_dict(payload["preprocessing"])
    return LSTMSequencePredictor(
        model=tf.keras.models.load_model(model_path),
        preprocessor=preprocessor,
        threshold=float(payload["validation_threshold"]),
        minimum_context_events=minimum_context_events,
    )
