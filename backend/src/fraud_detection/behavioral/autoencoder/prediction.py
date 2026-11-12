"""Reusable reconstruction-error inference for user activity sequences."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from fraud_detection.behavioral.autoencoder.model import masked_reconstruction_errors
from fraud_detection.behavioral.sequential.preprocessing import SequencePreprocessor
from fraud_detection.behavioral.sequential.sequences import (
    build_event_features,
    build_sequence_windows,
)


class AutoencoderModel(Protocol):
    def predict(self, values: np.ndarray, *, verbose: int = 0) -> np.ndarray: ...


@dataclass
class AutoencoderSequencePredictor:
    model: AutoencoderModel
    preprocessor: SequencePreprocessor
    threshold: float
    minimum_context_events: int = 3

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Reconstruction threshold must be finite and non-negative.")
        if self.minimum_context_events < 1:
            raise ValueError("minimum_context_events must be positive.")

    def predict(
        self, activities: Sequence[Mapping[str, Any]]
    ) -> dict[str, object]:
        """Return a reconstruction anomaly signal, never a fraud determination."""
        if not activities:
            raise ValueError("At least one activity is required for sequence inference.")
        frame = pd.DataFrame([dict(activity) for activity in activities])
        if "user_id" not in frame or frame["user_id"].nunique(dropna=False) != 1:
            raise ValueError("All activities in an autoencoder sequence must belong to one user.")

        events = build_event_features(frame, require_labels=False)
        if len(events) < self.minimum_context_events:
            return {
                "status": "insufficient_context",
                "event_count": len(events),
                "minimum_context_events": self.minimum_context_events,
                "reconstruction_error": None,
                "threshold": self.threshold,
                "behavioral_anomaly": None,
                "is_fraud_decision": False,
            }

        windows = build_sequence_windows(
            events[-self.preprocessor.sequence_length :],
            sequence_length=self.preprocessor.sequence_length,
        )
        if not windows:
            raise RuntimeError("Sequence window construction returned no inference window.")
        original = self.preprocessor.transform_windows(windows[-1:])
        reconstructed = np.asarray(self.model.predict(original, verbose=0))
        if reconstructed.shape != original.shape:
            raise ValueError(
                "Autoencoder output shape does not match input shape: "
                f"{reconstructed.shape} != {original.shape}."
            )
        error = float(masked_reconstruction_errors(original, reconstructed)[0])
        return {
            "status": "available",
            "event_count": len(events),
            "reconstruction_error": error,
            "threshold": self.threshold,
            "behavioral_anomaly": error >= self.threshold,
            "signal_type": "normal-activity-reconstruction-error",
            "is_fraud_decision": False,
        }


def load_autoencoder_predictor(
    model_path: Path,
    preprocessing_path: Path,
    *,
    minimum_context_events: int = 3,
) -> AutoencoderSequencePredictor:
    """Load a Keras autoencoder and its independently saved configuration."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to load the Phase 5 autoencoder. "
            "Install the backend 'tensorflow' extra."
        ) from exc
    payload = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
    if payload.get("model_type") != "normal-only-sequence-autoencoder":
        raise ValueError("Preprocessing file is not an autoencoder configuration.")
    preprocessor = SequencePreprocessor.from_dict(payload["preprocessing"])
    return AutoencoderSequencePredictor(
        model=tf.keras.models.load_model(model_path),
        preprocessor=preprocessor,
        threshold=float(payload["reconstruction_error_threshold"]),
        minimum_context_events=minimum_context_events,
    )
