"""Training-only normalization and categorical encoding for LSTM windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fraud_detection.behavioral.sequential.sequences import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SequenceWindow,
)


@dataclass
class SequencePreprocessor:
    sequence_length: int
    means: dict[str, float]
    scales: dict[str, float]
    categories: dict[str, list[str]]

    @classmethod
    def fit(
        cls,
        events: Sequence[Mapping[str, float | str]],
        *,
        sequence_length: int,
    ) -> SequencePreprocessor:
        if not events:
            raise ValueError("Cannot fit sequence preprocessing without training events.")
        numeric_values = np.asarray(
            [
                [float(event.get(name, 0.0)) for name in NUMERIC_FEATURES]
                for event in events
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric_values).all():
            raise ValueError("Training sequence features must be finite.")
        means_array = numeric_values.mean(axis=0)
        scales_array = numeric_values.std(axis=0)
        scales_array[np.isclose(scales_array, 0.0, atol=1e-8)] = 1.0
        categories = {
            name: sorted({str(event.get(name, "<MISSING>")) for event in events})
            for name in CATEGORICAL_FEATURES
        }
        return cls(
            sequence_length=sequence_length,
            means={name: float(means_array[index]) for index, name in enumerate(NUMERIC_FEATURES)},
            scales={name: float(scales_array[index]) for index, name in enumerate(NUMERIC_FEATURES)},
            categories=categories,
        )

    @property
    def feature_count(self) -> int:
        return len(NUMERIC_FEATURES) + sum(
            len(self.categories[name]) for name in CATEGORICAL_FEATURES
        ) + 1

    def transform_event(self, event: Mapping[str, Any]) -> np.ndarray:
        numeric = np.asarray(
            [
                (float(event.get(name, 0.0)) - self.means[name]) / self.scales[name]
                for name in NUMERIC_FEATURES
            ],
            dtype=np.float32,
        )
        categorical = []
        for name in CATEGORICAL_FEATURES:
            value = str(event.get(name, "<MISSING>"))
            vocabulary = self.categories[name]
            one_hot = np.zeros(len(vocabulary), dtype=np.float32)
            if value in vocabulary:
                one_hot[vocabulary.index(value)] = 1.0
            categorical.append(one_hot)
        return np.concatenate(
            [numeric, *categorical, np.ones(1, dtype=np.float32)]
        )

    def transform_windows(self, windows: Sequence[SequenceWindow]) -> np.ndarray:
        if not windows:
            return np.empty(
                (0, self.sequence_length, self.feature_count), dtype=np.float32
            )
        batch = np.zeros(
            (len(windows), self.sequence_length, self.feature_count),
            dtype=np.float32,
        )
        for batch_index, window in enumerate(windows):
            events = window.events[-self.sequence_length :]
            start_index = self.sequence_length - len(events)
            for offset, event in enumerate(events):
                batch[batch_index, start_index + offset] = self.transform_event(event)
        return batch

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "sequence_length": self.sequence_length,
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "means": self.means,
            "scales": self.scales,
            "categories": self.categories,
            "feature_count": self.feature_count,
            "timestep_masking": "left padding is the all-zero vector; valid timesteps include a 1 mask feature",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SequencePreprocessor:
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported sequence preprocessing format.")
        if payload.get("numeric_features") != list(NUMERIC_FEATURES):
            raise ValueError("Saved numeric sequence feature schema does not match.")
        if payload.get("categorical_features") != list(CATEGORICAL_FEATURES):
            raise ValueError("Saved categorical sequence feature schema does not match.")
        preprocessor = cls(
            sequence_length=int(payload["sequence_length"]),
            means={key: float(value) for key, value in payload["means"].items()},
            scales={key: float(value) for key, value in payload["scales"].items()},
            categories={
                key: [str(value) for value in values]
                for key, values in payload["categories"].items()
            },
        )
        if preprocessor.sequence_length < 1:
            raise ValueError("Saved sequence length must be positive.")
        if set(preprocessor.means) != set(NUMERIC_FEATURES):
            raise ValueError("Saved numeric means are incomplete.")
        if set(preprocessor.scales) != set(NUMERIC_FEATURES):
            raise ValueError("Saved numeric scales are incomplete.")
        if set(preprocessor.categories) != set(CATEGORICAL_FEATURES):
            raise ValueError("Saved categorical vocabularies are incomplete.")
        if any(value <= 0 for value in preprocessor.scales.values()):
            raise ValueError("Saved numeric feature scales must be positive.")
        return preprocessor
