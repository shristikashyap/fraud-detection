"""Train, evaluate, and persist the synthetic-data LSTM sequence model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.behavioral.sequential.metrics import (
    select_f1_threshold,
    sequence_metrics,
)
from fraud_detection.behavioral.sequential.preprocessing import SequencePreprocessor
from fraud_detection.behavioral.sequential.sequences import (
    LABEL_COLUMN,
    EventFeatures,
    SequenceWindow,
    build_event_features,
    build_sequence_windows,
    load_synthetic_events,
    user_aware_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "behavioral"
    / "synthetic_activities.csv"
)
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "behavioral_lstm.keras"
PREPROCESSING_PATH = (
    PROJECT_ROOT / "backend" / "models" / "behavioral_lstm_preprocessing.json"
)
REPORT_PATH = PROJECT_ROOT / "backend" / "reports" / "phase4_lstm_evaluation.json"


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for Phase 4 model training. "
            "Install the backend 'tensorflow' extra."
        ) from exc
    return tf


def build_lstm_model(sequence_length: int, feature_count: int, *, units: int = 16):
    if sequence_length < 1 or feature_count < 1 or units < 1:
        raise ValueError("LSTM dimensions must be positive.")
    tf = _tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, feature_count)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(units, dropout=0.1),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="behavioral_sequence_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _partition_events(
    frame: pd.DataFrame, users: tuple[str, ...], *, sequence_length: int
) -> tuple[list[EventFeatures], list[SequenceWindow]]:
    selected = frame[frame["user_id"].astype(str).isin(users)]
    events = build_event_features(selected, require_labels=True)
    windows = build_sequence_windows(events, sequence_length=sequence_length)
    if not windows:
        raise ValueError(f"No sequence examples found for users: {users}")
    return events, windows


def _arrays(windows: list[SequenceWindow], preprocessor: SequencePreprocessor):
    values = preprocessor.transform_windows(windows)
    labels = np.asarray([window.label for window in windows], dtype=np.int8)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Sequence targets must be binary.")
    return values, labels


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training sequence examples must contain both target classes.")
    total = int(counts.sum())
    return {
        0: float(total / (2 * counts[0])),
        1: float(total / (2 * counts[1])),
    }


def _scores(model: Any, values: np.ndarray) -> np.ndarray:
    outputs = np.asarray(model.predict(values, verbose=0)).reshape(-1)
    if outputs.shape[0] != values.shape[0]:
        raise ValueError("LSTM output count does not match the sequence batch.")
    if not np.isfinite(outputs).all() or ((outputs < 0) | (outputs > 1)).any():
        raise ValueError("LSTM returned invalid anomaly scores.")
    return outputs


def train_and_evaluate(
    *,
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    preprocessing_path: Path = PREPROCESSING_PATH,
    report_path: Path = REPORT_PATH,
    sequence_length: int = 12,
    units: int = 16,
    epochs: int = 40,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")

    frame = load_synthetic_events(str(data_path))
    user_splits = user_aware_split(frame, seed=seed)
    train_events, train_windows = _partition_events(
        frame, user_splits["train"], sequence_length=sequence_length
    )
    _, validation_windows = _partition_events(
        frame, user_splits["validation"], sequence_length=sequence_length
    )
    train_labels = np.asarray([window.label for window in train_windows], dtype=np.int8)
    preprocessor = SequencePreprocessor.fit(
        [event.values for event in train_events],
        sequence_length=sequence_length,
    )
    train_values, train_labels = _arrays(train_windows, preprocessor)
    validation_values, validation_labels = _arrays(
        validation_windows, preprocessor
    )

    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    model = build_lstm_model(
        sequence_length,
        preprocessor.feature_count,
        units=units,
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )
    history = model.fit(
        train_values,
        train_labels,
        validation_data=(validation_values, validation_labels),
        class_weight=_class_weights(train_labels),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[callback],
        verbose=0,
    )

    validation_scores = _scores(model, validation_values)
    threshold = select_f1_threshold(validation_labels, validation_scores)
    validation_report = sequence_metrics(
        validation_labels, validation_scores, threshold
    )

    model_path = Path(model_path)
    preprocessing_path = Path(preprocessing_path)
    report_path = Path(report_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    preprocessing_payload = {
        "format_version": 1,
        "preprocessing": preprocessor.to_dict(),
        "validation_threshold": threshold,
        "label": LABEL_COLUMN,
        "label_semantics": "Synthetic suspicious scenario target only; not a confirmed fraud label.",
        "excluded_input_columns": [
            "user_id",
            "scenario",
            "is_synthetic",
            LABEL_COLUMN,
            "transaction_status",
            "device_id",
            "location",
        ],
    }
    preprocessing_path.write_text(
        json.dumps(preprocessing_payload, indent=2), encoding="utf-8"
    )

    train_scores = _scores(model, train_values)
    train_report = sequence_metrics(train_labels, train_scores, threshold)

    # Test-user sequences and labels are accessed only after model fitting,
    # validation threshold selection, and artifact persistence are complete.
    _, test_windows = _partition_events(
        frame, user_splits["test"], sequence_length=sequence_length
    )
    test_values, test_labels = _arrays(test_windows, preprocessor)
    test_scores = _scores(model, test_values)
    test_report = sequence_metrics(test_labels, test_scores, threshold)

    report: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(data_path),
        "data_is_synthetic": True,
        "dataset_rows": int(len(frame)),
        "dataset_users": int(frame["user_id"].nunique()),
        "synthetic_label_counts": {
            "normal": int((frame[LABEL_COLUMN] == 0).sum()),
            "suspicious_scenario": int((frame[LABEL_COLUMN] == 1).sum()),
        },
        "split_method": "disjoint user-aware holdout; no user appears in multiple splits",
        "user_split": {name: list(users) for name, users in user_splits.items()},
        "sequence_length": sequence_length,
        "input_feature_count": preprocessor.feature_count,
        "feature_inputs": {
            "numeric": list(preprocessor.to_dict()["numeric_features"]),
            "categorical": list(preprocessor.to_dict()["categorical_features"]),
            "excluded": preprocessing_payload["excluded_input_columns"],
        },
        "class_weights": _class_weights(train_labels),
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "validation_threshold": threshold,
        "model_artifact": str(model_path),
        "preprocessing_artifact": str(preprocessing_path),
        "results": {
            "train": train_report,
            "validation": validation_report,
            "test": test_report,
        },
        "limitations": [
            "All sequences and targets are synthetic demonstration data.",
            "Only six synthetic users are available; the test partition contains one held-out user and one suspicious scenario.",
            "Reported metrics are unstable and do not establish real-world fraud performance.",
            "The supervised score estimates resemblance to the synthetic suspicious scenario labels, not confirmed fraud.",
            "Synthetic labels are used only as targets and excluded from input features.",
        ],
        "test_access_policy": "Test users were not used for preprocessing fit, model fit, or threshold selection.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--preprocessing-path", type=Path, default=PREPROCESSING_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train_and_evaluate(
        data_path=args.data_path,
        model_path=args.model_path,
        preprocessing_path=args.preprocessing_path,
        report_path=args.report_path,
        sequence_length=args.sequence_length,
        units=args.units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
