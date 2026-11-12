"""Train, threshold, evaluate, and persist the normal-only sequence autoencoder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.behavioral.autoencoder.model import (
    build_autoencoder,
    masked_reconstruction_errors,
)
from fraud_detection.behavioral.sequential.metrics import sequence_metrics
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
MODEL_PATH = (
    PROJECT_ROOT / "backend" / "models" / "behavioral_autoencoder.keras"
)
PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "behavioral_autoencoder_preprocessing.json"
)
REPORT_PATH = (
    PROJECT_ROOT / "backend" / "reports" / "phase5_autoencoder_evaluation.json"
)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for Phase 5 model training. "
            "Install the backend 'tensorflow' extra."
        ) from exc
    return tf


def _partition_windows(
    frame: pd.DataFrame,
    users: tuple[str, ...],
    *,
    sequence_length: int,
) -> tuple[list[EventFeatures], list[SequenceWindow]]:
    selected = frame[frame["user_id"].astype(str).isin(users)]
    events = build_event_features(selected, require_labels=True)
    return events, build_sequence_windows(events, sequence_length=sequence_length)


def _reconstruct(model: Any, values: np.ndarray) -> np.ndarray:
    reconstructed = np.asarray(model.predict(values, verbose=0))
    if reconstructed.shape != values.shape:
        raise ValueError(
            "Autoencoder output shape does not match input shape: "
            f"{reconstructed.shape} != {values.shape}."
        )
    if not np.isfinite(reconstructed).all():
        raise ValueError("Autoencoder returned non-finite reconstructions.")
    return reconstructed


def _threshold_from_normal_validation(
    errors: np.ndarray, *, quantile: float
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("validation_quantile must be between 0 and 1.")
    if errors.size == 0:
        raise ValueError("Normal validation sequences are required for thresholding.")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Validation reconstruction errors must be finite and non-negative.")
    return float(np.quantile(errors, quantile))


def train_and_evaluate(
    *,
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    preprocessing_path: Path = PREPROCESSING_PATH,
    report_path: Path = REPORT_PATH,
    sequence_length: int = 12,
    latent_units: int = 8,
    validation_quantile: float = 0.95,
    epochs: int = 50,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, object]:
    if sequence_length < 1 or latent_units < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("Sequence length, latent size, epochs, and batch size must be positive.")
    if not 0 < validation_quantile < 1:
        raise ValueError("validation_quantile must be between 0 and 1.")

    frame = load_synthetic_events(str(data_path))
    user_splits = user_aware_split(frame, seed=seed)
    train_events, train_windows = _partition_windows(
        frame, user_splits["train"], sequence_length=sequence_length
    )
    _, validation_windows = _partition_windows(
        frame, user_splits["validation"], sequence_length=sequence_length
    )

    # This experiment models only normal training events/windows.
    normal_train_events = [event for event in train_events if event.label == 0]
    normal_train_windows = [
        window for window in train_windows if window.label == 0
    ]
    normal_validation_windows = [
        window for window in validation_windows if window.label == 0
    ]
    if not normal_train_events or not normal_train_windows:
        raise ValueError("Normal training events and windows are required.")
    if not normal_validation_windows:
        raise ValueError("Normal validation windows are required for threshold selection.")

    preprocessor = SequencePreprocessor.fit(
        [event.values for event in normal_train_events],
        sequence_length=sequence_length,
    )
    train_values = preprocessor.transform_windows(normal_train_windows)
    validation_normal_values = preprocessor.transform_windows(
        normal_validation_windows
    )

    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    model = build_autoencoder(
        sequence_length,
        preprocessor.feature_count,
        latent_units=latent_units,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )
    history = model.fit(
        train_values,
        train_values,
        validation_data=(validation_normal_values, validation_normal_values),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=0,
    )

    validation_reconstructions = _reconstruct(model, validation_normal_values)
    validation_normal_errors = masked_reconstruction_errors(
        validation_normal_values, validation_reconstructions
    )
    threshold = _threshold_from_normal_validation(
        validation_normal_errors, quantile=validation_quantile
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
        "model_type": "normal-only-sequence-autoencoder",
        "preprocessing": preprocessor.to_dict(),
        "reconstruction_error_threshold": threshold,
        "validation_quantile": validation_quantile,
        "threshold_source": "normal validation-user reconstruction errors only",
        "excluded_input_columns": [
            "user_id",
            "scenario",
            "is_synthetic",
            LABEL_COLUMN,
            "transaction_status",
            "device_id",
            "location",
        ],
        "reconstruction_error": (
            "Mean squared error averaged over valid (non-padding) timesteps and features."
        ),
    }
    preprocessing_path.write_text(
        json.dumps(preprocessing_payload, indent=2), encoding="utf-8"
    )

    train_reconstructions = _reconstruct(model, train_values)
    train_errors = masked_reconstruction_errors(train_values, train_reconstructions)
    train_report = {
        "normal_training_rows": int(len(train_errors)),
        "reconstruction_error_summary": {
            "minimum": float(train_errors.min()),
            "median": float(np.median(train_errors)),
            "maximum": float(train_errors.max()),
        },
        "threshold": threshold,
    }

    # Test users are only evaluated after the normal-only fit and validation threshold.
    _, test_windows = _partition_windows(
        frame, user_splits["test"], sequence_length=sequence_length
    )
    test_values = preprocessor.transform_windows(test_windows)
    test_labels = np.asarray(
        [window.label for window in test_windows], dtype=np.int8
    )
    test_reconstructions = _reconstruct(model, test_values)
    test_errors = masked_reconstruction_errors(test_values, test_reconstructions)
    test_report = sequence_metrics(test_labels, test_errors, threshold)
    test_report["reconstruction_error_summary"] = {
        "normal_median": (
            float(np.median(test_errors[test_labels == 0]))
            if np.any(test_labels == 0)
            else None
        ),
        "suspicious_median": (
            float(np.median(test_errors[test_labels == 1]))
            if np.any(test_labels == 1)
            else None
        ),
    }
    validation_report = {
        "normal_validation_rows": int(len(validation_normal_errors)),
        "positive_rows_used_for_threshold": 0,
        "threshold_quantile": validation_quantile,
        "threshold": threshold,
        "normal_validation_error_summary": {
            "minimum": float(validation_normal_errors.min()),
            "median": float(np.median(validation_normal_errors)),
            "maximum": float(validation_normal_errors.max()),
        },
    }

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
        "training_policy": (
            "Autoencoder and preprocessing fit on normal training-user events/windows only."
        ),
        "sequence_length": sequence_length,
        "input_feature_count": preprocessor.feature_count,
        "latent_units": latent_units,
        "feature_inputs": {
            "numeric": list(preprocessor.to_dict()["numeric_features"]),
            "categorical": list(preprocessor.to_dict()["categorical_features"]),
            "excluded": preprocessing_payload["excluded_input_columns"],
        },
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "validation_quantile": validation_quantile,
        "reconstruction_error_threshold": threshold,
        "model_artifact": str(model_path),
        "preprocessing_artifact": str(preprocessing_path),
        "results": {
            "train_normal_only": train_report,
            "validation_normal_only": validation_report,
            "test_held_out_user": test_report,
        },
        "limitations": [
            "All sequences and targets are synthetic demonstration data.",
            "Only six fictional users are available; the held-out test contains one user and one suspicious scenario.",
            "Test classification and ranking metrics are unstable and do not establish real-world fraud performance.",
            "Synthetic suspicious scenario labels are evaluation targets only, not confirmed fraud labels.",
            "The reconstruction threshold is a validation-normal quantile and does not guarantee a target false-positive rate on future users.",
        ],
        "test_access_policy": (
            "Test-user data and labels were not used for preprocessing, model fitting, "
            "early stopping, or threshold selection."
        ),
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
    parser.add_argument("--latent-units", type=int, default=8)
    parser.add_argument("--validation-quantile", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train_and_evaluate(
        data_path=args.data_path,
        model_path=args.model_path,
        preprocessing_path=args.preprocessing_path,
        report_path=args.report_path,
        sequence_length=args.sequence_length,
        latent_units=args.latent_units,
        validation_quantile=args.validation_quantile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
