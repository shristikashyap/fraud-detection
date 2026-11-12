from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fraud_detection.behavioral.autoencoder.model import (
    build_autoencoder,
    masked_reconstruction_errors,
)
from fraud_detection.behavioral.autoencoder.prediction import (
    AutoencoderSequencePredictor,
    load_autoencoder_predictor,
)
from fraud_detection.behavioral.baseline import BaselineConfig, BaselineStore
from fraud_detection.behavioral.data import write_synthetic_dataset
from fraud_detection.behavioral.sequential.preprocessing import SequencePreprocessor
from fraud_detection.behavioral.sequential.sequences import LABEL_COLUMN, build_event_features
from fraud_detection.behavioral.autoencoder.training import (
    _threshold_from_normal_validation,
    train_and_evaluate,
)
from fraud_detection.behavioral.service import BehavioralLearningService


class ZeroReconstructionModel:
    def __init__(self):
        self.seen = None

    def predict(self, values, *, verbose=0):
        self.seen = values.copy()
        return np.zeros_like(values)


class LowRiskPredictor:
    def predict(self, transaction):
        return {
            "fraud_probability": 0.01,
            "risk_category": "LOW RISK",
            "prediction": "NOT FRAUD",
            "decision_threshold": 0.2,
        }


class FixedAutoencoderAnalyzer:
    def predict(self, activities):
        return {
            "status": "available",
            "event_count": len(activities),
            "reconstruction_error": 3.2,
            "threshold": 1.5,
            "behavioral_anomaly": True,
            "signal_type": "normal-activity-reconstruction-error",
            "is_fraud_decision": False,
        }


def activity(user_id, day, *, amount=25.0):
    timestamp = datetime(2026, 1, 1, 9, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "login_time": time(9, 0).isoformat(),
        "transaction_amount": amount,
        "transaction_frequency": 2,
        "device_type": "mobile",
        "device_id": "device-a",
        "location": "Boston",
        "transaction_type": "online",
        "activity_type": "login_and_transaction",
        "transaction_status": "completed",
        "is_synthetic": True,
        "scenario": "normal_history",
        LABEL_COLUMN: False,
    }


def test_masked_reconstruction_error_ignores_left_padding():
    original = np.asarray(
        [
            [[0, 0], [1, 1], [2, 1]],
        ],
        dtype=np.float32,
    )
    reconstructed = np.asarray(
        [
            [[100, 100], [0, 0], [0, 1]],
        ],
        dtype=np.float32,
    )

    errors = masked_reconstruction_errors(original, reconstructed)

    assert errors.tolist() == pytest.approx([1.5])


def test_autoencoder_predictor_returns_reconstruction_signal_not_fraud_decision():
    normal_events = build_event_features(
        pd.DataFrame([activity("train", index) for index in range(4)])
    )
    preprocessor = SequencePreprocessor.fit(
        [event.values for event in normal_events],
        sequence_length=4,
    )
    model = ZeroReconstructionModel()
    predictor = AutoencoderSequencePredictor(
        model=model,
        preprocessor=preprocessor,
        threshold=0.1,
        minimum_context_events=3,
    )
    recent = [
        {
            key: value
            for key, value in activity("inference", index).items()
            if key not in {"scenario", "is_synthetic", LABEL_COLUMN}
        }
        for index in range(3)
    ]

    result = predictor.predict(recent)

    assert result["status"] == "available"
    assert result["reconstruction_error"] > 0
    assert result["behavioral_anomaly"] is True
    assert result["is_fraud_decision"] is False
    assert model.seen.shape == (1, 4, preprocessor.feature_count)
    assert "scenario" not in recent[-1]
    assert "synthetic_suspicious_label" not in recent[-1]

    insufficient = predictor.predict(recent[:2])
    assert insufficient["status"] == "insufficient_context"


def test_normal_validation_quantile_sets_threshold_and_validates_inputs():
    errors = np.asarray([0.1, 0.2, 0.3, 0.4])

    threshold = _threshold_from_normal_validation(errors, quantile=0.75)

    assert threshold == pytest.approx(0.325)
    with pytest.raises(ValueError, match="between 0 and 1"):
        _threshold_from_normal_validation(errors, quantile=1.0)
    with pytest.raises(ValueError, match="required"):
        _threshold_from_normal_validation(np.asarray([]), quantile=0.95)


def test_phase3_service_keeps_autoencoder_anomaly_separate():
    store = BaselineStore()
    for day in range(5):
        store.add_trusted(activity("user-a", day))
    service = BehavioralLearningService(
        LowRiskPredictor(),
        config=BaselineConfig(minimum_trusted_activities=4, minimum_history_days=3),
        store=store,
        autoencoder_analyzer=FixedAutoencoderAnalyzer(),
    )
    current = activity("user-a", 7)
    previous = [activity("user-a", day) for day in range(3, 7)]

    result = service.assess(current, recent_activities=previous)

    assert result["initial_fraud_classifier"]["risk_category"] == "LOW RISK"
    assert result["autoencoder_analysis"]["behavioral_anomaly"] is True
    assert result["autoencoder_analysis"]["is_fraud_decision"] is False
    assert result["requires_additional_verification"] is False
    assert result["model_status"]["autoencoder"] == "available"


def test_autoencoder_trains_only_normal_users_and_saves_independent_artifacts(tmp_path):
    pytest.importorskip("tensorflow")
    data_path = write_synthetic_dataset(
        tmp_path / "synthetic.csv",
        seed=17,
        normal_days=3,
    )
    model_path = tmp_path / "autoencoder.keras"
    preprocessing_path = tmp_path / "preprocessing.json"
    report_path = tmp_path / "report.json"

    report = train_and_evaluate(
        data_path=data_path,
        model_path=model_path,
        preprocessing_path=preprocessing_path,
        report_path=report_path,
        sequence_length=4,
        latent_units=4,
        validation_quantile=0.9,
        epochs=1,
        batch_size=8,
        seed=17,
    )

    assert model_path.is_file()
    assert preprocessing_path.is_file()
    assert report_path.is_file()
    assert report["data_is_synthetic"] is True
    assert "normal training-user" in report["training_policy"]
    assert report["results"]["validation_normal_only"]["positive_rows_used_for_threshold"] == 0
    assert report["results"]["test_held_out_user"]["positive_rows"] == 1
    assert report["results"]["test_held_out_user"]["confusion_matrix"]["matrix"]
    assert "do not establish real-world fraud performance" in " ".join(
        report["limitations"]
    )
    heldout_user = report["user_split"]["test"][0]
    heldout_activities = (
        pd.read_csv(data_path)
        .query("user_id == @heldout_user")
        .sort_values("timestamp")
        .to_dict(orient="records")
    )
    predictor = load_autoencoder_predictor(
        model_path,
        preprocessing_path,
        minimum_context_events=3,
    )
    prediction = predictor.predict(heldout_activities)
    assert prediction["status"] == "available"
    assert prediction["is_fraud_decision"] is False


def test_autoencoder_model_has_encoder_and_decoder_shapes():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    model = build_autoencoder(sequence_length=4, feature_count=7, latent_units=3)

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 4, 7)
    assert model.output_shape == (None, 4, 7)
    assert "latent" in {layer.name for layer in model.layers}
