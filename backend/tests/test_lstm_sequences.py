from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fraud_detection.behavioral.data import generate_synthetic_activities
from fraud_detection.behavioral.sequential.preprocessing import SequencePreprocessor
from fraud_detection.behavioral.sequential.prediction import LSTMSequencePredictor
from fraud_detection.behavioral.sequential.sequences import (
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    build_event_features,
    build_sequence_windows,
    user_aware_split,
)
from fraud_detection.behavioral.service import BehavioralLearningService
from fraud_detection.behavioral.baseline import BaselineConfig, BaselineStore


class FixedScoreModel:
    def __init__(self, score=0.8):
        self.score = score
        self.seen_shape = None

    def predict(self, values, *, verbose=0):
        self.seen_shape = values.shape
        return np.asarray([[self.score]], dtype=np.float32)


class InitialPredictor:
    def predict(self, transaction):
        return {
            "fraud_probability": 0.01,
            "risk_category": "LOW RISK",
            "prediction": "NOT FRAUD",
            "decision_threshold": 0.2,
        }


class FixedSequenceAnalyzer:
    def predict(self, activities):
        return {
            "status": "available",
            "event_count": len(activities),
            "anomaly_score": 0.9,
            "threshold": 0.7,
            "behavioral_anomaly": True,
            "signal_type": "synthetic-supervised-sequence-anomaly",
            "is_fraud_decision": False,
        }


def _activity(user_id, day, *, amount=25.0, activity_type="login_and_transaction"):
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
        "activity_type": activity_type,
        "transaction_status": "completed",
        "is_synthetic": True,
        "scenario": "normal_history",
        LABEL_COLUMN: False,
    }


def test_sequence_windows_are_per_user_chronological_and_left_padded():
    frame = generate_synthetic_activities(normal_days=3)
    events = build_event_features(frame)
    windows = build_sequence_windows(events, sequence_length=4)

    assert len(windows) == len(frame)
    for user_id in frame["user_id"].unique():
        user_windows = [window for window in windows if window.user_id == user_id]
        assert [window.timestamp for window in user_windows] == sorted(
            window.timestamp for window in user_windows
        )
        assert all(len(window.events) <= 4 for window in user_windows)
    assert [window.label for window in windows].count(1) == 6
    assert "synthetic_suspicious_label" not in windows[-1].events[-1]
    assert "scenario" not in windows[-1].events[-1]


def test_features_capture_sequential_changes_without_identifiers_or_labels():
    frame = pd.DataFrame(
        [
            _activity("user-a", 0),
            {
                **_activity(
                    "user-a",
                    1,
                    amount=100,
                    activity_type="transfer",
                ),
                "device_id": "device-b",
                "location": "Seattle",
                "transaction_type": "transfer",
            },
        ]
    )

    events = build_event_features(frame)
    current = events[-1].values

    assert current["amount_change_log"] > 0
    assert current["interval_seconds_log"] > 0
    assert current["device_changed"] == 1
    assert current["device_new_to_user"] == 1
    assert current["location_changed"] == 1
    assert current["location_new_to_user"] == 1
    assert current["transaction_type_changed"] == 1
    assert current["activity_type_changed"] == 1
    assert current["activity_transition"] == "login_and_transaction\x1ftransfer"
    assert not {"user_id", "device_id", "location", "scenario", LABEL_COLUMN} & set(
        current
    )


def test_user_aware_split_is_disjoint_and_reproducible():
    frame = generate_synthetic_activities(normal_days=3)

    first = user_aware_split(frame, seed=13)
    second = user_aware_split(frame, seed=13)

    assert first == second
    assert [len(first[key]) for key in ("train", "validation", "test")] == [4, 1, 1]
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["validation"]) & set(first["test"]))


def test_preprocessor_uses_training_categories_only_and_round_trips():
    training = [_activity("train", index) for index in range(3)]
    heldout = _activity("heldout", 4, activity_type="heldout_only")
    training_events = build_event_features(pd.DataFrame(training))
    heldout_event = build_event_features(pd.DataFrame([heldout]))
    preprocessor = SequencePreprocessor.fit(
        [event.values for event in training_events],
        sequence_length=3,
    )

    assert preprocessor.scales["frequency_log"] == 1.0
    for field in CATEGORICAL_FEATURES:
        assert "heldout_only" not in preprocessor.categories[field]
    restored = SequencePreprocessor.from_dict(preprocessor.to_dict())
    assert restored.to_dict() == preprocessor.to_dict()

    event_vector = restored.transform_event(heldout_event[0].values)
    assert event_vector.shape == (preprocessor.feature_count,)
    assert np.isfinite(event_vector).all()


def test_reusable_predictor_returns_advisory_score_and_checks_user_sequence():
    training_events = build_event_features(
        pd.DataFrame([_activity("train", index) for index in range(4)])
    )
    preprocessor = SequencePreprocessor.fit(
        [event.values for event in training_events],
        sequence_length=4,
    )
    model = FixedScoreModel()
    predictor = LSTMSequencePredictor(model, preprocessor, threshold=0.6)
    history = [_activity("user-a", index) for index in range(3)]
    history[-1]["synthetic_suspicious_label"] = "not-a-label-used-for-inference"

    result = predictor.predict(history)

    assert result["status"] == "available"
    assert result["anomaly_score"] == pytest.approx(0.8)
    assert result["behavioral_anomaly"] is True
    assert result["is_fraud_decision"] is False
    assert model.seen_shape == (1, 4, preprocessor.feature_count)
    insufficient = predictor.predict(history[:2])
    assert insufficient["status"] == "insufficient_context"
    with pytest.raises(ValueError, match="one user"):
        predictor.predict([history[0], _activity("user-b", 1)])


def test_behavioral_service_keeps_lstm_signal_separate_from_classifier_decision():
    store = BaselineStore()
    for day in range(5):
        store.add_trusted(_activity("user-a", day))
    service = BehavioralLearningService(
        InitialPredictor(),
        config=BaselineConfig(minimum_trusted_activities=4, minimum_history_days=3),
        store=store,
        lstm_analyzer=FixedSequenceAnalyzer(),
    )
    incoming = _activity("user-a", 7)
    history = [_activity("user-a", day) for day in range(2, 5)]

    result = service.assess(incoming, recent_activities=history)

    assert result["state"] == "ESTABLISHED USER"
    assert result["lstm_sequence_analysis"]["behavioral_anomaly"] is True
    assert result["lstm_sequence_analysis"]["is_fraud_decision"] is False
    assert result["initial_fraud_classifier"]["risk_category"] == "LOW RISK"
    assert result["requires_additional_verification"] is False
    assert result["model_status"]["lstm"] == "available"


def test_lstm_training_and_artifact_persistence_on_small_synthetic_sample(tmp_path):
    pytest.importorskip("tensorflow")
    from fraud_detection.behavioral.data import write_synthetic_dataset
    from fraud_detection.behavioral.sequential.prediction import (
        load_sequence_predictor,
    )
    from fraud_detection.behavioral.sequential.training import train_and_evaluate

    data_path = write_synthetic_dataset(
        tmp_path / "synthetic.csv",
        seed=3,
        normal_days=3,
    )
    model_path = tmp_path / "lstm.keras"
    preprocessing_path = tmp_path / "preprocessing.json"
    report_path = tmp_path / "report.json"

    report = train_and_evaluate(
        data_path=data_path,
        model_path=model_path,
        preprocessing_path=preprocessing_path,
        report_path=report_path,
        sequence_length=4,
        units=4,
        epochs=1,
        batch_size=8,
        seed=3,
    )

    assert model_path.is_file()
    assert preprocessing_path.is_file()
    assert report_path.is_file()
    assert report["data_is_synthetic"] is True
    assert report["split_method"].startswith("disjoint user-aware")
    assert set(report["user_split"]["train"]).isdisjoint(
        report["user_split"]["test"]
    )
    assert report["results"]["test"]["positive_rows"] == 1
    assert "do not establish real-world fraud performance" in " ".join(
        report["limitations"]
    )

    loaded_data = pd.read_csv(data_path)
    heldout_user = report["user_split"]["test"][0]
    heldout_sequence = (
        loaded_data[loaded_data["user_id"] == heldout_user]
        .sort_values("timestamp")
        .to_dict(orient="records")
    )
    predictor = load_sequence_predictor(
        model_path,
        preprocessing_path,
        minimum_context_events=3,
    )
    prediction = predictor.predict(heldout_sequence)
    assert prediction["status"] == "available"
    assert prediction["is_fraud_decision"] is False
