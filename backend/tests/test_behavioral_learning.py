from datetime import datetime, time, timedelta, timezone

import pandas as pd

from fraud_detection.behavioral.baseline import (
    BaselineConfig,
    BaselineStore,
)
from fraud_detection.behavioral.data import (
    BEHAVIOR_COLUMNS,
    generate_synthetic_activities,
    write_synthetic_dataset,
)
from fraud_detection.behavioral.service import BehavioralLearningService


class LowRiskPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, transaction):
        self.calls.append(transaction)
        return {
            "fraud_probability": 0.02,
            "risk_category": "LOW RISK",
            "prediction": "NOT FRAUD",
            "decision_threshold": 0.2,
        }


class HighRiskPredictor(LowRiskPredictor):
    def predict(self, transaction):
        self.calls.append(transaction)
        return {
            "fraud_probability": 0.9,
            "risk_category": "HIGH RISK",
            "prediction": "FRAUD",
            "decision_threshold": 0.2,
        }


def service(*, profiles=None, config=None):
    predictor = LowRiskPredictor()
    return BehavioralLearningService(
        predictor,
        config=config or BaselineConfig(
            minimum_trusted_activities=4,
            minimum_history_days=3,
            minimum_sequence_transitions=2,
        ),
        store=profiles,
    ), predictor


def activity(
    *,
    user_id="user-a",
    day=0,
    hour=9,
    amount=40.0,
    frequency=2,
    device_id="device-a",
    location="Boston",
    activity_type="login_and_transaction",
):
    stamp = datetime(2026, 1, 1, hour, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "user_id": user_id,
        "timestamp": stamp.isoformat(),
        "login_time": time(hour, 0).isoformat(),
        "transaction_amount": amount,
        "transaction_frequency": frequency,
        "device_type": "mobile",
        "device_id": device_id,
        "location": location,
        "transaction_type": "online",
        "activity_type": activity_type,
        "transaction_status": "completed",
    }


def established_service():
    store = BaselineStore()
    for day, amount in enumerate((35.0, 40.0, 45.0, 50.0, 42.0)):
        store.add_trusted(activity(day=day, amount=amount))
    return service(profiles=store)


def test_new_user_uses_initial_classifier_without_behavioral_inference():
    analyzer, predictor = service()

    result = analyzer.assess(activity())

    assert result["state"] == "NEW USER"
    assert result["behavioral_analysis_active"] is False
    assert result["behavioral_anomalies"] == []
    assert result["initial_fraud_classifier"]["status"] == "available"
    assert len(predictor.calls) == 1
    assert result["activity_learned"] is False


def test_insufficient_trusted_history_stays_in_learning_and_updates_baseline():
    analyzer, _ = service()

    result = analyzer.assess(activity(), verified=True)

    assert result["state"] == "NEW USER"
    assert result["activity_learned"] is True
    assert result["baseline"]["trusted_activity_count"] == 1
    assert result["baseline"]["status"] == "LEARNING USER"


def test_verified_login_without_transaction_is_collected_without_fraud_label():
    analyzer, predictor = service()
    login = activity()
    login["activity_type"] = "login"
    login["transaction_amount"] = None
    login["transaction_frequency"] = None

    result = analyzer.assess(login, verified=True)

    assert result["initial_fraud_classifier"]["status"] == "not_applicable"
    assert result["activity_learned"] is True
    assert result["baseline"]["trusted_activity_count"] == 1
    assert result["baseline"]["login_hours"] == {9: 1}
    assert predictor.calls == []


def test_high_initial_risk_transaction_is_not_added_to_behavior_baseline():
    analyzer = BehavioralLearningService(
        HighRiskPredictor(),
        config=BaselineConfig(minimum_trusted_activities=4, minimum_history_days=3),
    )

    result = analyzer.assess(activity(), verified=True)

    assert result["initial_fraud_classifier"]["prediction"] == "FRAUD"
    assert result["activity_learned"] is False
    assert result["baseline"] is None


def test_established_user_normal_activity_matches_baseline_and_can_learn():
    analyzer, _ = established_service()

    result = analyzer.assess(activity(day=6, amount=44), verified=True)

    assert result["state"] == "ESTABLISHED USER"
    assert result["behavioral_analysis_active"] is True
    assert result["behavioral_anomalies"] == []
    assert result["requires_additional_verification"] is False
    assert result["activity_learned"] is True
    assert result["baseline"]["trusted_activity_count"] == 6
    assert result["baseline"]["typical_login_hours"] == [9]
    assert result["baseline"]["common_devices"] == {"device-a": 6}
    assert result["baseline"]["frequent_locations"] == {"Boston": 6}
    assert result["baseline"]["normal_activity_sequences"][
        "login_and_transaction\u001flogin_and_transaction"
    ] == 5


def test_single_deviation_is_explained_but_not_automatically_classified_as_fraud():
    analyzer, _ = established_service()

    result = analyzer.assess(activity(day=6, amount=1_000), verified=True)

    assert [item["code"] for item in result["behavioral_anomalies"]] == [
        "abnormal_transaction_amount"
    ]
    assert result["requires_additional_verification"] is False
    assert result["activity_learned"] is False
    assert result["initial_fraud_classifier"]["prediction"] == "NOT FRAUD"
    assert result["baseline"]["trusted_activity_count"] == 5


def test_multiple_contextual_anomalies_require_review_and_do_not_poison_baseline():
    analyzer, _ = established_service()

    result = analyzer.assess(
        activity(
            day=6,
            hour=3,
            amount=1_000,
            frequency=20,
            device_id="new-device",
            location="Unknown",
            activity_type="wire_transfer",
        ),
        verified=True,
    )

    anomaly_codes = {item["code"] for item in result["behavioral_anomalies"]}
    assert {
        "unusual_login_time",
        "new_device",
        "unexpected_location",
        "abnormal_transaction_amount",
        "unusual_transaction_frequency",
        "unusual_activity_sequence",
    } <= anomaly_codes
    assert result["requires_additional_verification"] is True
    assert result["activity_learned"] is False
    assert result["baseline"]["trusted_activity_count"] == 5


def test_synthetic_dataset_has_required_fields_labels_and_reproducibility(tmp_path):
    first = generate_synthetic_activities(seed=7, normal_days=4)
    second = generate_synthetic_activities(seed=7, normal_days=4)

    assert tuple(first.columns) == BEHAVIOR_COLUMNS
    assert first["is_synthetic"].all()
    assert first["user_id"].nunique() > 1
    assert {"normal_history", "multiple_contextual_anomalies"} <= set(first["scenario"])
    assert first.loc[
        first["scenario"] == "multiple_contextual_anomalies",
        "synthetic_suspicious_label",
    ].all()
    pd.testing.assert_frame_equal(first, second)

    destination = write_synthetic_dataset(tmp_path / "synthetic.csv", seed=7, normal_days=4)
    loaded = pd.read_csv(destination)
    assert len(loaded) == len(first)
    assert loaded["is_synthetic"].all()


def test_baseline_store_persists_profiles_and_requires_ordered_updates(tmp_path):
    store = BaselineStore()
    store.add_trusted(activity(day=0))
    store.add_trusted(activity(day=1))
    path = store.save(tmp_path / "profiles.json")

    restored = BaselineStore.load(path)

    assert restored.users() == ("user-a",)
    assert restored.get("user-a").trusted_activity_count == 2
    assert restored.get("user-a").average_transaction_amount == 40

    try:
        restored.add_trusted(activity(day=0, amount=80))
    except ValueError as exc:
        assert "timestamp order" in str(exc)
    else:
        raise AssertionError("Out-of-order trusted activity should be rejected.")
