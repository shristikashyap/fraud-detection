from datetime import datetime, time, timedelta, timezone

import pytest
import mongomock

from fraud_detection.api.app import create_app
from fraud_detection.api.database import MongoDatabase
from fraud_detection.api.runtime import APIComponents
from fraud_detection.behavioral.baseline import BaselineConfig, BaselineStore


class FixedClassifier:
    def __init__(self, probability=0.02):
        self.probability = probability
        self.calls = []

    def predict(self, transaction):
        self.calls.append(transaction)
        return {
            "fraud_probability": self.probability,
            "risk_category": "LOW RISK",
            "prediction": "NOT FRAUD",
            "decision_threshold": 0.2,
        }


class FixedLSTM:
    def predict(self, activities):
        return {
            "status": "available",
            "event_count": len(activities),
            "anomaly_score": 0.8,
            "threshold": 0.5,
            "behavioral_anomaly": True,
            "is_fraud_decision": False,
        }


class FixedAutoencoder:
    def predict(self, activities):
        return {
            "status": "available",
            "event_count": len(activities),
            "reconstruction_error": 2.0,
            "threshold": 1.0,
            "behavioral_anomaly": True,
            "is_fraud_decision": False,
        }


def activity(
    user_id="user-a",
    day=0,
    hour=9,
    amount=40.0,
    frequency=2,
    device_id="device-a",
    location="Boston",
):
    timestamp = datetime(2026, 1, 1, hour, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "user_id": user_id,
        "timestamp": timestamp.isoformat(),
        "login_time": time(hour, 0).isoformat(),
        "transaction_amount": amount,
        "transaction_frequency": frequency,
        "device_type": "mobile",
        "device_id": device_id,
        "location": location,
        "transaction_type": "online",
        "activity_type": "login_and_transaction",
        "transaction_status": "completed",
    }


def build_app(
    *,
    store=None,
    classifier=None,
    lstm=None,
    autoencoder=None,
    config=None,
):
    app_config = {
        "TESTING": True,
        "BASELINE_CONFIG": BaselineConfig(
            minimum_trusted_activities=4,
            minimum_history_days=3,
            minimum_sequence_transitions=2,
        ),
    }
    if config:
        app_config.update(config)
    components = APIComponents(
        predictor=classifier or FixedClassifier(),
        lstm_analyzer=lstm,
        autoencoder_analyzer=autoencoder,
        baseline_store=store or BaselineStore(),
        database=MongoDatabase(
            uri=None,
            database_name="fraud-detection-test",
            client=mongomock.MongoClient(tz_aware=True),
        ),
    )
    return create_app(app_config, components=components)


def test_health_endpoint_reports_loaded_component_availability():
    response = build_app(
        classifier=FixedClassifier(),
        lstm=FixedLSTM(),
        autoencoder=FixedAutoencoder(),
    ).test_client().get("/api/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["phase"] == "mongodb-persistence"
    assert body["cold_start_classifier_available"] is True
    assert body["model_inference_available"] is True
    assert body["models"]["lstm"]["available"] is True
    assert body["models"]["autoencoder"]["available"] is True
    assert body["database"]["status"] == "connected"


def test_missing_model_artifacts_are_reported_without_fabricated_inference(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "MODEL_DIR": tmp_path / "empty-model-directory",
            "BASELINE_STORE_PATH": tmp_path / "missing-profiles.json",
        }
    )
    client = app.test_client()

    health = client.get("/api/health").get_json()
    assert health["status"] == "degraded"
    assert health["cold_start_classifier_available"] is False
    assert all(not model["available"] for model in health["models"].values())

    response = client.post(
        "/api/analyze",
        json={"activity": activity()},
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["classifier"]["status"] == "unavailable"
    assert body["risk_assessment"]["status"] == "unavailable"
    assert body["risk_assessment"]["risk_score"] is None
    assert body["recommended_action"] == "manual_review_required"


def test_unconfigured_mongodb_keeps_api_alive_and_reports_persistence_failure():
    app = create_app(
        {"TESTING": True},
        components=APIComponents(
            predictor=FixedClassifier(),
            baseline_store=BaselineStore(),
            database=MongoDatabase(uri=None, database_name="fraud_detection"),
        ),
    )
    client = app.test_client()

    health = client.get("/api/health").get_json()
    analysis = client.post(
        "/api/analyze",
        json={"activity": activity()},
        headers={"Idempotency-Key": "offline-analysis-01"},
    )
    reports = client.get("/api/reports")

    assert health["backend_available"] is True
    assert health["status"] == "degraded"
    assert health["database"]["status"] == "not_configured"
    assert analysis.status_code == 200
    assert analysis.get_json()["risk_assessment"]["status"] == "available"
    assert analysis.get_json()["persistence"]["status"] == "unavailable"
    assert reports.status_code == 503
    assert reports.get_json()["error"]["code"] == "database_unavailable"


def test_new_user_uses_classifier_and_untrusted_activity_does_not_create_profile():
    classifier = FixedClassifier(probability=0.02)
    client = build_app(classifier=classifier).test_client()

    response = client.post(
        "/api/analyze",
        json={"activity": activity()},
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["user_id"] == "user-a"
    assert body["behavioral"]["state"] == "NEW USER"
    assert body["behavioral"]["analysis_active"] is False
    assert body["behavioral"]["activity_learned"] is False
    assert body["classifier"]["status"] == "available"
    assert body["lstm"]["status"] == "not_active_for_user_state"
    assert body["autoencoder"]["status"] == "not_active_for_user_state"
    assert body["risk_assessment"]["status"] == "available"
    assert body["risk_assessment"]["risk_category"] == "LOW RISK"
    assert len(classifier.calls) == 1
    assert client.get("/api/users").get_json()["count"] == 0


def test_idempotency_key_returns_saved_result_without_repeating_inference():
    classifier = FixedClassifier()
    client = build_app(classifier=classifier).test_client()

    first = client.post(
        "/api/analyze",
        json={"activity": activity()},
        headers={"Idempotency-Key": "retry-attempt-01"},
    )
    second = client.post(
        "/api/analyze",
        json={"activity": activity()},
        headers={"Idempotency-Key": "retry-attempt-01"},
    )
    conflicting = client.post(
        "/api/analyze",
        json={"activity": activity(amount=80.0)},
        headers={"Idempotency-Key": "retry-attempt-01"},
    )

    assert first.status_code == second.status_code == 200
    assert conflicting.status_code == 409
    assert conflicting.get_json()["error"]["code"] == "idempotency_conflict"
    assert first.get_json()["request_id"] == second.get_json()["request_id"]
    assert first.get_json()["persistence"]["status"] == "saved"
    assert len(classifier.calls) == 1
    assert client.get("/api/reports").get_json()["assessments_total"] == 1


def test_learning_user_keeps_initial_classifier_but_excludes_behavior_signals():
    store = BaselineStore()
    store.add_trusted(activity(day=0))
    client = build_app(store=store, lstm=FixedLSTM(), autoencoder=FixedAutoencoder()).test_client()

    response = client.post(
        "/api/analyze",
        json={"activity": activity(day=1)},
    )

    body = response.get_json()
    assert body["behavioral"]["state"] == "LEARNING USER"
    assert body["behavioral"]["analysis_active"] is False
    assert body["classifier"]["status"] == "available"
    assert body["lstm"]["status"] == "not_active_for_user_state"
    assert body["autoencoder"]["status"] == "not_active_for_user_state"
    assert store.get("user-a").trusted_activity_count == 1


def test_established_user_runs_all_components_and_records_review_alert():
    store = BaselineStore()
    for day in range(5):
        store.add_trusted(activity(day=day))
    app = build_app(
        store=store,
        lstm=FixedLSTM(),
        autoencoder=FixedAutoencoder(),
    )
    client = app.test_client()
    current = activity(
        day=7,
        hour=3,
        amount=1_000,
        frequency=20,
        device_id="new-device",
        location="Unknown",
    )
    history = [activity(day=day) for day in (4, 5, 6)]

    response = client.post(
        "/api/analyze",
        json={"activity": current, "recent_activities": history},
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["behavioral"]["state"] == "ESTABLISHED USER"
    assert body["classifier"]["status"] == "available"
    assert body["lstm"]["status"] == "available"
    assert body["autoencoder"]["status"] == "available"
    assert {item["code"] for item in body["behavioral"]["deviations"]} >= {
        "unusual_login_time",
        "new_device",
        "unexpected_location",
        "abnormal_transaction_amount",
        "unusual_transaction_frequency",
    }
    assert body["risk_assessment"]["available_signal_count"] == 4
    assert body["risk_assessment"]["risk_category"] in {"MEDIUM RISK", "HIGH RISK"}
    assert body["risk_assessment"]["fraud_determination"] == "not_made"
    assert body["risk_assessment"]["confirmed_fraud"] is None
    assert body["recommended_action"] == "additional_verification"
    assert body["behavioral"]["activity_learned"] is False
    assert store.get("user-a").trusted_activity_count == 5

    alerts = client.get("/api/alerts").get_json()
    assert alerts["count"] == 1
    assert alerts["alerts"][0]["user_id"] == "user-a"
    assert "transaction" not in alerts["alerts"][0]
    report = client.get("/api/reports").get_json()
    assert report["assessments_total"] == 1
    assert report["retained_alerts"] == 1


def test_profile_endpoints_return_summary_without_raw_device_or_location_values():
    store = BaselineStore()
    for day in range(5):
        store.add_trusted(activity(day=day))
    client = build_app(store=store).test_client()

    users = client.get("/api/users").get_json()
    profile_response = client.get("/api/users/user-a")
    profile = profile_response.get_json()

    assert users["count"] == 1
    assert users["users"][0]["known_device_count"] == 1
    assert profile_response.status_code == 200
    assert profile["state"] == "ESTABLISHED USER"
    assert profile["trusted_activity_count"] == 5
    assert "device-a" not in str(profile)
    assert "Boston" not in str(profile)
    assert "transaction_amounts" not in profile

    assert client.get("/api/users/missing").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"activity": {"user_id": "user-a"}},
        {"activity": {**activity(), "transaction_amount": -1}},
        {"activity": {**activity(), "timestamp": "2026-01-01T09:00:00"}},
        {"activity": {**activity(), "transaction_frequency": 1.5}},
        {"activity": {**activity(), "synthetic_suspicious_label": True}},
        {"activity": activity(), "verified": True},
        {"activity": activity(), "recent_activities": [activity("user-b")]},
        {
            "activity": activity(day=1),
            "recent_activities": [activity(day=2)],
        },
        {
            "activity": activity(day=3),
            "recent_activities": [activity(day=2), activity(day=1)],
        },
    ],
)
def test_analyze_rejects_invalid_or_untrusted_request_fields(payload):
    response = build_app().test_client().post("/api/analyze", json=payload)

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_analyze_requires_json_and_handles_malformed_json():
    client = build_app().test_client()

    not_json = client.post("/api/analyze", data="not json")
    malformed = client.post(
        "/api/analyze",
        data="{",
        content_type="application/json",
    )

    assert not_json.status_code == 400
    assert malformed.status_code == 400
    assert all("error" in result.get_json() for result in (not_json, malformed))


def test_alert_limit_validation_and_cors_preflight():
    app = build_app()
    client = app.test_client()

    invalid_limit = client.get("/api/alerts?limit=0")
    preflight = client.options(
        "/api/analyze",
        headers={"Origin": "http://localhost:5173"},
    )
    disallowed = client.get(
        "/api/health",
        headers={"Origin": "https://untrusted.example"},
    )

    assert invalid_limit.status_code == 400
    assert preflight.status_code == 200
    assert preflight.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert "Idempotency-Key" in preflight.headers["Access-Control-Allow-Headers"]
    assert "Access-Control-Allow-Origin" not in disallowed.headers


def test_transaction_history_is_paginated():
    client = build_app().test_client()
    for day in range(3):
        response = client.post("/api/analyze", json={"activity": activity(day=day)})
        assert response.status_code == 200

    page = client.get("/api/transactions?page=1&per_page=2").get_json()

    assert page["count"] == 3
    assert page["page"] == 1
    assert page["per_page"] == 2
    assert page["total_pages"] == 2
    assert len(page["transactions"]) == 2


def test_activity_and_recent_activity_limits_are_enforced():
    client = build_app(config={"MAX_RECENT_ACTIVITIES": 2}).test_client()
    response = client.post(
        "/api/analyze",
        json={
            "activity": activity(day=4),
            "recent_activities": [
                activity(day=0),
                activity(day=1),
                activity(day=2),
            ],
        },
    )

    assert response.status_code == 400
