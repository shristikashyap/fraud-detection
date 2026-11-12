from datetime import datetime, time, timedelta, timezone

import mongomock
import pytest
from pymongo.errors import ServerSelectionTimeoutError

from fraud_detection.api.config import default_config
from fraud_detection.api.database import DatabaseUnavailable, MongoDatabase
from fraud_detection.api.persistence import (
    MongoBaselineStore,
    MongoPersistence,
    ProfileConflict,
)
from fraud_detection.behavioral.baseline import BaselineConfig
from fraud_detection.behavioral.service import BehavioralLearningService


def activity(day=0, amount=40.0):
    timestamp = datetime(2026, 2, 1, 9, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "user_id": "mongo-user",
        "timestamp": timestamp.isoformat(),
        "login_time": time(9).isoformat(),
        "transaction_amount": amount,
        "transaction_frequency": 2,
        "device_type": "mobile",
        "device_id": "device-1",
        "location": "Boston",
        "transaction_type": "online",
        "activity_type": "login_and_transaction",
        "transaction_status": "completed",
    }


def make_database(client=None):
    return MongoDatabase(
        uri=None,
        database_name="fraud-detection-test",
        client=client or mongomock.MongoClient(tz_aware=True),
    )


def response(request_id, category="MEDIUM RISK"):
    return {
        "request_id": request_id,
        "user_id": "mongo-user",
        "transaction": {
            "timestamp": activity()["timestamp"],
            "amount": 40.0,
            "frequency": 2,
            "type": "online",
            "status": "completed",
        },
        "classifier": {"status": "available", "fraud_probability": 0.6},
        "behavioral": {
            "state": "ESTABLISHED USER",
            "requires_additional_verification": category == "MEDIUM RISK",
            "deviations": [],
        },
        "lstm": {"status": "not_configured"},
        "autoencoder": {"status": "not_configured"},
        "risk_assessment": {
            "status": "available",
            "risk_score": 42.0,
            "risk_category": category,
            "available_signal_count": 1,
        },
        "explanations": ["Additional verification is recommended."],
        "recommended_action": "additional_verification",
    }


def test_database_without_uri_reports_unconfigured_without_crashing():
    database = MongoDatabase(uri=None, database_name="fraud_detection")

    assert database.health() == {
        "available": False,
        "status": "not_configured",
        "database": None,
    }
    with pytest.raises(DatabaseUnavailable):
        database.collection("transactions")


def test_database_environment_configuration_uses_requested_defaults(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://example.invalid:27017")
    monkeypatch.delenv("MONGO_DB_NAME", raising=False)

    config = default_config()

    assert config["MONGO_URI"] == "mongodb://example.invalid:27017"
    assert config["MONGO_DB_NAME"] == "fraud_detection"


def test_connection_failure_is_reported_without_exposing_connection_details():
    class FailingAdmin:
        @staticmethod
        def command(_name):
            raise ServerSelectionTimeoutError("connection refused")

    class FailingClient:
        admin = FailingAdmin()

        @staticmethod
        def __getitem__(_name):
            return mongomock.MongoClient()["fraud_detection"]

    database = MongoDatabase(
        uri=None,
        database_name="fraud_detection",
        client=FailingClient(),
    )

    assert database.health()["available"] is False
    assert database.health()["status"] == "unavailable"
    assert "connection" not in str(database.health())


def test_analysis_risk_alert_and_transaction_persist_once_with_retry_key():
    client = mongomock.MongoClient(tz_aware=True)
    database = make_database(client)
    persistence = MongoPersistence(database)

    saved, duplicate = persistence.record_analysis(
        activity(),
        response("transaction-1"),
        idempotency_key="hashed-retry-key",
    )
    retry, was_duplicate = persistence.record_analysis(
        activity(),
        response("transaction-2"),
        idempotency_key="hashed-retry-key",
    )

    assert duplicate is False
    assert was_duplicate is True
    assert retry["request_id"] == saved["request_id"] == "transaction-1"
    assert saved["persistence"]["status"] == "saved"
    assert database.collection("transactions").count_documents({}) == 1
    assert database.collection("risk_assessments").count_documents({}) == 1
    assert database.collection("alerts").count_documents({}) == 1

    transaction = database.collection("transactions").find_one({})
    assert transaction["timestamp"].utcoffset() == timedelta(0)
    assert transaction["risk_score"] == 42.0
    assert transaction["analysis_response"]["classifier"]["fraud_probability"] == 0.6

    assert persistence.report()["assessments_total"] == 1
    alert_page = persistence.list_alerts(page=1, per_page=10)
    assert alert_page["count"] == 1
    assert alert_page["alerts"][0]["transaction_id"] == "transaction-1"


def test_transactions_and_alerts_are_paginated_and_survive_runtime_recreation():
    client = mongomock.MongoClient(tz_aware=True)
    first_runtime = MongoPersistence(make_database(client))
    for index in range(3):
        first_runtime.record_analysis(
            activity(day=index),
            response(f"transaction-{index}"),
        )

    restarted_runtime = MongoPersistence(make_database(client))
    transactions = restarted_runtime.list_transactions(page=2, per_page=2)
    alerts = restarted_runtime.list_alerts(page=2, per_page=1)

    assert transactions["count"] == 3
    assert transactions["total_pages"] == 2
    assert len(transactions["transactions"]) == 1
    assert alerts["count"] == 3
    assert alerts["total_pages"] == 3
    assert len(alerts["alerts"]) == 1
    assert restarted_runtime.report()["assessments_total"] == 3


def test_transaction_history_redacts_raw_device_and_location_values():
    client = mongomock.MongoClient(tz_aware=True)
    persistence = MongoPersistence(make_database(client))
    analysis = response("transaction-private")
    analysis["behavioral"]["deviations"] = [
        {
            "field": "device_id",
            "code": "new_device",
            "observed": "private-device-id",
            "expected": ["trusted-device-id"],
            "message": "Activity used an unfamiliar device.",
        },
        {
            "field": "location",
            "code": "new_location",
            "observed": "private-location",
            "expected": ["trusted-location"],
            "message": "Activity came from an unfamiliar location.",
        },
    ]
    persistence.record_analysis(activity(), analysis)

    listed = persistence.list_transactions(page=1, per_page=10)
    deviations = listed["transactions"][0]["behavioral"]["deviations"]

    assert deviations[0]["code"] == "new_device"
    assert deviations[0]["message"] == "Activity used an unfamiliar device."
    assert "observed" not in deviations[0]
    assert "expected" not in deviations[0]
    assert "observed" not in deviations[1]
    assert "expected" not in deviations[1]


def test_trusted_profiles_persist_and_reload_after_backend_restart():
    client = mongomock.MongoClient(tz_aware=True)
    persistence = MongoPersistence(make_database(client))
    original = MongoBaselineStore(persistence)
    assert original.profiles_loaded is True

    original.add_trusted(activity())
    restarted = MongoBaselineStore(MongoPersistence(make_database(client)))

    assert restarted.profiles_loaded is True
    assert restarted.users() == ("mongo-user",)
    assert restarted.get("mongo-user").trusted_activity_count == 1
    assert restarted.get("mongo-user").transaction_amounts == [40.0]
    profile_doc = persistence.database.collection("behavioral_profiles").find_one({})
    assert profile_doc["user_status"] == "LEARNING USER"
    assert profile_doc["last_updated"].utcoffset() == timedelta(0)


def test_existing_profile_is_not_overwritten_by_stale_baseline():
    client = mongomock.MongoClient(tz_aware=True)
    persistence = MongoPersistence(make_database(client))
    current = MongoBaselineStore(persistence)
    current.add_trusted(activity())
    stale = MongoBaselineStore(MongoPersistence(make_database(client)))

    current.add_trusted(activity(day=1))
    with pytest.raises(ProfileConflict, match="changed elsewhere"):
        stale.add_trusted(activity(day=1, amount=99.0))

    restored = MongoBaselineStore(MongoPersistence(make_database(client)))
    assert restored.get("mongo-user").trusted_activity_count == 2
    assert restored.get("mongo-user").transaction_amounts == [40.0, 40.0]


def test_profile_storage_does_not_learn_an_unverified_activity():
    store = MongoBaselineStore(MongoPersistence(make_database()))
    service = BehavioralLearningService(
        predictor=None,
        store=store,
        config=BaselineConfig(minimum_trusted_activities=3),
    )
    result = service.assess(activity(), verified=False)

    assert result["activity_learned"] is False
    assert store.users() == ()


def test_service_persists_only_an_explicitly_verified_login():
    client = mongomock.MongoClient(tz_aware=True)
    store = MongoBaselineStore(MongoPersistence(make_database(client)))
    service = BehavioralLearningService(
        predictor=None,
        store=store,
        config=BaselineConfig(minimum_trusted_activities=3),
    )
    login = activity()
    login["activity_type"] = "login"
    login["transaction_amount"] = None
    login["transaction_frequency"] = None

    result = service.assess(login, verified=True)
    restored = MongoBaselineStore(
        MongoPersistence(make_database(client))
    )

    assert result["activity_learned"] is True
    assert restored.get("mongo-user").trusted_activity_count == 1
