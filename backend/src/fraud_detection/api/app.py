"""Flask REST API for transaction, behavioral, and combined risk analysis."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import logging
import math
import re
from typing import Any
from uuid import uuid4

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, HTTPException

from fraud_detection.api.config import default_config
from fraud_detection.api.database import DatabaseUnavailable
from fraud_detection.api.persistence import IdempotencyConflict
from fraud_detection.api.runtime import APIComponents, APIRuntime, create_runtime
from fraud_detection.api.validation import (
    RequestValidationError,
    validate_analyze_payload,
)
from fraud_detection.behavioral.baseline import UserBaseline

logger = logging.getLogger(__name__)
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def create_app(
    config: Mapping[str, Any] | None = None,
    *,
    components: APIComponents | None = None,
) -> Flask:
    """Create the API app and load only server-configured model artifacts."""
    app = Flask(__name__)
    app.config.from_mapping(default_config())
    if config is not None:
        app.config.update(config)
    runtime = create_runtime(app.config, components)
    app.extensions["fraud_detection_runtime"] = runtime

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        allowed = app.config["CORS_ALLOWED_ORIGINS"]
        allowed_origins = (
            {item.strip() for item in allowed.split(",") if item.strip()}
            if isinstance(allowed, str)
            else set(allowed)
        )
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Idempotency-Key"
            )
            response.vary.add("Origin")
        return response

    @app.errorhandler(RequestValidationError)
    def invalid_request(error: RequestValidationError):
        return _error("invalid_request", str(error), 400)

    @app.errorhandler(DatabaseUnavailable)
    def database_unavailable(error: DatabaseUnavailable):
        return _error("database_unavailable", str(error), 503)

    @app.errorhandler(IdempotencyConflict)
    def idempotency_conflict(error: IdempotencyConflict):
        return _error("idempotency_conflict", str(error), 409)

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        status = error.code or 500
        code = (error.name or "http_error").lower().replace(" ", "_")
        return _error(code, error.description, status)

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception):
        logger.exception("Unhandled API error.", exc_info=error)
        return _error(
            "internal_error",
            "The request could not be completed.",
            500,
        )

    @app.get("/api/health")
    def health():
        database_status = runtime.database.check_health()
        classifier_available = bool(
            runtime.model_status.get("classifier", {}).get("available")
        )
        any_model_available = any(
            item.get("available") is True for item in runtime.model_status.values()
        )
        database_available = database_status["available"] is True
        return jsonify(
            {
                "status": (
                    "ok"
                    if classifier_available and database_available
                    else "degraded"
                ),
                "backend_available": True,
                "phase": "mongodb-persistence",
                "model_inference_available": any_model_available,
                "cold_start_classifier_available": classifier_available,
                "models": runtime.model_status,
                "database": database_status,
                "behavioral_profiles_available": runtime.profiles_loaded,
                "behavioral_profile_count": (
                    len(runtime.baseline_store.users())
                    if runtime.profiles_loaded
                    else None
                ),
            }
        )

    @app.post("/api/analyze")
    def analyze():
        if not request.is_json:
            raise RequestValidationError(
                "Content-Type must be application/json."
            )
        try:
            payload = request.get_json()
        except BadRequest as exc:
            raise RequestValidationError("Request body must contain valid JSON.") from exc
        activity, recent_activities = validate_analyze_payload(
            payload,
            max_recent_activities=int(app.config["MAX_RECENT_ACTIVITIES"]),
        )
        idempotency_key = request.headers.get("Idempotency-Key")
        request_fingerprint = _request_fingerprint(activity, recent_activities)
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not _IDEMPOTENCY_KEY.fullmatch(idempotency_key):
                raise RequestValidationError(
                    "'Idempotency-Key' must be 1–128 alphanumeric or . _ : - characters."
                )
            idempotency_key = hashlib.sha256(
                idempotency_key.encode("utf-8")
            ).hexdigest()
            try:
                existing = runtime.analysis_registry.get_idempotent_analysis(
                    idempotency_key,
                    request_fingerprint=request_fingerprint,
                )
            except DatabaseUnavailable:
                logger.warning(
                    "Idempotency lookup is unavailable; analysis will not be durable."
                )
            else:
                if existing is not None:
                    return jsonify(existing)

        # The API has no trusted-review workflow. Incoming events are inference
        # inputs only and must never be treated as verified baseline activity.
        behavioral = runtime.behavioral_service.assess(
            activity,
            verified=False,
            recent_activities=recent_activities,
        )
        if not runtime.profiles_loaded:
            behavioral["state"] = "PROFILE STORAGE UNAVAILABLE"
            behavioral["behavioral_analysis_active"] = False
        risk = runtime.risk_assessor.assess(behavioral)
        if not runtime.profiles_loaded:
            risk["limitations"] = [
                *risk.get("limitations", []),
                "Trusted behavioral profiles could not be loaded; personalized baseline state is unknown.",
            ]
        response: dict[str, object] = {
            "request_id": uuid4().hex,
            "user_id": activity["user_id"],
            "transaction": {
                "timestamp": activity["timestamp"],
                "amount": activity.get("transaction_amount"),
                "frequency": activity.get("transaction_frequency"),
                "type": activity.get("transaction_type"),
                "status": activity.get("transaction_status"),
            },
            "classifier": behavioral["initial_fraud_classifier"],
            "behavioral": {
                "state": behavioral["state"],
                "analysis_active": behavioral["behavioral_analysis_active"],
                "deviations": behavioral["behavioral_anomalies"],
                "requires_additional_verification": behavioral[
                    "requires_additional_verification"
                ],
                "activity_learned": False,
            },
            "lstm": behavioral["lstm_sequence_analysis"],
            "autoencoder": behavioral["autoencoder_analysis"],
            "risk_assessment": risk,
            "explanations": risk["explanations"],
            "recommended_action": risk["recommendation"],
        }
        try:
            stored_response, _ = runtime.analysis_registry.record_analysis(
                activity,
                response,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )
        except DatabaseUnavailable:
            logger.warning(
                "Analysis %s completed but MongoDB persistence failed.",
                response["request_id"],
            )
            response["transaction_id"] = response["request_id"]
            response["persistence"] = {
                "status": "unavailable",
                "message": "Analysis completed but could not be saved to persistent storage.",
            }
            return jsonify(response)
        return jsonify(stored_response)

    @app.get("/api/users")
    def users():
        if not runtime.profiles_loaded:
            raise DatabaseUnavailable(
                "Behavioral profiles could not be loaded from MongoDB."
            )
        profiles = [
            _public_profile(runtime.baseline_store.get(user_id), runtime)
            for user_id in runtime.baseline_store.users()
        ]
        return jsonify({"users": profiles, "count": len(profiles)})

    @app.get("/api/users/<user_id>")
    def user_profile(user_id: str):
        if not runtime.profiles_loaded:
            raise DatabaseUnavailable(
                "Behavioral profiles could not be loaded from MongoDB."
            )
        if not user_id or len(user_id) > 128 or user_id.strip() != user_id:
            raise RequestValidationError("user_id must be a non-empty ID of at most 128 characters.")
        profile = runtime.baseline_store.get(user_id)
        if profile is None:
            return _error("user_not_found", "No trusted profile exists for this user.", 404)
        return jsonify(_public_profile(profile, runtime))

    @app.get("/api/alerts")
    def alerts():
        page, per_page = _pagination(
            request.args.get("page", "1"),
            request.args.get("per_page", request.args.get("limit", "50")),
        )
        return jsonify(
            runtime.analysis_registry.list_alerts(page=page, per_page=per_page)
        )

    @app.get("/api/transactions")
    def transactions():
        page, per_page = _pagination(
            request.args.get("page", "1"),
            request.args.get("per_page", "20"),
        )
        return jsonify(
            runtime.analysis_registry.list_transactions(
                page=page,
                per_page=per_page,
            )
        )

    @app.get("/api/reports")
    def reports():
        return jsonify(runtime.analysis_registry.report())

    return app


def _public_profile(profile: UserBaseline | None, runtime: APIRuntime) -> dict[str, object]:
    if profile is None:
        raise ValueError("Cannot serialize a missing behavioral profile.")
    config = runtime.behavioral_service.config
    established = (
        profile.trusted_activity_count >= config.minimum_trusted_activities
        and profile.history_days >= config.minimum_history_days
    )
    typical_hours: list[int] = []
    if profile.login_hours:
        maximum = max(profile.login_hours.values())
        typical_hours = sorted(
            hour for hour, count in profile.login_hours.items() if count == maximum
        )
    return {
        "user_id": profile.user_id,
        "state": "ESTABLISHED USER" if established else "LEARNING USER",
        "trusted_activity_count": profile.trusted_activity_count,
        "history_days": profile.history_days,
        "average_transaction_amount": profile.average_transaction_amount,
        "transaction_amount_range": profile.transaction_amount_range,
        "average_transaction_frequency": profile.average_transaction_frequency,
        "typical_login_hours": typical_hours,
        "known_device_count": len(profile.device_counts),
        "known_location_count": len(profile.location_counts),
    }


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


def _pagination(page_value: str, per_page_value: str) -> tuple[int, int]:
    if not page_value.isdecimal() or int(page_value) < 1:
        raise RequestValidationError("'page' must be a positive integer.")
    if not per_page_value.isdecimal() or not 1 <= int(per_page_value) <= 100:
        raise RequestValidationError("'per_page' must be between 1 and 100.")
    return int(page_value), int(per_page_value)


def _request_fingerprint(
    activity: Mapping[str, Any],
    recent_activities: list[Mapping[str, Any]],
) -> str:
    payload = json.dumps(
        {"activity": activity, "recent_activities": recent_activities},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
