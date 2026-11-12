"""Strict JSON validation for activity analysis requests."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

from fraud_detection.behavioral.baseline import normalize_activity

_BASE_FIELDS = {
    "user_id",
    "timestamp",
    "login_time",
    "transaction_amount",
    "transaction_frequency",
    "device_type",
    "device_id",
    "location",
    "transaction_type",
    "activity_type",
    "transaction_status",
}
_COMPONENT_PATTERN = re.compile(r"v(?:[1-9]|1[0-9]|2[0-8])", re.IGNORECASE)
_TEXT_FIELDS = {
    "user_id": 128,
    "device_type": 64,
    "device_id": 128,
    "location": 128,
    "transaction_type": 64,
    "activity_type": 64,
    "transaction_status": 32,
}


class RequestValidationError(ValueError):
    """A client request violates the API input contract."""


def validate_analyze_payload(
    payload: Any, *, max_recent_activities: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise RequestValidationError("The JSON body must be an object.")
    unexpected = set(payload) - {"activity", "recent_activities"}
    if unexpected:
        raise RequestValidationError(
            f"Unsupported top-level fields: {sorted(map(str, unexpected))}."
        )
    if "activity" not in payload:
        raise RequestValidationError("The 'activity' object is required.")

    activity = _validate_activity(payload["activity"])
    history = payload.get("recent_activities", [])
    if not isinstance(history, list):
        raise RequestValidationError("'recent_activities' must be a JSON array.")
    if len(history) > max_recent_activities:
        raise RequestValidationError(
            f"'recent_activities' cannot contain more than {max_recent_activities} items."
        )
    recent = [_validate_activity(item) for item in history]

    user_id = activity["user_id"]
    current_time = normalize_activity(activity)["timestamp"]
    previous_time = None
    for index, previous in enumerate(recent):
        if previous["user_id"] != user_id:
            raise RequestValidationError(
                f"recent_activities[{index}].user_id must match activity.user_id."
            )
        previous_timestamp = normalize_activity(previous)["timestamp"]
        if previous_time is not None and previous_timestamp < previous_time:
            raise RequestValidationError(
                "'recent_activities' must be in chronological order."
            )
        if previous_timestamp >= current_time:
            raise RequestValidationError(
                "Every recent activity must precede the current activity."
            )
        previous_time = previous_timestamp
    return activity, recent


def _validate_activity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RequestValidationError("Each activity must be a JSON object.")
    fields = set(value)
    unknown = [
        str(field)
        for field in fields
        if field not in _BASE_FIELDS
        and not (
            isinstance(field, str)
            and _COMPONENT_PATTERN.fullmatch(field)
        )
    ]
    normalized_component_names = [
        str(field).lower()
        for field in fields
        if isinstance(field, str) and _COMPONENT_PATTERN.fullmatch(field)
    ]
    if unknown:
        raise RequestValidationError(f"Unsupported activity fields: {sorted(unknown)}.")
    if len(normalized_component_names) != len(set(normalized_component_names)):
        raise RequestValidationError("PCA component fields must not be duplicated.")
    missing = {"user_id", "timestamp", "activity_type"} - fields
    if missing:
        raise RequestValidationError(
            f"Activity is missing required fields: {sorted(missing)}."
        )

    activity = dict(value)
    for name, limit in _TEXT_FIELDS.items():
        field_value = activity.get(name)
        if name == "user_id":
            if not isinstance(field_value, str) or not field_value.strip():
                raise RequestValidationError("'user_id' must be a non-empty string.")
        if field_value is not None:
            if not isinstance(field_value, str):
                raise RequestValidationError(f"'{name}' must be a string.")
            if len(field_value.strip()) > limit:
                raise RequestValidationError(f"'{name}' exceeds {limit} characters.")
            if any(ord(character) < 32 for character in field_value):
                raise RequestValidationError(f"'{name}' must not contain control characters.")
            activity[name] = field_value.strip()

    amount = activity.get("transaction_amount")
    if amount is not None:
        activity["transaction_amount"] = _number(
            amount, "transaction_amount", minimum=0
        )
    frequency = activity.get("transaction_frequency")
    if frequency is not None:
        frequency = _number(frequency, "transaction_frequency", minimum=0)
        if not frequency.is_integer():
            raise RequestValidationError("'transaction_frequency' must be an integer.")
        activity["transaction_frequency"] = int(frequency)

    for field in fields:
        if isinstance(field, str) and _COMPONENT_PATTERN.fullmatch(field):
            component = activity[field]
            if component is not None:
                activity[field.lower()] = _number(component, field)
                if field != field.lower():
                    del activity[field]

    for optional_field in (
        "login_time",
        "transaction_amount",
        "transaction_frequency",
        "device_type",
        "device_id",
        "location",
        "transaction_type",
        "transaction_status",
    ):
        activity.setdefault(optional_field, None)

    try:
        normalized = normalize_activity(activity)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(str(exc)) from exc
    activity["user_id"] = normalized["user_id"]
    activity["timestamp"] = normalized["timestamp"].isoformat()
    if activity.get("login_time") is not None:
        activity["login_time"] = str(activity["login_time"]).strip()
    return activity


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestValidationError(f"'{name}' must be a JSON number.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise RequestValidationError(f"'{name}' must be finite.") from exc
    if not math.isfinite(number):
        raise RequestValidationError(f"'{name}' must be finite.")
    if minimum is not None and number < minimum:
        raise RequestValidationError(f"'{name}' must be at least {minimum}.")
    return number
