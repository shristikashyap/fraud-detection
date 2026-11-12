"""Trusted-only per-user behavioral profiles and configurable readiness."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class BaselineConfig:
    minimum_trusted_activities: int = 20
    minimum_history_days: float = 7.0
    typical_login_hour_tolerance: int = 1
    transaction_frequency_deviation: float = 2.0
    amount_mad_multiplier: float = 6.0
    minimum_sequence_transitions: int = 5
    multiple_anomaly_review_threshold: int = 2

    def __post_init__(self) -> None:
        if self.minimum_trusted_activities < 1:
            raise ValueError("minimum_trusted_activities must be positive.")
        if self.minimum_history_days < 0:
            raise ValueError("minimum_history_days must not be negative.")
        if not 0 <= self.typical_login_hour_tolerance <= 12:
            raise ValueError("typical_login_hour_tolerance must be between 0 and 12.")
        if self.transaction_frequency_deviation <= 1:
            raise ValueError("transaction_frequency_deviation must be greater than 1.")
        if self.amount_mad_multiplier <= 0:
            raise ValueError("amount_mad_multiplier must be positive.")
        if self.minimum_sequence_transitions < 1:
            raise ValueError("minimum_sequence_transitions must be positive.")
        if self.multiple_anomaly_review_threshold < 1:
            raise ValueError("multiple_anomaly_review_threshold must be positive.")


@dataclass
class UserBaseline:
    user_id: str
    trusted_activity_count: int = 0
    first_activity_at: str | None = None
    last_activity_at: str | None = None
    transaction_amounts: list[float] = field(default_factory=list)
    login_hours: Counter[int] = field(default_factory=Counter)
    device_counts: Counter[str] = field(default_factory=Counter)
    location_counts: Counter[str] = field(default_factory=Counter)
    transaction_frequencies: list[float] = field(default_factory=list)
    activity_counts: Counter[str] = field(default_factory=Counter)
    transition_counts: Counter[str] = field(default_factory=Counter)
    last_activity_type: str | None = None

    @property
    def history_days(self) -> float:
        if self.first_activity_at is None or self.last_activity_at is None:
            return 0.0
        start = datetime.fromisoformat(self.first_activity_at)
        end = datetime.fromisoformat(self.last_activity_at)
        return max(0.0, (end - start).total_seconds() / 86_400)

    @property
    def average_transaction_amount(self) -> float | None:
        if not self.transaction_amounts:
            return None
        return sum(self.transaction_amounts) / len(self.transaction_amounts)

    @property
    def transaction_amount_range(self) -> dict[str, float] | None:
        if not self.transaction_amounts:
            return None
        return {
            "minimum": min(self.transaction_amounts),
            "maximum": max(self.transaction_amounts),
        }

    @property
    def average_transaction_frequency(self) -> float | None:
        if not self.transaction_frequencies:
            return None
        return sum(self.transaction_frequencies) / len(self.transaction_frequencies)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["login_hours"] = dict(self.login_hours)
        values["device_counts"] = dict(self.device_counts)
        values["location_counts"] = dict(self.location_counts)
        values["activity_counts"] = dict(self.activity_counts)
        values["transition_counts"] = dict(self.transition_counts)
        values["history_days"] = self.history_days
        values["average_transaction_amount"] = self.average_transaction_amount
        values["transaction_amount_range"] = self.transaction_amount_range
        values["average_transaction_frequency"] = self.average_transaction_frequency
        return values

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> UserBaseline:
        """Restore a profile document while ignoring derived summary fields."""
        return cls(
            user_id=str(item["user_id"]),
            trusted_activity_count=int(item["trusted_activity_count"]),
            first_activity_at=item.get("first_activity_at"),
            last_activity_at=item.get("last_activity_at"),
            transaction_amounts=[
                float(value) for value in item.get("transaction_amounts", [])
            ],
            login_hours=Counter(
                {
                    int(key): int(value)
                    for key, value in item.get("login_hours", {}).items()
                }
            ),
            device_counts=Counter(
                {
                    str(key): int(value)
                    for key, value in item.get("device_counts", {}).items()
                }
            ),
            location_counts=Counter(
                {
                    str(key): int(value)
                    for key, value in item.get("location_counts", {}).items()
                }
            ),
            transaction_frequencies=[
                float(value) for value in item.get("transaction_frequencies", [])
            ],
            activity_counts=Counter(
                {
                    str(key): int(value)
                    for key, value in item.get("activity_counts", {}).items()
                }
            ),
            transition_counts=Counter(
                {
                    str(key): int(value)
                    for key, value in item.get("transition_counts", {}).items()
                }
            ),
            last_activity_type=item.get("last_activity_type"),
        )


class BaselineStore:
    """In-memory baselines with explicit trusted-only updates and JSON persistence."""

    def __init__(self) -> None:
        self._profiles: dict[str, UserBaseline] = {}

    def get(self, user_id: str) -> UserBaseline | None:
        return self._profiles.get(user_id)

    def users(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def load_profiles(self, profiles: Iterable[Mapping[str, Any]]) -> None:
        """Load stored profiles without replacing any profile already present."""
        restored: dict[str, UserBaseline] = {}
        for item in profiles:
            profile = UserBaseline.from_dict(item)
            if profile.user_id in self._profiles or profile.user_id in restored:
                raise ValueError(f"Duplicate behavioral baseline for user {profile.user_id}.")
            restored[profile.user_id] = profile
        self._profiles.update(restored)

    def add_trusted(self, activity: Mapping[str, Any]) -> UserBaseline:
        """Add a caller-verified activity; callers must gate suspicious events."""
        normalized = normalize_activity(activity)
        user_id = normalized["user_id"]
        timestamp = normalized["timestamp"]
        profile = self._profiles.setdefault(user_id, UserBaseline(user_id=user_id))

        if profile.last_activity_at is not None:
            previous = datetime.fromisoformat(profile.last_activity_at)
            if timestamp < previous:
                raise ValueError("Trusted activities must be added in timestamp order.")
            if profile.last_activity_type is not None:
                transition = _transition_key(
                    profile.last_activity_type, normalized["activity_type"]
                )
                profile.transition_counts[transition] += 1

        timestamp_text = timestamp.isoformat()
        if profile.first_activity_at is None:
            profile.first_activity_at = timestamp_text
        profile.last_activity_at = timestamp_text
        profile.trusted_activity_count += 1

        amount = normalized["transaction_amount"]
        if amount is not None:
            profile.transaction_amounts.append(amount)
        login_hour = normalized["login_hour"]
        if login_hour is not None:
            profile.login_hours[login_hour] += 1
        if normalized["device_id"] is not None:
            profile.device_counts[normalized["device_id"]] += 1
        if normalized["location"] is not None:
            profile.location_counts[normalized["location"]] += 1
        frequency = normalized["transaction_frequency"]
        if frequency is not None:
            profile.transaction_frequencies.append(float(frequency))
        profile.activity_counts[normalized["activity_type"]] += 1
        profile.last_activity_type = normalized["activity_type"]
        return profile

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "profiles": [profile.to_dict() for profile in self._profiles.values()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> BaselineStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or not isinstance(
            payload.get("profiles"), list
        ):
            raise ValueError("Unsupported or malformed behavioral baseline file.")
        store = cls()
        store.load_profiles(payload["profiles"])
        return store


def normalize_activity(activity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an activity record without mutating its input."""
    user_id = _required_text(activity, "user_id")
    activity_type = _required_text(activity, "activity_type")
    timestamp = _parse_timestamp(activity.get("timestamp"))
    amount = _optional_number(activity.get("transaction_amount"), "transaction_amount")
    if amount is not None and amount < 0:
        raise ValueError("transaction_amount must not be negative.")
    frequency = _optional_number(
        activity.get("transaction_frequency"), "transaction_frequency"
    )
    if frequency is not None and (frequency < 0 or not frequency.is_integer()):
        raise ValueError("transaction_frequency must be a non-negative integer.")
    login_hour = _parse_login_hour(activity.get("login_time"))

    return {
        "user_id": user_id,
        "timestamp": timestamp,
        "activity_type": activity_type,
        "transaction_amount": amount,
        "transaction_frequency": int(frequency) if frequency is not None else None,
        "login_hour": login_hour,
        "device_id": _optional_text(activity.get("device_id")),
        "location": _optional_text(activity.get("location")),
    }


def _required_text(activity: Mapping[str, Any], key: str) -> str:
    value = _optional_text(activity.get(key))
    if value is None:
        raise ValueError(f"{key} is required and must be non-empty.")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None


def _optional_number(value: Any, key: str) -> float | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric when provided.") from exc
    if not (float("-inf") < number < float("inf")):
        raise ValueError(f"{key} must be finite.")
    return number


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise ValueError("timestamp must be an ISO-8601 string or datetime.")
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO-8601 value.") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _parse_login_hour(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return value.hour
    if not isinstance(value, str):
        raise ValueError("login_time must be an ISO time string when provided.")
    try:
        return time.fromisoformat(value).hour
    except ValueError as exc:
        raise ValueError("login_time must be a valid ISO time string.") from exc


def _transition_key(previous: str, current: str) -> str:
    return f"{previous}\u001f{current}"
