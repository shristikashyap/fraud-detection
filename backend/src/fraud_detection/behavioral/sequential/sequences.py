"""Leakage-aware feature construction and per-user causal sequence windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LABEL_COLUMN = "synthetic_suspicious_label"
USER_COLUMN = "user_id"
TIMESTAMP_COLUMN = "timestamp"
NUMERIC_FEATURES = (
    "amount_log",
    "amount_change_log",
    "interval_seconds_log",
    "login_hour_sin",
    "login_hour_cos",
    "frequency_log",
    "device_changed",
    "device_new_to_user",
    "location_changed",
    "location_new_to_user",
    "transaction_type_changed",
    "activity_type_changed",
)
CATEGORICAL_FEATURES = (
    "transaction_type",
    "activity_type",
    "activity_transition",
)
REQUIRED_COLUMNS = (
    USER_COLUMN,
    TIMESTAMP_COLUMN,
    "login_time",
    "transaction_amount",
    "transaction_frequency",
    "device_id",
    "location",
    "transaction_type",
    "activity_type",
)


@dataclass(frozen=True)
class EventFeatures:
    user_id: str
    timestamp: datetime
    values: dict[str, float | str]
    label: int | None


@dataclass(frozen=True)
class SequenceWindow:
    user_id: str
    timestamp: datetime
    events: tuple[dict[str, float | str], ...]
    label: int | None


def load_synthetic_events(path: str) -> pd.DataFrame:
    """Load synthetic behavioral data and validate the event-label contract."""
    source = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Synthetic activity dataset is missing columns: {sorted(missing)}")
    if LABEL_COLUMN not in source.columns:
        raise ValueError(f"Synthetic activity dataset is missing '{LABEL_COLUMN}'.")
    if "is_synthetic" not in source.columns:
        raise ValueError("Behavioral sequence training requires explicit synthetic data.")

    synthetic = source["is_synthetic"].astype(str).str.strip().str.lower()
    if not synthetic.isin({"true", "1"}).all():
        raise ValueError("Phase 4 training accepts only explicitly synthetic events.")
    labels = source[LABEL_COLUMN].astype(str).str.strip().str.lower()
    if not labels.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Synthetic sequence labels must be binary.")
    source[LABEL_COLUMN] = labels.isin({"true", "1"}).astype("int8")
    return source


def user_aware_split(
    frame: pd.DataFrame, *, seed: int = 42
) -> dict[str, tuple[str, ...]]:
    """Assign complete users to disjoint train/validation/test partitions."""
    if USER_COLUMN not in frame:
        raise ValueError(f"Dataset is missing '{USER_COLUMN}'.")
    users = sorted(str(user) for user in frame[USER_COLUMN].dropna().unique())
    if len(users) < 3:
        raise ValueError("At least three users are required for user-aware splitting.")
    shuffled = np.random.default_rng(seed).permutation(users).tolist()
    validation_count = max(1, round(len(users) * 0.17))
    test_count = max(1, round(len(users) * 0.17))
    train_count = len(users) - validation_count - test_count
    if train_count < 1:
        raise ValueError("Not enough users remain for the training partition.")

    train_users = tuple(sorted(shuffled[:train_count]))
    validation_users = tuple(sorted(shuffled[train_count : train_count + validation_count]))
    test_users = tuple(sorted(shuffled[train_count + validation_count :]))
    if set(train_users) & set(validation_users) or set(train_users) & set(test_users):
        raise RuntimeError("User-aware partitions unexpectedly overlap.")
    return {
        "train": train_users,
        "validation": validation_users,
        "test": test_users,
    }


def build_event_features(frame: pd.DataFrame, *, require_labels: bool = True) -> list[EventFeatures]:
    """Build causal features from each user's prior events and current event only."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Behavioral events are missing columns: {sorted(missing)}")
    if require_labels and LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Behavioral events are missing '{LABEL_COLUMN}'.")

    ordered = frame.copy()
    ordered[USER_COLUMN] = ordered[USER_COLUMN].astype(str)
    ordered["_parsed_timestamp"] = pd.to_datetime(
        ordered[TIMESTAMP_COLUMN], utc=True, errors="coerce"
    )
    if ordered["_parsed_timestamp"].isna().any():
        raise ValueError("Every event must have a valid ISO-8601 timestamp.")
    ordered = ordered.sort_values(
        [USER_COLUMN, "_parsed_timestamp"], kind="mergesort"
    )

    results: list[EventFeatures] = []
    for user_id, user_frame in ordered.groupby(USER_COLUMN, sort=True):
        previous: pd.Series | None = None
        seen_devices: set[str] = set()
        seen_locations: set[str] = set()
        for _, row in user_frame.iterrows():
            timestamp: pd.Timestamp = row["_parsed_timestamp"]
            values = _feature_values(
                row,
                timestamp.to_pydatetime(),
                previous,
                seen_devices,
                seen_locations,
            )
            label = None
            if require_labels and LABEL_COLUMN in row.index and pd.notna(row[LABEL_COLUMN]):
                label = _parse_binary_label(row[LABEL_COLUMN])
            if require_labels and label is None:
                raise ValueError("Every training activity must have a binary synthetic label.")
            results.append(
                EventFeatures(
                    user_id=str(user_id),
                    timestamp=timestamp.to_pydatetime(),
                    values=values,
                    label=label,
                )
            )
            device = _optional_text(row.get("device_id"))
            location = _optional_text(row.get("location"))
            if device is not None:
                seen_devices.add(device)
            if location is not None:
                seen_locations.add(location)
            previous = row
    return results


def build_sequence_windows(
    events: Sequence[EventFeatures], *, sequence_length: int
) -> list[SequenceWindow]:
    """Create one left-padded causal window ending at every user's event."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    grouped: dict[str, list[EventFeatures]] = {}
    for event in events:
        grouped.setdefault(event.user_id, []).append(event)

    windows: list[SequenceWindow] = []
    for user_id, user_events in sorted(grouped.items()):
        user_events.sort(key=lambda event: event.timestamp)
        for index, event in enumerate(user_events):
            start = max(0, index - sequence_length + 1)
            windows.append(
                SequenceWindow(
                    user_id=user_id,
                    timestamp=event.timestamp,
                    events=tuple(item.values for item in user_events[start : index + 1]),
                    label=event.label,
                )
            )
    return windows


def _feature_values(
    row: pd.Series | Mapping[str, Any],
    timestamp: datetime,
    previous: pd.Series | None,
    seen_devices: set[str],
    seen_locations: set[str],
) -> dict[str, float | str]:
    amount = _optional_float(row.get("transaction_amount"))
    frequency = _optional_float(row.get("transaction_frequency"))
    device = _optional_text(row.get("device_id"))
    location = _optional_text(row.get("location"))
    transaction_type = _optional_text(row.get("transaction_type")) or "<MISSING>"
    activity_type = _optional_text(row.get("activity_type")) or "<MISSING>"
    login_hour = _parse_login_hour(row.get("login_time"), timestamp)

    previous_amount = (
        _optional_float(previous.get("transaction_amount"))
        if previous is not None
        else None
    )
    amount_delta = (
        amount - previous_amount
        if amount is not None and previous_amount is not None
        else 0.0
    )
    if previous is None:
        interval = 0.0
    else:
        previous_timestamp = pd.Timestamp(previous["_parsed_timestamp"]).to_pydatetime()
        interval = max(0.0, (timestamp - previous_timestamp).total_seconds())

    previous_device = _optional_text(previous.get("device_id")) if previous is not None else None
    previous_location = (
        _optional_text(previous.get("location")) if previous is not None else None
    )
    previous_activity = (
        _optional_text(previous.get("activity_type")) if previous is not None else None
    )
    previous_transaction = (
        _optional_text(previous.get("transaction_type")) if previous is not None else None
    )
    angle = login_hour * (2 * math.pi / 24)

    return {
        "amount_log": math.copysign(math.log1p(abs(amount)), amount)
        if amount
        else 0.0,
        "amount_change_log": math.copysign(math.log1p(abs(amount_delta)), amount_delta)
        if amount_delta
        else 0.0,
        "interval_seconds_log": math.log1p(interval),
        "login_hour_sin": math.sin(angle),
        "login_hour_cos": math.cos(angle),
        "frequency_log": math.log1p(max(0.0, frequency or 0.0)),
        "device_changed": float(
            previous_device is not None and device is not None and device != previous_device
        ),
        "device_new_to_user": float(device is not None and device not in seen_devices),
        "location_changed": float(
            previous_location is not None
            and location is not None
            and location != previous_location
        ),
        "location_new_to_user": float(
            location is not None and location not in seen_locations
        ),
        "transaction_type_changed": float(
            previous_transaction is not None
            and transaction_type != previous_transaction
        ),
        "activity_type_changed": float(
            previous_activity is not None and activity_type != previous_activity
        ),
        "transaction_type": transaction_type,
        "activity_type": activity_type,
        "activity_transition": (
            f"{previous_activity}\u001f{activity_type}"
            if previous_activity is not None
            else "<START>"
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Transaction amount/frequency must be numeric.") from exc
    if not math.isfinite(number):
        return None
    return number


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_login_hour(value: Any, timestamp: datetime) -> int:
    text = _optional_text(value)
    if text is None:
        return timestamp.hour
    try:
        return time.fromisoformat(text).hour
    except ValueError as exc:
        raise ValueError(f"Invalid login_time value: {text}") from exc


def _parse_binary_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return 1
    if text in {"false", "0"}:
        return 0
    raise ValueError("Synthetic sequence labels must be binary.")
