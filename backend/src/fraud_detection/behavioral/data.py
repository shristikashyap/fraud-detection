"""Reproducible synthetic behavioral event data for Phase 3 development."""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import random

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = PROJECT_ROOT / "backend" / "data" / "behavioral" / "synthetic_activities.csv"

BEHAVIOR_COLUMNS = (
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
    "is_synthetic",
    "scenario",
    "synthetic_suspicious_label",
)

_USER_PROFILES = (
    ("user-001", time(8, 0), "mobile", "phone-001", "Boston"),
    ("user-002", time(18, 30), "desktop", "laptop-002", "Chicago"),
    ("user-003", time(12, 0), "mobile", "phone-003", "Seattle"),
    ("user-004", time(9, 15), "tablet", "tablet-004", "Austin"),
    ("user-005", time(20, 0), "desktop", "laptop-005", "Denver"),
    ("user-006", time(7, 45), "mobile", "phone-006", "Portland"),
)


def generate_synthetic_activities(
    *,
    seed: int = 42,
    normal_days: int = 21,
    activities_per_day: int = 2,
    include_scenarios: bool = True,
) -> pd.DataFrame:
    """Create fictional user histories plus explicitly labeled demo anomalies.

    Scenario labels are ground-truth metadata for demonstrations only. The
    behavioral learning service never consumes them when building profiles or
    assessing incoming activity.
    """
    if normal_days < 2:
        raise ValueError("normal_days must be at least 2.")
    if activities_per_day < 1:
        raise ValueError("activities_per_day must be positive.")

    rng = random.Random(seed)
    now = datetime(2026, 1, 22, 12, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    start = now - timedelta(days=normal_days)

    for user_id, usual_login, device_type, device_id, location in _USER_PROFILES:
        amount_center = rng.uniform(35, 115)
        frequency = activities_per_day
        for day in range(normal_days):
            for index in range(activities_per_day):
                event_time = start + timedelta(
                    days=day,
                    hours=usual_login.hour,
                    minutes=usual_login.minute
                    + index * max(1, 24 // activities_per_day),
                )
                timestamp = event_time.isoformat()
                login_time = event_time.time().replace(tzinfo=None).isoformat(timespec="minutes")
                transaction_type = rng.choice(("card_present", "online", "transfer"))
                rows.append(
                    _activity_row(
                        user_id=user_id,
                        timestamp=timestamp,
                        login_time=login_time,
                        transaction_amount=round(
                            max(2.0, rng.gauss(amount_center, amount_center * 0.18)), 2
                        ),
                        transaction_frequency=frequency,
                        device_type=device_type,
                        device_id=device_id,
                        location=location,
                        transaction_type=transaction_type,
                        activity_type="login_and_transaction",
                        transaction_status="completed",
                        scenario="normal_history",
                        suspicious=False,
                    )
                )

        if include_scenarios:
            suspicious_time = now.replace(hour=23, minute=17)
            rows.append(
                _activity_row(
                    user_id=user_id,
                    timestamp=suspicious_time.isoformat(),
                    login_time="23:17",
                    transaction_amount=round(amount_center * 8.5, 2),
                    transaction_frequency=frequency * 5,
                    device_type="unknown",
                    device_id=f"new-device-{user_id}",
                    location="Unknown",
                    transaction_type="online",
                    activity_type="login_and_transaction",
                    transaction_status="pending_review",
                    scenario="multiple_contextual_anomalies",
                    suspicious=True,
                )
            )

    frame = pd.DataFrame(rows, columns=BEHAVIOR_COLUMNS)
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    frame["is_synthetic"] = frame["is_synthetic"].astype(bool)
    frame["synthetic_suspicious_label"] = frame["synthetic_suspicious_label"].astype(bool)
    return frame


def write_synthetic_dataset(
    destination: Path = DEFAULT_OUTPUT,
    *,
    seed: int = 42,
    normal_days: int = 21,
    activities_per_day: int = 2,
) -> Path:
    """Write generated fictional activities as CSV or Parquet."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_activities(
        seed=seed,
        normal_days=normal_days,
        activities_per_day=activities_per_day,
    )
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(destination, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    else:
        raise ValueError("Synthetic activity output must use CSV or Parquet.")
    return destination


def _activity_row(
    *,
    user_id: str,
    timestamp: str,
    login_time: str,
    transaction_amount: float,
    transaction_frequency: int,
    device_type: str,
    device_id: str,
    location: str,
    transaction_type: str,
    activity_type: str,
    transaction_status: str,
    scenario: str,
    suspicious: bool,
) -> dict[str, object]:
    return {
        "user_id": user_id,
        "timestamp": timestamp,
        "login_time": login_time,
        "transaction_amount": transaction_amount,
        "transaction_frequency": transaction_frequency,
        "device_type": device_type,
        "device_id": device_id,
        "location": location,
        "transaction_type": transaction_type,
        "activity_type": activity_type,
        "transaction_status": transaction_status,
        "is_synthetic": True,
        "scenario": scenario,
        "synthetic_suspicious_label": suspicious,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-days", type=int, default=21)
    parser.add_argument("--activities-per-day", type=int, default=2)
    args = parser.parse_args()
    path = write_synthetic_dataset(
        args.output,
        seed=args.seed,
        normal_days=args.normal_days,
        activities_per_day=args.activities_per_day,
    )
    print(f"Wrote synthetic behavioral activity data: {path}")


if __name__ == "__main__":
    main()
