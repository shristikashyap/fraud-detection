"""Cold-start routing and context-aware behavioral comparison."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from statistics import median
from typing import Any, Protocol, Sequence

from fraud_detection.behavioral.baseline import (
    BaselineConfig,
    BaselineStore,
    UserBaseline,
    normalize_activity,
)


class TransactionPredictor(Protocol):
    def predict(self, transaction: Mapping[str, Any]) -> dict[str, object]: ...


class SequenceAnalyzer(Protocol):
    def predict(
        self, activities: Sequence[Mapping[str, Any]]
    ) -> dict[str, object]: ...


class ReconstructionAnalyzer(Protocol):
    def predict(
        self, activities: Sequence[Mapping[str, Any]]
    ) -> dict[str, object]: ...


class BehavioralLearningService:
    """Keep initial classification active until a trusted personal baseline is ready."""

    def __init__(
        self,
        predictor: TransactionPredictor | None,
        *,
        config: BaselineConfig | None = None,
        store: BaselineStore | None = None,
        lstm_analyzer: SequenceAnalyzer | None = None,
        autoencoder_analyzer: ReconstructionAnalyzer | None = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or BaselineConfig()
        self.store = store or BaselineStore()
        self.lstm_analyzer = lstm_analyzer
        self.autoencoder_analyzer = autoencoder_analyzer

    def assess(
        self,
        activity: Mapping[str, Any],
        *,
        verified: bool = False,
        recent_activities: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, object]:
        """Assess an event and optionally learn it only when explicitly trusted.

        Verification is supplied by the calling application after its own
        review/authorization checks. Synthetic scenario labels are never read.
        """
        normalized = normalize_activity(activity)
        user_id = normalized["user_id"]
        profile = self.store.get(user_id)
        state = self._state(profile)
        anomalies = (
            self._compare(normalized, profile)
            if profile is not None and state == "ESTABLISHED USER"
            else []
        )
        classifier_result = self._classify(activity, normalized)
        lstm_result = self._analyze_sequence(
            activity,
            user_id=user_id,
            state=state,
            recent_activities=recent_activities,
        )
        autoencoder_result = self._analyze_autoencoder(
            activity,
            user_id=user_id,
            state=state,
            recent_activities=recent_activities,
        )
        requires_verification = (
            len(anomalies) >= self.config.multiple_anomaly_review_threshold
        )

        learned = False
        if verified and not anomalies and self._classifier_allows_learning(classifier_result):
            self.store.add_trusted(activity)
            learned = True

        updated_profile = self.store.get(user_id)
        return {
            "user_id": user_id,
            "state": state,
            "behavioral_analysis_active": state == "ESTABLISHED USER",
            "baseline": self._baseline_summary(updated_profile),
            "behavioral_anomalies": anomalies,
            "requires_additional_verification": requires_verification,
            "activity_learned": learned,
            "learning_status": (
                "trusted activity added"
                if learned
                else "not added; activity must be verified, non-anomalous, and not high-risk"
            ),
            "initial_fraud_classifier": classifier_result,
            "lstm_sequence_analysis": lstm_result,
            "autoencoder_analysis": autoencoder_result,
            "sequence_features": {
                "activity_type": normalized["activity_type"],
                "previous_trusted_activity_type": (
                    profile.last_activity_type if profile is not None else None
                ),
                "transition_observed": not any(
                    anomaly["code"] == "unusual_activity_sequence"
                    for anomaly in anomalies
                ),
            },
            "model_status": {
                "lstm": (
                    "not configured"
                    if self.lstm_analyzer is None
                    else str(lstm_result["status"])
                ),
                "autoencoder": (
                    "not configured"
                    if self.autoencoder_analyzer is None
                    else str(autoencoder_result["status"])
                ),
            },
        }

    def _analyze_sequence(
        self,
        activity: Mapping[str, Any],
        *,
        user_id: str,
        state: str,
        recent_activities: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, object]:
        if state != "ESTABLISHED USER":
            return {
                "status": "not_active_for_user_state",
                "behavioral_anomaly": None,
                "is_fraud_decision": False,
            }
        if self.lstm_analyzer is None:
            return {
                "status": "not_configured",
                "behavioral_anomaly": None,
                "is_fraud_decision": False,
            }

        sequence = self._validated_recent_sequence(
            activity, user_id=user_id, recent_activities=recent_activities
        )
        result = self.lstm_analyzer.predict([*sequence, activity])
        if result.get("is_fraud_decision") is not False:
            raise ValueError("LSTM analyzer must return an advisory, non-fraud decision.")
        return result

    def _analyze_autoencoder(
        self,
        activity: Mapping[str, Any],
        *,
        user_id: str,
        state: str,
        recent_activities: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, object]:
        if state != "ESTABLISHED USER":
            return {
                "status": "not_active_for_user_state",
                "behavioral_anomaly": None,
                "is_fraud_decision": False,
            }
        if self.autoencoder_analyzer is None:
            return {
                "status": "not_configured",
                "behavioral_anomaly": None,
                "is_fraud_decision": False,
            }

        sequence = self._validated_recent_sequence(
            activity, user_id=user_id, recent_activities=recent_activities
        )
        result = self.autoencoder_analyzer.predict([*sequence, activity])
        if result.get("is_fraud_decision") is not False:
            raise ValueError(
                "Autoencoder analyzer must return an advisory, non-fraud decision."
            )
        return result

    @staticmethod
    def _validated_recent_sequence(
        activity: Mapping[str, Any],
        *,
        user_id: str,
        recent_activities: Sequence[Mapping[str, Any]] | None,
    ) -> list[Mapping[str, Any]]:
        sequence = list(recent_activities or ())
        current_timestamp = normalize_activity(activity)["timestamp"]
        previous_timestamp = None
        for previous_activity in sequence:
            previous = normalize_activity(previous_activity)
            if previous["user_id"] != user_id:
                raise ValueError("Recent activities must belong to the assessed user.")
            if previous_timestamp is not None and previous["timestamp"] < previous_timestamp:
                raise ValueError("Recent activities must be in chronological order.")
            if previous["timestamp"] >= current_timestamp:
                raise ValueError("Recent activities must precede the current activity.")
            previous_timestamp = previous["timestamp"]
        return sequence

    def _state(self, profile: UserBaseline | None) -> str:
        if profile is None or profile.trusted_activity_count == 0:
            return "NEW USER"
        established = (
            profile.trusted_activity_count >= self.config.minimum_trusted_activities
            and profile.history_days >= self.config.minimum_history_days
        )
        return "ESTABLISHED USER" if established else "LEARNING USER"

    def _classify(
        self, activity: Mapping[str, Any], normalized: Mapping[str, Any]
    ) -> dict[str, object] | None:
        amount = normalized["transaction_amount"]
        if amount is None:
            return {
                "status": "not_applicable",
                "reason": "No transaction amount was provided for initial classification.",
            }
        if self.predictor is None:
            return {
                "status": "unavailable",
                "reason": "The initial transaction classifier is not loaded.",
            }

        timestamp: datetime = normalized["timestamp"]
        # The Phase 1 classifier uses elapsed seconds and a daily cyclic feature;
        # wall-clock epochs would be far outside its training range.
        transaction: dict[str, Any] = {
            "Time": (
                timestamp.hour * 3_600
                + timestamp.minute * 60
                + timestamp.second
                + timestamp.microsecond / 1_000_000
            ),
            "Amount": amount,
        }
        for component in range(1, 29):
            key = f"v{component}"
            if key in activity:
                transaction[key] = activity[key]
            elif f"V{component}" in activity:
                transaction[f"V{component}"] = activity[f"V{component}"]
        return {
            "status": "available",
            **self.predictor.predict(transaction),
        }

    @staticmethod
    def _classifier_allows_learning(
        classifier_result: dict[str, object] | None,
    ) -> bool:
        if classifier_result is None:
            return False
        if classifier_result.get("status") == "not_applicable":
            return True
        if classifier_result.get("status") != "available":
            return False
        return classifier_result.get("risk_category") == "LOW RISK"

    def _compare(
        self, activity: Mapping[str, Any], profile: UserBaseline
    ) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        login_hour = activity["login_hour"]
        if login_hour is not None and profile.login_hours:
            typical_hours = list(profile.login_hours)
            tolerance = self.config.typical_login_hour_tolerance
            closest_distance = min(
                min((login_hour - hour) % 24, (hour - login_hour) % 24)
                for hour in typical_hours
            )
            if closest_distance > tolerance:
                anomalies.append(
                    _anomaly(
                        "unusual_login_time",
                        "login_time",
                        f"{login_hour:02d}:00",
                        [f"{hour:02d}:00" for hour in sorted(typical_hours)],
                        "Login hour is outside the user's observed time window.",
                    )
                )

        device_id = activity["device_id"]
        if (
            device_id is not None
            and profile.device_counts
            and device_id not in profile.device_counts
        ):
            anomalies.append(
                _anomaly(
                    "new_device",
                    "device_id",
                    device_id,
                    sorted(profile.device_counts),
                    "Device has not appeared in trusted user activity.",
                )
            )

        location = activity["location"]
        if (
            location is not None
            and profile.location_counts
            and location not in profile.location_counts
        ):
            anomalies.append(
                _anomaly(
                    "unexpected_location",
                    "location",
                    location,
                    sorted(profile.location_counts),
                    "Location has not appeared in trusted user activity.",
                )
            )

        amount = activity["transaction_amount"]
        if amount is not None and profile.transaction_amounts:
            amount_median = median(profile.transaction_amounts)
            absolute_deviations = [
                abs(value - amount_median) for value in profile.transaction_amounts
            ]
            mad = median(absolute_deviations)
            amount_range = profile.transaction_amount_range
            observed_range = (
                amount_range["maximum"] - amount_range["minimum"]
                if amount_range is not None
                else 0.0
            )
            tolerance = max(
                mad * self.config.amount_mad_multiplier,
                observed_range * 0.25,
                1.0,
            )
            lower = amount_median - tolerance
            upper = amount_median + tolerance
            if amount < lower or amount > upper:
                anomalies.append(
                    _anomaly(
                        "abnormal_transaction_amount",
                        "transaction_amount",
                        amount,
                        {"lower_expected": lower, "upper_expected": upper},
                        "Amount is outside a robust range learned from trusted transactions.",
                    )
                )

        frequency = activity["transaction_frequency"]
        average_frequency = profile.average_transaction_frequency
        if frequency is not None and average_frequency is not None:
            deviation = self.config.transaction_frequency_deviation
            lower = max(0.0, average_frequency / deviation)
            upper = average_frequency * deviation
            if frequency < lower or frequency > upper:
                anomalies.append(
                    _anomaly(
                        "unusual_transaction_frequency",
                        "transaction_frequency",
                        frequency,
                        {"lower_expected": lower, "upper_expected": upper},
                        "Transaction frequency differs from the trusted user baseline.",
                    )
                )

        transition_total = sum(profile.transition_counts.values())
        previous_type = profile.last_activity_type
        current_type = activity["activity_type"]
        transition = (
            f"{previous_type}\u001f{current_type}"
            if previous_type is not None
            else None
        )
        if (
            transition_total >= self.config.minimum_sequence_transitions
            and transition is not None
            and profile.transition_counts.get(transition, 0) == 0
        ):
            anomalies.append(
                _anomaly(
                    "unusual_activity_sequence",
                    "activity_type",
                    current_type,
                    {
                        "previous_trusted_activity_type": previous_type,
                        "observed_transition_count": dict(profile.transition_counts),
                    },
                    "This activity transition was not observed in trusted sequences.",
                )
            )
        return anomalies

    def _baseline_summary(
        self, profile: UserBaseline | None
    ) -> dict[str, object] | None:
        if profile is None:
            return None
        typical_login_hours = []
        if profile.login_hours:
            maximum_count = max(profile.login_hours.values())
            typical_login_hours = [
                hour
                for hour, count in sorted(profile.login_hours.items())
                if count == maximum_count
            ]
        return {
            **profile.to_dict(),
            "typical_login_hours": typical_login_hours,
            "common_devices": dict(profile.device_counts.most_common()),
            "frequent_locations": dict(profile.location_counts.most_common()),
            "normal_activity_sequences": dict(profile.transition_counts),
            "minimum_trusted_activities": self.config.minimum_trusted_activities,
            "minimum_history_days": self.config.minimum_history_days,
            "status": self._state(profile),
        }


def _anomaly(
    code: str,
    field: str,
    observed: object,
    expected: object,
    message: str,
) -> dict[str, object]:
    return {
        "code": code,
        "field": field,
        "observed": observed,
        "expected": expected,
        "message": message,
    }
