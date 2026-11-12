"""Configurable, explainable combination of transaction and behavior signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class RiskAssessmentConfig:
    """Provisional triage weights and thresholds, not empirically calibrated."""

    classifier_weight: float = 0.45
    lstm_weight: float = 0.20
    autoencoder_weight: float = 0.20
    contextual_weight: float = 0.15
    medium_risk_threshold: float = 30.0
    high_risk_threshold: float = 70.0
    context_anomaly_saturation: float = 5.0

    def __post_init__(self) -> None:
        weights = (
            self.classifier_weight,
            self.lstm_weight,
            self.autoencoder_weight,
            self.contextual_weight,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Signal weights must be finite and non-negative.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one signal weight must be positive.")
        if (
            not math.isfinite(self.medium_risk_threshold)
            or not math.isfinite(self.high_risk_threshold)
            or not 0 <= self.medium_risk_threshold < self.high_risk_threshold <= 100
        ):
            raise ValueError(
                "Risk thresholds must satisfy 0 <= medium < high <= 100."
            )
        if (
            not math.isfinite(self.context_anomaly_saturation)
            or self.context_anomaly_saturation <= 0
        ):
            raise ValueError("context_anomaly_saturation must be positive and finite.")

    def weights(self) -> dict[str, float]:
        return {
            "initial_fraud_classifier": self.classifier_weight,
            "lstm_behavior": self.lstm_weight,
            "autoencoder": self.autoencoder_weight,
            "contextual_baseline": self.contextual_weight,
        }


class CombinedRiskAssessor:
    """Combine available component signals into a triage score, not a fraud verdict."""

    def __init__(self, config: RiskAssessmentConfig | None = None) -> None:
        self.config = config or RiskAssessmentConfig()

    def assess(
        self, behavioral_assessment: Mapping[str, Any] | None
    ) -> dict[str, object]:
        """Score the Phase 3 service response while preserving missingness."""
        if behavioral_assessment is None:
            assessment: Mapping[str, Any] = {}
        elif isinstance(behavioral_assessment, Mapping):
            assessment = behavioral_assessment
        else:
            raise TypeError("behavioral_assessment must be a mapping or None.")

        weights = self.config.weights()
        state = str(assessment.get("state", "UNKNOWN"))
        signals = {
            "initial_fraud_classifier": self._classifier_signal(
                assessment.get("initial_fraud_classifier"),
                weights["initial_fraud_classifier"],
            ),
            "lstm_behavior": self._lstm_signal(
                assessment.get("lstm_sequence_analysis"),
                weights["lstm_behavior"],
            ),
            "autoencoder": self._autoencoder_signal(
                assessment.get("autoencoder_analysis"),
                weights["autoencoder"],
            ),
            "contextual_baseline": self._context_signal(
                assessment,
                weights["contextual_baseline"],
            ),
        }

        eligible = [
            signal
            for signal in signals.values()
            if signal["available"] and signal["configured_weight"] > 0
        ]
        total_weight = sum(float(signal["configured_weight"]) for signal in eligible)
        if total_weight == 0:
            no_score_reason = (
                "No usable risk signals with positive configured weight are available."
            )
            for signal in signals.values():
                if signal["available"] and signal["configured_weight"] == 0:
                    signal["exclusion_reason"] = "Configured weight is zero."
            return {
                "status": "unavailable",
                "user_state": state,
                "risk_score": None,
                "risk_category": None,
                "score_semantics": "No score; unavailable signals are not low-risk evidence.",
                "signals": signals,
                "available_signal_count": 0,
                "explanations": [no_score_reason],
                "requires_additional_verification": True,
                "recommendation": "manual_review_required",
                "recommendation_reasons": [no_score_reason],
                "fraud_determination": "not_made",
                "confirmed_fraud": None,
                "configuration": self._configuration_payload(),
                "limitations": self._limitations(),
            }

        weighted_score = 0.0
        explanations: list[str] = []
        for name, signal in signals.items():
            if not signal["available"]:
                continue
            if signal["configured_weight"] == 0:
                signal["exclusion_reason"] = "Configured weight is zero."
                continue
            normalized_weight = float(signal["configured_weight"]) / total_weight
            contribution = normalized_weight * float(signal["normalized_score"])
            signal["normalized_weight"] = normalized_weight
            signal["contribution_points"] = contribution * 100.0
            weighted_score += contribution
            explanation = signal.get("explanation")
            if explanation:
                explanations.append(str(explanation))

        risk_score = min(100.0, max(0.0, weighted_score * 100.0))
        category = self._risk_category(risk_score)
        if category == "HIGH RISK":
            explanations.append(
                "The combined triage score meets the configured high-risk threshold."
            )
        elif category == "MEDIUM RISK":
            explanations.append(
                "The combined triage score meets the configured medium-risk threshold."
            )
        elif not explanations:
            explanations.append("Available signals are below the medium-risk threshold.")

        recommendation_reasons: list[str] = []
        if category in {"MEDIUM RISK", "HIGH RISK"}:
            recommendation_reasons.append(
                f"{category.title()} activity should receive additional verification."
            )
        if assessment.get("requires_additional_verification") is True:
            recommendation_reasons.append(
                "The behavioral baseline independently requested additional verification."
            )
        requires_verification = bool(recommendation_reasons)
        return {
            "status": "available",
            "user_state": state,
            "risk_score": risk_score,
            "risk_category": category,
            "score_semantics": (
                "Weighted triage score from 0 to 100; not a calibrated fraud probability."
            ),
            "signals": signals,
            "available_signal_count": len(eligible),
            "explanations": list(dict.fromkeys(explanations)),
            "requires_additional_verification": requires_verification,
            "recommendation": (
                "additional_verification"
                if requires_verification
                else "continue_monitoring"
            ),
            "recommendation_reasons": recommendation_reasons,
            "fraud_determination": "not_made",
            "confirmed_fraud": None,
            "configuration": self._configuration_payload(),
            "limitations": self._limitations(),
        }

    def _classifier_signal(
        self, result: Any, weight: float
    ) -> dict[str, object]:
        signal = self._unavailable(weight, result, "Classifier output is absent.")
        if not isinstance(result, Mapping):
            return signal
        status = result.get("status")
        if status != "available":
            signal["exclusion_reason"] = self._status_reason(
                "Classifier", status
            )
            return signal
        score = _finite_number(result.get("fraud_probability"), "fraud_probability")
        if not 0 <= score <= 1:
            raise ValueError("Classifier fraud_probability must be between 0 and 1.")
        threshold = result.get("decision_threshold")
        if threshold is None:
            signal["exclusion_reason"] = (
                "Classifier decision threshold is unavailable; the score cannot be "
                "normalized relative to its operating point."
            )
            return signal
        threshold = _finite_number(threshold, "decision_threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("Classifier decision_threshold must be between 0 and 1.")
        normalized_score = _threshold_relative_score(
            score, threshold, anomalous=result.get("prediction") == "FRAUD"
        )
        if result.get("prediction") == "FRAUD":
            explanation = (
                "The initial transaction classifier crossed its validation-selected "
                "decision threshold; its score is threshold-normalized for triage."
            )
        else:
            explanation = (
                "The initial classifier score is normalized relative to its "
                "validation-selected threshold; it is not a calibrated likelihood."
            )
        return self._available(
            weight,
            raw_value={
                "fraud_probability": score,
                "decision_threshold": threshold,
                "prediction": result.get("prediction"),
            },
            normalized_score=normalized_score,
            normalization="score / (score + validation-selected decision threshold)",
            explanation=explanation,
        )

    def _lstm_signal(self, result: Any, weight: float) -> dict[str, object]:
        signal = self._unavailable(weight, result, "LSTM output is absent.")
        if not isinstance(result, Mapping):
            return signal
        status = result.get("status")
        if status != "available":
            signal["exclusion_reason"] = self._status_reason("LSTM", status)
            return signal
        score = _finite_number(result.get("anomaly_score"), "anomaly_score")
        if not 0 <= score <= 1:
            raise ValueError("LSTM anomaly_score must be between 0 and 1.")
        threshold = result.get("threshold")
        if threshold is None:
            signal["exclusion_reason"] = (
                "LSTM anomaly threshold is unavailable; the score cannot be "
                "normalized relative to its operating point."
            )
            return signal
        threshold = _finite_number(threshold, "LSTM threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("LSTM threshold must be between 0 and 1.")
        normalized_score = _threshold_relative_score(
            score, threshold, anomalous=result.get("behavioral_anomaly") is True
        )
        explanation = (
            "The LSTM sequence signal indicates an unusual pattern."
            if result.get("behavioral_anomaly") is True
            else "The LSTM sequence score is normalized relative to its threshold."
        )
        return self._available(
            weight,
            raw_value={"anomaly_score": score, "threshold": threshold},
            normalized_score=normalized_score,
            normalization="score / (score + configured LSTM anomaly threshold)",
            explanation=explanation,
        )

    def _autoencoder_signal(self, result: Any, weight: float) -> dict[str, object]:
        signal = self._unavailable(weight, result, "Autoencoder output is absent.")
        if not isinstance(result, Mapping):
            return signal
        status = result.get("status")
        if status != "available":
            signal["exclusion_reason"] = self._status_reason("Autoencoder", status)
            return signal
        error = _finite_number(
            result.get("reconstruction_error"), "reconstruction_error"
        )
        threshold = _finite_number(result.get("threshold"), "autoencoder threshold")
        if error < 0 or threshold < 0:
            raise ValueError("Autoencoder error and threshold must be non-negative.")
        score = _threshold_relative_score(
            error,
            threshold,
            anomalous=result.get("behavioral_anomaly") is True,
        )
        explanation = (
            "Autoencoder reconstruction error meets or exceeds its validation threshold."
            if result.get("behavioral_anomaly") is True
            else "Autoencoder reconstruction error is normalized against its validation threshold."
        )
        return self._available(
            weight,
            raw_value={
                "reconstruction_error": error,
                "validation_threshold": threshold,
            },
            normalized_score=score,
            normalization="error / (error + validation threshold)",
            explanation=explanation,
        )

    def _context_signal(
        self, assessment: Mapping[str, Any], weight: float
    ) -> dict[str, object]:
        result = assessment.get("behavioral_anomalies")
        state = assessment.get("state")
        active = assessment.get("behavioral_analysis_active") is True
        if not active:
            return self._unavailable(
                weight,
                result,
                f"Personalized context analysis is inactive for state {state or 'UNKNOWN'}.",
            )
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
            return self._unavailable(
                weight,
                result,
                "Context anomaly results are absent or unavailable.",
            )
        anomalies = list(result)
        if any(not isinstance(item, Mapping) for item in anomalies):
            raise ValueError("Each contextual anomaly must be a mapping.")
        explanations = [
            str(item.get("message") or item.get("code") or "Context deviation detected.")
            for item in anomalies
        ]
        if explanations:
            explanation = "Contextual deviations: " + " ".join(explanations)
        else:
            explanation = "Established-user baseline comparison found no deviations."
        score = min(
            1.0,
            len(anomalies) / self.config.context_anomaly_saturation,
        )
        return self._available(
            weight,
            raw_value={
                "anomaly_count": len(anomalies),
                "anomaly_codes": [
                    str(item.get("code", "unknown")) for item in anomalies
                ],
            },
            normalized_score=score,
            normalization=(
                "min(context anomaly count / configured saturation count, 1)"
            ),
            explanation=explanation,
        )

    @staticmethod
    def _unavailable(
        weight: float, result: Any, reason: str
    ) -> dict[str, object]:
        status = result.get("status") if isinstance(result, Mapping) else None
        return {
            "available": False,
            "source_status": status,
            "raw_value": None,
            "normalized_score": None,
            "configured_weight": weight,
            "normalized_weight": None,
            "contribution_points": None,
            "normalization": None,
            "explanation": None,
            "exclusion_reason": reason,
        }

    @staticmethod
    def _available(
        weight: float,
        *,
        raw_value: object,
        normalized_score: float,
        normalization: str,
        explanation: str,
    ) -> dict[str, object]:
        return {
            "available": True,
            "source_status": "available",
            "raw_value": raw_value,
            "normalized_score": normalized_score,
            "configured_weight": weight,
            "normalized_weight": None,
            "contribution_points": None,
            "normalization": normalization,
            "explanation": explanation,
            "exclusion_reason": None,
        }

    @staticmethod
    def _status_reason(name: str, status: Any) -> str:
        if status is None:
            return f"{name} status is missing; the signal is excluded."
        return f"{name} output status is '{status}'; the signal is excluded."

    def _risk_category(self, score: float) -> str:
        if score >= self.config.high_risk_threshold:
            return "HIGH RISK"
        if score >= self.config.medium_risk_threshold:
            return "MEDIUM RISK"
        return "LOW RISK"

    def _configuration_payload(self) -> dict[str, object]:
        return {
            **asdict(self.config),
            "weight_policy": "weights are renormalized over available positive-weight signals",
        }

    @staticmethod
    def _limitations() -> list[str]:
        return [
            "The combined score is a configurable triage score, not a calibrated fraud probability.",
            "Phase 4 and Phase 5 behavioral models were evaluated on a small synthetic dataset; their signals are not validated on representative real-world behavior.",
            "An anomaly indicates deviation from learned or modeled behavior and does not confirm fraud.",
            "Unavailable signals are excluded from the weighted score and do not count as low-risk evidence.",
        ]


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _threshold_relative_score(
    score: float, threshold: float, *, anomalous: bool = False
) -> float:
    if threshold == 0:
        return 1.0 if score > 0 or anomalous else 0.0
    return score / (score + threshold)
