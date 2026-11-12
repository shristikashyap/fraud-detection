import pytest

from fraud_detection.behavioral.risk_assessment import (
    CombinedRiskAssessor,
    RiskAssessmentConfig,
)


def available_assessment():
    return {
        "state": "ESTABLISHED USER",
        "behavioral_analysis_active": True,
        "requires_additional_verification": False,
        "initial_fraud_classifier": {
            "status": "available",
            "fraud_probability": 0.4,
            "risk_category": "HIGH RISK",
            "prediction": "FRAUD",
            "decision_threshold": 0.3,
        },
        "lstm_sequence_analysis": {
            "status": "available",
            "anomaly_score": 0.8,
            "threshold": 0.7,
            "behavioral_anomaly": True,
            "is_fraud_decision": False,
        },
        "autoencoder_analysis": {
            "status": "available",
            "reconstruction_error": 2.0,
            "threshold": 1.0,
            "behavioral_anomaly": True,
            "is_fraud_decision": False,
        },
        "behavioral_anomalies": [
            {"code": "new_device", "message": "A new device was observed."},
            {"code": "unexpected_location", "message": "Location is new."},
        ],
    }


def test_combines_normalized_available_signals_and_reports_contributions():
    result = CombinedRiskAssessor().assess(available_assessment())

    assert result["status"] == "available"
    assert result["risk_category"] == "MEDIUM RISK"
    assert result["risk_score"] == pytest.approx(55.7142857143)
    assert result["fraud_determination"] == "not_made"
    assert result["confirmed_fraud"] is None
    signals = result["signals"]
    assert set(signals) == {
        "initial_fraud_classifier",
        "lstm_behavior",
        "autoencoder",
        "contextual_baseline",
    }
    assert all(signal["available"] for signal in signals.values())
    assert signals["initial_fraud_classifier"]["normalized_score"] == pytest.approx(
        0.4 / (0.4 + 0.3)
    )
    assert signals["lstm_behavior"]["normalized_score"] == pytest.approx(
        0.8 / (0.8 + 0.7)
    )
    assert sum(
        signal["contribution_points"] for signal in signals.values()
    ) == pytest.approx(result["risk_score"])
    assert signals["autoencoder"]["normalized_score"] == pytest.approx(2 / 3)
    assert signals["contextual_baseline"]["normalized_score"] == pytest.approx(0.4)
    assert result["requires_additional_verification"] is True
    assert any("new device" in reason for reason in result["explanations"])


def test_cold_start_uses_classifier_only_and_does_not_zero_fill_missing_signals():
    assessment = {
        "state": "NEW USER",
        "behavioral_analysis_active": False,
        "initial_fraud_classifier": {
            "status": "available",
            "fraud_probability": 0.1,
            "prediction": "NOT FRAUD",
            "decision_threshold": 0.3,
        },
        "lstm_sequence_analysis": {
            "status": "not_active_for_user_state",
            "behavioral_anomaly": None,
        },
        "autoencoder_analysis": {
            "status": "not_active_for_user_state",
            "behavioral_anomaly": None,
        },
        "behavioral_anomalies": [],
    }

    result = CombinedRiskAssessor().assess(assessment)

    assert result["risk_score"] == pytest.approx(25.0)
    assert result["risk_category"] == "LOW RISK"
    assert result["available_signal_count"] == 1
    assert result["signals"]["lstm_behavior"]["normalized_score"] is None
    assert result["signals"]["autoencoder"]["contribution_points"] is None
    assert result["signals"]["contextual_baseline"]["available"] is False
    assert result["requires_additional_verification"] is False


def test_single_context_deviation_is_not_confirmed_fraud_or_automatic_review():
    assessment = {
        "state": "ESTABLISHED USER",
        "behavioral_analysis_active": True,
        "behavioral_anomalies": [
            {"code": "new_device", "message": "A new device was observed."}
        ],
    }

    result = CombinedRiskAssessor().assess(assessment)

    assert result["risk_score"] == pytest.approx(20.0)
    assert result["risk_category"] == "LOW RISK"
    assert result["fraud_determination"] == "not_made"
    assert result["requires_additional_verification"] is False


def test_missing_or_unavailable_outputs_produce_no_score_not_low_risk():
    result = CombinedRiskAssessor().assess(
        {
            "state": "LEARNING USER",
            "behavioral_analysis_active": False,
            "initial_fraud_classifier": {"status": "not_applicable"},
            "lstm_sequence_analysis": {"status": "insufficient_context"},
            "autoencoder_analysis": {"status": "not_configured"},
        }
    )

    assert result["status"] == "unavailable"
    assert result["risk_score"] is None
    assert result["risk_category"] is None
    assert result["requires_additional_verification"] is True
    assert result["fraud_determination"] == "not_made"
    assert all(
        signal["normalized_score"] is None
        for signal in result["signals"].values()
    )


def test_baseline_verification_request_is_preserved_even_if_score_is_low():
    assessment = {
        "state": "ESTABLISHED USER",
        "behavioral_analysis_active": True,
        "requires_additional_verification": True,
        "initial_fraud_classifier": {
            "status": "available",
            "fraud_probability": 0.01,
            "prediction": "NOT FRAUD",
            "decision_threshold": 0.4,
        },
        "behavioral_anomalies": [
            {"code": "new_device", "message": "A new device was observed."}
        ],
    }

    result = CombinedRiskAssessor().assess(assessment)

    assert result["risk_category"] == "LOW RISK"
    assert result["requires_additional_verification"] is True
    assert result["recommendation"] == "additional_verification"


def test_custom_weights_and_thresholds_are_applied():
    config = RiskAssessmentConfig(
        classifier_weight=1,
        lstm_weight=0,
        autoencoder_weight=0,
        contextual_weight=0,
        medium_risk_threshold=20,
        high_risk_threshold=60,
    )
    assessment = {
        "state": "NEW USER",
        "behavioral_analysis_active": False,
        "initial_fraud_classifier": {
            "status": "available",
            "fraud_probability": 0.6,
            "decision_threshold": 0.3,
        },
    }

    result = CombinedRiskAssessor(config).assess(assessment)

    assert result["risk_score"] == pytest.approx(100 * (0.6 / (0.6 + 0.3)))
    assert result["risk_category"] == "HIGH RISK"
    assert result["signals"]["initial_fraud_classifier"]["normalized_weight"] == 1
    assert result["signals"]["lstm_behavior"]["contribution_points"] is None


@pytest.mark.parametrize(
    "config",
    [
        {"classifier_weight": -1},
        {"classifier_weight": 0, "lstm_weight": 0, "autoencoder_weight": 0, "contextual_weight": 0},
        {"medium_risk_threshold": 70, "high_risk_threshold": 30},
        {"context_anomaly_saturation": 0},
    ],
)
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        RiskAssessmentConfig(**config)


def test_invalid_component_scores_are_rejected():
    assessment = {
        "initial_fraud_classifier": {
            "status": "available",
            "fraud_probability": 1.1,
        }
    }

    with pytest.raises(ValueError, match="between 0 and 1"):
        CombinedRiskAssessor().assess(assessment)


def test_autoencoder_error_is_normalized_against_its_threshold():
    assessment = {
        "state": "ESTABLISHED USER",
        "autoencoder_analysis": {
            "status": "available",
            "reconstruction_error": 3.0,
            "threshold": 0.0,
        },
    }

    result = CombinedRiskAssessor().assess(assessment)

    assert result["signals"]["autoencoder"]["normalized_score"] == 1.0


def test_zero_threshold_respects_component_anomaly_indicator():
    result = CombinedRiskAssessor().assess(
        {
            "initial_fraud_classifier": {
                "status": "available",
                "fraud_probability": 0.0,
                "decision_threshold": 0.0,
                "prediction": "FRAUD",
            },
            "lstm_sequence_analysis": {
                "status": "available",
                "anomaly_score": 0.0,
                "threshold": 0.0,
                "behavioral_anomaly": True,
            },
            "autoencoder_analysis": {
                "status": "available",
                "reconstruction_error": 0.0,
                "threshold": 0.0,
                "behavioral_anomaly": True,
            },
        }
    )

    assert result["signals"]["initial_fraud_classifier"]["normalized_score"] == 1.0
    assert result["signals"]["lstm_behavior"]["normalized_score"] == 1.0
    assert result["signals"]["autoencoder"]["normalized_score"] == 1.0


def test_missing_operating_threshold_excludes_score_instead_of_using_raw_value():
    result = CombinedRiskAssessor().assess(
        {
            "state": "NEW USER",
            "initial_fraud_classifier": {
                "status": "available",
                "fraud_probability": 0.9,
            },
        }
    )

    assert result["status"] == "unavailable"
    assert result["risk_score"] is None
    assert "threshold is unavailable" in result["signals"][
        "initial_fraud_classifier"
    ]["exclusion_reason"]
