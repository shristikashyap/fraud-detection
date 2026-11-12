"""User-specific behavioral learning, separate from transaction classifiers."""

from fraud_detection.behavioral.risk_assessment import (
    CombinedRiskAssessor,
    RiskAssessmentConfig,
)

__all__ = ["CombinedRiskAssessor", "RiskAssessmentConfig"]
