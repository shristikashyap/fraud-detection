"""Classification metrics and validation-only decision-threshold tuning."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation threshold with the best F1, breaking ties by precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        raise ValueError("Threshold tuning requires at least two probability values.")

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = f1_values.max()
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_precision = precision[best_indices].max()
    best_indices = best_indices[
        np.isclose(precision[best_indices], best_precision)
    ]
    return float(thresholds[best_indices[-1]])


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    """Calculate thresholded metrics and ranking metrics from actual labels."""
    predictions = (probabilities >= threshold).astype("int8")
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["non_fraud", "fraud"],
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["non_fraud", "fraud"],
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": {
            "labels": ["non_fraud", "fraud"],
            "matrix": matrix.tolist(),
        },
        "classification_report": report,
        "classification_report_text": report_text,
        "rows": int(len(y_true)),
        "fraud_rows": int(np.asarray(y_true).sum()),
    }
