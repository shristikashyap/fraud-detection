"""Metrics for the synthetic supervised sequence-anomaly experiment."""

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


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation threshold selection requires both target classes.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Validation threshold selection requires score variation.")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = f1_values.max()
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_recall = recall[candidates].max()
    candidates = candidates[np.isclose(recall[candidates], best_recall)]
    return float(thresholds[candidates[0]])


def sequence_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, object]:
    predicted = (scores >= threshold).astype("int8")
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    has_both_classes = set(np.unique(labels)) == {0, 1}
    has_positive = bool(np.any(labels == 1))
    return {
        "rows": int(len(labels)),
        "positive_rows": int(np.asarray(labels).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1_score": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, scores)) if has_both_classes else None
        ),
        "pr_auc_average_precision": (
            float(average_precision_score(labels, scores)) if has_positive else None
        ),
        "confusion_matrix": {
            "labels": ["synthetic_normal", "synthetic_suspicious"],
            "matrix": matrix.tolist(),
        },
        "classification_report": classification_report(
            labels,
            predicted,
            labels=[0, 1],
            target_names=["synthetic_normal", "synthetic_suspicious"],
            zero_division=0,
            output_dict=True,
        ),
    }
