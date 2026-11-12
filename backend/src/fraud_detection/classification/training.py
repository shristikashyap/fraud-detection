"""Train, select, evaluate, and persist the initial fraud classifier."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.classification.metrics import (
    classification_metrics,
    tune_f1_threshold,
)
from fraud_detection.classification.prediction import FraudPredictor
from fraud_detection.data.features import TARGET_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "initial_fraud_classifier.joblib"
REPORT_PATH = PROJECT_ROOT / "backend" / "reports" / "phase2_evaluation.json"


def _candidate_estimators(
    y_train: pd.Series, random_state: int
) -> dict[str, Pipeline]:
    """Build candidates; optional XGBoost is included only when installed."""
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    if positive == 0 or negative == 0:
        raise ValueError("Training data must contain both fraud and non-fraud rows.")

    candidates = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        candidates["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        scale_pos_weight=negative / positive,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=random_state,
                ),
                ),
            ]
        )
    return candidates


def _load_split(name: str, data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    path = data_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared {name} split not found: {path}")
    frame = pd.read_parquet(path)
    if TARGET_NAME not in frame.columns:
        raise ValueError(f"Prepared {name} split is missing '{TARGET_NAME}'.")
    if frame.empty:
        raise ValueError(f"Prepared {name} split is empty.")
    features = frame.drop(columns=[TARGET_NAME])
    labels = frame[TARGET_NAME].astype("int8")
    if not labels.isin((0, 1)).all():
        raise ValueError(f"Prepared {name} labels must be encoded as 0 or 1.")
    return features, labels


def train_and_evaluate(
    *,
    data_dir: Path = PROCESSED_DIR,
    model_path: Path = MODEL_PATH,
    report_path: Path = REPORT_PATH,
    random_state: int = 42,
) -> dict[str, object]:
    """Select using validation, then load the held-out test split exactly once."""
    x_train, y_train = _load_split("train", data_dir)
    x_validation, y_validation = _load_split("validation", data_dir)
    if list(x_train.columns) != list(x_validation.columns):
        raise ValueError("Train and validation feature columns do not match.")

    candidate_models = _candidate_estimators(y_train, random_state)
    validation_reports: dict[str, dict[str, object]] = {}
    thresholds: dict[str, float] = {}
    validation_probabilities: dict[str, np.ndarray] = {}

    for name, estimator in candidate_models.items():
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        threshold = tune_f1_threshold(y_validation.to_numpy(), probabilities)
        thresholds[name] = threshold
        validation_probabilities[name] = probabilities
        validation_reports[name] = classification_metrics(
            y_validation.to_numpy(), probabilities, threshold
        )

    selected_name = max(
        candidate_models,
        key=lambda name: (
            float(validation_reports[name]["pr_auc_average_precision"]),
            float(validation_reports[name]["f1_score"]),
            float(validation_reports[name]["recall"]),
            float(validation_reports[name]["precision"]),
        ),
    )
    selected_estimator = candidate_models[selected_name]
    selected_threshold = thresholds[selected_name]
    predictor = FraudPredictor(
        estimator=selected_estimator,
        feature_columns=list(x_train.columns),
        threshold=selected_threshold,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, model_path)

    train_probabilities = selected_estimator.predict_proba(x_train)[:, 1]
    selected_validation_probabilities = validation_probabilities[selected_name]
    train_metrics = classification_metrics(
        y_train.to_numpy(), train_probabilities, selected_threshold
    )
    selected_validation_metrics = validation_reports[selected_name]

    # Test data is not accessed until model selection, threshold tuning, and persistence finish.
    x_test, y_test = _load_split("test", data_dir)
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("Train and test feature columns do not match.")
    test_probabilities = selected_estimator.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(
        y_test.to_numpy(), test_probabilities, selected_threshold
    )

    xgboost_available = "xgboost" in candidate_models
    report: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "Highest validation PR-AUC (average precision), then validation F1, "
            "recall, and precision in that order; "
            "each candidate threshold maximizes validation F1, ties by precision."
        ),
        "selected_model": selected_name,
        "model_artifact": str(model_path),
        "feature_columns": list(x_train.columns),
        "class_imbalance_strategy": {
            "logistic_regression": "class_weight=balanced",
            "random_forest": "class_weight=balanced_subsample",
            "xgboost": "scale_pos_weight=training negatives / training positives",
            "oversampling": "not used",
        },
        "xgboost": {
            "available": xgboost_available,
            "included_in_comparison": xgboost_available,
            "reason": None if xgboost_available else "xgboost is not installed.",
        },
        "validation_model_comparison": validation_reports,
        "selected_thresholds": thresholds,
        "selected_model_results": {
            "train": train_metrics,
            "validation": selected_validation_metrics,
            "test": test_metrics,
        },
        "test_access_policy": (
            "Test split loaded once after validation-based model and threshold selection."
        ),
        "risk_categories": {
            "high": f"probability >= selected validation threshold ({selected_threshold:.12g})",
            "medium": (
                "probability >= 0.5 * selected validation threshold and below it"
            ),
            "low": "probability < 0.5 * selected validation threshold",
        },
        "metric_definitions": {
            "pr_auc_average_precision": "Average precision, the PR-curve summary.",
            "confusion_matrix": "Rows are true non-fraud/fraud; columns are predicted non-fraud/fraud.",
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    arguments = parser.parse_args()
    report = train_and_evaluate(
        data_dir=arguments.data_dir,
        model_path=arguments.model_path,
        report_path=arguments.report_path,
        random_state=arguments.random_state,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
