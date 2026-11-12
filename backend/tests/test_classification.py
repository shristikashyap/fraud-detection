import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.classification.metrics import (
    classification_metrics,
    tune_f1_threshold,
)
from fraud_detection.classification.prediction import FraudPredictor
from fraud_detection.classification.prediction import load_predictor
from fraud_detection.classification.training import _candidate_estimators
from fraud_detection.data.features import engineer_transaction_features


def test_candidate_models_use_class_weights_and_training_fitted_preprocessors():
    candidates = _candidate_estimators(
        pd.Series([0] * 8 + [1] * 2), random_state=7
    )

    assert "logistic_regression" in candidates
    assert "random_forest" in candidates
    logistic = candidates["logistic_regression"]
    forest = candidates["random_forest"]
    assert logistic.named_steps["classifier"].class_weight == "balanced"
    assert forest.named_steps["classifier"].class_weight == "balanced_subsample"
    assert isinstance(logistic.named_steps["imputer"], SimpleImputer)
    assert isinstance(logistic.named_steps["scaler"], StandardScaler)


def test_validation_threshold_and_metrics_are_calculated_from_input():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = tune_f1_threshold(labels, probabilities)
    metrics = classification_metrics(labels, probabilities, threshold)

    assert 0 < threshold <= 1
    assert metrics["accuracy"] == 1
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["f1_score"] == 1
    assert metrics["roc_auc"] == 1
    assert metrics["pr_auc_average_precision"] == 1
    assert metrics["confusion_matrix"]["matrix"] == [[2, 0], [0, 2]]
    assert metrics["classification_report"]["fraud"]["recall"] == 1
    assert "non_fraud" in metrics["classification_report_text"]


def test_predictor_accepts_unlabeled_cold_start_transaction():
    training = pd.DataFrame(
        {
            "Time": [10, 20, 30, 40, 50, 60],
            "Amount": [5, 10, 15, 100, 150, 200],
            "V1": [0.1, 0.2, 0.3, 1.0, 1.1, 1.2],
            "Class": [0, 0, 0, 1, 1, 1],
        }
    )
    x_train, y_train = (
        training.drop(columns="Class"),
        training["Class"],
    )
    feature_columns = list(engineer_transaction_features(x_train).columns)
    x_features = engineer_transaction_features(x_train, feature_columns)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=0)),
        ]
    ).fit(x_features, y_train)
    predictor = FraudPredictor(
        estimator=model,
        feature_columns=feature_columns,
        threshold=0.6,
    )

    result = predictor.predict({"Time": 70, "Amount": 120, "V1": 1.0})

    assert 0 <= result["fraud_probability"] <= 1
    assert result["decision_threshold"] == 0.6
    assert result["prediction"] in {"FRAUD", "NOT FRAUD"}
    assert result["risk_category"] in {"LOW RISK", "MEDIUM RISK", "HIGH RISK"}


def test_predictor_rejects_empty_or_invalid_transaction_and_threshold():
    with pytest.raises(ValueError, match="between 0 and 1"):
        FraudPredictor(Pipeline([]), ["amount"], threshold=1.1)

    predictor = FraudPredictor(Pipeline([]), ["amount"], threshold=0.5)
    with pytest.raises(ValueError, match="must not be empty"):
        predictor.predict({})
    with pytest.raises(ValueError, match="amount"):
        predictor.predict({"Time": 10})


def test_saved_predictor_can_be_loaded_and_used(tmp_path):
    features = pd.DataFrame({"Time": [1, 2, 3, 4], "Amount": [1, 2, 20, 40]})
    labels = pd.Series([0, 0, 1, 1])
    model_features = engineer_transaction_features(features)
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", LogisticRegression(random_state=0)),
        ]
    ).fit(model_features, labels)
    predictor = FraudPredictor(
        estimator=estimator,
        feature_columns=list(model_features.columns),
        threshold=0.5,
    )
    artifact_path = tmp_path / "predictor.joblib"
    joblib.dump(predictor, artifact_path)

    loaded = load_predictor(artifact_path)

    assert loaded.predict({"Time": 5, "Amount": 30})["prediction"] in {
        "FRAUD",
        "NOT FRAUD",
    }
