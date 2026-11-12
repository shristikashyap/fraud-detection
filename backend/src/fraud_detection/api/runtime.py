"""Model loading and shared runtime composition for the Flask API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

from fraud_detection.behavioral.autoencoder.prediction import (
    load_autoencoder_predictor,
)
from fraud_detection.api.database import MongoDatabase
from fraud_detection.api.persistence import MongoBaselineStore, MongoPersistence
from fraud_detection.behavioral.baseline import BaselineConfig, BaselineStore
from fraud_detection.behavioral.risk_assessment import (
    CombinedRiskAssessor,
    RiskAssessmentConfig,
)
from fraud_detection.behavioral.sequential.prediction import load_sequence_predictor
from fraud_detection.behavioral.service import BehavioralLearningService
from fraud_detection.classification.prediction import load_predictor

logger = logging.getLogger(__name__)


@dataclass
class APIComponents:
    """Optional injection point for tests or an embedding application."""

    predictor: Any | None = None
    lstm_analyzer: Any | None = None
    autoencoder_analyzer: Any | None = None
    baseline_store: BaselineStore | None = None
    database: MongoDatabase | None = None
    analysis_registry: Any | None = None
    model_status: Mapping[str, Mapping[str, object]] | None = None


@dataclass
class APIRuntime:
    behavioral_service: BehavioralLearningService
    risk_assessor: CombinedRiskAssessor
    baseline_store: BaselineStore
    analysis_registry: Any
    model_status: dict[str, dict[str, object]]
    database: MongoDatabase
    profiles_loaded: bool


def create_runtime(
    app_config: Mapping[str, Any], components: APIComponents | None = None
) -> APIRuntime:
    if components is None:
        components = _load_components(app_config)

    database = components.database or MongoDatabase(
        uri=app_config.get("MONGO_URI"),
        database_name=str(app_config.get("MONGO_DB_NAME", "fraud_detection")),
        server_selection_timeout_ms=int(
            app_config.get("MONGO_SERVER_SELECTION_TIMEOUT_MS", 2_000)
        )
    )
    persistence = MongoPersistence(database)
    baseline_config = app_config.get("BASELINE_CONFIG") or BaselineConfig()
    if components.baseline_store is not None:
        baseline_store = components.baseline_store
        profiles_loaded = True
    else:
        baseline_store = MongoBaselineStore(
            persistence,
            legacy_path=str(app_config["BASELINE_STORE_PATH"]),
            config=baseline_config,
        )
        profiles_loaded = baseline_store.profiles_loaded

    model_status = (
        {name: dict(value) for name, value in components.model_status.items()}
        if components.model_status is not None
        else {
            "classifier": _injected_status(components.predictor),
            "lstm": _injected_status(components.lstm_analyzer),
            "autoencoder": _injected_status(components.autoencoder_analyzer),
        }
    )
    behavioral_service = BehavioralLearningService(
        components.predictor,
        config=baseline_config,
        store=baseline_store,
        lstm_analyzer=components.lstm_analyzer,
        autoencoder_analyzer=components.autoencoder_analyzer,
    )
    return APIRuntime(
        behavioral_service=behavioral_service,
        risk_assessor=CombinedRiskAssessor(
            app_config.get("RISK_ASSESSMENT_CONFIG") or RiskAssessmentConfig()
        ),
        baseline_store=baseline_store,
        analysis_registry=components.analysis_registry or persistence,
        model_status=model_status,
        database=database,
        profiles_loaded=profiles_loaded,
    )


def _load_components(app_config: Mapping[str, Any]) -> APIComponents:
    model_dir = Path(app_config["MODEL_DIR"])
    model_status: dict[str, dict[str, object]] = {}

    def load_optional(
        name: str, filenames: tuple[str, ...], loader: Any
    ) -> Any | None:
        paths = [model_dir / filename for filename in filenames]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            model_status[name] = {
                "available": False,
                "status": "missing",
                "missing_artifacts": missing,
            }
            return None
        try:
            loaded = loader(*paths)
        except Exception as exc:
            logger.exception("Could not load %s model artifacts.", name)
            model_status[name] = {
                "available": False,
                "status": "load_error",
                "error_type": type(exc).__name__,
            }
            return None
        model_status[name] = {"available": True, "status": "loaded"}
        return loaded

    predictor = load_optional(
        "classifier",
        ("initial_fraud_classifier.joblib",),
        lambda path: load_predictor(path),
    )
    lstm = load_optional(
        "lstm",
        ("behavioral_lstm.keras", "behavioral_lstm_preprocessing.json"),
        lambda model, config: load_sequence_predictor(model, config),
    )
    autoencoder = load_optional(
        "autoencoder",
        (
            "behavioral_autoencoder.keras",
            "behavioral_autoencoder_preprocessing.json",
        ),
        lambda model, config: load_autoencoder_predictor(model, config),
    )
    return APIComponents(
        predictor=predictor,
        lstm_analyzer=lstm,
        autoencoder_analyzer=autoencoder,
        model_status=model_status,
    )


def _injected_status(component: Any | None) -> dict[str, object]:
    return {
        "available": component is not None,
        "status": "injected" if component is not None else "unavailable",
    }
