"""MongoDB persistence for analyses, alerts, reports, and trusted profiles."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Mapping

from pymongo.errors import DuplicateKeyError, PyMongoError

from fraud_detection.api.database import DatabaseUnavailable, MongoDatabase
from fraud_detection.behavioral.baseline import (
    BaselineConfig,
    BaselineStore,
    UserBaseline,
    normalize_activity,
)

logger = logging.getLogger(__name__)


class ProfileConflict(RuntimeError):
    """A trusted profile changed since the process loaded its baseline."""


class IdempotencyConflict(RuntimeError):
    """An idempotency key was reused with a different request body."""


class MongoPersistence:
    """Perform idempotent writes and bounded reads against MongoDB collections."""

    def __init__(self, database: MongoDatabase) -> None:
        self.database = database

    def get_idempotent_analysis(
        self,
        idempotency_key: str,
        *,
        request_fingerprint: str,
    ) -> dict[str, object] | None:
        try:
            document = self.database.collection("transactions").find_one(
                {"idempotency_key": idempotency_key},
            )
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc
        if document is None:
            return None
        self._check_fingerprint(document, request_fingerprint)
        return document["analysis_response"]

    @staticmethod
    def _check_fingerprint(
        transaction: Mapping[str, Any],
        request_fingerprint: str | None,
    ) -> None:
        stored_fingerprint = transaction.get("request_fingerprint")
        if (
            request_fingerprint is not None
            and stored_fingerprint is not None
            and stored_fingerprint != request_fingerprint
        ):
            raise IdempotencyConflict(
                "This Idempotency-Key was already used with a different request."
            )

    def record_analysis(
        self,
        activity: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> tuple[dict[str, object], bool]:
        """Persist the complete response and return it, including duplicate status."""
        transaction_id = str(response["request_id"])
        existing = self._find_transaction(
            idempotency_key=idempotency_key,
            request_id=transaction_id,
        )
        if existing is not None:
            self._check_fingerprint(existing, request_fingerprint)
            self._persist_related_records(existing)
            return dict(existing["analysis_response"]), True

        normalized = normalize_activity(activity)
        created_at = datetime.now(timezone.utc)
        analysis_response = dict(response)
        analysis_response["transaction_id"] = transaction_id
        analysis_response["persistence"] = {
            "status": "saved",
            "transaction_id": transaction_id,
        }
        transaction: dict[str, Any] = {
            "request_id": transaction_id,
            "transaction_id": transaction_id,
            "user_id": normalized["user_id"],
            "transaction_amount": normalized["transaction_amount"],
            "transaction_frequency": normalized["transaction_frequency"],
            "timestamp": normalized["timestamp"],
            "transaction_type": activity.get("transaction_type"),
            "activity_type": normalized["activity_type"],
            "transaction_status": activity.get("transaction_status"),
            "login_time": activity.get("login_time"),
            "device_type": activity.get("device_type"),
            "device_id": activity.get("device_id"),
            "location": activity.get("location"),
            "analysis_status": str(
                analysis_response.get("risk_assessment", {}).get("status", "unknown")
            ),
            "risk_score": analysis_response.get("risk_assessment", {}).get("risk_score"),
            "risk_category": analysis_response.get("risk_assessment", {}).get(
                "risk_category"
            ),
            "created_at": created_at,
            "analysis_response": analysis_response,
        }
        if idempotency_key is not None:
            transaction["idempotency_key"] = idempotency_key
        if request_fingerprint is not None:
            transaction["request_fingerprint"] = request_fingerprint

        try:
            self.database.collection("transactions").insert_one(transaction)
        except DuplicateKeyError:
            existing = self._find_transaction(
                idempotency_key=idempotency_key,
                request_id=transaction_id,
            )
            if existing is None:
                raise
            self._check_fingerprint(existing, request_fingerprint)
            self._persist_related_records(existing)
            return dict(existing["analysis_response"]), True
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc

        self._persist_related_records(
            {
                **transaction,
                "analysis_response": analysis_response,
                "created_at": created_at,
            }
        )
        return analysis_response, False

    def list_transactions(
        self, *, page: int, per_page: int
    ) -> dict[str, object]:
        collection = self.database.collection("transactions")
        try:
            total = collection.count_documents({})
            documents = list(
                collection.find({}, {"analysis_response": 1, "_id": 0})
                .sort("created_at", -1)
                .skip((page - 1) * per_page)
                .limit(per_page)
            )
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc
        return {
            "transactions": [
                _public_history_response(document["analysis_response"])
                for document in documents
            ],
            "count": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    def list_alerts(self, *, page: int, per_page: int) -> dict[str, object]:
        collection = self.database.collection("alerts")
        try:
            total = collection.count_documents({})
            documents = list(
                collection.find({}, {"_id": 0})
                .sort("created_at", -1)
                .skip((page - 1) * per_page)
                .limit(per_page)
            )
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc
        for document in documents:
            document["created_at"] = _utc_iso(document.get("created_at"))
        return {
            "alerts": documents,
            "count": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    def report(self) -> dict[str, object]:
        collection = self.database.collection("risk_assessments")
        try:
            total = collection.count_documents({})
            unavailable = collection.count_documents({"risk_category": None})
            category_counts = {
                category: collection.count_documents({"risk_category": category})
                for category in ("LOW RISK", "MEDIUM RISK", "HIGH RISK")
            }
            recent = list(
                collection.find(
                    {},
                    {
                        "created_at": 1,
                        "risk_category": 1,
                        "combined_risk_score": 1,
                        "available_signal_count": 1,
                        "recommended_action": 1,
                        "_id": 0,
                    },
                )
                .sort("created_at", -1)
                .limit(50)
            )
            retained_alerts = self.database.collection("alerts").count_documents({})
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc

        return {
            "assessments_total": total,
            "retained_assessments": total,
            "risk_category_counts": category_counts,
            "unavailable_assessments": unavailable,
            "retained_alerts": retained_alerts,
            "recent_assessments": [
                {
                    "created_at": _utc_iso(item.get("created_at")),
                    "risk_category": item.get("risk_category"),
                    "risk_score": item.get("combined_risk_score"),
                    "available_signal_count": item.get("available_signal_count", 0),
                    "recommended_action": item.get("recommended_action"),
                }
                for item in recent
            ],
            "storage": "mongodb; persistent across backend restarts",
        }

    def load_profiles(self) -> list[dict[str, Any]]:
        try:
            documents = list(self.database.collection("behavioral_profiles").find({}))
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc
        profiles = []
        for document in documents:
            document.pop("_id", None)
            profiles.append(document)
        return profiles

    def import_profiles_if_missing(
        self,
        profiles: list[Mapping[str, Any]],
        *,
        config: BaselineConfig,
    ) -> bool:
        collection = self.database.collection("behavioral_profiles")
        imported = False
        try:
            for item in profiles:
                profile = UserBaseline.from_dict(item)
                if collection.find_one({"user_id": profile.user_id}) is not None:
                    continue
                try:
                    collection.insert_one(
                        _profile_document(profile, config=config)
                    )
                except DuplicateKeyError:
                    if collection.find_one({"user_id": profile.user_id}) is None:
                        raise
                    continue
                imported = True
        except DuplicateKeyError as exc:
            raise ProfileConflict(
                "A behavioral profile was added during baseline migration."
            ) from exc
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc
        return imported

    def save_profile(
        self,
        profile: UserBaseline,
        *,
        expected_activity_count: int,
        config: BaselineConfig,
    ) -> None:
        document = _profile_document(profile, config=config)
        collection = self.database.collection("behavioral_profiles")
        if expected_activity_count == 0:
            try:
                collection.insert_one(document)
            except DuplicateKeyError as exc:
                raise ProfileConflict(
                    "A behavioral profile already exists; reload before updating it."
                ) from exc
            except PyMongoError as exc:
                raise self._unavailable(exc) from exc
            return

        try:
            result = collection.replace_one(
                {
                    "user_id": profile.user_id,
                    "trusted_activity_count": expected_activity_count,
                },
                document,
                upsert=False,
            )
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc
        if result.matched_count != 1:
            raise ProfileConflict(
                "The behavioral profile changed elsewhere; reload before updating it."
            )

    def _find_transaction(
        self, *, idempotency_key: str | None, request_id: str
    ) -> dict[str, Any] | None:
        query = (
            {"idempotency_key": idempotency_key}
            if idempotency_key is not None
            else {"request_id": request_id}
        )
        try:
            return self.database.collection("transactions").find_one(query)
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc

    def _persist_related_records(self, transaction: Mapping[str, Any]) -> None:
        response = transaction["analysis_response"]
        risk = response.get("risk_assessment") or {}
        transaction_id = str(transaction["transaction_id"])
        created_at = transaction["created_at"]
        risk_record = {
            "transaction_id": transaction_id,
            "user_id": transaction["user_id"],
            "classifier_signal": response.get("classifier"),
            "behavioral_signal": response.get("behavioral"),
            "lstm_signal": response.get("lstm"),
            "autoencoder_signal": response.get("autoencoder"),
            "combined_risk_score": risk.get("risk_score"),
            "risk_category": risk.get("risk_category"),
            "explanations": response.get("explanations", []),
            "recommended_action": response.get("recommended_action"),
            "available_signal_count": risk.get("available_signal_count", 0),
            "risk_assessment": risk,
            "created_at": created_at,
        }
        try:
            self.database.collection("risk_assessments").replace_one(
                {"transaction_id": transaction_id}, risk_record, upsert=True
            )
            if _requires_alert(response):
                reasons = list(response.get("explanations", []))
                self.database.collection("alerts").update_one(
                    {"alert_id": transaction_id},
                    {
                        "$setOnInsert": {
                            "alert_id": transaction_id,
                            "user_id": transaction["user_id"],
                            "transaction_id": transaction_id,
                            "risk_category": risk.get("risk_category"),
                            "risk_score": risk.get("risk_score"),
                            "explanation": reasons,
                            "reasons": reasons,
                            "recommended_action": response.get("recommended_action"),
                            "created_at": created_at,
                            "status": "open",
                        }
                    },
                    upsert=True,
                )
        except PyMongoError as exc:
            raise self._unavailable(exc) from exc

    def _unavailable(self, error: PyMongoError) -> DatabaseUnavailable:
        self.database.mark_unavailable(error)
        logger.warning("MongoDB operation failed (%s).", type(error).__name__)
        return DatabaseUnavailable("MongoDB persistence is currently unavailable.")


class MongoBaselineStore(BaselineStore):
    """Cache MongoDB profiles and persist only explicitly trusted updates."""

    def __init__(
        self,
        persistence: MongoPersistence,
        *,
        legacy_path: str | None = None,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__()
        self.persistence = persistence
        self.config = config or BaselineConfig()
        self.profiles_loaded = False
        if persistence.database.available:
            try:
                profiles = persistence.load_profiles()
                if legacy_path is not None:
                    path = Path(legacy_path)
                    if path.is_file():
                        legacy_store = BaselineStore.load(path)
                        legacy_profiles = []
                        for user_id in legacy_store.users():
                            profile = legacy_store.get(user_id)
                            if profile is not None:
                                legacy_profiles.append(profile.to_dict())
                        persistence.import_profiles_if_missing(
                            legacy_profiles,
                            config=self.config,
                        )
                        profiles = persistence.load_profiles()
                self.load_profiles(profiles)
            except (DatabaseUnavailable, ProfileConflict, ValueError):
                logger.warning("Behavioral profiles could not be loaded from MongoDB.")
            else:
                self.profiles_loaded = True

    def add_trusted(self, activity: Mapping[str, Any]) -> UserBaseline:
        user_id = normalize_activity(activity)["user_id"]
        previous = self.get(user_id)
        previous_count = previous.trusted_activity_count if previous else 0
        before = deepcopy(self._profiles)
        profile = super().add_trusted(activity)
        try:
            self.persistence.save_profile(
                profile,
                expected_activity_count=previous_count,
                config=self.config,
            )
        except (DatabaseUnavailable, ProfileConflict):
            self._profiles = before
            raise
        return profile


def _requires_alert(response: Mapping[str, Any]) -> bool:
    risk = response.get("risk_assessment")
    if not isinstance(risk, Mapping):
        return False
    behavioral = response.get("behavioral")
    return (
        risk.get("risk_category") in {"MEDIUM RISK", "HIGH RISK"}
        or (
            isinstance(behavioral, Mapping)
            and behavioral.get("requires_additional_verification") is True
        )
    )


def _utc_iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _profile_document(
    profile: UserBaseline,
    *,
    config: BaselineConfig,
) -> dict[str, Any]:
    document = profile.to_dict()
    document["login_hours"] = {
        str(hour): count for hour, count in profile.login_hours.items()
    }
    is_established = (
        profile.trusted_activity_count >= config.minimum_trusted_activities
        and profile.history_days >= config.minimum_history_days
    )
    document["user_status"] = (
        "ESTABLISHED USER" if is_established else "LEARNING USER"
    )
    document["last_updated"] = datetime.now(timezone.utc)
    return document


def _public_history_response(
    response: Mapping[str, Any],
) -> dict[str, object]:
    public = deepcopy(dict(response))
    behavioral = public.get("behavioral")
    if not isinstance(behavioral, dict):
        return public
    deviations = behavioral.get("deviations")
    if not isinstance(deviations, list):
        return public
    for deviation in deviations:
        if isinstance(deviation, dict) and deviation.get("field") in {
            "device_id",
            "location",
        }:
            deviation.pop("observed", None)
            deviation.pop("expected", None)
    return public
