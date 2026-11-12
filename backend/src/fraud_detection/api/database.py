"""MongoDB connection lifecycle and index setup for the Flask API."""

from __future__ import annotations

import logging
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class DatabaseUnavailable(RuntimeError):
    """A MongoDB operation cannot be completed."""


class MongoDatabase:
    """Own one reusable MongoClient and expose safe database health status."""

    def __init__(
        self,
        *,
        uri: str | None,
        database_name: str,
        server_selection_timeout_ms: int = 2_000,
        client: Any | None = None,
    ) -> None:
        self.database_name = database_name
        self._client = client
        self._database = None
        self._available = False
        self._status = "not_configured"
        self._indexes_ready = False
        if self._client is None and not uri:
            return
        try:
            if self._client is None:
                self._client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=server_selection_timeout_ms,
                    tz_aware=True,
                )
            self._database = self._client[database_name]
            self.check_health()
        except (DatabaseUnavailable, PyMongoError, ValueError) as exc:
            self._record_failure(exc)

    @property
    def available(self) -> bool:
        return self._available

    def collection(self, name: str):
        if self._database is None:
            raise DatabaseUnavailable("MongoDB is not configured.")
        return self._database[name]

    def check_health(self) -> dict[str, object]:
        if self._client is None or self._database is None:
            return self.health()
        try:
            self._client.admin.command("ping")
            if not self._indexes_ready:
                self._create_indexes()
        except (DatabaseUnavailable, PyMongoError) as exc:
            self._record_failure(exc)
        else:
            self._available = True
            self._status = "connected"
        return self.health()

    def health(self) -> dict[str, object]:
        return {
            "available": self._available,
            "status": self._status,
            "database": self.database_name if self._available else None,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def mark_unavailable(self, error: Exception) -> None:
        self._record_failure(error)

    def _create_indexes(self) -> None:
        try:
            self.collection("transactions").create_index(
                [("request_id", ASCENDING)], unique=True, name="uq_transaction_request"
            )
            self.collection("transactions").create_index(
                [("idempotency_key", ASCENDING)],
                unique=True,
                partialFilterExpression={"idempotency_key": {"$type": "string"}},
                name="uq_transaction_idempotency",
            )
            self.collection("transactions").create_index(
                [("user_id", ASCENDING), ("created_at", DESCENDING)],
                name="ix_transactions_user_created",
            )
            self.collection("transactions").create_index(
                [("created_at", DESCENDING)], name="ix_transactions_created"
            )
            self.collection("risk_assessments").create_index(
                [("transaction_id", ASCENDING)],
                unique=True,
                name="uq_risk_assessment_transaction",
            )
            self.collection("risk_assessments").create_index(
                [("created_at", DESCENDING)], name="ix_risk_assessments_created"
            )
            self.collection("risk_assessments").create_index(
                [("risk_category", ASCENDING)],
                name="ix_risk_assessments_category",
            )
            self.collection("alerts").create_index(
                [("alert_id", ASCENDING)], unique=True, name="uq_alert_id"
            )
            self.collection("alerts").create_index(
                [("created_at", DESCENDING)], name="ix_alerts_created"
            )
            self.collection("alerts").create_index(
                [("status", ASCENDING), ("created_at", DESCENDING)],
                name="ix_alerts_status_created",
            )
            self.collection("behavioral_profiles").create_index(
                [("user_id", ASCENDING)], unique=True, name="uq_behavioral_profile_user"
            )
            self._indexes_ready = True
            self._available = True
        except PyMongoError as exc:
            self._record_failure(exc)
            raise DatabaseUnavailable(
                "MongoDB indexes could not be initialized."
            ) from exc

    def _record_failure(self, error: Exception) -> None:
        self._available = False
        self._status = "unavailable"
        logger.warning("MongoDB is unavailable (%s).", type(error).__name__)
