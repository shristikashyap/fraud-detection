"""Thread-safe, process-local summaries for API reports and review alerts."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import threading
from typing import Any, Mapping
from uuid import uuid4

from fraud_detection.api.persistence import IdempotencyConflict


class AnalysisRegistry:
    """Test adapter retaining bounded summaries without a MongoDB dependency."""

    def __init__(self, *, analysis_limit: int = 1000, alert_limit: int = 500) -> None:
        if analysis_limit < 1 or alert_limit < 1:
            raise ValueError("Analysis and alert retention limits must be positive.")
        self._analyses: deque[dict[str, object]] = deque(maxlen=analysis_limit)
        self._alerts: deque[dict[str, object]] = deque(maxlen=alert_limit)
        self._category_counts: Counter[str] = Counter()
        self._unavailable_count = 0
        self._total_count = 0
        self._transactions: deque[dict[str, object]] = deque(
            maxlen=analysis_limit
        )
        self._idempotent: dict[str, dict[str, object]] = {}
        self._fingerprints: dict[str, str] = {}
        self._lock = threading.RLock()

    def get_idempotent_analysis(
        self,
        idempotency_key: str,
        *,
        request_fingerprint: str,
    ) -> dict[str, object] | None:
        with self._lock:
            response = self._idempotent.get(idempotency_key)
            if (
                response is not None
                and self._fingerprints.get(idempotency_key) != request_fingerprint
            ):
                raise IdempotencyConflict(
                    "This Idempotency-Key was already used with a different request."
                )
            return dict(response) if response is not None else None

    def record_analysis(
        self,
        activity: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> tuple[dict[str, object], bool]:
        with self._lock:
            if idempotency_key is not None and idempotency_key in self._idempotent:
                if (
                    request_fingerprint is not None
                    and self._fingerprints.get(idempotency_key)
                    != request_fingerprint
                ):
                    raise IdempotencyConflict(
                        "This Idempotency-Key was already used with a different request."
                    )
                return dict(self._idempotent[idempotency_key]), True
            saved_response = dict(response)
            transaction_id = str(saved_response["request_id"])
            saved_response["transaction_id"] = transaction_id
            saved_response["persistence"] = {
                "status": "test_memory",
                "transaction_id": transaction_id,
            }
            self._transactions.appendleft(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "analysis_response": saved_response,
                }
            )
            self.record(str(activity["user_id"]), saved_response)
            if idempotency_key is not None:
                self._idempotent[idempotency_key] = saved_response
                if request_fingerprint is not None:
                    self._fingerprints[idempotency_key] = request_fingerprint
            return saved_response, False

    def record(
        self, user_id: str, response: Mapping[str, Any]
    ) -> dict[str, object] | None:
        risk = response["risk_assessment"]
        if not isinstance(risk, Mapping):
            raise TypeError("Risk assessment response must be a mapping.")
        category = risk.get("risk_category")
        score = risk.get("risk_score")
        created_at = datetime.now(timezone.utc).isoformat()
        summary: dict[str, object] = {
            "created_at": created_at,
            "risk_category": category,
            "risk_score": score,
            "recommended_action": risk.get("recommendation"),
            "available_signal_count": risk.get("available_signal_count", 0),
        }
        with self._lock:
            self._total_count += 1
            if category in {"LOW RISK", "MEDIUM RISK", "HIGH RISK"}:
                self._category_counts[str(category)] += 1
            else:
                self._unavailable_count += 1
            self._analyses.append(summary)

            if (
                category in {"MEDIUM RISK", "HIGH RISK"}
                or response.get("behavioral", {}).get(
                    "requires_additional_verification"
                )
                is True
            ):
                alert: dict[str, object] = {
                    "alert_id": uuid4().hex,
                    "created_at": created_at,
                    "user_id": user_id,
                    "risk_category": category,
                    "risk_score": score,
                    "recommended_action": risk.get("recommendation"),
                    "reasons": list(risk.get("explanations", [])),
                    "status": "requires_additional_verification",
                }
                self._alerts.append(alert)
                return dict(alert)
        return None

    def alerts(self, *, limit: int) -> list[dict[str, object]]:
        with self._lock:
            return [dict(alert) for alert in list(self._alerts)[-limit:][::-1]]

    def list_alerts(self, *, page: int, per_page: int) -> dict[str, object]:
        with self._lock:
            alerts = list(reversed(self._alerts))
            start = (page - 1) * per_page
            selected = [dict(alert) for alert in alerts[start : start + per_page]]
            total = len(alerts)
        return {
            "alerts": selected,
            "count": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    def list_transactions(
        self, *, page: int, per_page: int
    ) -> dict[str, object]:
        with self._lock:
            records = list(self._transactions)
            start = (page - 1) * per_page
            selected = [
                dict(item["analysis_response"])
                for item in records[start : start + per_page]
            ]
            total = self._total_count
        return {
            "transactions": selected,
            "count": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }

    def report(self) -> dict[str, object]:
        with self._lock:
            return {
                "assessments_total": self._total_count,
                "retained_assessments": len(self._analyses),
                "risk_category_counts": {
                    category: self._category_counts[category]
                    for category in ("LOW RISK", "MEDIUM RISK", "HIGH RISK")
                },
                "unavailable_assessments": self._unavailable_count,
                "retained_alerts": len(self._alerts),
                "recent_assessments": [dict(item) for item in reversed(self._analyses)],
                "storage": "test-only process memory",
            }
