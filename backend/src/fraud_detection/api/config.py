"""Default API configuration using server-owned paths and a CORS allowlist."""

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[3]


def default_config() -> dict[str, object]:
    return {
        "MODEL_DIR": BACKEND_ROOT / "models",
        "BASELINE_STORE_PATH": (
            BACKEND_ROOT / "data" / "behavioral" / "trusted_profiles.json"
        ),
        "CORS_ALLOWED_ORIGINS": (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ),
        "MAX_CONTENT_LENGTH": 64 * 1024,
        "MAX_RECENT_ACTIVITIES": 50,
        "ANALYSIS_RETENTION": 1000,
        "ALERT_RETENTION": 500,
        "MONGO_URI": os.environ.get("MONGO_URI"),
        "MONGO_DB_NAME": os.environ.get("MONGO_DB_NAME", "fraud_detection"),
        "MONGO_SERVER_SELECTION_TIMEOUT_MS": 2_000,
    }
