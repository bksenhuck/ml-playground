"""Deploy configuration — reads GCS and Cloud Run settings from environment.

All values come from environment variables (or a .env file loaded by
python-dotenv before this module is imported).
"""
from __future__ import annotations

import os
from pathlib import Path

# Try to load .env from project root (no-op if python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]


class DeploySettings:
    # ── GCP ──────────────────────────────────────────────────────────────────
    GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")
    GCP_REGION: str = os.environ.get("GCP_REGION", "us-central1")

    # ── Cloud Run ─────────────────────────────────────────────────────────────
    CLOUDRUN_SERVICE: str = os.environ.get("CLOUDRUN_SERVICE", "ml-playground")
    # Full image URI, e.g. gcr.io/my-project/ml-playground:latest
    # Falls back to gcr.io/{GCP_PROJECT_ID}/ml-playground if not set.
    GCR_IMAGE: str = os.environ.get("GCR_IMAGE", "")

    # ── GCS artifacts ─────────────────────────────────────────────────────────
    # gs://your-bucket/mlflow.db
    GCS_MLFLOW_URI: str = os.environ.get("GCS_MLFLOW_URI", "")

    # ── Local paths ───────────────────────────────────────────────────────────
    @classmethod
    def get_mlflow_db_path(cls) -> Path:
        return ROOT / "mlflow.db"

    @classmethod
    def get_docker_image(cls) -> str:
        if cls.GCR_IMAGE:
            return cls.GCR_IMAGE
        if cls.GCP_PROJECT_ID:
            return f"gcr.io/{cls.GCP_PROJECT_ID}/ml-playground"
        raise ValueError(
            "Set GCR_IMAGE or GCP_PROJECT_ID in your .env"
        )

    @classmethod
    def validate(cls) -> None:
        """Raise if required deploy vars are missing."""
        missing = []
        if not cls.GCP_PROJECT_ID:
            missing.append("GCP_PROJECT_ID")
        if not cls.GCS_MLFLOW_URI:
            missing.append("GCS_MLFLOW_URI")
        if missing:
            raise EnvironmentError(
                f"Missing required env vars: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in the values."
            )


settings = DeploySettings()
