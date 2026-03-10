"""Deploy configuration — reads Cloud Run settings from environment.

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


class DeploySettings:
    # ── GCP ──────────────────────────────────────────────────────────────────
    GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")
    GCP_REGION: str = os.environ.get("GCP_REGION", "us-central1")

    # ── Cloud Run ────────────────────────────────────────────────────────────
    CLOUDRUN_SERVICE: str = os.environ.get(
        "CLOUDRUN_SERVICE", "ml-playground"
    )
    # Full image URI — falls back to Artifact Registry if not set.
    GCR_IMAGE: str = os.environ.get("GCR_IMAGE", "")

    # ── GCS model paths ──────────────────────────────────────────────────────
    GCS_MODEL_PATH: str = os.environ.get("GCS_MODEL_PATH", "")
    GCS_GUARD_PATH: str = os.environ.get("GCS_GUARD_PATH", "")
    GCS_MLFLOW_URI: str = os.environ.get("GCS_MLFLOW_URI", "")

    @classmethod
    def get_docker_image(cls) -> str:
        if cls.GCR_IMAGE:
            return cls.GCR_IMAGE
        if cls.GCP_PROJECT_ID:
            ar = f"{cls.GCP_REGION}-docker.pkg.dev"
            return f"{ar}/{cls.GCP_PROJECT_ID}/ml-playground/ml-playground"
        raise ValueError(
            "Set GCR_IMAGE or GCP_PROJECT_ID in your .env"
        )

    @classmethod
    def validate(cls) -> None:
        """Raise if required deploy vars are missing."""
        missing = [
            name for name, val in [
                ("GCP_PROJECT_ID", cls.GCP_PROJECT_ID),
                ("GCS_MODEL_PATH", cls.GCS_MODEL_PATH),
                ("GCS_GUARD_PATH", cls.GCS_GUARD_PATH),
            ] if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required env vars: {', '.join(missing)}\n"
                "Add them to your .env file or export before running."
            )


settings = DeploySettings()
