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

    # ── Cloud Run ─────────────────────────────────────────────────────────────
    CLOUDRUN_SERVICE: str = os.environ.get("CLOUDRUN_SERVICE", "ml-playground")
    # Full image URI, e.g. gcr.io/my-project/ml-playground:latest
    # Falls back to gcr.io/{GCP_PROJECT_ID}/ml-playground if not set.
    GCR_IMAGE: str = os.environ.get("GCR_IMAGE", "")

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
        if not cls.GCP_PROJECT_ID:
            raise EnvironmentError(
                "Missing required env var: GCP_PROJECT_ID\n"
                "Add it to your .env file or export it before running."
            )


settings = DeploySettings()
