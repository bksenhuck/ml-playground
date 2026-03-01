"""LLM configuration — reads environment variables with optional .env support.

Architecture note:
  All config is centralised here so every module imports from one place.
  python-dotenv is optional: if not installed the app still works as long
  as the env vars are set another way (Docker, shell export, etc.).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(key: str, default: bool = True) -> bool:
    """Parse a boolean environment variable. Missing key returns *default*."""
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMConfig:
    use_gemini: bool   # Whether Gemini is enabled at all
    project_id: str    # GCP project ID (empty → Gemini disabled)
    region: str        # Vertex AI region
    model: str         # Gemini model name

    @property
    def gemini_ready(self) -> bool:
        """True only when Gemini is enabled AND a project ID is provided."""
        return self.use_gemini and bool(self.project_id)


def load_config() -> LLMConfig:
    """Load LLM configuration from environment variables.

    Optionally reads a .env file via python-dotenv (override=False so real
    env vars always win). Falls back silently when dotenv is not installed.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass  # dotenv optional — not a hard requirement

    return LLMConfig(
        use_gemini=_env_bool("USE_GEMINI", default=True),
        project_id=os.environ.get("GCP_PROJECT_ID", ""),
        # Accept both GCP_REGION (new) and GCP_LOCATION (legacy) so existing
        # setups that set GCP_LOCATION continue to work without changes.
        region=(
            os.environ.get("GCP_REGION")
            or os.environ.get("GCP_LOCATION", "us-central1")
        ),
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
    )
