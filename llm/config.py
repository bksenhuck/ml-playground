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
    use_llm: bool      # Whether Qwen is enabled
    model_path: str    # Local or GCS path for Qwen weights
    project_id: str    # GCP project ID (for GCS downloads)

    @property
    def llm_ready(self) -> bool:
        """True only when LLM is enabled."""
        return self.use_llm


def load_config() -> LLMConfig:
    """Load LLM configuration from environment variables.

    Optionally reads a .env file via python-dotenv (override=False).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    use_llm = _env_bool("USE_LLM", default=True)
    model_path = os.environ.get("MODEL_PATH", "./models/Qwen/Qwen2.5-0.5B-Instruct")
    project_id = os.environ.get("GCP_PROJECT_ID", "")

    return LLMConfig(
        use_llm=use_llm,
        model_path=model_path,
        project_id=project_id,
    )
