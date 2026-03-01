"""Vertex AI Gemini client — thin, fault-tolerant wrapper.

Design principle: this class NEVER raises. Every failure path returns None
so the caller (service.py) can fall back to the local engine cleanly.

Requirements:
  pip install google-cloud-aiplatform
  gcloud auth application-default login   # or set GOOGLE_APPLICATION_CREDENTIALS
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 512


class GeminiClient:
    """Wraps Vertex AI Gemini with lazy initialisation and safe error handling."""

    def __init__(self, project_id: str, region: str, model: str) -> None:
        self._project = project_id
        self._region = region
        self._model = model
        self._vertex_ready = False  # flipped to True on first successful init

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_vertex(self) -> bool:
        """Initialise Vertex AI once per instance. Returns False on any error."""
        if self._vertex_ready:
            return True
        try:
            import vertexai  # noqa: PLC0415
            vertexai.init(project=self._project, location=self._region)
            self._vertex_ready = True
            return True
        except ImportError:
            logger.warning(
                "google-cloud-aiplatform not installed — Gemini unavailable. "
                "Run: pip install google-cloud-aiplatform"
            )
            return False
        except Exception as exc:
            logger.warning("Vertex AI initialisation failed: %s", exc)
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_output_tokens: int = _MAX_OUTPUT_TOKENS) -> str | None:
        """Send *prompt* to Gemini and return the response text.

        Returns:
            Response string on success, or None if the call fails for any
            reason (network error, quota, invalid credentials, etc.).
        """
        if not self._init_vertex():
            return None

        try:
            from vertexai.generative_models import GenerationConfig, GenerativeModel  # noqa: PLC0415

            model = GenerativeModel(self._model)
            response = model.generate_content(
                prompt,
                generation_config=GenerationConfig(max_output_tokens=max_output_tokens),
            )
            text = response.text
            if not text or not text.strip():
                logger.warning("Gemini returned an empty response.")
                return None
            return text.strip()

        except Exception as exc:
            # Log the specific error so operators can diagnose quota / auth issues
            # without crashing the whole application.
            logger.warning("Gemini generate_content failed: %s", exc)
            return None
