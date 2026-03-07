"""Qwen 2.5 0.5B client wrapper — local weights orchestration layer.

Design principle: this class NEVER raises. Every failure path returns None
so the caller (service.py) can fall back to the local engine cleanly.

The HuggingFace pipeline is loaded once and cached process-wide
(_CLIENT_CACHE) so repeated questions don't reload the model.

Requirements:
  pip install transformers torch accelerate
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MAX_NEW_TOKENS = 512
_DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH", "./models/Qwen/Qwen2.5-0.5B-Instruct"
)

# Process-level cache: model_path → QwenClient (already initialised)
_CLIENT_CACHE: dict[str, "QwenClient"] = {}


class QwenClient:
    """Wraps Qwen 2.5 with Hugging Face transformers."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        self._model_path = model_path
        self._pipe = None
        self._model_ready = False

    # ── Singleton factory ──────────────────────────────────────────────────

    @classmethod
    def get_or_create(
        cls, model_path: str = _DEFAULT_MODEL_PATH
    ) -> "QwenClient":
        """Cached client for *model_path*, created and loaded once."""
        if model_path not in _CLIENT_CACHE:
            client = cls(model_path)
            client._init_model()
            _CLIENT_CACHE[model_path] = client
        return _CLIENT_CACHE[model_path]

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_model(self) -> bool:
        """Load Qwen once per instance. Returns False on any error."""
        if self._model_ready:
            return True

        print(f"[QwenClient] Checking path: {self._model_path}", flush=True)
        if not os.path.exists(self._model_path):
            print(
                f"[QwenClient] Model NOT found: {self._model_path}",
                flush=True,
            )
            logger.warning(
                "[QwenClient] Model path '%s' not found. "
                "Run: python llm/download_model.py",
                self._model_path,
            )
            return False
        print("[QwenClient] Model found — loading pipeline...", flush=True)

        # Lazy-import so the app starts normally even without torch
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        except ImportError:
            print("[QwenClient] transformers not installed!", flush=True)
            logger.error(
                "[QwenClient] 'transformers' not installed. "
                "Run: pip install transformers torch accelerate"
            )
            return False

        try:
            logger.info(
                "[QwenClient] Loading Qwen from '%s'…", self._model_path
            )
            self._pipe = hf_pipeline(
                "text-generation",
                model=self._model_path,
                dtype="auto",
                device_map="auto",
            )
            self._model_ready = True
            print("[QwenClient] Qwen 2.5 loaded successfully.", flush=True)
            logger.info("[QwenClient] Qwen 2.5 loaded successfully.")
            return True
        except Exception as exc:
            print(f"[QwenClient] Load failed: {exc}", flush=True)
            logger.error("[QwenClient] Initialisation failed: %s", exc)
            return False

    # ── Inference ──────────────────────────────────────────────────────────

    def generate(
        self, prompt: str, max_new_tokens: int = _MAX_NEW_TOKENS
    ) -> str | None:
        """Send *prompt* to Qwen and return the response text, or None."""
        if not self._model_ready or self._pipe is None:
            print("[QwenClient] generate() — model not ready.", flush=True)
            logger.warning(
                "[QwenClient] generate() called but model is not ready."
            )
            return None

        try:
            from llm.guardrails import build_system_prompt  # noqa: PLC0415
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": prompt},
            ]
            from transformers import GenerationConfig  # noqa: PLC0415
            gen_cfg = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            outputs = self._pipe(messages, generation_config=gen_cfg)
            text = outputs[0]["generated_text"][-1]["content"]
            if not text or not text.strip():
                logger.warning("[QwenClient] Empty response from model.")
                return None
            return text.strip()
        except Exception as exc:
            print(f"[QwenClient] generate() failed: {exc}", flush=True)
            logger.warning("[QwenClient] generate() failed: %s", exc)
            return None
