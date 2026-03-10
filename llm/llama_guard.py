"""Llama Guard 3 1B client — input and output safety moderation.

Architecture:
  - Singleton per model path (same pattern as QwenClient).
  - NEVER raises: every failure path returns a safe default so the
    caller can continue with the regex guardrails as a fallback.
  - Lazy-loaded: the model is NOT imported at module level, so the app
    starts normally even if the weights are not present.

Setup:
  1. Accept the Meta license at https://huggingface.co/meta-llama/Llama-Guard-3-1B
  2. Set LLAMA_GUARD_PATH in .env  (or use the default below).
  3. Run: python llm/download_model.py --model llama-guard

Llama Guard 3 output format:
  safe            -> the content is safe
  unsafe\\nS<N>    -> category S<N> violated (see SAFETY_CATEGORIES)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_GUARD_PATH = os.environ.get(
    "LLAMA_GUARD_PATH",
    "./models/meta-llama/Llama-Guard-3-1B",
)

# Process-level cache
_GUARD_CACHE: dict[str, "LlamaGuardClient"] = {}

# Safety categories defined by Meta (LlamaGuard-3)
SAFETY_CATEGORIES = """S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex-Related Crimes.
S4: Child Sexual Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Suicide & Self-Harm.
S12: Sexual Content.
S13: Elections.
S14: Code Interpreter Abuse."""

_CLASSIFY_PROMPT = """\
[INST] Task: Check if there is unsafe content in the 'User' message \
according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: {message}
<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of \
violated categories. [/INST]"""


class LlamaGuardClient:
    """Wraps Llama Guard 3 1B for content safety classification."""

    def __init__(self, model_path: str = _DEFAULT_GUARD_PATH) -> None:
        self._model_path = model_path
        self._pipe = None
        self._ready = False

    # ── Singleton factory ──────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @classmethod
    def get_or_create(
        cls, model_path: str = _DEFAULT_GUARD_PATH
    ) -> "LlamaGuardClient":
        """Return (or create) a cached client for *model_path*."""
        if model_path not in _GUARD_CACHE:
            client = cls(model_path)
            client._init_model()
            _GUARD_CACHE[model_path] = client
        return _GUARD_CACHE[model_path]

    # ── Init ──────────────────────────────────────────────────────────────

    def _init_model(self) -> bool:
        """Load Llama Guard once. Returns False on any error."""
        if self._ready:
            return True

        print(
            f"[LlamaGuard] Checking path: {self._model_path}", flush=True
        )
        if not os.path.exists(self._model_path):
            logger.warning(
                "[LlamaGuard] Model not found at '%s'. "
                "Falling back to regex-only guardrails.",
                self._model_path,
            )
            return False

        # Verify torch is importable first (pipeline depends on it)
        try:
            import torch  # noqa: PLC0415
            print(f"[LlamaGuard] torch {torch.__version__} OK", flush=True)
        except Exception as exc:
            import traceback
            logger.error("[LlamaGuard] torch not importable: %s\n%s", exc, traceback.format_exc())
            return False

        try:
            # Use direct submodule import to bypass the lazy-loader in __init__.py
            from transformers.pipelines import pipeline as hf_pipeline  # noqa: PLC0415
        except ImportError as exc:
            import traceback
            logger.error("[LlamaGuard] transformers.pipeline not importable: %s\n%s", exc, traceback.format_exc())
            return False

        try:
            print("[LlamaGuard] Loading Llama Guard…", flush=True)
            self._pipe = hf_pipeline(
                "text-generation",
                model=self._model_path,
                torch_dtype="auto",
                device_map={"": "cpu"},
            )
            self._ready = True
            print("[LlamaGuard] Loaded successfully.", flush=True)
            logger.info("[LlamaGuard] Loaded successfully.")
            return True
        except Exception as exc:
            print(f"[LlamaGuard] Load failed: {exc}", flush=True)
            logger.error("[LlamaGuard] Load failed: %s", exc)
            return False

    # ── Classify ──────────────────────────────────────────────────────────

    def classify(self, text: str) -> tuple[bool, str | None]:
        """Classify *text* for safety.

        Returns:
            ``(is_safe, violated_category)``
            If the model is not ready, returns ``(True, None)`` —
            fail-open so the regex layer stays as the last resort.
        """
        if not self._ready or self._pipe is None:
            logger.debug(
                "[LlamaGuard] Model not ready — skipping moderation."
            )
            return True, None

        try:
            prompt = _CLASSIFY_PROMPT.format(
                categories=SAFETY_CATEGORIES,
                message=text,
            )
            out = self._pipe(
                prompt,
                max_new_tokens=20,
                temperature=1.0,
                do_sample=False,
            )
            generated = out[0]["generated_text"]
            # Strip the input prompt to get only the model's response
            response = generated[len(prompt):].strip().lower()
            if response.startswith("unsafe"):
                lines = response.split("\n")
                category = lines[1].strip() if len(lines) > 1 else None
                logger.warning(
                    "[LlamaGuard] [!] UNSAFE CONTENT DETECTED: category='%s' | text_preview='%.60s'",
                    category, text,
                )
                return False, category
            
            logger.info("[LlamaGuard] Content is SAFE.")
            return True, None
        except Exception as exc:
            logger.error("[LlamaGuard] ERROR: classify() failed with: %s", exc)
            return True, None  # fail-open


# ── Module-level helper ───────────────────────────────────────────────────────

def is_available(model_path: str = _DEFAULT_GUARD_PATH) -> bool:
    """Return True if the Llama Guard model is already loaded and ready.

    Non-blocking: never triggers a model load.  The background preload thread
    (in app.py) is the only place that loads the model.  This prevents a
    concurrent torch import on the request thread while the preload is running.
    """
    client = _GUARD_CACHE.get(model_path)
    return client is not None and client.is_ready


def moderate(
    text: str,
    model_path: str = _DEFAULT_GUARD_PATH,
) -> tuple[bool, str | None]:
    """Convenience wrapper — classify *text* using the cached client."""
    return LlamaGuardClient.get_or_create(model_path).classify(text)
