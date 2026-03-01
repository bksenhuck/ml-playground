"""LLM insight service — hybrid orchestration layer.

Architecture:
  1. Trim runs_df to max 5 rows x relevant columns (context window efficiency).
  2. If Gemini is enabled and configured -> try GeminiClient.generate().
     On success -> return the response directly.
     On failure (network, quota, auth, empty response) -> fall through.
  3. Always fall back to the local rule-based engine so the assistant
     returns a useful answer even without any internet access.

Logging:
  - "Using Gemini"              -- cloud path taken
  - "Falling back to local engine" -- local fallback taken
"""
from __future__ import annotations

import logging
import pandas as pd

from llm.config import load_config
from llm.fallback_engine import generate_fallback_insight
from llm.prompts import build_experiment_prompt

logger = logging.getLogger(__name__)

# Columns forwarded to the LLM / fallback engine as context
_CONTEXT_COLS = [
    "run_name", "model", "n_features", "scaling", "class_weight",
    "poly_features", "test_size",
    "metric_accuracy", "metric_precision", "metric_recall",
    "metric_f1", "metric_roc_auc",
]
_MAX_RUNS = 5


def generate_insight(
    question: str, runs_df: pd.DataFrame
) -> tuple[str, str]:
    """Return ``(answer, source)`` where source is ``"llm"`` or ``"local"``.

    This is the single public entry point called by the Dash callback.
    It always returns a non-empty answer string.

    Args:
        question: User's free-text question about their experiments.
        runs_df:  DataFrame of runs to include as context (already filtered
                  to selected rows by the callback).

    Returns:
        Tuple of (insight text in Brazilian Portuguese, source label).
    """
    # -- Input validation -----------------------------------------------------
    if not question or not question.strip():
        return (
            "Por favor, escreva uma pergunta sobre seus experimentos.",
            "local",
        )

    if runs_df is None or runs_df.empty:
        return (
            "Nenhum experimento disponível. "
            "Execute experimentos e selecione ao menos um na tabela.",
            "local",
        )

    # -- Context trimming -----------------------------------------------------
    keep = [c for c in _CONTEXT_COLS if c in runs_df.columns]
    context_df = runs_df[keep].head(_MAX_RUNS).copy()
    q = question.strip()

    # -- Cloud path: Gemini ---------------------------------------------------
    cfg = load_config()
    if cfg.gemini_ready:
        logger.info("Using Gemini (%s / %s)", cfg.project_id, cfg.model)
        response = _try_gemini(cfg, q, context_df)
        if response:
            return response, "llm"
        logger.info(
            "Falling back to local engine (Gemini returned no response)"
        )
    else:
        logger.info(
            "Falling back to local engine (Gemini not configured — "
            "set USE_GEMINI=true and GCP_PROJECT_ID to enable)"
        )

    # -- Local fallback: rule-based engine ------------------------------------
    return generate_fallback_insight(q, context_df), "local"


# -- Private helpers ----------------------------------------------------------

def _try_gemini(
    cfg, question: str, context_df: pd.DataFrame
) -> str | None:
    """Attempt a Gemini call. Returns response text or None on any failure."""
    try:
        from llm.client_gemini import GeminiClient  # noqa: PLC0415
        client = GeminiClient(cfg.project_id, cfg.region, cfg.model)
        prompt = build_experiment_prompt(question, context_df)
        return client.generate(prompt)
    except Exception as exc:
        logger.warning("Gemini call error: %s", exc)
        return None
