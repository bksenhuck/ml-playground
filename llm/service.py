"""LLM insight service — orchestration layer.

Architecture:
  1. Trim runs_df to max 5 rows x relevant columns (context window efficiency).
  2. Guardrails pre-filter: Llama Guard (optional) → regex topic filter.
  3. If USE_LLM=true → try Qwen 2.5 0.5B (local weights, CPU inference).
     On success → return the response.
     On failure → fall through to local rule-based engine.
  4. Local fallback: rule-based engine returns a useful answer without any
     model loaded.

Logging:
  - "All guardrails PASSED"          -- question cleared for LLM
  - "USE_LLM=true — Qwen"            -- Qwen path taken
  - "Qwen fell back to local engine" -- local fallback taken
"""
from __future__ import annotations

import logging
import pandas as pd

from llm.config import load_config
from llm.fallback_engine import generate_fallback_insight
from llm.guardrails import check_input
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
    """Return ``(answer, source)`` where source is ``"llm"`` or ``"local"``."""
    print(f"[LLM] generate_insight called: '{question[:60]}'", flush=True)
    logger.info("[LLM] generate_insight called: '%s'", question[:60])

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

    # -- Guardrail pre-filter -------------------------------------------------
    q = question.strip()

    # Layer 1: Llama Guard (optional).  If not yet loaded, skip it and rely on
    # the regex layer below — the LLM still runs.  Blocking on LlamaGuard made
    # the assistant permanently unusable because loading takes 15+ min on CPU.
    from llm.llama_guard import is_available, moderate
    if is_available():
        is_safe, category = moderate(q)
        if not is_safe:
            logger.warning(
                "[LLM] Llama Guard BLOCKED: category='%s' | q='%s'",
                category, q[:60],
            )
            return (
                f"Sua pergunta foi sinalizada como insegura "
                f"(Categoria: {category}). Por favor, reformule.",
                "local",
            )
    else:
        logger.info("[LLM] Llama Guard not yet loaded — using regex guardrails only.")

    # Layer 2: Regex & Topic Filter
    allowed, refusal = check_input(q)
    if not allowed:
        logger.info("[LLM] Regex Guardrail blocked question: '%s'", q[:60])
        return refusal, "local"

    logger.info("[LLM] All guardrails PASSED for: '%s'", q[:60])

    # -- Context trimming -----------------------------------------------------
    keep = [c for c in _CONTEXT_COLS if c in runs_df.columns]
    context_df = runs_df[keep].head(_MAX_RUNS).copy()

    # -- Hybrid Path: Qwen ----------------------------------------------------
    cfg = load_config()
    if cfg.llm_ready:
        print(f"[LLM] USE_LLM=true — Qwen. path={cfg.model_path}", flush=True)
        logger.info("[LLM] USE_LLM=true — Qwen. path=%s", cfg.model_path)
        response = _try_qwen(cfg, q, context_df)
        if response:
            return response, "llm"
        print("[LLM] Qwen fell back to local engine", flush=True)
        logger.warning("[LLM] Qwen returned None — falling back")
    else:
        logger.info(
            "[LLM] USE_LLM=false — local engine only "
            "(set USE_LLM=true to enable Qwen)"
        )

    # -- Local fallback: rule-based engine ------------------------------------
    return generate_fallback_insight(q, context_df), "local"




# -- Private helpers ----------------------------------------------------------

def _try_qwen(
    cfg, question: str, context_df: pd.DataFrame
) -> str | None:
    """Attempt Qwen 2.5 call. Returns response text or None on failure."""
    try:
        from llm.client_qwen import QwenClient  # noqa: PLC0415
        # Use cached singleton — avoids reloading the model on every question
        client = QwenClient.get_or_create(cfg.model_path)
        prompt = build_experiment_prompt(question, context_df)
        return client.generate(prompt)
    except Exception as exc:
        logger.warning("[LLM] Qwen client raised an unexpected error: %s", exc)
        return None
