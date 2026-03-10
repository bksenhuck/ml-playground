"""Prompt builders for ML experiment insights.

Kept deliberately concise to minimise token usage while giving Qwen enough
context to produce a useful, actionable response.
"""
from __future__ import annotations

import pandas as pd

# Columns included in the context table sent to the LLM
_PARAM_COLS = ["run_name", "model", "dataset", "n_features", "scaling",
               "class_weight", "poly_features", "test_size"]
_METRIC_COLS = ["metric_accuracy", "metric_precision", "metric_recall",
                "metric_f1", "metric_roc_auc"]
_MAX_ROWS = 5


def build_experiment_prompt(question: str, runs_df: pd.DataFrame) -> str:
    """Build a prompt from experiment data and the user question."""
    keep = [c for c in _PARAM_COLS + _METRIC_COLS if c in runs_df.columns]
    summary = runs_df[keep].head(_MAX_ROWS).copy()
    rename = {c: c.replace("metric_", "") for c in _METRIC_COLS}
    summary = summary.rename(columns=rename)

    # Simple key: value format works better than markdown tables for small models
    runs_text = ""
    for i, (_, row) in enumerate(summary.iterrows(), 1):
        runs_text += f"Experimento {i}:\n"
        for col, val in row.items():
            if pd.notna(val) and val != "" and val is not None:
                runs_text += f"  {col}: {val}\n"
        runs_text += "\n"

    return (
        f"Dados dos experimentos:\n\n"
        f"{runs_text}"
        f"Pergunta: {question}\n\n"
        f"Responda diretamente com sua análise e conclusão. "
        f"Não repita os dados acima na resposta."
    )
