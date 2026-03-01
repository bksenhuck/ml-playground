"""Prompt builders for ML experiment insights.

Kept deliberately concise to minimise token usage while giving Gemini enough
context to produce a useful, actionable response.
"""
from __future__ import annotations

import pandas as pd

# Columns included in the context table sent to the LLM
_PARAM_COLS  = ["run_name", "model", "n_features", "scaling", "class_weight", "poly_features", "test_size"]
_METRIC_COLS = ["metric_accuracy", "metric_precision", "metric_recall", "metric_f1", "metric_roc_auc"]
_MAX_ROWS    = 5


def build_experiment_prompt(question: str, runs_df: pd.DataFrame) -> str:
    """Build a token-efficient prompt from experiment data and the user question.

    Args:
        question: The user's free-text question.
        runs_df:  DataFrame of runs to include as context (up to _MAX_ROWS used).

    Returns:
        A ready-to-send prompt string in Brazilian Portuguese.
    """
    keep    = [c for c in _PARAM_COLS + _METRIC_COLS if c in runs_df.columns]
    summary = runs_df[keep].head(_MAX_ROWS).copy()

    # Strip "metric_" prefix for readability inside the table
    summary = summary.rename(columns={c: c.replace("metric_", "") for c in _METRIC_COLS})

    try:
        table_str = summary.to_markdown(index=False)
    except Exception:
        table_str = summary.to_string(index=False)

    return f"""Você é um engenheiro de ML sênior. Analise os experimentos abaixo e responda a pergunta.

## Contexto da plataforma
- Dataset: Titanic (classificação binária — survived 0/1)
- Modelos: Logistic Regression (logreg), Random Forest (rf)
- Métricas: accuracy, precision, recall, f1, roc_auc

## Experimentos selecionados
{table_str}

## Pergunta
{question}

## Instruções
Responda em português (Brasil). Seja técnico e conciso (máx. 200 palavras).
Foque em insights acionáveis: o que os números significam, quais parâmetros ajustar
e qual seria o próximo experimento recomendado.
"""
