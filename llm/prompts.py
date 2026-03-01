"""Prompt builders for ML experiment insights."""
from __future__ import annotations

import pandas as pd

# Columns to include in the context summary sent to the LLM
_PARAM_COLS = ["run_name", "model", "n_features", "scaling", "class_weight", "poly_features", "test_size"]
_METRIC_COLS = ["metric_accuracy", "metric_precision", "metric_recall", "metric_f1", "metric_roc_auc"]


def build_experiment_prompt(question: str, runs_df: pd.DataFrame) -> str:
    """Build a structured prompt from experiment data and the user question.

    Args:
        question: The user's free-text question.
        runs_df: DataFrame of selected runs (up to 5 rows used).

    Returns:
        A ready-to-send prompt string.
    """
    keep = [c for c in _PARAM_COLS + _METRIC_COLS if c in runs_df.columns]
    summary = runs_df[keep].head(5).copy()

    # Rename metric columns for readability in the table
    rename = {c: c.replace("metric_", "") for c in _METRIC_COLS}
    summary = summary.rename(columns=rename)

    try:
        table_str = summary.to_markdown(index=False)
    except Exception:
        table_str = summary.to_string(index=False)

    return f"""Você é um engenheiro de ML sênior analisando resultados de experimentos.

## Contexto
- Dataset: Titanic (classificação binária — survived 0/1)
- Plataforma: ML Playground (scikit-learn + MLflow)
- Modelos disponíveis: Logistic Regression (logreg), Random Forest (rf)
- Métricas: accuracy, precision, recall, f1, roc_auc

## Experimentos Selecionados
{table_str}

## Pergunta do Usuário
{question}

## Instruções
Responda em português (Brasil) de forma técnica e concisa (máximo 200 palavras).
Foque em insights acionáveis: o que os números significam, quais parâmetros mudar,
e qual seria o próximo experimento recomendado.
"""
