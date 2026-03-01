"""LLM insight service: orchestrates prompt building and the Gemini call."""
from __future__ import annotations

import pandas as pd

from llm.client import generate
from llm.prompts import build_experiment_prompt


def generate_insight(question: str, runs_df: pd.DataFrame) -> str:
    """Return a Gemini-generated insight for the given question and run data.

    Args:
        question: User's free-text question about their experiments.
        runs_df: DataFrame of runs to include as context (typically selected rows).

    Returns:
        A text response from the LLM, or a helpful fallback message.
    """
    if not question or not question.strip():
        return "Por favor, escreva uma pergunta sobre seus experimentos."

    if runs_df is None or runs_df.empty:
        return (
            "Nenhum experimento disponível no contexto. "
            "Execute alguns experimentos e, opcionalmente, selecione linhas na tabela."
        )

    prompt = build_experiment_prompt(question.strip(), runs_df)
    return generate(prompt)
