"""Vertex AI Gemini client wrapper.

Requires:
  - google-cloud-aiplatform installed
  - GCP_PROJECT_ID environment variable set to your GCP project
  - GCP_LOCATION environment variable (defaults to "us-central1")
  - Application Default Credentials configured:
      gcloud auth application-default login
"""
from __future__ import annotations

import os

_PROJECT = os.environ.get("GCP_PROJECT_ID", "")
_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
_MODEL_NAME = "gemini-2.0-flash"


def generate(prompt: str, max_output_tokens: int = 512) -> str:
    """Call Vertex AI Gemini and return the text response.

    Args:
        prompt: Full prompt string to send to the model.
        max_output_tokens: Hard cap on response length.

    Returns:
        Model response text, or an error message string.
    """
    if not _PROJECT:
        return (
            "Variável de ambiente GCP_PROJECT_ID não configurada. "
            "Defina-a para habilitar o assistente."
        )

    try:
        import vertexai
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        vertexai.init(project=_PROJECT, location=_LOCATION)
        model = GenerativeModel(_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(max_output_tokens=max_output_tokens),
        )
        return response.text
    except ImportError:
        return (
            "Pacote google-cloud-aiplatform não instalado. "
            "Execute: pip install google-cloud-aiplatform"
        )
    except Exception as exc:
        return f"Erro ao chamar Vertex AI: {exc}"
