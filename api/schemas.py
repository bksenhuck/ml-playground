"""Pydantic request / response schemas for the assistant API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    question: str = Field(..., min_length=1, max_length=2000)
    # Caller may supply run dicts directly (from the Dash dcc.Store) …
    runs: list[dict[str, Any]] | None = Field(
        default=None,
        description="Run records from the Dash experiment store.",
    )
    # … or MLflow run IDs so the server can retrieve them itself.
    run_ids: list[str] | None = Field(
        default=None,
        description="MLflow run IDs. Ignored when 'runs' is supplied.",
    )


class ChatResponse(BaseModel):
    """Response returned to the caller."""

    answer: str
    source: str = Field(
        description="'llm' | 'local' | 'blocked'"
    )
    blocked: bool = False
    block_reason: str | None = None
