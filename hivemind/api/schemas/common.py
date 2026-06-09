"""Shared API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMConfigSchema(BaseModel):
    provider: str = Field(..., examples=["anthropic"])
    model: str = Field(..., examples=["claude-opus-4-8"])
    temperature: float | None = None
    max_tokens: int = 4096
    top_p: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ProblemDetail(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    request_id: str | None = None
    conversation_id: str | None = None
