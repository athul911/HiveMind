"""Provider-agnostic LLM contracts.

A single normalized request/response/stream-event shape lets the rest of the system
treat every provider identically (dependency inversion). Adapters translate to and from
their native SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Per-agent LLM configuration, persisted as JSON on the agent."""

    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int = 4096
    top_p: float | None = None
    # Provider-specific knobs (e.g. anthropic effort, azure deployment overrides).
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        known = {"provider", "model", "temperature", "max_tokens", "top_p"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            provider=data["provider"],
            model=data["model"],
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens", 4096),
            top_p=data.get("top_p"),
            extra={**data.get("extra", {}), **extra},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "extra": self.extra,
        }


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    # For assistant tool-call turns and tool-result turns.
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(slots=True)
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(slots=True)
class LLMRequest:
    config: LLMConfig
    messages: list[Message]
    system: str | None = None
    tools: list[ToolSchema] = field(default_factory=list)
    tool_choice: Literal["auto", "any", "none"] = "auto"


# ---- streaming events ----------------------------------------------------


@dataclass(slots=True)
class TextDelta:
    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class ThinkingDelta:
    text: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(slots=True)
class ToolCallEvent:
    tool_call: ToolCall
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class UsageEvent:
    usage: Usage
    type: Literal["usage"] = "usage"


@dataclass(slots=True)
class DoneEvent:
    stop_reason: str
    type: Literal["done"] = "done"


LLMStreamEvent = TextDelta | ThinkingDelta | ToolCallEvent | UsageEvent | DoneEvent


@dataclass(slots=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str


@runtime_checkable
class LLMProvider(Protocol):
    """All providers implement streaming-first; ``complete`` aggregates the stream."""

    name: str

    def stream(self, request: LLMRequest):  # -> AsyncIterator[LLMStreamEvent]
        ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
