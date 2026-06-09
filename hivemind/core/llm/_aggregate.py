"""Shared helper: build an :class:`LLMResponse` by consuming a provider stream."""

from __future__ import annotations

from collections.abc import AsyncIterator

from hivemind.core.llm.base import (
    DoneEvent,
    LLMResponse,
    LLMStreamEvent,
    TextDelta,
    ToolCallEvent,
    Usage,
    UsageEvent,
)


async def aggregate_stream(stream: AsyncIterator[LLMStreamEvent]) -> LLMResponse:
    """Collect a streaming response into a single :class:`LLMResponse`."""
    text_parts: list[str] = []
    tool_calls = []
    usage = Usage()
    stop_reason = "end_turn"
    async for event in stream:
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, ToolCallEvent):
            tool_calls.append(event.tool_call)
        elif isinstance(event, UsageEvent):
            usage = event.usage
        elif isinstance(event, DoneEvent):
            stop_reason = event.stop_reason
    return LLMResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=stop_reason,
    )
