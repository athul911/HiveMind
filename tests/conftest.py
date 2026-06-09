"""Shared test fixtures and fakes."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from hivemind.config import Settings
from hivemind.core.llm.base import (
    DoneEvent,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
    UsageEvent,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        auth_disabled=True,
        otel_enabled=False,
        database_url="postgresql+asyncpg://test/test",
        sandbox_backend="subprocess",
    )


class ScriptedProvider:
    """A fake LLMProvider that replays a queue of scripted turns.

    Each scripted turn is a list of stream events. ``stream`` pops the next turn; this lets a
    test simulate "call tool, then answer" sequences.
    """

    name = "scripted"

    def __init__(self, turns: list[list[LLMStreamEvent]]) -> None:
        self._turns = list(turns)
        self.calls: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        events = self._turns.pop(0) if self._turns else [DoneEvent(stop_reason="end_turn")]
        for ev in events:
            yield ev

    async def complete(self, request: LLMRequest) -> LLMResponse:
        text_parts, tool_calls, usage, stop = [], [], Usage(), "end_turn"
        async for ev in self.stream(request):
            if isinstance(ev, TextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, ToolCallEvent):
                tool_calls.append(ev.tool_call)
            elif isinstance(ev, UsageEvent):
                usage = ev.usage
            elif isinstance(ev, DoneEvent):
                stop = ev.stop_reason
        return LLMResponse("".join(text_parts), tool_calls, usage, stop)


class ScriptedFactory:
    def __init__(self, provider: ScriptedProvider) -> None:
        self._provider = provider

    def create(self, _config) -> ScriptedProvider:
        return self._provider


def text_turn(text: str) -> list[LLMStreamEvent]:
    return [TextDelta(text=text), UsageEvent(Usage(10, 5)), DoneEvent("end_turn")]


def tool_turn(name: str, args: dict, call_id: str = "call_1") -> list[LLMStreamEvent]:
    return [
        ToolCallEvent(ToolCall(id=call_id, name=name, arguments=args)),
        UsageEvent(Usage(10, 5)),
        DoneEvent("tool_use"),
    ]
