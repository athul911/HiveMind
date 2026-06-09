"""Tests for the resilience wrapper, conditional evaluation, and conversation compaction."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from hivemind.config import Settings
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.errors import LLMProviderError
from hivemind.core.graph.conditional import evaluate_condition
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.llm.base import (
    DoneEvent,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    TextDelta,
    Usage,
)
from hivemind.core.llm.resilience import ResilientProvider
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.tools.registry import ToolRegistry
from hivemind.services import conversation as convo_mod
from hivemind.services.conversation import ConversationService

from tests import fakes
from tests.conftest import ScriptedFactory, ScriptedProvider, text_turn

# ---- ResilientProvider -----------------------------------------------------

class _FlakyProvider:
    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.attempts = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise LLMProviderError("transient")
        return LLMResponse("ok", [], Usage(1, 1), "end_turn")

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise LLMProviderError("transient")
        yield TextDelta("ok")
        yield DoneEvent("end_turn")


async def test_resilient_retries_then_succeeds():
    inner = _FlakyProvider(fail_times=2)
    provider = ResilientProvider(inner, max_retries=3, base_delay=0.0)
    resp = await provider.complete(LLMRequest(config=_cfg(), messages=[]))
    assert resp.text == "ok"
    assert inner.attempts == 3


async def test_resilient_gives_up_and_opens_circuit():
    inner = _FlakyProvider(fail_times=99)
    provider = ResilientProvider(inner, max_retries=1, base_delay=0.0, breaker_threshold=1)
    with pytest.raises(LLMProviderError):
        await provider.complete(LLMRequest(config=_cfg(), messages=[]))
    # Circuit is now open → next call fast-fails without touching the inner provider.
    attempts_before = inner.attempts
    with pytest.raises(LLMProviderError):
        await provider.complete(LLMRequest(config=_cfg(), messages=[]))
    assert inner.attempts == attempts_before


async def test_resilient_stream_retries_cold_failure():
    inner = _FlakyProvider(fail_times=1)
    provider = ResilientProvider(inner, max_retries=2, base_delay=0.0)
    out = [ev async for ev in provider.stream(LLMRequest(config=_cfg(), messages=[]))]
    assert any(isinstance(e, TextDelta) for e in out)


def _cfg():
    from hivemind.core.llm.base import LLMConfig

    return LLMConfig(provider="flaky", model="m")


# ---- Conditional evaluation ------------------------------------------------

def _deps(provider):
    return GraphDeps(
        settings=Settings(otel_enabled=False),
        agents=AgentRegistry(),
        agent_factory=AgentFactory(ToolRegistry(), SkillRegistry()),
        llm_factory=ScriptedFactory(provider),
        tools=ToolRegistry(),
    )


async def test_condition_true_and_false():
    yes = await evaluate_condition(_deps(ScriptedProvider([text_turn("YES")])), "needs review", "x")
    assert yes[0] is True
    no = await evaluate_condition(_deps(ScriptedProvider([text_turn("NO")])), "needs review", "x")
    assert no[0] is False


# ---- Conversation compaction ----------------------------------------------

async def test_load_history_windows_when_no_llm(monkeypatch):
    monkeypatch.setattr(convo_mod, "MessageRepository", fakes.FakeMessageRepo)
    fakes.reset_fakes()
    fakes.FakeMessageRepo.store["c1"] = [
        fakes.FakeMessage("user", f"m{i}") for i in range(10)
    ]
    # No deps → windowing only (no summary message prepended).
    svc = ConversationService(
        fakes.FakeDatabase(), fakes.FakeRunner([]), ttl_seconds=60, history_limit=4, deps=None
    )
    history = await svc.load_history("c1")
    assert len(history) == 4
    assert [m.content for m in history] == ["m6", "m7", "m8", "m9"]


async def test_load_history_summarizes_when_llm_available(monkeypatch):
    monkeypatch.setattr(convo_mod, "MessageRepository", fakes.FakeMessageRepo)
    fakes.reset_fakes()
    fakes.FakeMessageRepo.store["c2"] = [
        fakes.FakeMessage("user", f"m{i}") for i in range(10)
    ]
    deps = _deps(ScriptedProvider([text_turn("a summary")]))
    svc = ConversationService(
        fakes.FakeDatabase(),
        fakes.FakeRunner([]),
        ttl_seconds=60,
        history_limit=4,
        deps=deps,
        compaction_enabled=True,
    )
    history = await svc.load_history("c2")
    # 4 recent + 1 prepended summary marker.
    assert len(history) == 5
    assert history[0].content.startswith("[Summary of earlier conversation]")
    assert "a summary" in history[0].content
