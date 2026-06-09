"""Sub-agent runner: depth limit + checkpoint-backed idempotent restore."""

from __future__ import annotations

import pytest
from hivemind.config import Settings
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.context import RequestContext
from hivemind.core.errors import ValidationError
from hivemind.core.graph import subagent_runner as sr
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.graph.subagent_runner import SubAgentRunnerImpl
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.tools.registry import ToolRegistry

from tests.conftest import ScriptedFactory, ScriptedProvider, text_turn
from tests.fakes import FakeDatabase


class _EphemeralRepo:
    """In-memory ephemeral-agent store keyed by id, with get + checkpoint."""

    rows: dict = {}

    def __init__(self, _session) -> None: ...

    async def get(self, ephemeral_id):
        from types import SimpleNamespace

        row = _EphemeralRepo.rows.get(ephemeral_id)
        return SimpleNamespace(**row) if row else None

    async def checkpoint(self, ephemeral_id, conv, definition, checkpoint, ttl):
        _EphemeralRepo.rows[ephemeral_id] = {
            "id": ephemeral_id,
            "definition": definition,
            "checkpoint": checkpoint,
        }


def _deps(provider, *, max_depth=2):
    return GraphDeps(
        settings=Settings(otel_enabled=False, subagent_max_depth=max_depth),
        agents=AgentRegistry(),
        agent_factory=AgentFactory(ToolRegistry(), SkillRegistry()),
        llm_factory=ScriptedFactory(provider),
        tools=ToolRegistry(),
    )


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _EphemeralRepo.rows.clear()
    monkeypatch.setattr(sr, "EphemeralAgentRepository", _EphemeralRepo)
    yield
    _EphemeralRepo.rows.clear()


async def test_subagent_runs_and_checkpoints():
    runner = SubAgentRunnerImpl(_deps(ScriptedProvider([text_turn("sub result")])), FakeDatabase())
    out = await runner.run_subagent(
        {"system_prompt": "be a sub", "tool_names": []},
        "do a thing",
        RequestContext(conversation_id="c1"),
    )
    assert out["result"] == "sub result"
    # A completed checkpoint was written.
    rows = _EphemeralRepo.rows.values()
    assert any(r["checkpoint"] and r["checkpoint"]["result"] == "sub result" for r in rows)


async def test_subagent_restores_from_checkpoint_without_rerun():
    # Second spawn with identical (conversation, prompt, task) must NOT call the LLM again.
    provider = ScriptedProvider([text_turn("first run")])
    runner = SubAgentRunnerImpl(_deps(provider), FakeDatabase())
    ctx = RequestContext(conversation_id="c1")
    defn = {"system_prompt": "p", "tool_names": []}

    first = await runner.run_subagent(defn, "task", ctx)
    assert first["result"] == "first run"
    calls_after_first = len(provider.calls)

    second = await runner.run_subagent(defn, "task", ctx)  # identical → restored
    assert second["result"] == "first run"
    assert len(provider.calls) == calls_after_first  # no extra LLM call


async def test_subagent_depth_limit_enforced():
    runner = SubAgentRunnerImpl(_deps(ScriptedProvider([]), max_depth=2), FakeDatabase())
    deep_ctx = RequestContext(conversation_id="c1", subagent_depth=2)
    with pytest.raises(ValidationError):
        await runner.run_subagent({"system_prompt": "p", "tool_names": []}, "t", deep_ctx)
