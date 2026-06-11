"""Graph execution with a mocked LLM and the in-memory checkpointer (no external services)."""

from __future__ import annotations

from hivemind.config import Settings
from hivemind.core.agents.agent import Agent
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.graph.checkpointer import memory_checkpointer
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.graph.runner import GraphRunner
from hivemind.core.llm.base import LLMConfig, Message
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.tools.registry import ToolRegistry

from tests.conftest import ScriptedFactory, ScriptedProvider, text_turn


def _runner(provider: ScriptedProvider) -> GraphRunner:
    tools = ToolRegistry()
    skills = SkillRegistry()
    registry = AgentRegistry()
    registry.add(
        Agent(
            name="helper",
            description="A general helper.",
            system_prompt="You help.",
            llm_config=LLMConfig(provider="scripted", model="m"),
        )
    )
    deps = GraphDeps(
        settings=Settings(otel_enabled=False, supervisor_max_iterations=10),
        agents=registry,
        agent_factory=AgentFactory(tools, skills),
        llm_factory=ScriptedFactory(provider),
        tools=tools,
    )
    return GraphRunner(deps, memory_checkpointer())


async def test_graph_runs_single_agent_and_emits_done():
    provider = ScriptedProvider([text_turn("Hello from the helper.")])
    runner = _runner(provider)
    collected = []
    async for event in runner.run(
        thread_id="conv-1", history=[], user_message="hi", mode="sse"
    ):
        collected.append(event)

    types = [e.type for e in collected]
    assert "routing_decision" in types
    assert "text_delta" in types
    done = [e for e in collected if e.type == "done"]
    assert done and done[0].data["final"] == "Hello from the helper."


async def test_graph_preserves_history():
    provider = ScriptedProvider([text_turn("ack")])
    runner = _runner(provider)
    history = [Message(role="user", content="earlier"), Message(role="assistant", content="ok")]
    async for _ in runner.run(
        thread_id="conv-2", history=history, user_message="now", mode="sse"
    ):
        pass
    # The agent turn should have seen the prior history plus the new message.
    seen = provider.calls[-1].messages
    assert any(m.content == "earlier" for m in seen)
    assert seen[-1].content == "now"
