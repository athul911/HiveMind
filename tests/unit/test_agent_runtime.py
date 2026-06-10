from __future__ import annotations

from hivemind.config import Settings
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.context import RequestContext
from hivemind.core.graph.agent_runtime import run_agent_turn
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.llm.base import LLMConfig, Message
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.tools.base import BaseTool, ToolResult
from hivemind.core.tools.registry import ToolRegistry

from tests.conftest import ScriptedFactory, ScriptedProvider, text_turn, tool_turn


class AddTool(BaseTool):
    name = "add"
    description = "Add two numbers."
    input_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }

    async def run(self, args, ctx):
        return ToolResult(content={"sum": args["a"] + args["b"]})


def _deps(provider: ScriptedProvider) -> GraphDeps:
    tools = ToolRegistry()
    tools.register(AddTool())
    skills = SkillRegistry()
    return GraphDeps(
        settings=Settings(otel_enabled=False),
        agents=AgentRegistry(),
        agent_factory=AgentFactory(tools, skills),
        llm_factory=ScriptedFactory(provider),
        tools=tools,
    )


async def test_agent_turn_simple_answer():
    provider = ScriptedProvider([text_turn("The answer is 42.")])
    deps = _deps(provider)
    events: list = []
    final, _appended, usage = await run_agent_turn(
        deps=deps,
        system_prompt="You are helpful.",
        tool_names=[],
        llm_config=LLMConfig(provider="scripted", model="m"),
        conversation=[Message(role="user", content="hi")],
        ctx=RequestContext(),
        emit=events.append,
    )
    assert final == "The answer is 42."
    assert usage.output_tokens == 5
    assert any(e.type == "text_delta" for e in events)


class FlakyTool(BaseTool):
    """Fails on the first call, succeeds on the second — simulates a bad-then-fixed query."""

    name = "lookup"
    description = "Look something up."
    input_schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, args, ctx):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('column "wrong" does not exist')
        return ToolResult(content={"ok": True})


async def test_tool_failure_is_fed_back_and_agent_retries():
    from hivemind.core.graph.deps import GraphDeps

    tools = ToolRegistry()
    flaky = FlakyTool()
    tools.register(flaky)
    provider = ScriptedProvider(
        [
            tool_turn("lookup", {"q": "bad"}, "c1"),  # round 1: model calls tool → it fails
            tool_turn("lookup", {"q": "good"}, "c2"),  # round 2: model retries after seeing error
            text_turn("Done."),  # round 3: model answers
        ]
    )
    deps = GraphDeps(
        settings=Settings(otel_enabled=False),
        agents=AgentRegistry(),
        agent_factory=AgentFactory(tools, SkillRegistry()),
        llm_factory=ScriptedFactory(provider),
        tools=tools,
    )
    events_seen: list = []
    final, _appended, _usage = await run_agent_turn(
        deps=deps,
        system_prompt="Use tools; fix errors.",
        tool_names=["lookup"],
        llm_config=LLMConfig(provider="scripted", model="m"),
        conversation=[Message(role="user", content="do it")],
        ctx=RequestContext(conversation_id="c1"),
        emit=events_seen.append,
    )
    assert final == "Done."
    assert flaky.calls == 2  # it retried after the failure
    # The first tool_result carried the error back to the model.
    tool_results = [e for e in events_seen if e.type == "tool_result"]
    assert tool_results[0].data["result"].get("is_error") is True
    assert "does not exist" in tool_results[0].data["result"]["error"]


def _empty_turn():
    from hivemind.core.llm.base import DoneEvent, Usage, UsageEvent

    return [UsageEvent(Usage(1, 0)), DoneEvent("stop")]  # no text, no tool call


async def test_empty_completion_is_nudged_once_then_recovers():
    # Round 1 empty → nudge → round 2 answers.
    provider = ScriptedProvider([_empty_turn(), text_turn("Recovered answer.")])
    deps = _deps(provider)
    seen: list = []
    final, _appended, _usage = await run_agent_turn(
        deps=deps,
        system_prompt="p",
        tool_names=[],
        llm_config=LLMConfig(provider="scripted", model="m"),
        conversation=[Message(role="user", content="hi")],
        ctx=RequestContext(),
        emit=seen.append,
    )
    assert final == "Recovered answer."
    assert any(e.type == "empty_completion_retry" for e in seen)


async def test_empty_completion_gives_up_after_one_nudge():
    # Two empties in a row → single nudge, then stop cleanly (no infinite loop).
    provider = ScriptedProvider([_empty_turn(), _empty_turn(), text_turn("never reached")])
    deps = _deps(provider)
    seen: list = []
    final, _appended, _usage = await run_agent_turn(
        deps=deps,
        system_prompt="p",
        tool_names=[],
        llm_config=LLMConfig(provider="scripted", model="m"),
        conversation=[Message(role="user", content="hi")],
        ctx=RequestContext(),
        emit=seen.append,
    )
    assert final == ""
    assert sum(e.type == "empty_completion_retry" for e in seen) == 1
    assert any(e.type == "agent_finished" for e in seen)
    assert len(provider.calls) == 2  # nudged exactly once, didn't reach the third turn


async def test_agent_turn_runs_tool_then_answers():
    provider = ScriptedProvider([tool_turn("add", {"a": 2, "b": 3}), text_turn("The sum is 5.")])
    deps = _deps(provider)
    events: list = []
    final, appended, _usage = await run_agent_turn(
        deps=deps,
        system_prompt="Use tools.",
        tool_names=["add"],
        llm_config=LLMConfig(provider="scripted", model="m"),
        conversation=[Message(role="user", content="2+3?")],
        ctx=RequestContext(conversation_id="c1"),
        emit=events.append,
    )
    assert final == "The sum is 5."
    types = [e.type for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    # The tool result message was appended for the follow-up turn.
    assert any(m.role == "tool" and "sum" in m.content for m in appended)
