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
