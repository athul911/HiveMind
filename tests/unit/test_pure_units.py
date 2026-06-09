"""Unit tests for pure, dependency-free logic across core/ and api/."""

from __future__ import annotations

import pytest
from hivemind.core.agents.agent import Agent
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.context import RequestContext, bind_context, reset_context
from hivemind.core.errors import (
    BudgetExceededError,
    HiveMindError,
    NotFoundError,
    UnsafeSQLError,
)
from hivemind.core.graph import events
from hivemind.core.graph.state import GraphState, dict_to_msg, msg_to_dict
from hivemind.core.llm._aggregate import aggregate_stream
from hivemind.core.llm.base import (
    DoneEvent,
    LLMConfig,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from hivemind.core.llm.ollama_provider import _to_ollama_messages
from hivemind.core.llm.openai_provider import _to_openai_messages

# ---- errors ----------------------------------------------------------------


def test_problem_detail_shape_includes_context():
    token = bind_context(RequestContext(request_id="r-1", conversation_id="c-1"))
    try:
        problem = NotFoundError("missing", agent_id="a1").to_problem()
    finally:
        reset_context(token)
    assert problem["status"] == 404
    assert problem["type"].endswith("/not-found")
    assert problem["request_id"] == "r-1"
    assert problem["conversation_id"] == "c-1"
    assert problem["agent_id"] == "a1"


def test_error_hierarchy_status_codes():
    assert UnsafeSQLError().status_code == 400
    assert BudgetExceededError().status_code == 429
    assert issubclass(UnsafeSQLError, HiveMindError)


# ---- usage / aggregate -----------------------------------------------------


def test_usage_addition():
    total = Usage(1, 2) + Usage(3, 4)
    assert total.input_tokens == 4
    assert total.output_tokens == 6


async def test_aggregate_stream_collects_text_tools_usage():
    async def gen():
        yield TextDelta("Hello ")
        yield TextDelta("world")
        yield ToolCallEvent(ToolCall(id="1", name="t", arguments={"x": 1}))
        yield UsageEvent(Usage(7, 3))
        yield DoneEvent("end_turn")

    resp = await aggregate_stream(gen())
    assert resp.text == "Hello world"
    assert resp.tool_calls[0].name == "t"
    assert resp.usage.input_tokens == 7
    assert resp.stop_reason == "end_turn"


# ---- message translation ---------------------------------------------------


def test_openai_message_translation_handles_tool_turns():
    msgs = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1})],
        ),
        Message(role="tool", content='{"sum":1}', tool_call_id="c1", name="add"),
    ]
    out = _to_openai_messages(msgs, system="sys")
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "add"
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "c1"


def test_ollama_message_translation():
    out = _to_ollama_messages([Message(role="user", content="hi")], system="s")
    assert out == [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]


# ---- state round-trip ------------------------------------------------------


def test_message_dict_roundtrip():
    m = Message(
        role="assistant",
        content="x",
        tool_calls=[ToolCall(id="1", name="t", arguments={"k": "v"})],
    )
    again = dict_to_msg(msg_to_dict(m))
    assert again.role == "assistant"
    assert again.tool_calls[0].name == "t"
    assert again.tool_calls[0].arguments == {"k": "v"}


def test_graph_state_reducers():
    # _append accumulates agent_outputs; _replace overwrites scalars.
    from hivemind.core.graph.state import _append, _replace

    assert _append([1], [2, 3]) == [1, 2, 3]
    assert _replace("old", "new") == "new"
    assert GraphState.__annotations__  # TypedDict is well-formed


# ---- events ----------------------------------------------------------------


def test_event_constructors():
    assert events.tool_call("t", {"a": 1}, "id1").data["name"] == "t"
    assert events.done("final").data["final"] == "final"
    assert events.error("boom", "X").data["error_type"] == "X"
    assert events.routing_decision({"mode": "single"}).type == "routing_decision"


# ---- agent serialization + registry ---------------------------------------


def test_agent_roundtrip_and_registry():
    agent = Agent(
        name="a",
        system_prompt="p",
        llm_config=LLMConfig(provider="anthropic", model="claude-opus-4-8"),
        tool_names=("sql_query",),
        skill_names=("postgres-optimization",),
        description="desc",
    )
    again = Agent.from_dict(agent.to_dict())
    assert again.name == "a"
    assert again.tool_names == ("sql_query",)
    assert again.llm_config.model == "claude-opus-4-8"

    reg = AgentRegistry()
    reg.add(agent)
    assert reg.get(agent.id).name == "a"
    assert reg.get_by_name("a") is agent
    assert reg.routing_table()[0]["agent_id"] == agent.id
    reg.remove(agent.id)
    with pytest.raises(NotFoundError):
        reg.get(agent.id)
