"""LangGraph state schema.

The state is JSON-serializable so the Postgres checkpointer can snapshot it after every
node. Messages are stored as plain dicts (role/content/tool_calls) and converted to
:class:`hivemind.core.llm.base.Message` at the boundary.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from hivemind.core.llm.base import Message, ToolCall


def _replace(_old: Any, new: Any) -> Any:
    return new


def _append(old: list, new: list) -> list:
    return [*old, *new]


class GraphState(TypedDict, total=False):
    user_message: str
    messages: Annotated[list[dict], _replace]
    route: Annotated[dict, _replace]
    agent_outputs: Annotated[list[dict], _append]
    final_response: Annotated[str, _replace]
    iterations: Annotated[int, _replace]
    tokens_used: Annotated[int, _replace]


def msg_to_dict(m: Message) -> dict:
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
        ],
        "tool_call_id": m.tool_call_id,
        "name": m.name,
    }


def dict_to_msg(d: dict) -> Message:
    return Message(
        role=d["role"],
        content=d.get("content", ""),
        tool_calls=[
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in d.get("tool_calls", [])
        ],
        tool_call_id=d.get("tool_call_id"),
        name=d.get("name"),
    )
