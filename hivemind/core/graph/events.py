"""Typed events emitted by the graph runner.

These are the canonical events serialized to SSE (interactive mode) and buffered to the
task-event log (queue mode). Each carries enough context to render incremental progress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class GraphEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}


def conversation(conversation_id: str, request_id: str | None = None) -> GraphEvent:
    """First event on an SSE stream: tells the client the conversation id to reuse."""
    return GraphEvent(
        "conversation", {"conversation_id": conversation_id, "request_id": request_id}
    )


def node_start(node: str, agent_id: str | None = None) -> GraphEvent:
    return GraphEvent("node_start", {"node": node, "agent_id": agent_id})


def node_end(node: str, agent_id: str | None = None) -> GraphEvent:
    return GraphEvent("node_end", {"node": node, "agent_id": agent_id})


def routing_decision(plan: dict) -> GraphEvent:
    return GraphEvent("routing_decision", {"plan": plan})


def text_delta(text: str, agent_id: str | None = None) -> GraphEvent:
    return GraphEvent("text_delta", {"text": text, "agent_id": agent_id})


def tool_call(name: str, arguments: dict, call_id: str) -> GraphEvent:
    return GraphEvent("tool_call", {"name": name, "arguments": arguments, "id": call_id})


def tool_result(name: str, payload: dict, call_id: str) -> GraphEvent:
    return GraphEvent("tool_result", {"name": name, "result": payload, "id": call_id})


def message(role: str, content: str, agent_id: str | None = None) -> GraphEvent:
    return GraphEvent("message", {"role": role, "content": content, "agent_id": agent_id})


def usage(input_tokens: int, output_tokens: int) -> GraphEvent:
    return GraphEvent("usage", {"input_tokens": input_tokens, "output_tokens": output_tokens})


def error(detail: str, error_type: str = "error") -> GraphEvent:
    return GraphEvent("error", {"detail": detail, "error_type": error_type})


def done(final: str) -> GraphEvent:
    return GraphEvent("done", {"final": final})


Mode = Literal["sse", "queue"]


def serialize(event: GraphEvent) -> dict:
    return asdict(event)
