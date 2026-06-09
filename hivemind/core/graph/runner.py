"""GraphRunner — the single entry point both the API (SSE) and worker (queue) call.

Wraps ``graph.astream(..., stream_mode="custom")`` and yields typed :class:`GraphEvent`s.
Resumes from the Postgres checkpoint keyed by ``thread_id == conversation_id``. Records the
workflow-duration metric and enforces the recursion limit.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from hivemind.core.graph import events
from hivemind.core.graph.builder import build_graph
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.graph.state import msg_to_dict
from hivemind.core.llm.base import Message
from hivemind.observability.logging import get_logger
from hivemind.observability.tracing import record_workflow_duration, span

logger = get_logger("hivemind.graph.runner")


class GraphRunner:
    def __init__(self, deps: GraphDeps, checkpointer) -> None:
        self._deps = deps
        self._graph = build_graph(deps, checkpointer)

    async def run(
        self,
        *,
        conversation_id: str,
        history: list[Message],
        user_message: str,
        mode: str = "sse",
    ) -> AsyncIterator[events.GraphEvent]:
        """Stream the workflow as typed events. ``history`` excludes the new user message."""
        messages = [*history, Message(role="user", content=user_message)]
        state = {
            "user_message": user_message,
            "messages": [msg_to_dict(m) for m in messages],
            "agent_outputs": [],
            "iterations": 0,
            "tokens_used": 0,
        }
        config = {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": self._deps.settings.supervisor_max_iterations,
        }
        started = time.perf_counter()
        with span("graph.run", **{"graph.mode": mode}):
            try:
                async for chunk in self._graph.astream(state, config, stream_mode="custom"):
                    yield events.GraphEvent(type=chunk["type"], data=chunk.get("data", {}))
            except Exception as exc:
                logger.error("graph.run_failed", error=str(exc))
                yield events.error(str(exc), error_type=type(exc).__name__)
            finally:
                record_workflow_duration(time.perf_counter() - started, mode=mode)
