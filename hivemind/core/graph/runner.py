"""GraphRunner — the single entry point both the API (SSE) and worker (queue) call.

Wraps ``graph.astream(..., stream_mode="custom")`` and yields typed :class:`GraphEvent`s,
keyed by a checkpoint ``thread_id`` (the queue ``task_id`` for resumable tasks; the
conversation id for in-process SSE). ``resume`` continues an interrupted run from its
checkpoint after a crash + redelivery, instead of re-running from the start. Records the
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
        self._checkpointer = checkpointer
        self._graph = build_graph(deps, checkpointer)

    async def delete_checkpoint(self, thread_id: str) -> None:
        """Best-effort removal of a thread's checkpoints once they're no longer needed.

        Queue threads are keyed by ``task_id`` and aren't used across turns, so a task's
        checkpoint is dead weight once the task reaches a terminal state. Idempotent — safe to
        call on an already-deleted or never-created thread — and never fatal to the caller.
        """
        try:
            await self._checkpointer.adelete_thread(thread_id)
        except Exception as exc:
            logger.debug("graph.checkpoint_gc_failed", thread_id=thread_id, error=str(exc))

    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._deps.settings.supervisor_max_iterations,
        }

    async def is_resumable(self, thread_id: str) -> bool:
        """True if a checkpoint for ``thread_id`` has pending nodes (an interrupted run).

        A fresh/never-run thread or a completed run has no pending next node → not resumable.
        """
        try:
            snapshot = await self._graph.aget_state(self._config(thread_id))
        except Exception as exc:  # no checkpoint / saver hiccup → treat as not resumable
            logger.debug("graph.is_resumable_failed", thread_id=thread_id, error=str(exc))
            return False
        return bool(snapshot.next)

    async def run(
        self,
        *,
        thread_id: str,
        history: list[Message],
        user_message: str,
        mode: str = "sse",
    ) -> AsyncIterator[events.GraphEvent]:
        """Stream a fresh turn as typed events. ``history`` excludes the new user message."""
        messages = [*history, Message(role="user", content=user_message)]
        state = {
            "user_message": user_message,
            "messages": [msg_to_dict(m) for m in messages],
            "agent_outputs": [],
            "iterations": 0,
            "tokens_used": 0,
        }
        async for ev in self._astream(state, thread_id, mode, "graph.run"):
            yield ev

    async def resume(
        self, *, thread_id: str, mode: str = "queue"
    ) -> AsyncIterator[events.GraphEvent]:
        """Resume an interrupted run from its checkpoint (``input=None`` continues pending work)."""
        async for ev in self._astream(None, thread_id, mode, "graph.resume"):
            yield ev

    async def _astream(
        self, state, thread_id: str, mode: str, span_name: str
    ) -> AsyncIterator[events.GraphEvent]:
        config = self._config(thread_id)
        started = time.perf_counter()
        with span(span_name, **{"graph.mode": mode}):
            try:
                async for chunk in self._graph.astream(state, config, stream_mode="custom"):
                    yield events.GraphEvent(type=chunk["type"], data=chunk.get("data", {}))
            except Exception as exc:
                logger.error("graph.run_failed", error=str(exc))
                yield events.error(str(exc), error_type=type(exc).__name__)
            finally:
                record_workflow_duration(time.perf_counter() - started, mode=mode)
