"""Selects SSE (interactive) vs queue (long-running) execution mode.

Drivers, in order: an explicit ``stream: false`` flag forces queue mode; otherwise a simple
heuristic estimates workflow length (number of registered agents as a proxy for fan-out /
pipeline depth) against the configurable ``WORKFLOW_ASYNC_THRESHOLD_STEPS``.
"""

from __future__ import annotations

from hivemind.config import Settings


class ModeSelector:
    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.workflow_async_threshold_steps

    def select(self, *, stream: bool, agent_count: int) -> str:
        if not stream:
            return "queue"
        if agent_count > self._threshold:
            return "queue"
        return "sse"
