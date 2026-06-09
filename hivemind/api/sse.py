"""Helpers to serialize graph events as Server-Sent Events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from hivemind.core.graph import events


def format_sse(event: events.GraphEvent) -> dict:
    """Shape an SSE message for ``sse_starlette.EventSourceResponse``."""
    return {"event": event.type, "data": json.dumps(event.to_dict())}


async def graph_event_sse(stream: AsyncIterator[events.GraphEvent]) -> AsyncIterator[dict]:
    async for event in stream:
        yield format_sse(event)
