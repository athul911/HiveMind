"""Helpers to serialize graph events as Server-Sent Events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from hivemind.core.graph import events


def format_sse(event: events.GraphEvent) -> dict:
    """Shape an SSE message for ``sse_starlette.EventSourceResponse``.

    ``default=str`` is a safety net so an unexpected non-JSON-native value in an event
    payload (e.g. a Decimal slipping through a tool result) can never break the stream.
    """
    return {"event": event.type, "data": json.dumps(event.to_dict(), default=str)}


async def graph_event_sse(stream: AsyncIterator[events.GraphEvent]) -> AsyncIterator[dict]:
    async for event in stream:
        yield format_sse(event)
