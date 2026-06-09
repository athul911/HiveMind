"""Async task streaming + status endpoints.

``/stream`` replays buffered events from the client's last offset (the ``Last-Event-ID``
header or ``?after`` query) and then tails the live channel — so a client can disconnect and
reconnect without losing events.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, Query
from sse_starlette.sse import EventSourceResponse

from hivemind.api.deps import AppCtx, CurrentUser, EventBuffer
from hivemind.api.schemas.chat import TaskStatusResponse
from hivemind.core.errors import NotFoundError
from hivemind.db.repository import TaskRepository

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def task_status(task_id: str, app: AppCtx, user: CurrentUser):
    async with app.db.session() as session:
        task = await TaskRepository(session).get(task_id)
    if task is None:
        raise NotFoundError(f"Task not found: {task_id}")
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        result=task.result,
        error=task.error,
        usage=task.usage,
    )


@router.get("/{task_id}/stream")
async def task_stream(
    task_id: str,
    app: AppCtx,
    buffer: EventBuffer,
    user: CurrentUser,
    after: int = Query(default=0, description="Resume after this event sequence number."),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    async with app.db.session() as session:
        task = await TaskRepository(session).get(task_id)
    if task is None:
        raise NotFoundError(f"Task not found: {task_id}")

    after_seq = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

    async def generator():
        import json

        async for seq, event in buffer.replay_and_tail(task_id, after_seq=after_seq):
            yield {
                "id": str(seq),
                "event": event.type,
                "data": json.dumps(event.to_dict(), default=str),
            }

    return EventSourceResponse(generator())
