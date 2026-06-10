"""Async task streaming + status + cancel endpoints.

``/stream`` replays buffered events from the client's last offset (the ``Last-Event-ID``
header or ``?after`` query) and then tails the live channel — so a client can disconnect and
reconnect without losing events. All endpoints enforce that the task's conversation belongs
to the caller. ``/cancel`` releases the conversation's turn lock so the user can proceed.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, Query
from sse_starlette.sse import EventSourceResponse

from hivemind.api.authz import assert_conversation_access
from hivemind.api.deps import AppCtx, CurrentUser, EventBuffer
from hivemind.api.schemas.chat import TaskStatusResponse
from hivemind.core.errors import NotFoundError
from hivemind.db.repository import ConversationRepository, TaskRepository

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

_TERMINAL = {"completed", "failed", "cancelled"}


async def _load_owned_task(app, user, task_id: str):
    """Load a task and assert the caller owns its conversation, else 404/403."""
    async with app.db.session() as session:
        task = await TaskRepository(session).get(task_id)
        if task is None:
            raise NotFoundError(f"Task not found: {task_id}")
        convo = await ConversationRepository(session).get(task.conversation_id)
    if convo is not None:
        assert_conversation_access(convo, user, app.settings)
    return task


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def task_status(task_id: str, app: AppCtx, user: CurrentUser):
    task = await _load_owned_task(app, user, task_id)
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
    await _load_owned_task(app, user, task_id)
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


@router.post("/{task_id}/cancel", response_model=TaskStatusResponse)
async def cancel_task(task_id: str, app: AppCtx, user: CurrentUser):
    """Cancel a task and release its conversation's lock so the user can send a new query.

    Cancellation is cooperative/best-effort: a worker already mid-run finishes its current
    step, but the task is marked cancelled and the conversation is unlocked immediately.
    Already-finished tasks are returned unchanged.
    """
    task = await _load_owned_task(app, user, task_id)
    if task.status not in _TERMINAL:
        async with app.db.session() as session:
            await TaskRepository(session).set_status(task_id, "cancelled")
            await ConversationRepository(session).release_lock(task.conversation_id)
        task.status = "cancelled"
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        result=task.result,
        error=task.error,
        usage=task.usage,
    )
