"""Chat completions + conversation lifecycle endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, Request
from sse_starlette.sse import EventSourceResponse

from hivemind.api.authz import assert_conversation_access
from hivemind.api.deps import AppCtx, CurrentUser, Dispatcher
from hivemind.api.schemas.chat import (
    AsyncTaskResponse,
    ChatCompletionRequest,
    ConversationSummary,
    TaskSummary,
)
from hivemind.api.sse import format_sse, graph_event_sse
from hivemind.core.context import get_context, update_context
from hivemind.core.errors import ConflictError, NotFoundError
from hivemind.core.graph import events
from hivemind.db.repository import ConversationRepository, MessageRepository, TaskRepository
from hivemind.services.mode_selector import ModeSelector

router = APIRouter(tags=["chat"])


async def _ensure_and_lock(app, user, conversation_id: str, agent_id: str | None) -> None:
    """Create-or-load the conversation, enforce ownership, and acquire its turn lock.

    Raises 403 if the conversation belongs to another user, 409 if it has ended or a turn is
    already in progress (the conversation is "paused" until that turn finishes or is
    cancelled). The acquire is atomic, so concurrent requests can't both proceed.
    """
    async with app.db.session() as session:
        repo = ConversationRepository(session)
        convo = await repo.get(conversation_id)
        if convo is None:
            convo = await repo.create(
                conversation_id, user.user_id, agent_id, app.settings.ephemeral_agent_ttl_seconds
            )
        else:
            assert_conversation_access(convo, user, app.settings)
            if convo.status == "ended":
                raise ConflictError("This conversation has ended.")
        if not await repo.acquire_lock(conversation_id):
            raise ConflictError(
                "A turn is already in progress for this conversation. Wait for it to finish "
                "or cancel it before sending another message."
            )


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    app: AppCtx,
    dispatcher: Dispatcher,
    user: CurrentUser,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """Send a message. Returns an SSE stream (interactive) or a task_id (long-running)."""
    conversation_id = body.conversation_id or str(uuid.uuid4())
    update_context(conversation_id=conversation_id, agent_id=body.agent_id)
    user_message = body.latest_user_message

    selector = ModeSelector(app.settings)
    mode = selector.select(stream=body.stream, agent_count=len(app.agents.list()))

    # Idempotent-retry short-circuit: if this request carries an Idempotency-Key we've already
    # seen, return that same task instead of acquiring the lock (which would otherwise 409,
    # since the original request still holds it). Ownership-checked so a colliding key from
    # another user can't fetch someone else's task.
    if mode == "queue" and idempotency_key:
        existing = await _task_for_key(app, idempotency_key)
        if existing is not None:
            await _assert_task_owner(app, user, existing)
            return _task_response(existing.task_id, existing.conversation_id, existing.status)

    # Ownership + single-in-flight lock. The lock is released by the worker (queue) or in the
    # SSE generator's finally (below); a crashed holder is recovered by the scheduler.
    await _ensure_and_lock(app, user, conversation_id, body.agent_id)

    if mode == "queue":
        try:
            task_id = await dispatcher.dispatch(
                conversation_id=conversation_id,
                user_id=user.user_id,
                agent_id=body.agent_id,
                user_message=user_message,
                idempotency_key=idempotency_key,
            )
        except Exception:
            await _release(app, conversation_id)  # don't leave the conversation locked
            raise
        return _task_response(task_id, conversation_id, "queued")

    stream = app.conversations.stream(
        conversation_id=conversation_id,
        user_id=user.user_id,
        agent_id=body.agent_id,
        user_message=user_message,
        mode="sse",
    )
    ctx = get_context()
    request_id = ctx.request_id if ctx else None

    async def sse_stream():
        # First frame announces the conversation id so a first-time caller can capture it
        # and reuse it on follow-up turns (also exposed as the x-conversation-id header).
        try:
            yield format_sse(events.conversation(conversation_id, request_id))
            async for frame in graph_event_sse(stream):
                yield frame
        finally:
            await _release(app, conversation_id)  # always unlock when the stream ends

    return EventSourceResponse(
        sse_stream(),
        headers={"x-conversation-id": conversation_id},
    )


async def _release(app, conversation_id: str) -> None:
    async with app.db.session() as session:
        await ConversationRepository(session).release_lock(conversation_id)


def _task_response(task_id: str, conversation_id: str, status: str) -> AsyncTaskResponse:
    return AsyncTaskResponse(
        task_id=task_id,
        conversation_id=conversation_id,
        status=status,
        stream_url=f"/v1/tasks/{task_id}/stream",
        status_url=f"/v1/tasks/{task_id}/status",
    )


async def _task_for_key(app, idempotency_key: str):
    async with app.db.session() as session:
        return await TaskRepository(session).get_by_idempotency_key(idempotency_key)


async def _assert_task_owner(app, user, task) -> None:
    """Ownership check for an existing task, via its conversation."""
    async with app.db.session() as session:
        convo = await ConversationRepository(session).get(task.conversation_id)
    if convo is not None:
        assert_conversation_access(convo, user, app.settings)


@router.get("/v1/conversations", response_model=list[ConversationSummary])
async def list_conversations(app: AppCtx, user: CurrentUser):
    """List the current user's conversations, most-recently-active first."""
    async with app.db.session() as session:
        convos = await ConversationRepository(session).list_for_user(user.user_id)
    return [
        ConversationSummary(
            id=c.id,
            status=c.status,
            agent_id=c.agent_id,
            created_at=c.created_at.isoformat() if c.created_at else "",
            updated_at=c.updated_at.isoformat() if c.updated_at else "",
        )
        for c in convos
    ]


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, app: AppCtx, user: CurrentUser):
    async with app.db.session() as session:
        convo = await ConversationRepository(session).get(conversation_id)
        if convo is None:
            raise NotFoundError(f"Conversation not found: {conversation_id}")
        assert_conversation_access(convo, user, app.settings)
        messages = await MessageRepository(session).history(conversation_id)
    return {
        "id": convo.id,
        "user_id": convo.user_id,
        "status": convo.status,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in messages
        ],
    }


@router.get("/v1/conversations/{conversation_id}/tasks", response_model=list[TaskSummary])
async def list_conversation_tasks(conversation_id: str, app: AppCtx, user: CurrentUser):
    """List a conversation's tasks, newest first — how a returning client rediscovers a task.

    Status-independent by design: the most recent entry is the one to reconnect to, and
    because task events are durably logged, reconnecting to its `stream_url` replays the full
    history even if it already completed.
    """
    async with app.db.session() as session:
        convo = await ConversationRepository(session).get(conversation_id)
        if convo is None:
            raise NotFoundError(f"Conversation not found: {conversation_id}")
        assert_conversation_access(convo, user, app.settings)
        tasks = await TaskRepository(session).list_for_conversation(conversation_id)
    return [
        TaskSummary(
            task_id=t.task_id,
            status=t.status,
            created_at=t.created_at.isoformat() if t.created_at else "",
            completed_at=t.completed_at.isoformat() if t.completed_at else None,
            stream_url=f"/v1/tasks/{t.task_id}/stream",
            status_url=f"/v1/tasks/{t.task_id}/status",
        )
        for t in tasks
    ]


@router.delete("/v1/conversations/{conversation_id}", status_code=202)
async def end_conversation(conversation_id: str, app: AppCtx, user: CurrentUser):
    async with app.db.session() as session:
        convo = await ConversationRepository(session).get(conversation_id)
        if convo is None:
            raise NotFoundError(f"Conversation not found: {conversation_id}")
        assert_conversation_access(convo, user, app.settings)
    await app.conversations.end(conversation_id)
    return {"status": "ended", "conversation_id": conversation_id}
