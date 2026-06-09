"""Chat completions + conversation lifecycle endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, Request
from sse_starlette.sse import EventSourceResponse

from hivemind.api.deps import AppCtx, CurrentUser, Dispatcher
from hivemind.api.schemas.chat import AsyncTaskResponse, ChatCompletionRequest
from hivemind.api.sse import format_sse, graph_event_sse
from hivemind.core.context import get_context, update_context
from hivemind.core.errors import NotFoundError
from hivemind.core.graph import events
from hivemind.db.repository import ConversationRepository, MessageRepository
from hivemind.services.mode_selector import ModeSelector

router = APIRouter(tags=["chat"])


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

    if mode == "queue":
        task_id = await dispatcher.dispatch(
            conversation_id=conversation_id,
            user_id=user.user_id,
            agent_id=body.agent_id,
            user_message=user_message,
            idempotency_key=idempotency_key,
        )
        return AsyncTaskResponse(
            task_id=task_id,
            conversation_id=conversation_id,
            stream_url=f"/v1/tasks/{task_id}/stream",
            status_url=f"/v1/tasks/{task_id}/status",
        )

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
        yield format_sse(events.conversation(conversation_id, request_id))
        async for frame in graph_event_sse(stream):
            yield frame

    return EventSourceResponse(
        sse_stream(),
        headers={"x-conversation-id": conversation_id},
    )


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, app: AppCtx, user: CurrentUser):
    async with app.db.session() as session:
        convo = await ConversationRepository(session).get(conversation_id)
        if convo is None:
            raise NotFoundError(f"Conversation not found: {conversation_id}")
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


@router.delete("/v1/conversations/{conversation_id}", status_code=202)
async def end_conversation(conversation_id: str, app: AppCtx, user: CurrentUser):
    await app.conversations.end(conversation_id)
    return {"status": "ended", "conversation_id": conversation_id}
