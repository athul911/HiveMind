"""RabbitMQ worker — consumes tasks and runs the LangGraph workflow.

For each task it binds a :class:`RequestContext`, runs the same ``ConversationService`` the
API uses, and buffers every event to the :class:`TaskEventBuffer` (durable + live). On
SIGTERM it stops accepting new deliveries and lets the in-flight task finish (graceful
shutdown). Task handling is idempotent so at-least-once redelivery is safe.
"""

from __future__ import annotations

import asyncio
import signal

import redis.asyncio as aioredis

from hivemind.bootstrap import build_context
from hivemind.config import get_settings
from hivemind.core.context import RequestContext, bind_context, reset_context
from hivemind.db.repository import TaskRepository
from hivemind.observability.logging import get_logger
from hivemind.workers.broker import TaskBroker
from hivemind.workers.events import TaskEventBuffer

logger = get_logger("hivemind.worker")


async def _handle_task(payload: dict, app, buffer: TaskEventBuffer) -> None:
    task_id = payload["task_id"]
    conversation_id = payload["conversation_id"]
    ctx = RequestContext(
        conversation_id=conversation_id,
        user_id=payload.get("user_id"),
        agent_id=payload.get("agent_id"),
        task_id=task_id,
    )
    token = bind_context(ctx)
    seq = 0
    final_text = ""
    usage: dict = {}
    try:
        async with app.db.session() as session:
            await TaskRepository(session).set_status(task_id, "running")
        async for event in app.conversations.stream(
            conversation_id=conversation_id,
            user_id=payload.get("user_id", "system"),
            agent_id=payload.get("agent_id"),
            user_message=payload["user_message"],
            mode="queue",
        ):
            seq += 1
            if event.type == "done":
                final_text = event.data.get("final", "")
            elif event.type == "usage":
                usage = event.data
            await buffer.publish(task_id, seq, event)
        async with app.db.session() as session:
            await TaskRepository(session).set_status(
                task_id, "completed", result={"final": final_text}, usage=usage
            )
    except Exception as exc:
        logger.error("task.failed", task_id=task_id, error=str(exc))
        from hivemind.core.graph import events as ev

        seq += 1
        await buffer.publish(task_id, seq, ev.error(str(exc), error_type=type(exc).__name__))
        async with app.db.session() as session:
            await TaskRepository(session).set_status(task_id, "failed", error=str(exc))
    finally:
        # Always release the conversation's turn lock so it isn't stuck "running" after the
        # task ends (completed, failed, or — if a cancel raced in — already unlocked).
        from hivemind.db.repository import ConversationRepository

        async with app.db.session() as session:
            await ConversationRepository(session).release_lock(conversation_id)
        reset_context(token)


async def run_worker() -> None:
    settings = get_settings()
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    async with build_context(settings) as app:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        buffer = TaskEventBuffer(app.db, redis)
        broker = TaskBroker(settings.rabbitmq_url, settings.rabbitmq_task_queue)
        await broker.connect()

        async def handler(payload: dict) -> None:
            await _handle_task(payload, app, buffer)

        await broker.consume(handler)
        logger.info("worker.started")
        await stop_event.wait()  # graceful: in-flight message.process() completes its ack
        logger.info("worker.stopping")
        await broker.close()
        await redis.aclose()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
