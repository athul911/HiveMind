"""Task dispatch — the API side of queue mode.

Creates an idempotent ``tasks`` row and publishes a message to RabbitMQ. Returns the
``task_id`` the client uses to stream/poll. Idempotency-Key support means a retried request
returns the same task instead of double-dispatching.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from hivemind.db.repository import TaskRepository
from hivemind.db.session import Database
from hivemind.observability.logging import get_logger
from hivemind.workers.broker import TaskBroker

logger = get_logger("hivemind.dispatcher")


@dataclass
class TaskDispatcher:
    db: Database
    broker: TaskBroker

    async def dispatch(
        self,
        *,
        conversation_id: str,
        user_id: str,
        agent_id: str | None,
        user_message: str,
        idempotency_key: str | None = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        async with self.db.session() as session:
            task = await TaskRepository(session).create_idempotent(
                task_id, conversation_id, idempotency_key
            )
            task_id = task.task_id  # may be an existing task for the same idempotency key
            already_dispatched = task.status != "queued"

        if already_dispatched:
            logger.info("task.idempotent_hit", task_id=task_id)
            return task_id

        await self.broker.publish(
            {
                "task_id": task_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "user_message": user_message,
            }
        )
        logger.info("task.dispatched", task_id=task_id, conversation_id=conversation_id)
        return task_id
