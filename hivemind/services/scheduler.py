"""Background cleanup scheduler.

Periodically garbage-collects expired ephemeral sub-agents and ended/expired conversations
(and their task-event logs). Runs as an asyncio task inside the API/worker lifespan, and is
also exposed as a one-shot ``run_once`` for the Kubernetes CronJob.
"""

from __future__ import annotations

import asyncio
from datetime import UTC

from hivemind.db.repository import (
    ConversationRepository,
    EphemeralAgentRepository,
)
from hivemind.db.session import Database
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.scheduler")


class CleanupScheduler:
    def __init__(
        self,
        db: Database,
        *,
        interval_seconds: int,
        artifacts=None,
        lock_stale_seconds: int = 900,
    ) -> None:
        self._db = db
        self._interval = interval_seconds
        self._artifacts = artifacts  # ArtifactStore; enables artifact GC. Optional for tests.
        self._lock_stale_seconds = lock_stale_seconds
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def run_once(self) -> dict[str, int]:
        """Perform one cleanup pass. Returns counts of GC'd records.

        Expires overdue conversations (marking them ended + dropping their ephemeral agents
        and artifact directories), deletes expired ephemeral-agent rows, and releases stale
        conversation locks left behind by a crashed turn/worker.
        """
        from datetime import datetime, timedelta

        artifacts_deleted = 0
        async with self._db.session() as session:
            ephemeral = await EphemeralAgentRepository(session).delete_expired()
            convo_repo = ConversationRepository(session)
            ephemeral_repo = EphemeralAgentRepository(session)
            stale_cutoff = datetime.now(UTC) - timedelta(seconds=self._lock_stale_seconds)
            locks_released = await convo_repo.reset_stale_locks(stale_cutoff)
            expired = await convo_repo.list_expired()
            for convo in expired:
                await convo_repo.set_status(convo.id, "ended")
                await ephemeral_repo.delete_for_conversation(convo.id)
                if self._artifacts is not None and self._artifacts.delete_conversation(convo.id):
                    artifacts_deleted += 1
        logger.info(
            "cleanup.pass",
            ephemeral_deleted=ephemeral,
            conversations_expired=len(expired),
            artifacts_deleted=artifacts_deleted,
            stale_locks_released=locks_released,
        )
        return {
            "ephemeral_deleted": ephemeral,
            "conversations_expired": len(expired),
            "artifacts_deleted": artifacts_deleted,
            "stale_locks_released": locks_released,
        }

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("cleanup.error", error=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                continue

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
