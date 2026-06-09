"""Repository layer — the only place that issues ORM/SQL queries.

Business logic depends on these repositories, never on the ORM directly. Read and write
operations are kept as distinct methods (command/query separation). Task and checkpoint
writes are idempotent to survive retries.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from hivemind.db.models import (
    AgentModel,
    ConversationModel,
    EphemeralAgentModel,
    MessageModel,
    SkillModel,
    TaskEventModel,
    TaskModel,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentRepository:
    """Persistence for immutable agent definitions."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(self, agent: AgentModel) -> AgentModel:
        self._s.add(agent)
        await self._s.flush()
        return agent

    async def get(self, agent_id: str) -> AgentModel | None:
        return await self._s.get(AgentModel, agent_id)

    async def get_by_name(self, name: str) -> AgentModel | None:
        stmt = (
            select(AgentModel)
            .where(AgentModel.name == name, AgentModel.decommissioned.is_(False))
            .order_by(AgentModel.version.desc())
            .limit(1)
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def list_active(self) -> list[AgentModel]:
        stmt = select(AgentModel).where(AgentModel.decommissioned.is_(False))
        return list((await self._s.execute(stmt)).scalars().all())

    async def decommission(self, agent_id: str) -> None:
        await self._s.execute(
            update(AgentModel).where(AgentModel.id == agent_id).values(decommissioned=True)
        )


class SkillRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert(self, name: str, description: str, body: str, version: int = 1) -> SkillModel:
        existing = await self.get_by_name(name)
        if existing is not None:
            existing.description = description
            existing.body = body
            existing.version = version
            await self._s.flush()
            return existing
        skill = SkillModel(name=name, description=description, body=body, version=version)
        self._s.add(skill)
        await self._s.flush()
        return skill

    async def get_by_name(self, name: str) -> SkillModel | None:
        stmt = (
            select(SkillModel)
            .where(SkillModel.name == name)
            .order_by(SkillModel.version.desc())
            .limit(1)
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def list_all(self) -> list[SkillModel]:
        return list((await self._s.execute(select(SkillModel))).scalars().all())


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(
        self, conversation_id: str, user_id: str, agent_id: str | None, ttl_seconds: int
    ) -> ConversationModel:
        convo = ConversationModel(
            id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            ttl_expires_at=_utcnow() + timedelta(seconds=ttl_seconds),
        )
        self._s.add(convo)
        await self._s.flush()
        return convo

    async def get(self, conversation_id: str) -> ConversationModel | None:
        return await self._s.get(ConversationModel, conversation_id)

    async def get_or_create(
        self, conversation_id: str, user_id: str, agent_id: str | None, ttl_seconds: int
    ) -> ConversationModel:
        existing = await self.get(conversation_id)
        if existing is not None:
            return existing
        return await self.create(conversation_id, user_id, agent_id, ttl_seconds)

    async def set_status(self, conversation_id: str, status: str) -> None:
        await self._s.execute(
            update(ConversationModel)
            .where(ConversationModel.id == conversation_id)
            .values(status=status, updated_at=_utcnow())
        )

    async def list_expired(self, now: datetime | None = None) -> list[ConversationModel]:
        now = now or _utcnow()
        stmt = select(ConversationModel).where(
            ConversationModel.ttl_expires_at.is_not(None),
            ConversationModel.ttl_expires_at < now,
            ConversationModel.status != "ended",
        )
        return list((await self._s.execute(stmt)).scalars().all())


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        tool_calls: dict | None = None,
        tool_results: dict | None = None,
        agent_id: str | None = None,
    ) -> MessageModel:
        msg = MessageModel(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            agent_id=agent_id,
        )
        self._s.add(msg)
        await self._s.flush()
        return msg

    async def history(self, conversation_id: str, limit: int = 200) -> list[MessageModel]:
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at.asc())
            .limit(limit)
        )
        return list((await self._s.execute(stmt)).scalars().all())


class EphemeralAgentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def checkpoint(
        self,
        ephemeral_id: str,
        parent_conversation_id: str,
        definition: dict,
        checkpoint: dict | None,
        ttl_seconds: int,
    ) -> None:
        """Idempotent upsert of a sub-agent definition + checkpoint."""
        stmt = pg_insert(EphemeralAgentModel).values(
            id=ephemeral_id,
            parent_conversation_id=parent_conversation_id,
            definition=definition,
            checkpoint=checkpoint,
            expires_at=_utcnow() + timedelta(seconds=ttl_seconds),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[EphemeralAgentModel.id],
            set_={"definition": definition, "checkpoint": checkpoint},
        )
        await self._s.execute(stmt)

    async def get(self, ephemeral_id: str) -> EphemeralAgentModel | None:
        return await self._s.get(EphemeralAgentModel, ephemeral_id)

    async def list_for_conversation(self, conversation_id: str) -> list[EphemeralAgentModel]:
        stmt = select(EphemeralAgentModel).where(
            EphemeralAgentModel.parent_conversation_id == conversation_id
        )
        return list((await self._s.execute(stmt)).scalars().all())

    async def delete_for_conversation(self, conversation_id: str) -> int:
        result = await self._s.execute(
            delete(EphemeralAgentModel).where(
                EphemeralAgentModel.parent_conversation_id == conversation_id
            )
        )
        return result.rowcount or 0

    async def delete_expired(self, now: datetime | None = None) -> int:
        now = now or _utcnow()
        result = await self._s.execute(
            delete(EphemeralAgentModel).where(
                EphemeralAgentModel.expires_at.is_not(None),
                EphemeralAgentModel.expires_at < now,
            )
        )
        return result.rowcount or 0


class TaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create_idempotent(
        self,
        task_id: str,
        conversation_id: str,
        idempotency_key: str | None,
    ) -> TaskModel:
        """Create a task, or return the existing one for the same idempotency key."""
        if idempotency_key:
            existing = await self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
        task = TaskModel(
            task_id=task_id,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
            status="queued",
        )
        self._s.add(task)
        await self._s.flush()
        return task

    async def get(self, task_id: str) -> TaskModel | None:
        return await self._s.get(TaskModel, task_id)

    async def get_by_idempotency_key(self, key: str) -> TaskModel | None:
        stmt = select(TaskModel).where(TaskModel.idempotency_key == key).limit(1)
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def set_status(
        self,
        task_id: str,
        status: str,
        *,
        result: dict | None = None,
        error: str | None = None,
        usage: dict | None = None,
    ) -> None:
        values: dict = {"status": status}
        if status in ("completed", "failed"):
            values["completed_at"] = _utcnow()
        if result is not None:
            values["result"] = result
        if error is not None:
            values["error"] = error
        if usage is not None:
            values["usage"] = usage
        await self._s.execute(
            update(TaskModel).where(TaskModel.task_id == task_id).values(**values)
        )


class TaskEventRepository:
    """Durable, replayable per-task event log (source of record for SSE replay)."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def append(self, task_id: str, seq: int, event_type: str, payload: dict) -> None:
        """Idempotent append keyed on (task_id, seq)."""
        stmt = (
            pg_insert(TaskEventModel)
            .values(task_id=task_id, seq=seq, event_type=event_type, payload=payload)
            .on_conflict_do_nothing(index_elements=["task_id", "seq"])
        )
        await self._s.execute(stmt)
        await self._s.execute(
            update(TaskModel).where(TaskModel.task_id == task_id).values(last_event_seq=seq)
        )

    async def replay(self, task_id: str, after_seq: int = 0) -> list[TaskEventModel]:
        stmt = (
            select(TaskEventModel)
            .where(TaskEventModel.task_id == task_id, TaskEventModel.seq > after_seq)
            .order_by(TaskEventModel.seq.asc())
        )
        return list((await self._s.execute(stmt)).scalars().all())

    async def delete_for_task(self, task_id: str) -> int:
        result = await self._s.execute(
            delete(TaskEventModel).where(TaskEventModel.task_id == task_id)
        )
        return result.rowcount or 0
