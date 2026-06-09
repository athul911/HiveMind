"""SQLAlchemy ORM models (async).

These tables are the durable record of agents, skills, conversations, messages, tasks,
the task-event replay log, and ephemeral sub-agents. LangGraph owns its own checkpoint
tables (created by ``AsyncPostgresSaver``); they are not modeled here.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base with a portable JSONB type."""

    type_annotation_map = {dict: JSON}


def _uuid() -> str:
    return str(uuid.uuid4())


class AgentModel(Base):
    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_agent_name_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    system_prompt: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    tool_names: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    skill_names: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    llm_config: Mapped[dict] = mapped_column(JSONB, default=dict)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)
    decommissioned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SkillModel(Base):
    __tablename__ = "skills"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_skill_name_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    description: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationModel(Base):
    __tablename__ = "conversations"

    # String(64) to match every column that references it (FK type parity).
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    ttl_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    messages: Mapped[list[MessageModel]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageModel(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tool_results: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[ConversationModel] = relationship(back_populates="messages")


class EphemeralAgentModel(Base):
    __tablename__ = "ephemeral_agents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    parent_conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    definition: Mapped[dict] = mapped_column(JSONB)
    checkpoint: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )


class TaskModel(Base):
    __tablename__ = "tasks"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_task_idempotency_key"),)

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    usage: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_event_seq: Mapped[int] = mapped_column(Integer, default=0)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TaskEventModel(Base):
    __tablename__ = "task_events"
    __table_args__ = (
        UniqueConstraint("task_id", "seq", name="uq_task_event_seq"),
        Index("ix_task_events_task_seq", "task_id", "seq"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
