"""baseline schema

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("tool_names", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column(
            "skill_names", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"
        ),
        sa.Column("llm_config", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("immutable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("decommissioned", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("name", "version", name="uq_agent_name_version"),
    )
    op.create_table(
        "skills",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("name", "version", name="uq_skill_name_version"),
    )
    op.create_table(
        "conversations",
        # String(64) to match messages.conversation_id / tasks.conversation_id (FK parity).
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False, index=True),
        sa.Column("agent_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="active", index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("ttl_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        sa.Column("tool_results", postgresql.JSONB(), nullable=True),
        sa.Column("agent_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_messages_conversation_created", "messages", ["conversation_id", "created_at"]
    )
    op.create_table(
        "ephemeral_agents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("parent_conversation_id", sa.String(64), nullable=False, index=True),
        sa.Column("definition", postgresql.JSONB(), nullable=False),
        sa.Column("checkpoint", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True, index=True),
    )
    op.create_table(
        "tasks",
        sa.Column("task_id", sa.String(64), primary_key=True),
        sa.Column("conversation_id", sa.String(64), nullable=False, index=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued", index=True),
        sa.Column("usage", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("last_event_seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("idempotency_key", name="uq_task_idempotency_key"),
    )
    op.create_table(
        "task_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(64), nullable=False, index=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("task_id", "seq", name="uq_task_event_seq"),
    )
    op.create_index("ix_task_events_task_seq", "task_events", ["task_id", "seq"])


def downgrade() -> None:
    op.drop_table("task_events")
    op.drop_table("tasks")
    op.drop_table("ephemeral_agents")
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("skills")
    op.drop_table("agents")
