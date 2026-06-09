"""LangGraph checkpointer wiring (Postgres).

Uses the official ``langgraph-checkpoint-postgres`` ``AsyncPostgresSaver`` as the source of
truth for graph state — it snapshots after every node, enabling crash recovery and resume.
The saver owns its own tables (created by ``setup()``); these are separate from HiveMind's
domain tables. For tests / no-DB runs, an in-memory saver is provided.
"""

from __future__ import annotations

from contextlib import asynccontextmanager


def _to_psycopg_dsn(dsn: str) -> str:
    """AsyncPostgresSaver uses psycopg, not asyncpg — normalize the scheme."""
    return dsn.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )


@asynccontextmanager
async def open_checkpointer(dsn: str):
    """Async context manager yielding a set-up AsyncPostgresSaver."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(_to_psycopg_dsn(dsn)) as saver:
        await saver.setup()
        yield saver


def memory_checkpointer():
    """In-memory checkpointer for tests and ephemeral runs."""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
