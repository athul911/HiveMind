"""Alembic migration environment (async).

The database URL comes from :class:`Settings` (env), not alembic.ini, so the same
configuration source drives the app and migrations.
"""

from __future__ import annotations

import asyncio

from alembic import context
from hivemind.config import get_settings
from hivemind.db.models import Base
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config
target_metadata = Base.metadata


def _dsn() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_dsn(), poolclass=None)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
