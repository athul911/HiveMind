"""Async engine + session factory management.

The engine and sessionmaker are created once per process (in the app/worker lifespan)
and handed to repositories — no module-level global engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@dataclass
class Database:
    """Owns the async engine and session factory for its lifetime."""

    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]

    @classmethod
    def create(cls, dsn: str, *, echo: bool = False, pool_size: int = 10) -> Database:
        engine = create_async_engine(
            dsn,
            echo=echo,
            pool_size=pool_size,
            max_overflow=pool_size,
            pool_pre_ping=True,
        )
        maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        return cls(engine=engine, sessionmaker=maker)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session that commits on success and rolls back on error."""
        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        await self.engine.dispose()
