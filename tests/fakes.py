"""Reusable in-memory fakes for unit-testing services, workers, and API routes.

These let us exercise route handlers and service logic without a live Postgres / RabbitMQ /
Redis. Repository classes are monkeypatched per-module (they're instantiated as
``Repo(session)`` inside the code under test), so the fake :class:`FakeDatabase` only needs
to yield a sentinel session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from hivemind.core.graph import events


class FakeDatabase:
    """A Database stand-in whose ``session()`` yields a sentinel (repos are monkeypatched)."""

    def __init__(self) -> None:
        self.engine = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[object]:
        yield object()

    async def dispose(self) -> None:
        return None


class FakeRunner:
    """A GraphRunner stand-in that replays a fixed list of events."""

    def __init__(self, events_to_emit: list[events.GraphEvent]) -> None:
        self._events = events_to_emit
        self.calls: list[dict] = []

    async def run(self, **kwargs: Any) -> AsyncIterator[events.GraphEvent]:
        self.calls.append(kwargs)
        for ev in self._events:
            yield ev


@dataclass
class FakeMessage:
    role: str
    content: str


class FakeMessageRepo:
    """Records added messages in a class-level store and returns canned history."""

    store: dict[str, list[FakeMessage]] = {}

    def __init__(self, _session: object) -> None: ...

    async def add(self, conversation_id, role, content, **kwargs):
        FakeMessageRepo.store.setdefault(conversation_id, []).append(FakeMessage(role, content))
        return FakeMessage(role, content)

    async def history(self, conversation_id, limit: int = 200):
        return FakeMessageRepo.store.get(conversation_id, [])


class FakeConversationRepo:
    created: list[str] = []
    statuses: dict[str, str] = {}

    def __init__(self, _session: object) -> None: ...

    async def get_or_create(self, conversation_id, *a, **k):
        FakeConversationRepo.created.append(conversation_id)
        return object()

    async def set_status(self, conversation_id, status):
        FakeConversationRepo.statuses[conversation_id] = status

    async def list_expired(self, now=None):
        return []


class FakeEphemeralRepo:
    deleted: list[str] = []

    def __init__(self, _session: object) -> None: ...

    async def delete_for_conversation(self, conversation_id):
        FakeEphemeralRepo.deleted.append(conversation_id)
        return 1

    async def delete_expired(self, now=None):
        return 2


@dataclass
class FakeTask:
    task_id: str
    conversation_id: str = "c1"
    status: str = "queued"
    result: dict | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)


class FakeTaskRepo:
    tasks: dict[str, FakeTask] = {}

    def __init__(self, _session: object) -> None: ...

    async def create_idempotent(self, task_id, conversation_id, idempotency_key):
        task = FakeTaskRepo.tasks.get(task_id) or FakeTask(task_id, conversation_id)
        FakeTaskRepo.tasks[task_id] = task
        return task

    async def get(self, task_id):
        return FakeTaskRepo.tasks.get(task_id)

    async def set_status(self, task_id, status, **kwargs):
        if task_id in FakeTaskRepo.tasks:
            FakeTaskRepo.tasks[task_id].status = status


class FakeBroker:
    def __init__(self) -> None:
        self.published: list[dict] = []

    async def connect(self) -> None: ...

    async def publish(self, message: dict) -> None:
        self.published.append(message)

    async def consume(self, handler) -> None:
        return None

    async def close(self) -> None: ...


class FakeRedis:
    """Minimal Redis Streams stand-in supporting xadd/xread used by the event buffer."""

    def __init__(self) -> None:
        self._streams: dict[str, list[tuple[str, dict]]] = {}
        self._counter = 0

    async def xadd(self, key, fields, **kwargs):
        self._counter += 1
        entry_id = f"{self._counter}-0"
        self._streams.setdefault(key, []).append((entry_id, dict(fields)))
        return entry_id

    async def xread(self, streams, block=0, count=50):
        out = []
        for key, after in streams.items():
            entries = [e for e in self._streams.get(key, []) if e[0] > after]
            if entries:
                out.append((key, entries[:count]))
        return out

    async def incr(self, key):
        return 1

    async def expire(self, key, ttl):
        return True

    async def aclose(self) -> None: ...


def reset_fakes() -> None:
    """Clear class-level fake stores between tests."""
    FakeMessageRepo.store.clear()
    FakeConversationRepo.created.clear()
    FakeConversationRepo.statuses.clear()
    FakeEphemeralRepo.deleted.clear()
    FakeTaskRepo.tasks.clear()
