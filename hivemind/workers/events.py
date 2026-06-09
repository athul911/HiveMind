"""Per-task event buffer for queue-mode workflows.

Each event is written to two places:
  * the durable ``task_events`` table — the **replay source of record**, queried when a
    client connects/reconnects to catch up from its last offset;
  * a **Redis Stream** (``task:{task_id}``) — the live fan-out channel the SSE endpoint
    tails for new events with low latency.

Redis Streams (not pub/sub) are used because pub/sub cannot replay from an offset; combined
with the DB log this gives reconnect-safe streaming.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import redis.asyncio as aioredis

from hivemind.core.graph import events
from hivemind.db.repository import TaskEventRepository
from hivemind.db.session import Database

_TERMINAL = {"done", "error"}


def _stream_key(task_id: str) -> str:
    return f"hivemind:task:{task_id}"


class TaskEventBuffer:
    def __init__(self, db: Database, redis: aioredis.Redis) -> None:
        self._db = db
        self._redis = redis

    async def publish(self, task_id: str, seq: int, event: events.GraphEvent) -> None:
        """Persist (durable) then fan out (live) a single event."""
        payload = event.to_dict()
        async with self._db.session() as session:
            await TaskEventRepository(session).append(task_id, seq, event.type, payload)
        await self._redis.xadd(
            _stream_key(task_id),
            {"seq": seq, "data": json.dumps(payload)},
            maxlen=10_000,
            approximate=True,
        )

    async def replay_and_tail(
        self, task_id: str, *, after_seq: int = 0, idle_timeout_ms: int = 30_000
    ) -> AsyncIterator[tuple[int, events.GraphEvent]]:
        """Yield (seq, event): durable replay after ``after_seq``, then tail the live stream.

        Stops after emitting a terminal event (``done``/``error``) or after an idle timeout
        with no new events (e.g. the producer crashed).
        """
        last_seq = after_seq
        terminal_seen = False

        # 1. Durable replay catches the client up to the present.
        async with self._db.session() as session:
            rows = await TaskEventRepository(session).replay(task_id, after_seq)
        for row in rows:
            last_seq = row.seq
            event = events.GraphEvent(type=row.event_type, data=row.payload.get("data", {}))
            yield row.seq, event
            if row.event_type in _TERMINAL:
                terminal_seen = True
        if terminal_seen:
            return

        # 2. Tail the Redis stream for events newer than what we replayed.
        last_id = "0-0"
        while not terminal_seen:
            resp = await self._redis.xread(
                {_stream_key(task_id): last_id}, block=idle_timeout_ms, count=50
            )
            if not resp:
                break  # idle timeout — producer likely finished or died
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    last_id = entry_id
                    seq = int(fields["seq"])
                    if seq <= last_seq:
                        continue
                    last_seq = seq
                    payload = json.loads(fields["data"])
                    event = events.GraphEvent(type=payload["type"], data=payload.get("data", {}))
                    yield seq, event
                    if event.type in _TERMINAL:
                        terminal_seen = True
