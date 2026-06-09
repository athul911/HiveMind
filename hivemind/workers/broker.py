"""RabbitMQ broker helpers (aio-pika).

A thin wrapper for publishing task messages and consuming them. The queue is durable and
messages are persistent so a broker restart doesn't lose work. Consumers ack only after the
task finishes, so a crashed worker's task is redelivered (at-least-once); idempotent task
handling makes redelivery safe.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.broker")


class TaskBroker:
    def __init__(self, url: str, queue_name: str) -> None:
        self._url = url
        self._queue_name = queue_name
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=1)
        await self._channel.declare_queue(self._queue_name, durable=True)
        logger.info("broker.connected", queue=self._queue_name)

    async def publish(self, message: dict) -> None:
        if self._channel is None:
            raise RuntimeError("Broker not connected.")
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            ),
            routing_key=self._queue_name,
        )

    async def consume(self, handler: Callable[[dict], Awaitable[None]]) -> None:
        """Consume forever, invoking ``handler`` per message. Acks on success only."""
        if self._channel is None:
            raise RuntimeError("Broker not connected.")
        queue = await self._channel.declare_queue(self._queue_name, durable=True)

        async def _on_message(message: AbstractIncomingMessage) -> None:
            async with message.process(requeue=True):
                payload = json.loads(message.body.decode("utf-8"))
                await handler(payload)

        await queue.consume(_on_message)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
