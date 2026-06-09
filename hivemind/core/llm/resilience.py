"""Resilience wrapper for LLM providers: retry-with-backoff + circuit breaker.

Wraps any :class:`LLMProvider`. Transient upstream failures are retried with exponential
backoff; a per-provider circuit breaker trips after a configurable number of consecutive
failures and fast-fails subsequent calls for a cooldown window, so one provider outage
doesn't stall every workflow.

Streaming policy: a stream is only retried if it fails **before** emitting its first event
(a cold failure). Once tokens have streamed, a mid-stream error propagates — we never replay
partial output.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from hivemind.core.errors import LLMProviderError
from hivemind.core.llm.base import LLMProvider, LLMRequest, LLMResponse, LLMStreamEvent
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.llm.resilience")


class CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: float) -> None:
        self._threshold = threshold
        self._reset_seconds = reset_seconds
        self._failures = 0
        self._opened_until = 0.0

    def check(self, name: str) -> None:
        if self._opened_until and time.monotonic() < self._opened_until:
            raise LLMProviderError(f"Circuit open for provider {name!r}; failing fast.")

    def record_success(self) -> None:
        self._failures = 0
        self._opened_until = 0.0

    def record_failure(self, name: str) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_until = time.monotonic() + self._reset_seconds
            logger.warning("llm.circuit_open", provider=name, cooldown_s=self._reset_seconds)


class ResilientProvider:
    """Decorates a provider with retry + circuit-breaker behavior."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int = 2,
        base_delay: float = 0.5,
        breaker_threshold: int = 5,
        breaker_reset_s: float = 30.0,
    ) -> None:
        self._inner = inner
        self.name = inner.name
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._breaker = CircuitBreaker(breaker_threshold, breaker_reset_s)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        attempt = 0
        while True:
            self._breaker.check(self.name)
            yielded = False
            try:
                async for event in self._inner.stream(request):
                    yielded = True
                    yield event
                self._breaker.record_success()
                return
            except Exception as exc:
                # Mid-stream failures can't be safely retried (would replay output).
                if yielded or attempt >= self._max_retries:
                    self._breaker.record_failure(self.name)
                    raise self._wrap(exc) from exc
                attempt += 1
                logger.warning("llm.retry", provider=self.name, attempt=attempt, error=str(exc))
                await asyncio.sleep(self._base_delay * (2 ** (attempt - 1)))

    async def complete(self, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            self._breaker.check(self.name)
            try:
                response = await self._inner.complete(request)
                self._breaker.record_success()
                return response
            except Exception as exc:
                if attempt >= self._max_retries:
                    self._breaker.record_failure(self.name)
                    raise self._wrap(exc) from exc
                attempt += 1
                logger.warning("llm.retry", provider=self.name, attempt=attempt, error=str(exc))
                await asyncio.sleep(self._base_delay * (2 ** (attempt - 1)))

    @staticmethod
    def _wrap(exc: Exception) -> LLMProviderError:
        return exc if isinstance(exc, LLMProviderError) else LLMProviderError(str(exc))
