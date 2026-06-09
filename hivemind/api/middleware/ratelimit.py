"""Redis-backed per-user token-bucket rate limiting.

Keyed on the authenticated ``user_id``. A fixed-window counter in Redis with a 60s TTL is
simple and adequate; it degrades open (allows the request) if Redis is unavailable so a
cache outage doesn't take down the API.
"""

from __future__ import annotations

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hivemind.core.errors import RateLimitError
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.ratelimit")

_EXEMPT = {"/health", "/healthz", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis: aioredis.Redis, *, per_minute: int) -> None:
        super().__init__(app)
        self._redis = redis
        self._limit = per_minute

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _EXEMPT:
            return await call_next(request)
        principal = getattr(request.state, "principal", None)
        if principal is None:
            return await call_next(request)

        key = f"hivemind:rl:{principal.user_id}"
        try:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, 60)
            if count > self._limit:
                exc = RateLimitError("Rate limit exceeded.", limit=self._limit)
                return JSONResponse(
                    status_code=exc.status_code,
                    content=exc.to_problem(),
                    media_type="application/problem+json",
                    headers={"Retry-After": "60"},
                )
        except Exception as exc:
            logger.warning("ratelimit.unavailable", error=str(exc))
        return await call_next(request)
