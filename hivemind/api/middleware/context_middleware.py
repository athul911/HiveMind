"""Request context + auth + structured access logging middleware.

Runs first in the stack. It authenticates the bearer token, binds a
:class:`RequestContext` (with the OTel trace id when available), logs the request, and
clears the context on the way out. Health endpoints bypass auth.
"""

from __future__ import annotations

import time

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hivemind.api.auth import TokenVerifier
from hivemind.core.context import RequestContext, bind_context, reset_context
from hivemind.core.errors import AuthenticationError
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.http")

_PUBLIC_PATHS = {"/health", "/healthz", "/readyz", "/metrics", "/docs", "/openapi.json", "/redoc"}


def _is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS


class ContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, verifier: TokenVerifier) -> None:
        super().__init__(app)
        self._verifier = verifier

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_public(request.url.path):
            return await call_next(request)

        try:
            principal = self._authenticate(request)
        except AuthenticationError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.to_problem(),
                media_type="application/problem+json",
            )

        trace_id = None
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            trace_id = format(span.get_span_context().trace_id, "032x")

        ctx = RequestContext(
            user_id=principal.user_id,
            conversation_id=request.headers.get("x-conversation-id"),
            trace_id=trace_id,
        )
        token = bind_context(ctx)
        request.state.principal = principal
        started = time.perf_counter()
        try:
            logger.info("http.request", method=request.method, path=request.url.path)
            response = await call_next(request)
            logger.info(
                "http.response",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_s=round(time.perf_counter() - started, 4),
            )
            response.headers["x-request-id"] = ctx.request_id
            return response
        finally:
            reset_context(token)

    def _authenticate(self, request: Request):
        header = request.headers.get("authorization", "")
        token = header.split(" ", 1)[1].strip() if header.lower().startswith("bearer ") else ""
        # When auth is disabled (local dev), no header is required — the verifier returns a
        # local-dev principal. Otherwise a Bearer token is mandatory.
        if not token and not self._verifier.auth_disabled:
            raise AuthenticationError("Missing or malformed Authorization header.")
        return self._verifier.verify(token)
