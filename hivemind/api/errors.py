"""Exception handlers producing RFC 7807 problem+json responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from hivemind.core.context import get_context
from hivemind.core.errors import HiveMindError
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.errors")

_PROBLEM = "application/problem+json"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HiveMindError)
    async def _hivemind(_request: Request, exc: HiveMindError):
        return JSONResponse(
            status_code=exc.status_code, content=exc.to_problem(), media_type=_PROBLEM
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError):
        ctx = get_context()
        return JSONResponse(
            status_code=422,
            media_type=_PROBLEM,
            content={
                "type": "https://hivemind.dev/errors/validation-error",
                "title": "Validation Error",
                "status": 422,
                "detail": "Request validation failed.",
                "errors": exc.errors(),
                "request_id": ctx.request_id if ctx else None,
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        ctx = get_context()
        logger.error("unhandled_error", error=str(exc), error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            media_type=_PROBLEM,
            content={
                "type": "https://hivemind.dev/errors/internal-error",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred.",
                "request_id": ctx.request_id if ctx else None,
            },
        )
