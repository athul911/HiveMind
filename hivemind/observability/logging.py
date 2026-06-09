"""structlog configuration emitting JSON to stdout, enriched with request context.

Logs are emitted to stdout for collection by Kubernetes log aggregators (Loki/ELK).
Every entry is enriched with the active :class:`RequestContext` fields.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from hivemind.core.context import get_context


def _add_request_context(_logger: Any, _method: str, event_dict: dict) -> dict:
    """structlog processor: merge the active request context into every event."""
    ctx = get_context()
    if ctx is not None:
        for key, value in ctx.to_log_dict().items():
            event_dict.setdefault(key, value)
    return event_dict


_SECRET_HINTS = ("api_key", "apikey", "token", "secret", "password", "authorization", "dsn")
_REDACTED = "***redacted***"


def _redact_secrets(_logger: Any, _method: str, event_dict: dict) -> dict:
    """structlog processor: redact values for keys that look secret-bearing.

    Defense-in-depth — code shouldn't log secrets, but this guarantees an accidental
    ``logger.info("x", token=...)`` never leaks the value to stdout.
    """
    for key in list(event_dict.keys()):
        lowered = key.lower()
        if any(hint in lowered for hint in _SECRET_HINTS) and event_dict[key] not in (None, ""):
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure structlog + stdlib logging once at startup.

    Args:
        level: root log level (e.g. ``"INFO"``, ``"DEBUG"``).
        json_logs: emit JSON (production) or pretty console output (local dev).
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_request_context,
        _redact_secrets,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, sqlalchemy) through the same handler.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelNamesMapping().get(level.upper(), logging.INFO),
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
