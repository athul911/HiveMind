"""Per-request context propagated via :mod:`contextvars`.

A :class:`RequestContext` carries identity and tracing identifiers through the entire
async call stack without threading arguments everywhere and without global mutable
state. ``contextvars`` are copied into child tasks, so concurrent requests never bleed
into one another. Both the structlog processor and the OTel span enricher read the
current context, so every log line and span is automatically tagged.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field, replace

_ctx_var: ContextVar[RequestContext | None] = ContextVar("hivemind_request_context", default=None)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Immutable bundle of identifiers carried through one logical request/task."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    trace_id: str | None = None
    subagent_depth: int = 0  # how many spawn_subagent hops deep we are

    def to_log_dict(self) -> dict[str, str]:
        """Return only the populated, non-default fields, for log/span enrichment."""
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None and not (k == "subagent_depth" and v == 0)
        }


def bind_context(ctx: RequestContext) -> Token:
    """Bind ``ctx`` as the current context. Returns a token for :func:`reset_context`."""
    return _ctx_var.set(ctx)


def get_context() -> RequestContext | None:
    """Return the current context, or ``None`` if none is bound."""
    return _ctx_var.get()


def require_context() -> RequestContext:
    """Return the current context or raise if unset (programmer error)."""
    ctx = _ctx_var.get()
    if ctx is None:
        raise RuntimeError("No RequestContext bound in this execution scope.")
    return ctx


def update_context(**fields: str | None) -> Token:
    """Bind a copy of the current context with ``fields`` overridden."""
    current = _ctx_var.get() or RequestContext()
    return _ctx_var.set(replace(current, **fields))


def reset_context(token: Token) -> None:
    """Restore the context to the state captured by ``token``."""
    _ctx_var.reset(token)


def clear_context() -> None:
    """Clear the current context (used at task boundaries)."""
    _ctx_var.set(None)
