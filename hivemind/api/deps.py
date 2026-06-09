"""FastAPI dependency providers.

Shared resources live on ``app.state`` (set during lifespan). These helpers expose them via
``Depends`` so handlers never touch global state.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from hivemind.api.auth import Principal
from hivemind.bootstrap import AppContext
from hivemind.core.errors import AuthenticationError, AuthorizationError
from hivemind.workers.dispatcher import TaskDispatcher
from hivemind.workers.events import TaskEventBuffer


def get_app_context(request: Request) -> AppContext:
    return request.app.state.context


def _principal_scopes(principal: Principal) -> set[str]:
    """Extract scopes from common JWT claim shapes (``scope``, ``scopes``, ``permissions``)."""
    claims = principal.claims
    scopes: set[str] = set()
    raw = claims.get("scope") or claims.get("scp")
    if isinstance(raw, str):
        scopes.update(raw.split())
    for key in ("scopes", "permissions", "roles"):
        value = claims.get(key)
        if isinstance(value, list):
            scopes.update(str(v) for v in value)
    return scopes


def require_admin(request: Request) -> Principal:
    """Authorize agent-management routes. No-op unless ``rbac_enabled`` is set."""
    principal = get_principal(request)
    settings = request.app.state.context.settings
    if not settings.rbac_enabled:
        return principal
    if settings.admin_scope not in _principal_scopes(principal):
        raise AuthorizationError(
            "Agent management requires the admin scope.", required_scope=settings.admin_scope
        )
    return principal


def get_dispatcher(request: Request) -> TaskDispatcher:
    return request.app.state.dispatcher


def get_event_buffer(request: Request) -> TaskEventBuffer:
    return request.app.state.event_buffer


def get_principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise AuthenticationError("Not authenticated.")
    return principal


AppCtx = Annotated[AppContext, Depends(get_app_context)]
Dispatcher = Annotated[TaskDispatcher, Depends(get_dispatcher)]
EventBuffer = Annotated[TaskEventBuffer, Depends(get_event_buffer)]
CurrentUser = Annotated[Principal, Depends(get_principal)]
AdminUser = Annotated[Principal, Depends(require_admin)]
