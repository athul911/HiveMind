"""Authorization helpers for conversation/task access.

A conversation is private to the user who created it. Access by anyone else is denied
unless explicitly allowed — currently the only explicit exception is an operator holding the
admin scope (when RBAC is enabled). This is the seam where future sharing rules would go.
"""

from __future__ import annotations

from typing import Protocol

from hivemind.api.auth import Principal
from hivemind.api.deps import _principal_scopes
from hivemind.config import Settings
from hivemind.core.errors import AuthorizationError


class _OwnedResource(Protocol):
    user_id: str


def assert_conversation_access(
    conversation: _OwnedResource, principal: Principal, settings: Settings
) -> None:
    """Raise ``AuthorizationError`` unless ``principal`` may access ``conversation``."""
    if conversation.user_id == principal.user_id:
        return
    if settings.rbac_enabled and settings.admin_scope in _principal_scopes(principal):
        return  # explicit operator override
    raise AuthorizationError("You do not have access to this conversation.")
