"""Conversation ownership checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hivemind.api.auth import Principal
from hivemind.api.authz import assert_conversation_access
from hivemind.config import Settings
from hivemind.core.errors import AuthorizationError


def _convo(user_id: str):
    return SimpleNamespace(user_id=user_id)


def test_owner_allowed():
    p = Principal(user_id="alice", claims={"sub": "alice"})
    assert_conversation_access(_convo("alice"), p, Settings())  # no raise


def test_other_user_denied():
    p = Principal(user_id="bob", claims={"sub": "bob"})
    with pytest.raises(AuthorizationError):
        assert_conversation_access(_convo("alice"), p, Settings())


def test_admin_scope_bypass_when_rbac_enabled():
    p = Principal(user_id="bob", claims={"sub": "bob", "scope": "hivemind:admin"})
    settings = Settings(rbac_enabled=True, admin_scope="hivemind:admin")
    assert_conversation_access(_convo("alice"), p, settings)  # admin override, no raise


def test_admin_scope_ignored_when_rbac_disabled():
    p = Principal(user_id="bob", claims={"sub": "bob", "scope": "hivemind:admin"})
    with pytest.raises(AuthorizationError):
        assert_conversation_access(_convo("alice"), p, Settings(rbac_enabled=False))
