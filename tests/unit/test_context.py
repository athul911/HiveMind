from __future__ import annotations

import asyncio

from hivemind.core.context import (
    RequestContext,
    bind_context,
    clear_context,
    get_context,
    reset_context,
    update_context,
)


def test_bind_and_get():
    ctx = RequestContext(user_id="u1", conversation_id="c1")
    token = bind_context(ctx)
    try:
        got = get_context()
        assert got is not None
        assert got.user_id == "u1"
        assert got.conversation_id == "c1"
    finally:
        reset_context(token)


def test_update_context_overrides_fields():
    token = bind_context(RequestContext(user_id="u1"))
    try:
        t2 = update_context(agent_id="a1")
        assert get_context().agent_id == "a1"
        assert get_context().user_id == "u1"
        reset_context(t2)
    finally:
        reset_context(token)


def test_to_log_dict_omits_none():
    ctx = RequestContext(request_id="r1", user_id="u1")
    d = ctx.to_log_dict()
    assert d["request_id"] == "r1"
    assert d["user_id"] == "u1"
    assert "agent_id" not in d


async def test_context_isolated_across_tasks():
    results: dict[str, str | None] = {}

    async def worker(name: str) -> None:
        bind_context(RequestContext(user_id=name))
        await asyncio.sleep(0.01)
        results[name] = get_context().user_id

    await asyncio.gather(worker("a"), worker("b"))
    assert results == {"a": "a", "b": "b"}
    clear_context()
