"""API route tests with an injected fake AppContext (no live Postgres/RabbitMQ/Redis).

We build the real FastAPI app via ``create_app`` (real middleware, routers, error handlers,
schemas) but skip the lifespan and set ``app.state`` to in-memory fakes. DB-touching route
handlers have their repository classes monkeypatched. This exercises the full HTTP surface
deterministically and offline. (These are unit-level despite living under integration/.)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from asgi_lifespan import LifespanManager  # noqa: F401  (kept for parity; not used here)
from hivemind.api.routes import chat as chat_route
from hivemind.api.routes import tasks as tasks_route
from hivemind.config import Settings
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.graph import events
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.skills.skill import Skill
from hivemind.core.tools.base import BaseTool, ToolResult
from hivemind.core.tools.registry import ToolRegistry
from hivemind.main import create_app
from httpx import ASGITransport, AsyncClient

from tests import fakes


class _Noop(BaseTool):
    name = "sql_query"
    description = "Query."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, args, ctx):
        return ToolResult(content={})


class _FakeConversations:
    def __init__(self) -> None:
        self.ended: list[str] = []

    async def stream(self, **kwargs) -> AsyncIterator[events.GraphEvent]:
        yield events.text_delta("hello")
        yield events.message("assistant", "hello")
        yield events.done("hello")

    async def end(self, conversation_id: str) -> None:
        self.ended.append(conversation_id)


class _FakeAgentService:
    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    async def create(self, agent):
        self._registry.add(agent)
        return agent

    async def decommission(self, agent_id: str) -> None:
        self._registry.remove(agent_id)


class _FakeDispatcher:
    async def dispatch(self, **kwargs) -> str:
        return "task-123"


class _FakeBuffer:
    async def replay_and_tail(self, task_id, *, after_seq=0):
        yield 1, events.text_delta("x")
        yield 2, events.done("done")


@pytest.fixture
def client(monkeypatch):
    fakes.reset_fakes()
    settings = Settings(auth_disabled=True, environment="test", otel_enabled=False)

    tools = ToolRegistry()
    tools.register(_Noop())
    skills = SkillRegistry()
    skills.register(Skill(name="postgres-optimization", description="d", body="b"))
    registry = AgentRegistry()
    factory = AgentFactory(tools, skills)

    ctx = SimpleNamespace(
        settings=settings,
        tools=tools,
        skills=skills,
        agents=registry,
        agent_service=_FakeAgentService(registry),
        deps=SimpleNamespace(agent_factory=factory),
        conversations=_FakeConversations(),
        db=fakes.FakeDatabase(),
    )

    # Route modules that touch the DB directly: swap their repository classes for fakes.
    monkeypatch.setattr(chat_route, "ConversationRepository", _ConvoRepoForGet)
    monkeypatch.setattr(chat_route, "MessageRepository", fakes.FakeMessageRepo)
    monkeypatch.setattr(tasks_route, "TaskRepository", fakes.FakeTaskRepo)

    app = create_app(settings)
    app.state.context = ctx
    app.state.dispatcher = _FakeDispatcher()
    app.state.event_buffer = _FakeBuffer()

    transport = ASGITransport(app=app)
    return AsyncClient(
        transport=transport, base_url="http://t", headers={"Authorization": "Bearer x"}
    ), ctx


class _ConvoRepoForGet:
    def __init__(self, _session) -> None: ...

    async def get(self, conversation_id):
        return SimpleNamespace(id=conversation_id, user_id="u", status="active")


async def test_health_is_public(client):
    ac, _ = client
    async with ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_list_and_get_tools(client):
    ac, _ = client
    async with ac:
        resp = await ac.get("/v1/tools")
        assert resp.status_code == 200
        assert {t["name"] for t in resp.json()} == {"sql_query"}
        one = await ac.get("/v1/tools/sql_query")
        assert one.status_code == 200
        missing = await ac.get("/v1/tools/ghost")
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/problem+json")


async def test_list_skills(client):
    ac, _ = client
    async with ac:
        resp = await ac.get("/v1/skills")
        detail = await ac.get("/v1/skills/postgres-optimization")
    assert resp.status_code == 200
    assert detail.json()["name"] == "postgres-optimization"


async def test_agent_crud(client):
    ac, _ctx = client
    payload = {
        "name": "my-agent",
        "system_prompt": "You help.",
        "llm_config": {"provider": "anthropic", "model": "claude-opus-4-8"},
        "tool_names": ["sql_query"],
        "skill_names": ["postgres-optimization"],
    }
    async with ac:
        created = await ac.post("/v1/agents", json=payload)
        assert created.status_code == 201
        agent_id = created.json()["id"]
        listed = await ac.get("/v1/agents")
        assert any(a["id"] == agent_id for a in listed.json())
        got = await ac.get(f"/v1/agents/{agent_id}")
        assert got.json()["name"] == "my-agent"
        deleted = await ac.delete(f"/v1/agents/{agent_id}")
        assert deleted.status_code == 204


async def test_agent_create_rejects_unknown_tool(client):
    ac, _ = client
    payload = {
        "name": "bad",
        "system_prompt": "p",
        "llm_config": {"provider": "anthropic", "model": "m"},
        "tool_names": ["ghost"],
    }
    async with ac:
        resp = await ac.post("/v1/agents", json=payload)
    assert resp.status_code == 400


async def test_chat_sse_mode(client):
    ac, _ = client
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    async with ac:
        resp = await ac.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "done" in resp.text


async def test_chat_queue_mode_returns_task(client):
    ac, _ = client
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    async with ac:
        resp = await ac.post("/v1/chat/completions", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert body["task_id"] == "task-123"
    assert body["stream_url"].endswith("/stream")


async def test_conversation_get_and_delete(client):
    ac, ctx = client
    async with ac:
        got = await ac.get("/v1/conversations/conv-9")
        assert got.status_code == 200
        deleted = await ac.delete("/v1/conversations/conv-9")
    assert deleted.status_code == 202
    assert "conv-9" in ctx.conversations.ended


async def test_task_status_and_stream(client):
    ac, _ = client
    fakes.FakeTaskRepo.tasks["task-123"] = fakes.FakeTask("task-123", status="completed")
    async with ac:
        status = await ac.get("/v1/tasks/task-123/status")
        assert status.status_code == 200
        assert status.json()["status"] == "completed"
        stream = await ac.get("/v1/tasks/task-123/stream")
    assert stream.status_code == 200
    assert "done" in stream.text


async def test_missing_auth_header_rejected():
    settings = Settings(auth_disabled=False, environment="test", otel_enabled=False)
    app = create_app(settings)
    app.state.context = SimpleNamespace(settings=settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        resp = await ac.get("/v1/tools")
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/problem+json")
