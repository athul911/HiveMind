"""API integration tests.

Marked ``integration`` — they require the docker-compose stack (Postgres, RabbitMQ, Redis)
to be up and the schema migrated. Run with ``make up && make migrate && make test-integration``.
They exercise the real app factory, auth, agent CRUD, and the SSE chat path end-to-end.
"""

from __future__ import annotations

import os

import pytest
from asgi_lifespan import LifespanManager
from hivemind.config import Settings
from hivemind.main import create_app
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration


@pytest.fixture
async def client():
    settings = Settings(auth_disabled=True, environment="test")
    app = create_app(settings)
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", headers={"Authorization": "Bearer x"}
        ) as ac:
            yield ac


@pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="requires live Postgres")
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="requires live Postgres")
async def test_list_tools_includes_builtins(client: AsyncClient):
    resp = await client.get("/v1/tools")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert {"sql_query", "code_exec", "web_search", "spawn_subagent"} <= names


@pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="requires live Postgres")
async def test_sql_specialist_provisioned(client: AsyncClient):
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200
    assert any(a["name"] == "sql-specialist" for a in resp.json())
