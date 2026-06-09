"""AgentService.ensure_exists: provision once, reconcile on provider/model drift."""

from __future__ import annotations

import pytest
from hivemind.core.agents.agent import Agent
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.llm.base import LLMConfig
from hivemind.db.models import AgentModel
from hivemind.services import agent_service as svc_mod
from hivemind.services.agent_service import AgentService

from tests.fakes import FakeDatabase


def _builtin(provider: str, model: str) -> Agent:
    return Agent(
        name="sql-specialist",
        system_prompt="p",
        llm_config=LLMConfig(provider=provider, model=model),
    )


def _model(provider: str, model: str, version: int = 1) -> AgentModel:
    return AgentModel(
        id=f"id-v{version}",
        name="sql-specialist",
        version=version,
        system_prompt="p",
        description="",
        tool_names=[],
        skill_names=[],
        llm_config={"provider": provider, "model": model, "max_tokens": 4096},
        immutable=True,
    )


class _Repo:
    """Fake AgentRepository backed by a class-level dict."""

    by_name: dict = {}
    added: list = []
    decommissioned: list = []

    def __init__(self, _session) -> None: ...

    async def get_by_name(self, name):
        return _Repo.by_name.get(name)

    async def get(self, agent_id):
        return next((m for m in _Repo.by_name.values() if m.id == agent_id), object())

    async def add(self, model):
        _Repo.added.append(model)
        _Repo.by_name[model.name] = model
        return model

    async def decommission(self, agent_id):
        _Repo.decommissioned.append(agent_id)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    _Repo.by_name.clear()
    _Repo.added.clear()
    _Repo.decommissioned.clear()
    monkeypatch.setattr(svc_mod, "AgentRepository", _Repo)
    yield


async def test_ensure_exists_creates_when_absent():
    svc = AgentService(FakeDatabase(), AgentRegistry())
    agent = await svc.ensure_exists(_builtin("openai", "gpt-4o"))
    assert agent.llm_config.provider == "openai"
    assert len(_Repo.added) == 1


async def test_ensure_exists_reuses_when_config_matches():
    _Repo.by_name["sql-specialist"] = _model("openai", "gpt-4o")
    svc = AgentService(FakeDatabase(), AgentRegistry())
    agent = await svc.ensure_exists(_builtin("openai", "gpt-4o"))
    assert agent.llm_config.provider == "openai"
    assert _Repo.added == []  # nothing re-created
    assert _Repo.decommissioned == []


async def test_ensure_exists_reprovisions_on_provider_drift():
    # Persisted as ollama (old default); operator switched to openai.
    _Repo.by_name["sql-specialist"] = _model("ollama", "llama3.1", version=1)
    registry = AgentRegistry()
    svc = AgentService(FakeDatabase(), registry)
    agent = await svc.ensure_exists(_builtin("openai", "gpt-4o"))
    assert agent.llm_config.provider == "openai"
    assert agent.version == 2  # new version provisioned
    assert _Repo.decommissioned == ["id-v1"]  # old one retired
    assert registry.get_by_name("sql-specialist").llm_config.provider == "openai"
