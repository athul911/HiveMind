"""Agent management service.

Bridges the persistence layer (``AgentRepository``) and the in-memory ``AgentRegistry``.
On create, an agent is persisted and immediately registered. On startup all persisted
agents are hydrated into the registry. Agents are immutable — there is no update path.
"""

from __future__ import annotations

from dataclasses import replace

from hivemind.core.agents.agent import Agent
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.errors import NotFoundError
from hivemind.db.models import AgentModel
from hivemind.db.repository import AgentRepository
from hivemind.db.session import Database
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.agents")


class AgentService:
    def __init__(self, db: Database, registry: AgentRegistry) -> None:
        self._db = db
        self._registry = registry

    async def create(self, agent: Agent) -> Agent:
        model = AgentModel(
            id=agent.id,
            name=agent.name,
            version=agent.version,
            description=agent.description,
            system_prompt=agent.system_prompt,
            tool_names=list(agent.tool_names),
            skill_names=list(agent.skill_names),
            llm_config=agent.llm_config.to_dict(),
            immutable=agent.immutable,
        )
        async with self._db.session() as session:
            await AgentRepository(session).add(model)
        self._registry.add(agent)
        logger.info("agent.created", agent_id=agent.id, name=agent.name)
        return agent

    async def decommission(self, agent_id: str) -> None:
        async with self._db.session() as session:
            repo = AgentRepository(session)
            if await repo.get(agent_id) is None:
                raise NotFoundError(f"Agent not found: {agent_id}", agent_id=agent_id)
            await repo.decommission(agent_id)
        self._registry.remove(agent_id)
        logger.info("agent.decommissioned", agent_id=agent_id)

    async def hydrate(self) -> int:
        """Load all active agents from the database into the registry."""
        async with self._db.session() as session:
            models = await AgentRepository(session).list_active()
        for model in models:
            self._registry.add(_model_to_agent(model))
        logger.info("agents.hydrated", count=len(models))
        return len(models)

    async def ensure_exists(self, agent: Agent) -> Agent:
        """Provision/reconcile a built-in agent.

        If no agent with this name exists, create it. If one exists but its LLM provider/model
        has drifted from the desired config (e.g. the operator switched ``LLM_DEFAULT_PROVIDER``
        from ollama to openai), provision a **new version** with the current config and retire
        the old one. This keeps system-managed built-ins in sync with configuration without
        violating per-version immutability.
        """
        async with self._db.session() as session:
            existing = await AgentRepository(session).get_by_name(agent.name)
        if existing is None:
            return await self.create(agent)

        hydrated = _model_to_agent(existing)
        desired = agent.llm_config
        current = hydrated.llm_config
        if (current.provider, current.model) == (desired.provider, desired.model):
            self._registry.add(hydrated)
            return hydrated

        logger.info(
            "agent.reprovision",
            name=agent.name,
            from_provider=current.provider,
            to_provider=desired.provider,
            from_model=current.model,
            to_model=desired.model,
        )
        new_version = replace(agent, version=existing.version + 1)
        await self.decommission(existing.id)
        return await self.create(new_version)


def _model_to_agent(model: AgentModel) -> Agent:
    from hivemind.core.llm.base import LLMConfig

    return Agent(
        id=model.id,
        name=model.name,
        description=model.description,
        system_prompt=model.system_prompt,
        tool_names=tuple(model.tool_names),
        skill_names=tuple(model.skill_names),
        llm_config=LLMConfig.from_dict(model.llm_config),
        version=model.version,
        immutable=model.immutable,
    )
