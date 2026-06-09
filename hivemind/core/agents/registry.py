"""In-memory agent registry.

Holds the live set of agents for routing and execution. Hydrated from Postgres on startup
(API + worker) and updated when agents are created/decommissioned via the API.
"""

from __future__ import annotations

from hivemind.core.agents.agent import Agent
from hivemind.core.errors import NotFoundError


class AgentRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, Agent] = {}
        self._by_name: dict[str, Agent] = {}

    def add(self, agent: Agent) -> None:
        self._by_id[agent.id] = agent
        self._by_name[agent.name] = agent

    def remove(self, agent_id: str) -> None:
        agent = self._by_id.pop(agent_id, None)
        if agent is not None:
            self._by_name.pop(agent.name, None)

    def get(self, agent_id: str) -> Agent:
        agent = self._by_id.get(agent_id)
        if agent is None:
            raise NotFoundError(f"Agent not found: {agent_id}", agent_id=agent_id)
        return agent

    def get_optional(self, agent_id: str) -> Agent | None:
        return self._by_id.get(agent_id)

    def get_by_name(self, name: str) -> Agent | None:
        return self._by_name.get(name)

    def list(self) -> list[Agent]:
        return list(self._by_id.values())

    def routing_table(self) -> list[dict]:
        """Compact view the supervisor uses to choose agents."""
        return [
            {
                "agent_id": a.id,
                "name": a.name,
                "description": a.description or a.system_prompt[:200],
            }
            for a in self._by_id.values()
        ]
