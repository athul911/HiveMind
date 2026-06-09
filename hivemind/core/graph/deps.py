"""Shared dependency container for graph execution.

Bundles the registries/factories the graph nodes need. Created once per process and passed
explicitly to the graph builder — no module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass

from hivemind.config import Settings
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.llm.factory import LLMProviderFactory
from hivemind.core.tools.registry import ToolRegistry


@dataclass
class GraphDeps:
    settings: Settings
    agents: AgentRegistry
    agent_factory: AgentFactory
    llm_factory: LLMProviderFactory
    tools: ToolRegistry
