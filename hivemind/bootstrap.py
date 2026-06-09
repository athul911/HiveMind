"""Composition root.

Wires every component together with explicit dependency injection — no global mutable
state. Shared by the API app factory and the worker. The returned :class:`AppContext` owns
the lifetimes of the engine, checkpointer, and background scheduler.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from hivemind.config import Settings
from hivemind.core.agents.builtin import sql_specialist_agent
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.agents.registry import AgentRegistry
from hivemind.core.graph.checkpointer import memory_checkpointer, open_checkpointer
from hivemind.core.graph.deps import GraphDeps
from hivemind.core.graph.runner import GraphRunner
from hivemind.core.graph.subagent_runner import SubAgentRunnerImpl
from hivemind.core.llm.factory import LLMProviderFactory
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.tools.code_tool import CodeExecTool
from hivemind.core.tools.registry import ToolRegistry
from hivemind.core.tools.sandbox import build_sandbox
from hivemind.core.tools.sql_tool import SQLTool
from hivemind.core.tools.subagent_tool import SpawnSubAgentTool
from hivemind.core.tools.web_search_tool import WebSearchTool
from hivemind.db.session import Database
from hivemind.observability.logging import configure_logging, get_logger
from hivemind.observability.tracing import setup_telemetry
from hivemind.services.agent_service import AgentService
from hivemind.services.artifact_store import ArtifactStore
from hivemind.services.conversation import ConversationService
from hivemind.services.scheduler import CleanupScheduler

logger = get_logger("hivemind.bootstrap")


@dataclass
class AppContext:
    settings: Settings
    db: Database
    tools: ToolRegistry
    skills: SkillRegistry
    agents: AgentRegistry
    agent_service: AgentService
    deps: GraphDeps
    runner: GraphRunner
    conversations: ConversationService
    artifacts: ArtifactStore
    scheduler: CleanupScheduler


def _build_registries(
    settings: Settings, db: Database
) -> tuple[ToolRegistry, SkillRegistry, AgentRegistry, GraphDeps, ArtifactStore]:
    artifacts = ArtifactStore(settings.artifact_base_path)
    sandbox = build_sandbox(settings)

    tools = ToolRegistry()
    tools.register(SQLTool(settings, artifacts))
    tools.register(CodeExecTool(sandbox, artifacts, timeout_s=settings.sandbox_timeout_s))
    tools.register(WebSearchTool())

    skills = SkillRegistry()
    skills.load_directory(settings.skills_dir)

    agents = AgentRegistry()
    agent_factory = AgentFactory(tools, skills)
    llm_factory = LLMProviderFactory(settings)
    deps = GraphDeps(
        settings=settings,
        agents=agents,
        agent_factory=agent_factory,
        llm_factory=llm_factory,
        tools=tools,
    )

    # The sub-agent spawner closes the loop: it needs deps, which needs the tool registry.
    subagent_runner = SubAgentRunnerImpl(deps, db)
    tools.register(SpawnSubAgentTool(subagent_runner))
    return tools, skills, agents, deps, artifacts


@contextlib.asynccontextmanager
async def build_context(
    settings: Settings, *, use_memory_checkpointer: bool = False
) -> AsyncIterator[AppContext]:
    """Build and tear down the full application context."""
    configure_logging(settings.log_level, json_logs=settings.environment != "local")
    setup_telemetry(settings)

    db = Database.create(settings.database_url)
    tools, skills, agents, deps, artifacts = _build_registries(settings, db)

    agent_service = AgentService(db, agents)
    await agent_service.hydrate()
    await agent_service.ensure_exists(sql_specialist_agent(settings))

    checkpoint_cm = (
        _null_checkpointer()
        if use_memory_checkpointer
        else open_checkpointer(settings.database_url)
    )
    async with checkpoint_cm as checkpointer:
        runner = GraphRunner(deps, checkpointer)
        conversations = ConversationService(
            db,
            runner,
            ttl_seconds=settings.ephemeral_agent_ttl_seconds,
            deps=deps,
            history_limit=settings.conversation_history_limit,
            compaction_enabled=settings.conversation_compaction_enabled,
        )
        scheduler = CleanupScheduler(
            db,
            interval_seconds=settings.cleanup_interval_seconds,
            artifacts=artifacts,
        )
        ctx = AppContext(
            settings=settings,
            db=db,
            tools=tools,
            skills=skills,
            agents=agents,
            agent_service=agent_service,
            deps=deps,
            runner=runner,
            conversations=conversations,
            artifacts=artifacts,
            scheduler=scheduler,
        )
        try:
            logger.info("bootstrap.ready", agents=len(agents.list()), tools=len(tools.names()))
            yield ctx
        finally:
            await scheduler.stop()
            await db.dispose()


@contextlib.asynccontextmanager
async def _null_checkpointer() -> AsyncIterator[object]:
    yield memory_checkpointer()
