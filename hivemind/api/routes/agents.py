"""Agent management endpoints. Agents are immutable after creation."""

from __future__ import annotations

from fastapi import APIRouter

from hivemind.api.deps import AdminUser, AppCtx, CurrentUser
from hivemind.api.schemas.agents import AgentResponse, CreateAgentRequest
from hivemind.api.schemas.common import LLMConfigSchema
from hivemind.core.agents.agent import Agent
from hivemind.core.llm.base import LLMConfig

router = APIRouter(prefix="/v1/agents", tags=["agents"])


def _to_response(agent: Agent) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        system_prompt=agent.system_prompt,
        tool_names=list(agent.tool_names),
        skill_names=list(agent.skill_names),
        llm_config=LLMConfigSchema(**agent.llm_config.to_dict()),
        version=agent.version,
        immutable=agent.immutable,
    )


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent(body: CreateAgentRequest, app: AppCtx, user: AdminUser):
    agent = app.deps.agent_factory.build(
        name=body.name,
        system_prompt=body.system_prompt,
        description=body.description,
        llm_config=LLMConfig.from_dict(body.llm_config.model_dump()),
        tool_names=body.tool_names,
        skill_names=body.skill_names,
    )
    created = await app.agent_service.create(agent)
    return _to_response(created)


@router.get("", response_model=list[AgentResponse])
async def list_agents(app: AppCtx, user: CurrentUser):
    return [_to_response(a) for a in app.agents.list()]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, app: AppCtx, user: CurrentUser):
    return _to_response(app.agents.get(agent_id))


@router.delete("/{agent_id}", status_code=204)
async def decommission_agent(agent_id: str, app: AppCtx, user: AdminUser):
    await app.agent_service.decommission(agent_id)
