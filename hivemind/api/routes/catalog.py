"""Tool and skill catalog endpoints (read-only)."""

from __future__ import annotations

from fastapi import APIRouter

from hivemind.api.deps import AppCtx, CurrentUser
from hivemind.api.schemas.agents import SkillResponse, ToolResponse
from hivemind.core.errors import NotFoundError

router = APIRouter(tags=["catalog"])


@router.get("/v1/tools", response_model=list[ToolResponse])
async def list_tools(app: AppCtx, user: CurrentUser):
    return [
        ToolResponse(name=t.name, description=t.description, input_schema=t.input_schema)
        for t in app.tools.list()
    ]


@router.get("/v1/tools/{name}", response_model=ToolResponse)
async def get_tool(name: str, app: AppCtx, user: CurrentUser):
    if not app.tools.has(name):
        raise NotFoundError(f"Tool not found: {name}")
    tool = app.tools.get(name)
    return ToolResponse(
        name=tool.name, description=tool.description, input_schema=tool.input_schema
    )


@router.get("/v1/skills", response_model=list[SkillResponse])
async def list_skills(app: AppCtx, user: CurrentUser):
    return [
        SkillResponse(name=s.name, description=s.description, version=s.version)
        for s in app.skills.list()
    ]


@router.get("/v1/skills/{name}", response_model=SkillResponse)
async def get_skill(name: str, app: AppCtx, user: CurrentUser):
    skill = app.skills.get(name)
    if skill is None:
        raise NotFoundError(f"Skill not found: {name}")
    return SkillResponse(name=skill.name, description=skill.description, version=skill.version)
