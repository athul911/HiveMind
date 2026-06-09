"""Agent / tool / skill API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hivemind.api.schemas.common import LLMConfigSchema


class CreateAgentRequest(BaseModel):
    name: str
    system_prompt: str
    llm_config: LLMConfigSchema
    description: str = ""
    tool_names: list[str] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    tool_names: list[str]
    skill_names: list[str]
    llm_config: LLMConfigSchema
    version: int
    immutable: bool


class ToolResponse(BaseModel):
    name: str
    description: str
    input_schema: dict


class SkillResponse(BaseModel):
    name: str
    description: str
    version: int
