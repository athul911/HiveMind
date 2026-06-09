from __future__ import annotations

import pytest
from hivemind.core.agents.factory import AgentFactory
from hivemind.core.errors import ValidationError
from hivemind.core.llm.base import LLMConfig
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.skills.skill import Skill
from hivemind.core.tools.base import BaseTool, ToolResult
from hivemind.core.tools.registry import ToolRegistry


class NoopTool(BaseTool):
    name = "noop"
    description = "noop"
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, args, ctx):
        return ToolResult(content={})


@pytest.fixture
def factory() -> AgentFactory:
    tools = ToolRegistry()
    tools.register(NoopTool())
    skills = SkillRegistry()
    skills.register(Skill(name="sk", description="d", body="full body text"))
    return AgentFactory(tools, skills)


def _cfg() -> LLMConfig:
    return LLMConfig(provider="anthropic", model="claude-opus-4-8")


def test_build_validates_tools(factory: AgentFactory):
    with pytest.raises(ValidationError):
        factory.build(name="a", system_prompt="p", llm_config=_cfg(), tool_names=["ghost"])


def test_build_validates_skills(factory: AgentFactory):
    with pytest.raises(ValidationError):
        factory.build(name="a", system_prompt="p", llm_config=_cfg(), skill_names=["ghost"])


def test_prepare_injects_skills_and_resolves_tools(factory: AgentFactory):
    agent = factory.build(
        name="a",
        system_prompt="You are A.",
        llm_config=_cfg(),
        tool_names=["noop"],
        skill_names=["sk"],
    )
    prepared = factory.prepare(agent)
    assert "You are A." in prepared.effective_system_prompt
    assert "full body text" in prepared.effective_system_prompt
    assert [s.name for s in prepared.tool_schemas] == ["noop"]
