"""Agent factory — composes effective system prompts and resolves capabilities.

The factory turns a stored/registered :class:`Agent` into its runtime form: the effective
system prompt (persona + injected skills) and the resolved LLM tool schemas for its bound
tools. It depends on the skill and tool registries via injection.
"""

from __future__ import annotations

from dataclasses import dataclass

from hivemind.core.agents.agent import Agent
from hivemind.core.llm.base import LLMConfig, ToolSchema
from hivemind.core.skills.injection import build_skill_prompt
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class PreparedAgent:
    """An agent resolved for execution."""

    agent: Agent
    effective_system_prompt: str
    tool_schemas: tuple[ToolSchema, ...]

    @property
    def llm_config(self) -> LLMConfig:
        return self.agent.llm_config


class AgentFactory:
    def __init__(self, tools: ToolRegistry, skills: SkillRegistry) -> None:
        self._tools = tools
        self._skills = skills

    def prepare(self, agent: Agent) -> PreparedAgent:
        """Build the runtime form of ``agent``."""
        prompt = agent.system_prompt
        skill_section = build_skill_prompt(self._skills, list(agent.skill_names))
        if skill_section:
            prompt = f"{prompt}\n\n{skill_section}"
        schemas = tuple(self._tools.schemas_for(list(agent.tool_names)))
        return PreparedAgent(agent=agent, effective_system_prompt=prompt, tool_schemas=schemas)

    def build(
        self,
        *,
        name: str,
        system_prompt: str,
        llm_config: LLMConfig,
        description: str = "",
        tool_names: list[str] | None = None,
        skill_names: list[str] | None = None,
    ) -> Agent:
        """Construct a new immutable agent, validating bound tools/skills exist."""
        tool_names = tool_names or []
        skill_names = skill_names or []
        unknown_tools = [t for t in tool_names if not self._tools.has(t)]
        if unknown_tools:
            from hivemind.core.errors import ValidationError

            raise ValidationError(f"Unknown tools: {unknown_tools}")
        unknown_skills = [s for s in skill_names if self._skills.get(s) is None]
        if unknown_skills:
            from hivemind.core.errors import ValidationError

            raise ValidationError(f"Unknown skills: {unknown_skills}")
        return Agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            llm_config=llm_config,
            tool_names=tuple(tool_names),
            skill_names=tuple(skill_names),
        )
