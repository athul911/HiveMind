"""Immutable agent value object.

An agent is composed of a system prompt, a subset of tool names, a subset of skill names,
and an LLM configuration. Agents are immutable: any change is a new version. They are
serializable to/from the ``agents`` table.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from hivemind.core.llm.base import LLMConfig


@dataclass(frozen=True, slots=True)
class Agent:
    name: str
    system_prompt: str
    llm_config: LLMConfig
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    tool_names: tuple[str, ...] = ()
    skill_names: tuple[str, ...] = ()
    version: int = 1
    immutable: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tool_names": list(self.tool_names),
            "skill_names": list(self.skill_names),
            "llm_config": self.llm_config.to_dict(),
            "version": self.version,
            "immutable": self.immutable,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Agent:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            system_prompt=data["system_prompt"],
            tool_names=tuple(data.get("tool_names", [])),
            skill_names=tuple(data.get("skill_names", [])),
            llm_config=LLMConfig.from_dict(data["llm_config"]),
            version=data.get("version", 1),
            immutable=data.get("immutable", True),
        )
