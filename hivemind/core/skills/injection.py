"""Skill → system-prompt injection with progressive disclosure.

To keep token usage bounded, we always inject a compact index of each bound skill's
``name`` + ``description``, then append the full body of the bound skills. For very large
skill sets, callers can cap how many full bodies are inlined; the rest remain discoverable
via their index entry (and could be fetched on demand by a ``read_skill`` capability).
"""

from __future__ import annotations

from hivemind.core.skills.registry import SkillRegistry

_HEADER = "## Available Skills\nYou have been equipped with the following skills:\n"


def build_skill_prompt(
    registry: SkillRegistry, skill_names: list[str], *, max_full_bodies: int = 20
) -> str:
    """Return the skills section to append to an agent's system prompt.

    Args:
        registry: source of skill definitions.
        skill_names: skills bound to the agent.
        max_full_bodies: how many full skill bodies to inline (rest are index-only).
    """
    skills = [s for name in skill_names if (s := registry.get(name)) is not None]
    if not skills:
        return ""

    lines = [_HEADER]
    for skill in skills:
        lines.append(f"- **{skill.name}**: {skill.description}")
    lines.append("")

    for skill in skills[:max_full_bodies]:
        lines.append(f"### Skill: {skill.name}\n{skill.body}\n")

    return "\n".join(lines).strip()
