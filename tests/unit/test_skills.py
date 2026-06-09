from __future__ import annotations

import pytest
from hivemind.core.skills.injection import build_skill_prompt
from hivemind.core.skills.registry import SkillRegistry
from hivemind.core.skills.skill import Skill

_MD = """\
---
name: demo-skill
description: A demo skill.
version: 2
---
Body line one.
Body line two.
"""


def test_skill_from_markdown():
    skill = Skill.from_markdown(_MD)
    assert skill.name == "demo-skill"
    assert skill.description == "A demo skill."
    assert skill.version == 2
    assert "Body line one." in skill.body


def test_skill_missing_frontmatter_raises():
    with pytest.raises(ValueError):
        Skill.from_markdown("no frontmatter here")


def test_registry_load_directory(tmp_path):
    (tmp_path / "s.md").write_text(_MD)
    reg = SkillRegistry()
    assert reg.load_directory(tmp_path) == 1
    assert reg.get("demo-skill") is not None


def test_injection_includes_index_and_body():
    reg = SkillRegistry()
    reg.register(Skill.from_markdown(_MD))
    prompt = build_skill_prompt(reg, ["demo-skill"])
    assert "**demo-skill**: A demo skill." in prompt
    assert "Body line one." in prompt


def test_injection_empty_for_no_skills():
    reg = SkillRegistry()
    assert build_skill_prompt(reg, []) == ""
    assert build_skill_prompt(reg, ["missing"]) == ""
