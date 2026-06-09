"""In-memory skill registry, hydrated from disk and/or the database."""

from __future__ import annotations

from pathlib import Path

from hivemind.core.skills.skill import Skill
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.skills")


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def load_directory(self, directory: str | Path) -> int:
        """Load all ``*.md`` skills from a directory. Returns the count loaded."""
        path = Path(directory)
        if not path.exists():
            return 0
        count = 0
        for md in sorted(path.glob("*.md")):
            try:
                skill = Skill.from_file(md)
                self.register(skill)
                count += 1
            except Exception as exc:
                logger.warning("skill.load_failed", file=str(md), error=str(exc))
        logger.info("skills.loaded", directory=str(path), count=count)
        return count
