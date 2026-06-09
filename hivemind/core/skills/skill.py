"""Skill model + markdown loader.

A skill is a markdown file with YAML frontmatter:

    ---
    name: postgres-optimization
    description: When and how to use EXPLAIN/ANALYZE and index strategies.
    version: 1
    ---
    <full instructional body...>

Skills are bound to agents at creation and injected into the system prompt at agent-load
time (see :mod:`hivemind.core.skills.injection`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import frontmatter


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    body: str
    version: int = 1

    @classmethod
    def from_markdown(cls, raw: str) -> Skill:
        post = frontmatter.loads(raw)
        meta = post.metadata
        name = meta.get("name")
        description = meta.get("description")
        if not name or not description:
            raise ValueError("Skill frontmatter must include 'name' and 'description'.")
        return cls(
            name=str(name),
            description=str(description),
            body=post.content.strip(),
            version=int(meta.get("version", 1)),
        )

    @classmethod
    def from_file(cls, path: Path) -> Skill:
        return cls.from_markdown(path.read_text(encoding="utf-8"))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "body": self.body,
            "version": self.version,
        }
