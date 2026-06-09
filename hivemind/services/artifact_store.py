"""Artifact storage on a mounted persistent volume.

Tool outputs (files, CSVs, data) are written under a configurable base path, namespaced
by ``conversation_id/task_id/tool_name/`` to prevent collisions. Tools return a structured
``artifact_ref`` rather than raw bytes, so large data never re-enters prompts. Path
traversal is blocked by jailing every resolved path under the namespace directory.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    size_bytes: int
    mime_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "artifact_ref",
            "path": self.path,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
        }


class ArtifactStore:
    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    def namespace_dir(self, conversation_id: str, task_id: str | None, tool_name: str) -> Path:
        """Return (creating if needed) the namespaced directory for a tool's outputs."""
        ns = self._base / _safe(conversation_id) / _safe(task_id or "sync") / _safe(tool_name)
        ns.mkdir(parents=True, exist_ok=True)
        return ns

    def delete_conversation(self, conversation_id: str) -> bool:
        """Remove all artifacts for a conversation (called by the cleanup scheduler)."""
        import shutil

        target = (self._base / _safe(conversation_id)).resolve()
        self._assert_within(target, self._base)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            return True
        return False

    def write_bytes(
        self,
        conversation_id: str,
        task_id: str | None,
        tool_name: str,
        filename: str,
        data: bytes,
    ) -> ArtifactRef:
        ns = self.namespace_dir(conversation_id, task_id, tool_name)
        target = (ns / _safe_filename(filename)).resolve()
        self._assert_within(target, ns)
        target.write_bytes(data)
        return self.describe(target)

    def write_text(
        self,
        conversation_id: str,
        task_id: str | None,
        tool_name: str,
        filename: str,
        text: str,
    ) -> ArtifactRef:
        return self.write_bytes(conversation_id, task_id, tool_name, filename, text.encode("utf-8"))

    def describe(self, path: Path) -> ArtifactRef:
        mime, _ = mimetypes.guess_type(str(path))
        return ArtifactRef(
            path=str(path),
            size_bytes=path.stat().st_size if path.exists() else 0,
            mime_type=mime or "application/octet-stream",
        )

    def read_ref(self, ref_path: str) -> bytes:
        """Read an artifact referenced by path, with jail enforcement."""
        target = Path(ref_path).resolve()
        self._assert_within(target, self._base)
        return target.read_bytes()

    def _assert_within(self, target: Path, root: Path) -> None:
        if not str(target).startswith(str(root.resolve())):
            raise ValueError(f"Path traversal blocked: {target} escapes {root}")


def _safe(component: str) -> str:
    return "".join(c for c in component if c.isalnum() or c in "-_.") or "default"


def _safe_filename(filename: str) -> str:
    # Reject any path components outright rather than silently sanitizing — a filename
    # containing a separator or '..' is a traversal attempt and should fail loudly.
    if "/" in filename or "\\" in filename or filename in (".", "..") or not filename:
        raise ValueError(f"Invalid artifact filename: {filename!r}")
    return Path(filename).name
