"""Artifact storage on a mounted persistent volume.

Tool outputs (files, CSVs, data) are written under a configurable base path, namespaced
by ``conversation_id/task_id/tool_name/`` to prevent collisions. Tools return a structured
``artifact_ref`` rather than raw bytes, so large data never re-enters prompts. Path
traversal is blocked by jailing every resolved path under the namespace directory.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    size_bytes: int
    mime_type: str
    # An owner-authenticated download URL. Present only when a public base URL is configured.
    # The link resolves to ``GET /v1/artifacts/{id}``, which requires the owner's bearer token
    # (the id encodes the relative path; the endpoint derives ownership from it). The model
    # sees this in the tool result and includes it in its answer when relevant.
    download_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": "artifact_ref",
            "path": self.path,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
        }
        if self.download_url:
            d["download_url"] = self.download_url
        return d


class ArtifactStore:
    def __init__(self, base_path: str, *, public_base_url: str = "") -> None:
        self._base = Path(base_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)
        self._public_base_url = public_base_url.rstrip("/")

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
            download_url=self._download_url(str(path)),
        )

    def _download_url(self, path: str) -> str | None:
        if not self._public_base_url:
            return None
        return f"{self._public_base_url}/v1/artifacts/{self._encode_id(path)}"

    def _encode_id(self, path: str) -> str:
        """Encode an artifact's base-relative path as a URL-safe id (not a credential).

        The id only *names* the artifact; access is authorized by the download endpoint,
        which requires the owner's bearer token. The path is stored relative to the base so a
        forged id can at most point within the artifact tree (and is jail-checked on resolve).
        """
        rel = self.resolve(path).relative_to(self._base).as_posix()
        return base64.urlsafe_b64encode(rel.encode()).decode().rstrip("=")

    def read_ref(self, ref_path: str) -> bytes:
        """Read an artifact referenced by path, with jail enforcement."""
        return self.resolve(ref_path).read_bytes()

    def resolve(self, ref_path: str) -> Path:
        """Resolve a path to a real file under the artifact base, or raise (jail enforced)."""
        target = Path(ref_path).resolve()
        self._assert_within(target, self._base)
        return target

    def resolve_id(self, artifact_id: str) -> Path:
        """Decode a download id to its jailed file path. Raises ValueError if malformed."""
        try:
            padded = artifact_id + "=" * (-len(artifact_id) % 4)
            rel = base64.urlsafe_b64decode(padded.encode()).decode()
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError(f"Malformed artifact id: {artifact_id}") from exc
        return self.resolve(str(self._base / rel))

    def owner_conversation_id(self, path: Path) -> str | None:
        """The conversation id that owns an artifact, derived from its namespace directory.

        Layout is ``base/{conversation}/{task}/{tool}/file``; the first path component under
        the base is the (sanitized) conversation id. Used by the download endpoint to check
        the caller owns the artifact. Returns None if the path isn't under a conversation dir.
        """
        try:
            parts = self.resolve(str(path)).relative_to(self._base).parts
        except ValueError:
            return None
        return parts[0] if parts else None

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
