"""Sandbox abstraction for executing untrusted, LLM-generated code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@runtime_checkable
class Sandbox(Protocol):
    """Runs ``code`` with ``artifact_dir`` available for output files."""

    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult: ...
