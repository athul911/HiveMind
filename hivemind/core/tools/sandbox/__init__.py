"""Sandbox backends + factory."""

from __future__ import annotations

from hivemind.config import Settings
from hivemind.core.tools.sandbox.base import Sandbox, SandboxResult


def build_sandbox(settings: Settings) -> Sandbox:
    """Construct the configured sandbox backend."""
    if settings.sandbox_backend == "docker":
        from hivemind.core.tools.sandbox.docker_sandbox import DockerSandbox

        return DockerSandbox(
            image=settings.sandbox_image,
            memory=settings.sandbox_memory,
            cpus=settings.sandbox_cpus,
            pids_limit=settings.sandbox_pids_limit,
        )
    from hivemind.core.tools.sandbox.subprocess_sandbox import SubprocessSandbox

    return SubprocessSandbox()


__all__ = ["Sandbox", "SandboxResult", "build_sandbox"]
