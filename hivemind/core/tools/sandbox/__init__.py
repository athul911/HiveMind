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
    if settings.sandbox_backend == "microsandbox":
        from hivemind.core.tools.sandbox.microsandbox_sandbox import MicrosandboxSandbox

        return MicrosandboxSandbox(
            image=settings.sandbox_image,
            memory_mib=settings.microsandbox_memory_mib,
            cpus=settings.sandbox_cpus,
        )
    from hivemind.core.tools.sandbox.subprocess_sandbox import SubprocessSandbox

    return SubprocessSandbox()


__all__ = ["Sandbox", "SandboxResult", "build_sandbox"]
