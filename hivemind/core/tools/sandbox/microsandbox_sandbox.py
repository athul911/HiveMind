"""microsandbox-backed sandbox — one ephemeral microVM per execution.

Uses the microsandbox ``Sandbox`` SDK (microVMs via libkrun, stronger isolation than a shared
kernel). The host ``artifact_dir`` is **bind-mounted** into the guest at ``/artifacts`` (just
like the Docker backend's volume), so code writes output files there and they land back on the
host — no copy-out needed. A wall-clock timeout kills the exec.

Deployment notes (inherent to microsandbox, not configurable away):
  * Requires ``microsandbox>=0.5`` (the maturin/PyO3 build that exports ``Sandbox``). It's on
    PyPI but ships **Linux-only wheels** (libkrun/KVM), so install it in the Linux worker
    image — not on a macOS dev host, where pip resolves only the older 0.1.x pure-python line
    that exposes ``PythonSandbox`` instead. The import is lazy, so this module is only needed
    when the ``microsandbox`` backend is actually selected.
  * Host bind mounts require the microsandbox **server to be co-located** with the worker
    (shared filesystem), and a server reachable per the SDK's own connection config.

The exact ``ExecOutput`` field names and the ``volumes`` dict keying are taken from the SDK
docs; if a future SDK revision changes them, the two marked spots below are where to adjust.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from hivemind.core.errors import SandboxError
from hivemind.core.tools.sandbox.base import SandboxResult
from hivemind.observability.logging import get_logger

logger = get_logger("hivemind.sandbox.microsandbox")

_GUEST_DIR = "/artifacts"


class MicrosandboxSandbox:
    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        memory_mib: int = 512,
        cpus: float = 1.0,
        _sandbox_cls=None,  # injectable for tests; lazily imported from microsandbox otherwise
        _mount=None,  # injectable (host_path -> mount config); lazily built otherwise
    ) -> None:
        self._image = image
        self._memory_mib = memory_mib
        self._cpus = max(1, round(cpus))
        self._sandbox_cls = _sandbox_cls
        self._mount = _mount

    def _load(self):
        """Resolve the SDK ``Sandbox`` class + a host-bind-mount factory (lazy import)."""
        if self._sandbox_cls is not None and self._mount is not None:
            return self._sandbox_cls, self._mount
        try:
            from microsandbox import Sandbox
            from microsandbox.types import MountConfig, MountKind
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK installed
            raise SandboxError(
                "microsandbox backend selected but the 'microsandbox' SDK (the maturin build "
                "exporting `Sandbox`) is not installed."
            ) from exc

        def mount(host_path: str):
            return MountConfig(kind=MountKind.BIND, bind=host_path, readonly=False)

        return (self._sandbox_cls or Sandbox), (self._mount or mount)

    async def run(self, code: str, *, artifact_dir: Path, timeout_s: int) -> SandboxResult:
        sandbox_cls, mount = self._load()
        # Tiny one-time host FS prep before booting the VM; sync Path is fine (cf. subprocess).
        artifact_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        # Written on the host; visible in the guest via the bind mount. _collect_artifacts in
        # the code tool skips '_code.py', so it isn't surfaced as a generated artifact.
        (artifact_dir / "_code.py").write_text(code, encoding="utf-8")

        name = f"hivemind-{_token()}"
        # volumes dict is keyed by the GUEST mount path -> source mount config.  [adjust here]
        volumes = {_GUEST_DIR: mount(str(artifact_dir))}
        try:
            sandbox = await sandbox_cls.create(
                name,
                image=self._image,
                cpus=self._cpus,
                memory=self._memory_mib,
                workdir=_GUEST_DIR,
                volumes=volumes,
            )
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"Could not start microsandbox VM: {exc}") from exc

        try:
            async with sandbox as sb:
                return await self._exec(sb, timeout_s)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"microsandbox execution failed: {exc}") from exc

    async def _exec(self, sb, timeout_s: int) -> SandboxResult:
        try:
            # Hard wall-clock backstop around the SDK's own per-exec timeout.
            out = await asyncio.wait_for(
                sb.exec(
                    "python",
                    [f"{_GUEST_DIR}/_code.py"],
                    cwd=_GUEST_DIR,
                    timeout=float(timeout_s),
                ),
                timeout=timeout_s + 10,
            )
        except TimeoutError:
            return SandboxResult(exit_code=124, stdout="", stderr="timed out", timed_out=True)

        # ExecOutput: stdout_text / stderr_text / exit_code.  [adjust here if the SDK changes]
        exit_code = int(getattr(out, "exit_code", 0) or 0)
        return SandboxResult(
            exit_code=exit_code,
            stdout=getattr(out, "stdout_text", "") or "",
            stderr=getattr(out, "stderr_text", "") or "",
            timed_out=exit_code == 124,
        )


def _token() -> str:
    import uuid

    return uuid.uuid4().hex[:12]
