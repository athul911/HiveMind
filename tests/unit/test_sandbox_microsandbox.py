"""microsandbox backend — verified with a fake Sandbox SDK (no real microVM/server)."""

from __future__ import annotations

from hivemind.core.tools.sandbox.microsandbox_sandbox import MicrosandboxSandbox


class _ExecOut:
    def __init__(self, stdout_text, stderr_text, exit_code):
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        self.exit_code = exit_code


def _fake_sandbox_cls(record: dict, *, hang: bool = False):
    class _Sandbox:
        @classmethod
        async def create(cls, name, **kwargs):
            record["create"] = (name, kwargs)
            return cls()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def exec(self, cmd, args=None, *, cwd=None, timeout=None):  # noqa: ASYNC109
            record.setdefault("exec", []).append((cmd, args, cwd, timeout))
            if hang:
                raise TimeoutError  # simulate the exec exceeding its deadline
            return _ExecOut("hello\n", "", 0)

    return _Sandbox


def _mount(host_path):  # stand-in for microsandbox.types MountConfig(kind=BIND, bind=...)
    return {"bind": host_path}


async def test_microsandbox_runs_and_bind_mounts_artifacts(tmp_path):
    record: dict = {}
    sb = MicrosandboxSandbox(
        image="python:3.11-slim",
        memory_mib=256,
        cpus=2,
        _sandbox_cls=_fake_sandbox_cls(record),
        _mount=_mount,
    )
    res = await sb.run("print('hello')", artifact_dir=tmp_path, timeout_s=5)

    assert res.exit_code == 0 and res.stdout == "hello\n" and not res.timed_out
    # Code is written to the host dir (visible in the guest via the bind mount).
    assert (tmp_path / "_code.py").read_text() == "print('hello')"

    _name, kwargs = record["create"]
    assert kwargs["image"] == "python:3.11-slim"
    assert kwargs["memory"] == 256 and kwargs["cpus"] == 2
    assert kwargs["workdir"] == "/artifacts"
    # Host artifact_dir bind-mounted to the guest's /artifacts.
    assert kwargs["volumes"] == {"/artifacts": {"bind": str(tmp_path)}}

    cmd, args, cwd, _timeout = record["exec"][0]
    assert cmd == "python" and args == ["/artifacts/_code.py"] and cwd == "/artifacts"


async def test_microsandbox_timeout(tmp_path):
    record: dict = {}
    sb = MicrosandboxSandbox(
        _sandbox_cls=_fake_sandbox_cls(record, hang=True), _mount=_mount
    )
    res = await sb.run("while True: pass", artifact_dir=tmp_path, timeout_s=1)
    assert res.timed_out and res.exit_code == 124
