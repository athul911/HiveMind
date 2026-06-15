"""SubprocessSandbox command construction: plain vs bubblewrap, and graceful fallback.

The bwrap capability probe is bypassed by setting the cached ``_bwrap_ok`` flag, so these
tests are deterministic and don't require bubblewrap/userns (or even Linux).
"""

from __future__ import annotations

import sys

from hivemind.core.tools.sandbox.subprocess_sandbox import SubprocessSandbox


async def test_plain_argv_when_isolation_none(tmp_path):
    sb = SubprocessSandbox(isolation="none")
    argv = await sb._argv(tmp_path / "_code.py", tmp_path)
    assert argv == [sys.executable, "-I", str(tmp_path / "_code.py")]


async def test_bwrap_argv_when_namespaces_available(tmp_path):
    sb = SubprocessSandbox(isolation="namespaces")
    sb._bwrap_ok = True  # pretend the probe passed
    argv = await sb._argv(tmp_path / "_code.py", tmp_path)

    assert argv[0] == "bwrap"
    assert "--unshare-all" in argv  # all namespaces incl. network
    assert "--share-net" not in argv  # => no network egress
    # The artifact dir is the only writable bind, mounted at the guest path.
    bind = argv.index("--bind")
    assert argv[bind + 1 : bind + 3] == [str(tmp_path), "/artifacts"]
    assert "--chdir" in argv and argv[argv.index("--chdir") + 1] == "/artifacts"
    # Read-only runtime exposed (e.g. /usr), never read-write.
    assert "--ro-bind-try" in argv and "/usr" in argv
    # Ends by running the interpreter on the in-sandbox code path.
    assert argv[-3:] == [sys.executable, "-I", "/artifacts/_code.py"]


async def test_falls_back_to_plain_when_bwrap_unavailable(tmp_path):
    sb = SubprocessSandbox(isolation="namespaces")
    sb._bwrap_ok = False  # probe failed (no bwrap / userns)
    argv = await sb._argv(tmp_path / "_code.py", tmp_path)
    assert argv == [sys.executable, "-I", str(tmp_path / "_code.py")]
