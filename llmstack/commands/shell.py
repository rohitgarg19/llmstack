"""``llmstack shell`` -- drop into the env-prepared subshell.

In local mode, refuses to spawn unless the daemons are running so the
user gets a clear error instead of a half-broken shell with
``OPENCODE_CONFIG`` set but nothing answering on :10101.

In remote mode (``$LLMSTACK_REMOTE_URL`` set), probes the remote
``/health`` endpoint instead and labels the channel as ``external``.
"""

from __future__ import annotations

import sys

from llmstack.commands._helpers import is_running, port_responds
from llmstack.paths import (
    SWAP_PORT,
    ensure_state_dirs,
    is_remote,
    read_marker,
    remote_url,
)
from llmstack.shell_env import spawn_subshell


def _print_help() -> None:
    print("usage: llmstack shell")


def run(args: list[str]) -> int:
    for arg in args:
        if arg in ("-h", "--help"):
            _print_help()
            return 0
        print(f"[!] unknown arg to shell: {arg}")
        return 2

    paths = ensure_state_dirs()

    if is_remote():
        url = remote_url()
        assert url is not None
        if not port_responds(f"{url}/health", timeout=5.0):
            print(f"[!] remote {url}/health did not respond.", file=sys.stderr)
            print("    proceeding anyway; opencode will surface the error on first request.", file=sys.stderr)
        spawn_subshell("external")
        return 0  # unreachable

    if not is_running(paths.swap_pid) and not port_responds(f"http://127.0.0.1:{SWAP_PORT}/health"):
        raise SystemExit("stack is not running. start it first: llmstack start")

    mark = read_marker(paths.active_marker)
    channel = mark.channel if mark else "shared"
    spawn_subshell(channel)
    return 0  # unreachable
