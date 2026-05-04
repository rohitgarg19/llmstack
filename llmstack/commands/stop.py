"""``llmstack stop`` -- tear down the singleton router + llama-swap daemons.

Three layers, in order:

  1. SIGTERM/SIGKILL the pids in ``<state>/router.pid`` and
     ``<state>/llama-swap.pid`` (if any).
  2. ``pkill`` by pattern as a cross-project safety net for daemons that
     were started from another project's ``.llmstack/``.
  3. ``pkill`` any orphaned ``llama-server`` children spawned by
     llama-swap.

In **remote mode** (``$LLMSTACK_REMOTE_URL`` set) there are no local
daemons to tear down -- we just clear the active-channel marker so
``status`` no longer reports the connection.
"""

from __future__ import annotations

from llmstack.commands._helpers import (
    kill_pid,
    pgrep_describe,
    pkill,
    read_pid,
)
from llmstack.paths import is_remote, remote_url, resolve


def _print_help() -> None:
    print("usage: llmstack stop")


def run(args: list[str]) -> int:
    for arg in args:
        if arg in ("-h", "--help"):
            _print_help()
            return 0
        print(f"[!] unknown arg to stop: {arg}")
        return 2

    paths = resolve()

    if is_remote():
        url = remote_url()
        if paths.active_marker.is_file():
            paths.active_marker.unlink(missing_ok=True)
            print(f"[OK] disconnected from {url} (active-channel cleared).")
        else:
            print(f"[=] not connected to any remote llmstack. ($LLMSTACK_REMOTE_URL={url})")
        print("    note: nothing local was running. To stop the *remote* daemons, run")
        print("          'llmstack stop' on the host that started them.")
        return 0

    for name, pid_file in (("router", paths.router_pid), ("llama-swap", paths.swap_pid)):
        pid = read_pid(pid_file)
        if pid is not None:
            print(f"[*] stopping {name} (pid {pid})")
            kill_pid(pid)
        pid_file.unlink(missing_ok=True)

    cross_project = pgrep_describe(r"llama-swap --config|llmstack\.app")
    if cross_project.strip():
        print("[*] stopping daemons by name (no local pid files, started elsewhere):")
        for line in cross_project.splitlines():
            print(f"    {line}")
        pkill(r"llama-swap --config")
        pkill(r"llmstack\.app")

    # Orphaned llama-server children (shouldn't happen, but cheap insurance)
    pkill(r"llama-server.*--alias (code-fast|code-smart|plan|plan-uncensored)")

    paths.active_marker.unlink(missing_ok=True)
    print("[OK] stopped.")
    return 0
