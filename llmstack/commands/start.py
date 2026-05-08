"""``llmstack start`` -- bring up the stack and enter the env-prepared subshell.

The channel is **decided at install time** and persisted to
``.llmstack/default-channel`` -- ``start`` reads that marker and never
re-derives the channel from the environment. Three channels exist:

  *Local* (``current`` / ``next``)
    Generate ``llama-swap.yaml`` for the chosen channel, launch
    llama-swap + the FastAPI router locally, and drop into a subshell
    with ``OPENCODE_CONFIG`` exported. The yaml is regenerated on every
    fresh launch so it always reflects the live ``models.ini``; if the
    daemons are already up under our pid files we leave the loaded yaml
    alone. The ``--current`` / ``--next`` flags pick which of the two
    local channels to launch *for this run* (without rewriting the
    marker -- only ``install`` does that).

    Daemon state has two branches:
      (a) local pid file says daemons are up   -> idempotent, channel-checked,
                                                  no yaml regeneration
      (b) nothing in the pid file              -> regenerate yaml, launch
                                                  fresh. If port :10102 is
                                                  already in use by *another*
                                                  process (typically another
                                                  project on this host) we
                                                  refuse: the user should
                                                  ``llmstack install --external``
                                                  to wire this project as a
                                                  thin client of those
                                                  daemons, or stop them
                                                  first.

  *External* (``external``)
    Don't launch anything; verify the marker's ``/health`` endpoint is
    reachable and drop into the subshell with ``LLMSTACK_CHANNEL=external``.
    The URL was pinned by ``llmstack install --external [URL]``; an
    external install with no URL defaults to the local router
    (``http://127.0.0.1:10101``), which is the laptop-with-N-projects
    case where one project owns the daemons and the others are clients.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from llmstack.commands._helpers import (
    is_running,
    port_responds,
    read_pid,
    spawn_daemon,
)
from llmstack.generators import render_to
from llmstack.generators.llama_swap import render as render_yaml
from llmstack.generators.llama_swap import validate as validate_yaml
from llmstack.tiers import load_tiers
from llmstack.paths import (
    DEFAULT_REMOTE_URL,
    ROUTER_PORT,
    SWAP_PORT,
    ChannelMark,
    ensure_state_dirs,
    read_marker,
    write_marker,
)
from llmstack.shell_env import spawn_subshell
from llmstack.tiers import load_tiers


def _print_help() -> None:
    print("usage: llmstack start [--current | --next] [--detach]")


def _queued_next_tiers() -> list[str]:
    """Names of every tier that has *some* queued upgrade target.

    Backend-aware: gguf tiers with ``hf_file_next`` qualify, and so do
    bedrock tiers with ``aws_model_id_next``. Used to short-circuit
    ``--next`` when nothing's queued.
    """
    return sorted(t.name for t in load_tiers().values() if t.has_next)


def _start_remote(detach: bool, url: str) -> int:
    """Client-mode start: just verify the remote and drop into the shell.

    ``url`` is the remote-router base URL pinned by ``install --external``
    into ``default-channel``. It is *not* re-derived from the environment
    here -- the marker is canonical post-install, and silently following
    a stale env var would lie to the user about which remote opencode is
    actually wired to (the URL is baked into ``opencode.json`` at install
    time).

    The reachability probe hits ``GET /models.ini`` rather than a
    dedicated ``/health`` endpoint -- a 200 there proves both that the
    router is up and that it actually has a config worth talking to,
    which is what the thin client needs. The router has no separate
    ``/health`` route.
    """
    paths = ensure_state_dirs()

    if not paths.opencode_json.is_file():
        raise SystemExit(
            f"no .llmstack/opencode.json in {paths.work_dir} -- run: llmstack install --external\n"
            f"    (or `llmstack install --external {url}` to keep this remote URL)"
        )

    print(f"[*] external llmstack: {url}")
    probe_url = f"{url}/models.ini"
    if port_responds(probe_url, timeout=5.0):
        print(f"[OK] {probe_url} responds.")
    else:
        print(f"[!] {probe_url} did not respond -- is the remote stack up?", file=sys.stderr)
        print("    proceeding anyway; opencode will surface the error on first request.", file=sys.stderr)

    write_marker(paths.active_marker, ChannelMark("external", url))

    print()
    print("[OK] client mode (channel: external).")
    print()
    print(f"  router       {url}     (external)")
    print()
    print("Try:")
    print(f"  curl -s {url}/v1/models | jq '.data[].id'")
    print(f"  curl -s {url}/models.ini | head")
    print()
    print("Disconnect:")
    print("  exit                # leave the subshell (daemons are external, nothing to stop)")

    if detach:
        return 0

    # Same "spawn only when no active env" rule as the local-mode path.
    if os.environ.get("LLMSTACK_ACTIVE") == "1":
        cur_chan = os.environ.get("LLMSTACK_CHANNEL", "?")
        if cur_chan == "external":
            print("[=] already active as external client -- env is up to date.")
        else:
            print(
                f"[*] switching to external client ({cur_chan} -> external); env in "
                "this shell is now stale."
            )
            print("    refresh prompt + env in this shell:")
            print('        eval "$(llmstack reload)"')
        return 0

    spawn_subshell("external")
    return 0  # unreachable


def run(args: list[str]) -> int:
    requested: str | None = None
    detach = False
    for arg in args:
        if arg == "--next":
            requested = "next"
        elif arg == "--current":
            requested = "current"
        elif arg in ("--detach", "--no-shell"):
            detach = True
        elif arg in ("-h", "--help"):
            _print_help()
            return 0
        else:
            print(f"[!] unknown arg to start: {arg} (try --next, --current, --detach, -h)")
            return 2

    paths = ensure_state_dirs()
    default = read_marker(paths.default_marker)

    # External installs short-circuit to the thin-client path. The URL
    # is taken from the marker (set by ``install --external``); we
    # never re-derive it from the env.
    if default and default.channel == "external":
        if requested is not None:
            print(
                "[!] --current / --next have no effect for external installs "
                "(no daemons to launch).",
                file=sys.stderr,
            )
        url = (default.url or "").rstrip("/") or DEFAULT_REMOTE_URL
        return _start_remote(detach, url)

    # Local mode -- decide which of current/next to launch.
    if requested is not None:
        channel = requested
    elif default and default.channel in ("current", "next"):
        channel = default.channel
    else:
        channel = "current"

    if not paths.llama_swap_bin.exists() or not os.access(paths.llama_swap_bin, os.X_OK):
        raise SystemExit(f"missing {paths.llama_swap_bin} (run: llmstack setup)")
    if not paths.opencode_json.is_file():
        raise SystemExit(f"no .llmstack/opencode.json in {paths.work_dir} -- run: llmstack install")

    tiers = load_tiers()
    has_gguf = any(t.is_gguf for t in tiers.values())

    if has_gguf:
        if is_running(paths.swap_pid):
            launch_daemons = False
            live_mark = read_marker(paths.active_marker)
            live = live_mark.channel if live_mark else channel
            if live != channel:
                print(
                    f"[!] llama-swap is already running in '{live}' channel; "
                    f"refusing to also start '{channel}'. Stop the stack first:",
                    file=sys.stderr,
                )
                print("\n      llmstack stop", file=sys.stderr)
                print(f"      llmstack start --{channel}\n", file=sys.stderr)
                return 1
        elif port_responds(f"http://127.0.0.1:{SWAP_PORT}/health"):
            # Something is already listening on :10102, but it isn't ours
            # (no pid file in this project's state dir). The pre-flag flow
            # silently joined as "shared", which was a footgun: a `stop`
            # from this project would tear down the other project's
            # daemons and we couldn't bring them back without local
            # tooling. Instead, refuse and tell the user how to wire this
            # project as a proper thin client.
            print(
                f"[!] port :{SWAP_PORT} is already in use (daemons started by "
                "another project on this host).",
                file=sys.stderr,
            )
            print("    This project is installed for local mode -- it expects to own", file=sys.stderr)
            print("    those daemons. To run as a thin client of the running stack:", file=sys.stderr)
            print("", file=sys.stderr)
            print("        llmstack install --external", file=sys.stderr)
            print("", file=sys.stderr)
            print("    (--external defaults to http://127.0.0.1:10101, the local router.)", file=sys.stderr)
            print("    To take over instead, stop the running daemons first:", file=sys.stderr)
            print("", file=sys.stderr)
            print("        llmstack stop && llmstack start", file=sys.stderr)
            return 1
        else:
            launch_daemons = True
    else:
        launch_daemons = True

    if launch_daemons:
        if not has_gguf:
            print("[*] bedrock-only config -- skipping llama-swap")
        elif channel == "next":
            queued = _queued_next_tiers()
            if not queued:
                print(
                    "[!] no tiers have hf_file_next or aws_model_id_next set in models.ini -- "
                    "nothing to do.",
                    file=sys.stderr,
                )
                print(
                    "    add a *_next line to a tier and re-run, or use --current.",
                    file=sys.stderr,
                )
                return 1
            print(f"[*] generating next-channel yaml -> {paths.llama_swap_yaml}")
            print(f"    queued upgrade tiers: {' '.join(queued)}")
        else:
            print(f"[*] generating yaml -> {paths.llama_swap_yaml}")
        if has_gguf:
            render_to(
                paths.llama_swap_yaml,
                render=lambda p: Path(p).write_text(render_yaml(use_next=(channel == "next"))),
                validate=validate_yaml,
            )

    print(f"[*] channel: {channel}  ({paths.llama_swap_yaml.name})")

    if launch_daemons:
        if has_gguf:
            print(f"[*] starting llama-swap on :{SWAP_PORT}")
            spawn_daemon(
                [
                    str(paths.llama_swap_bin),
                    "--config", str(paths.llama_swap_yaml),
                    "--listen", f"127.0.0.1:{SWAP_PORT}",
                ],
                log=paths.log_dir / "llama-swap.log",
                pid_file=paths.swap_pid,
            )
            write_marker(paths.active_marker, ChannelMark(channel))
            time.sleep(1)
            if not is_running(paths.swap_pid):
                print(f"[!] llama-swap failed to start. Check {paths.log_dir}/llama-swap.log")
                paths.swap_pid.unlink(missing_ok=True)
                paths.active_marker.unlink(missing_ok=True)
                return 1
            print(f"    pid {read_pid(paths.swap_pid)}")

        print(f"[*] starting router on :{ROUTER_PORT}")
        env = os.environ.copy()
        if has_gguf:
            env.setdefault("LLAMA_SWAP_URL", f"http://127.0.0.1:{SWAP_PORT}")
        env.setdefault("ROUTER_HOST", "127.0.0.1")
        env.setdefault("ROUTER_PORT", str(ROUTER_PORT))
        # Lock-step with the gguf --use-next swap: bedrock tiers in the
        # router pick aws_model_id_next when this flag is set.
        if channel == "next":
            env["LLMSTACK_USE_NEXT"] = "1"
        else:
            env.pop("LLMSTACK_USE_NEXT", None)
        spawn_daemon(
            [sys.executable, "-m", "llmstack.app"],
            log=paths.log_dir / "router.log",
            pid_file=paths.router_pid,
            env=env,
        )
        time.sleep(1)
        if not is_running(paths.router_pid):
            print(f"[!] router failed to start. Check {paths.log_dir}/router.log")
            paths.router_pid.unlink(missing_ok=True)
            return 1
        print(f"    pid {read_pid(paths.router_pid)}")
      else:
            if has_gguf:
                print(f"[=] llama-swap already running (pid {read_pid(paths.swap_pid)}, channel {channel})")
            else:
                print("[=] bedrock-only config -- llama-swap not used")
            if is_running(paths.router_pid):
                print(f"[=] router already running (pid {read_pid(paths.router_pid)})")

    other = "next" if channel == "current" else "current"
    print()
    print(f"[OK] stack is up (channel: {channel}).")
    print()
    print(f'  router       http://127.0.0.1:{ROUTER_PORT}     (OpenAI-compatible, "auto" routing)')
    if has_gguf:
        print(f"  llama-swap   http://127.0.0.1:{SWAP_PORT}     (raw model endpoints + UI)")
    print()
    print("Try:")
    print(f"  curl -s http://127.0.0.1:{ROUTER_PORT}/v1/models | jq '.data[].id'")
    print(f"  curl -s http://127.0.0.1:{ROUTER_PORT}/models.ini | head")
    print()
    print("Logs:")
    if has_gguf:
        print(f"  tail -f {paths.log_dir}/llama-swap.log")
    print(f"  tail -f {paths.log_dir}/router.log")
    print()
    print("Switch channel (requires stop first):")
    print(f"  llmstack restart --{other}")
    print()
    print("Stop:")
    print("  llmstack stop")

    if detach:
        return 0

    # Only spawn a subshell when the env isn't already wired up. Two cases:
    #   - Hook installed + sourced: cd-ing into a project sets
    #     LLMSTACK_ACTIVE=1 and friends. start just brings up daemons --
    #     no need to nest another shell.
    #   - Inside a previously-spawned llmstack shell: same deal.
    # For users who haven't run `eval "$(llmstack activate <shell>)"`,
    # spawn so they at least get OPENCODE_CONFIG / channel exports for
    # this terminal.
    if os.environ.get("LLMSTACK_ACTIVE") == "1":
        cur_chan = os.environ.get("LLMSTACK_CHANNEL", "?")
        if cur_chan == channel:
            print(f"[=] already active in '{channel}' -- env is up to date.")
        else:
            # Daemons + active-channel marker are already on the new
            # channel. The current shell's env + PROMPT lag behind --
            # `llmstack reload` emits the eval-able snippet to fix that
            # without nesting a subshell.
            print(
                f"[*] channel switched ({cur_chan} -> {channel}); env in this shell "
                "is now stale."
            )
            print("    refresh prompt + env in this shell:")
            print('        eval "$(llmstack reload)"')
        return 0

    spawn_subshell(channel)
    return 0  # unreachable: spawn_subshell execvps
