"""``llmstack start`` -- bring up the stack and enter the env-prepared subshell.

Two top-level modes:

  *Local mode* (default)
    Generate ``llama-swap.yaml`` for the chosen channel, launch
    llama-swap + the FastAPI router locally, and drop into a subshell
    with ``OPENCODE_CONFIG`` exported. The yaml is regenerated on every
    fresh launch so it always reflects the live ``models.ini``; if the
    daemons are already up we leave their loaded yaml alone.

    Channel resolution (highest priority first):
      1. explicit ``--current`` / ``--next`` on the command line
      2. ``.llmstack/default-channel`` pinned by the last ``install``
      3. hard-coded fallback: ``current``

    Daemon state has three branches:
      (a) local pid file says daemons are up      -> idempotent, channel-checked,
                                                     no yaml regeneration
      (b) port 10102 responds but no local pid    -> daemons started by
                                                     another project on this
                                                     host; reuse them.
                                                     Channel label: **shared**.
      (c) nothing                                 -> regenerate yaml, launch fresh

  *Remote / client mode* -- when ``$LLMSTACK_REMOTE_URL`` is set
    Don't launch anything; verify the remote ``/health`` endpoint is
    reachable and drop into the subshell with ``LLMSTACK_CHANNEL=external``.
    Useful for laptops talking to a beefy desktop's llmstack.
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
from llmstack.paths import (
    ROUTER_PORT,
    SWAP_PORT,
    ChannelMark,
    ensure_state_dirs,
    is_remote,
    read_marker,
    remote_url,
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


def _start_remote(detach: bool) -> int:
    """Client-mode start: just verify the remote and drop into the shell."""
    paths = ensure_state_dirs()
    url = remote_url()
    assert url is not None  # guarded by caller

    if not paths.opencode_json.is_file():
        raise SystemExit(
            f"no .llmstack/opencode.json in {paths.work_dir} -- run: llmstack install\n"
            f"    (with LLMSTACK_REMOTE_URL={url} so opencode is wired to the remote)"
        )

    print(f"[*] remote llmstack: {url}")
    if port_responds(f"{url}/health", timeout=5.0):
        print(f"[OK] {url}/health responds.")
    else:
        print(f"[!] {url}/health did not respond -- is the remote stack up?", file=sys.stderr)
        print("    proceeding anyway; opencode will surface the error on first request.", file=sys.stderr)

    write_marker(paths.active_marker, ChannelMark("external", url))

    print()
    print("[OK] client mode (channel: external).")
    print()
    print(f"  router       {url}     (remote)")
    print()
    print("Try:")
    print(f"  curl -s {url}/v1/models | jq '.data[].id'")
    print(f"  curl -s {url}/health | jq '.tiers'")
    print()
    print("Disconnect:")
    print("  exit                # leave the subshell (daemons are remote, nothing to stop)")

    if detach:
        return 0

    if os.environ.get("LLMSTACK_ACTIVE") == "1":
        if os.environ.get("LLMSTACK_CHANNEL") == "external":
            print("[=] already in an 'external' llmstack shell -- no reload needed.")
            return 0
        print(
            f"[*] switching to external client ({os.environ.get('LLMSTACK_CHANNEL')} -> external); "
            "reloading shell..."
        )
        spawn_subshell("external", reload=True)
        return 0

    spawn_subshell("external")
    return 0  # unreachable


def run(args: list[str]) -> int:
    channel: str | None = None
    detach = False
    for arg in args:
        if arg == "--next":
            channel = "next"
        elif arg == "--current":
            channel = "current"
        elif arg in ("--detach", "--no-shell"):
            detach = True
        elif arg in ("-h", "--help"):
            _print_help()
            return 0
        else:
            print(f"[!] unknown arg to start: {arg} (try --next, --current, --detach, -h)")
            return 2

    if is_remote():
        if channel is not None:
            print(
                "[!] --current / --next are local-mode flags and have no effect when "
                "LLMSTACK_REMOTE_URL is set; the remote picks the channel.",
                file=sys.stderr,
            )
        return _start_remote(detach)

    paths = ensure_state_dirs()

    if channel is None:
        default = read_marker(paths.default_marker)
        # `external` in default-channel means the user previously installed
        # this project in remote mode -- but if they got here without
        # LLMSTACK_REMOTE_URL set, fall through to "current" rather than
        # silently picking up a stale URL.
        if default and default.channel in ("current", "next"):
            channel = default.channel
    if not channel:
        channel = "current"

    if not paths.llama_swap_bin.exists() or not os.access(paths.llama_swap_bin, os.X_OK):
        raise SystemExit(f"missing {paths.llama_swap_bin} (run: llmstack setup)")
    if not paths.opencode_json.is_file():
        raise SystemExit(f"no .llmstack/opencode.json in {paths.work_dir} -- run: llmstack install")

    launch_daemons = True
    shared_daemons = False
    if is_running(paths.swap_pid):
        launch_daemons = False
        live_mark = read_marker(paths.active_marker)
        live = live_mark.channel if live_mark else "current"
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
        launch_daemons = False
        shared_daemons = True
        print(f"[*] daemons already up on :{ROUTER_PORT}/:{SWAP_PORT} (started by another project on this host)")
        print("    will reuse them. Use 'llmstack stop' from any project to stop.")

    if launch_daemons:
        if channel == "next":
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
        render_to(
            paths.llama_swap_yaml,
            render=lambda p: Path(p).write_text(render_yaml(use_next=(channel == "next"))),
            validate=validate_yaml,
        )

    if shared_daemons:
        print("[*] channel: shared (whatever the running daemons were started with)")
    else:
        print(f"[*] channel: {channel}  ({paths.llama_swap_yaml.name})")

    if launch_daemons:
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
    elif not shared_daemons:
        print(f"[=] llama-swap already running (pid {read_pid(paths.swap_pid)}, channel {channel})")
        if is_running(paths.router_pid):
            print(f"[=] router already running (pid {read_pid(paths.router_pid)})")

    other = "next" if channel == "current" else "current"
    channel_label = "shared" if shared_daemons else channel
    print()
    print(f"[OK] stack is up (channel: {channel_label}).")
    print()
    print(f'  router       http://127.0.0.1:{ROUTER_PORT}     (OpenAI-compatible, "auto" routing)')
    print(f"  llama-swap   http://127.0.0.1:{SWAP_PORT}     (raw model endpoints + UI)")
    print()
    print("Try:")
    print(f"  curl -s http://127.0.0.1:{ROUTER_PORT}/v1/models | jq '.data[].id'")
    print(f"  curl -s http://127.0.0.1:{ROUTER_PORT}/health | jq '.tiers'")
    print()
    print("Logs:")
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

    # spawn_subshell expects the channel as written into active-marker so the
    # prompt prefix matches reality. For shared daemons we keep the *real*
    # local channel ("current"/"next") since the env we're entering is
    # genuinely on that channel from this project's POV.
    if os.environ.get("LLMSTACK_ACTIVE") == "1":
        if os.environ.get("LLMSTACK_CHANNEL") == channel:
            print(f"[=] already in the '{channel}' llmstack shell -- no reload needed.")
            return 0
        print(
            f"[*] channel changed ({os.environ.get('LLMSTACK_CHANNEL')} -> {channel}); "
            "reloading shell..."
        )
        spawn_subshell(channel, reload=True)
        return 0

    spawn_subshell(channel)
    return 0  # unreachable: spawn_subshell execvps
