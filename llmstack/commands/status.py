"""``llmstack status`` -- show channel, pids, ``/v1/models``, llama-server load.

Three states the channel can be in:

  * ``current`` / ``next``        -- local stack we (or another project on
                                     this host) own. Pid files + port
                                     probes both relevant.
  * ``shared``                    -- local daemons running but no pid file
                                     in this project's ``.llmstack/`` --
                                     i.e. another project on the same host
                                     started them.
  * ``external``                  -- ``$LLMSTACK_REMOTE_URL`` is set; we
                                     are a thin client of a remote stack.
                                     Skip all local checks; just probe the
                                     remote.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

import yaml

from llmstack.commands._helpers import is_running, pgrep, port_responds, read_pid
from llmstack.paths import (
    ROUTER_PORT,
    SWAP_PORT,
    is_remote,
    read_marker,
    remote_url,
    resolve,
)


def _print_help() -> None:
    print("usage: llmstack status")


def _check_local(name: str, url: str) -> None:
    paths = resolve()
    pid_file = paths.state_dir / f"{name}.pid"
    pid = read_pid(pid_file) if pid_file.is_file() else None
    alive = pid is not None and is_running(pid_file)
    responds = port_responds(url, timeout=3.0)

    if alive:
        status = f"pid {pid:<7}"
    elif responds:
        status = "shared"
    else:
        status = "DOWN"
    suffix = f"OK {url}" if responds else f"no response @ {url}"
    print(f"  {name:<12} {status:<11}  {suffix}")


def _list_models(base: str) -> None:
    print()
    print("current models in /v1/models:")
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as resp:
            data = json.load(resp)
        for m in data.get("data", []):
            print(f"  - {m.get('id')}")
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
        print(f"  (no response @ {base}/v1/models)")


def _print_remote_status(paths) -> int:
    url = remote_url()
    assert url is not None
    print(f"stack status (channel: external -- remote {url}):")
    print(f"  work dir      {paths.work_dir}")
    responds = port_responds(f"{url}/health", timeout=3.0)
    suffix = f"OK {url}/health" if responds else f"no response @ {url}/health"
    status = "external" if responds else "DOWN"
    print(f"  {'router':<12} {status:<11}  {suffix}")

    print()
    if paths.opencode_json.is_file():
        print(f"  opencode      {paths.opencode_json}")
        if paths.agents_local.is_file():
            print(f"  instructions  {paths.agents_local}")
    else:
        print("  opencode      (not generated for this work dir; run: llmstack install)")

    if os.environ.get("LLMSTACK_ACTIVE") == "1":
        cfg = os.environ.get("OPENCODE_CONFIG", "?")
        chan = os.environ.get("LLMSTACK_CHANNEL", "?")
        print(f"  in-shell      OPENCODE_CONFIG={cfg}, LLMSTACK_CHANNEL={chan}")

    if responds:
        _list_models(url)
    return 0


def run(args: list[str]) -> int:
    for arg in args:
        if arg in ("-h", "--help"):
            _print_help()
            return 0
        print(f"[!] unknown arg to status: {arg}")
        return 2

    paths = resolve()

    if is_remote():
        return _print_remote_status(paths)

    mark = read_marker(paths.active_marker)
    if mark:
        channel = mark.channel
        if mark.url:
            channel = f"{channel} (remote: {mark.url})"
    elif port_responds(f"http://127.0.0.1:{SWAP_PORT}/health"):
        channel = "shared (started by another project on this host)"
    else:
        channel = "current (or stopped)"

    print(f"stack status (channel: {channel}):")
    print(f"  work dir      {paths.work_dir}")
    _check_local("router", f"http://127.0.0.1:{ROUTER_PORT}/health")
    _check_local("llama-swap", f"http://127.0.0.1:{SWAP_PORT}/health")

    print()
    if paths.opencode_json.is_file():
        print(f"  opencode      {paths.opencode_json}")
        if paths.agents_local.is_file():
            print(f"  instructions  {paths.agents_local}")
    else:
        print("  opencode      (not generated for this work dir; run: llmstack install)")

    if os.environ.get("LLMSTACK_ACTIVE") == "1":
        cfg = os.environ.get("OPENCODE_CONFIG", "?")
        chan = os.environ.get("LLMSTACK_CHANNEL", "?")
        print(f"  in-shell      OPENCODE_CONFIG={cfg}, LLMSTACK_CHANNEL={chan}")

    _list_models(f"http://127.0.0.1:{ROUTER_PORT}")

    print()
    print("loaded llama-server processes:")
    pids = pgrep(r"llama-server.*--alias")
    if pids:
        try:
            ps = subprocess.run(
                ["ps", "-o", "pid,rss,command", "-p", ",".join(str(p) for p in pids)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for i, line in enumerate(ps.stdout.splitlines()):
                if i == 0:
                    print(line)
                    continue
                cols = line.split()
                if len(cols) >= 3:
                    try:
                        rss_mb = int(cols[1]) // 1024
                        cols[1] = f"{rss_mb} MB"
                    except ValueError:
                        pass
                print(" ".join(cols))
        except (OSError, subprocess.SubprocessError):
            print("  (ps failed)")
    else:
        print("  (none loaded)")

    if channel.split()[0] == "next" and paths.llama_swap_yaml.is_file():
        print()
        print(f"next-channel swaps (from {paths.llama_swap_yaml.name}):")
        try:
            cfg = yaml.safe_load(paths.llama_swap_yaml.read_text())
            for name, m in (cfg.get("models") or {}).items():
                md = m.get("metadata") or {}
                if md.get("channel") != "next":
                    continue
                active = "?"
                for line in (m.get("cmd") or "").splitlines():
                    s = line.strip()
                    if s.startswith("-hff ") and not s.lstrip().startswith("#"):
                        active = s[len("-hff "):].strip()
                        break
                print(f"  {name:<18}  -> {active}  ({md.get('quant', '?')}, {md.get('size_gb', '?')} GB)")
        except (OSError, yaml.YAMLError):
            pass
    return 0
