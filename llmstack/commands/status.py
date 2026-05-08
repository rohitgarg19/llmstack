"""``llmstack status`` -- show channel, pids, ``/v1/models``, llama-server load.

The channel comes from ``.llmstack/default-channel`` (pinned by
``install``). Two top-level reporting paths:

  * ``current`` / ``next``  -- local install. Check pid files + port
                               probes for our daemons. If port :10102
                               responds without a pid file in *this*
                               project's ``.llmstack/``, the daemons
                               belong to another project on this host;
                               we report that as "(other)" so the user
                               knows the local daemons aren't ours --
                               it's not an error, but also not
                               something this project can ``stop``
                               cleanly.
  * ``external``            -- thin-client install. Skip all local
                               checks; probe the remote-router URL
                               from the marker.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

import yaml

from llmstack._platform import IS_WINDOWS
from llmstack.commands._helpers import is_running, pgrep, port_responds, read_pid
from llmstack.paths import (
    DEFAULT_REMOTE_URL,
    ROUTER_PORT,
    SWAP_PORT,
    read_marker,
    resolve,
)
from llmstack.tiers import load_tiers


def _print_help() -> None:
    print("usage: llmstack status")


def _check_local(name: str, url: str) -> None:
    """Report on a local daemon (router/llama-swap).

    ``alive`` (we own the process via pid file) is the happy path.
    ``responds`` without ``alive`` means the port is in use but the
    process isn't ours -- another project on this host owns it. We
    surface that as ``(other)`` rather than ``shared`` because there's
    no special "shared" mode anymore: a local install can't manage
    daemons it didn't spawn. ``llmstack install --external`` is the
    documented way to consume those daemons cleanly.
    """
    paths = resolve()
    pid_file = paths.state_dir / f"{name}.pid"
    pid = read_pid(pid_file) if pid_file.is_file() else None
    alive = pid is not None and is_running(pid_file)
    responds = port_responds(url, timeout=3.0)

    if alive:
        status = f"pid {pid:<7}"
    elif responds:
        status = "(other)"
    else:
        status = "DOWN"
    suffix = f"OK {url}" if responds else f"no response @ {url}"
    print(f"  {name:<12} {status:<11}  {suffix}")


def _print_process_table(pids: list[int]) -> None:
    """Render ``pid / rss_mb / command`` for each pid (cross-OS).

    POSIX: ``ps -o pid,rss,command`` (rss is in KB, we humanise to MB).
    Windows: ``tasklist /FI "PID eq ..." /FO CSV`` (image name + memory
    usage). Both branches print a header row.
    """
    if IS_WINDOWS:
        rows: list[tuple[str, str, str]] = []
        for pid in pids:
            try:
                proc = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if proc.returncode != 0 or not proc.stdout.strip():
                continue
            import csv
            for fields in csv.reader(proc.stdout.splitlines()):
                if len(fields) < 5:
                    continue
                image, pid_str, _session, _sid, mem = fields[0], fields[1], fields[2], fields[3], fields[4]
                if not pid_str.isdigit():
                    continue
                rss_mb = mem.replace(",", "").replace(" K", "").strip()
                try:
                    rss_mb = f"{int(rss_mb) // 1024} MB"
                except ValueError:
                    pass
                rows.append((pid_str, rss_mb, image))
        if not rows:
            print("  (tasklist returned nothing)")
            return
        print(f"  {'PID':<8} {'RSS':<10} COMMAND")
        for pid_str, rss, cmd in rows:
            print(f"  {pid_str:<8} {rss:<10} {cmd}")
        return

    try:
        ps = subprocess.run(
            ["ps", "-o", "pid,rss,command", "-p", ",".join(str(p) for p in pids)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        print("  (ps failed)")
        return
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


def _print_remote_status(paths, url: str) -> int:
    print(f"stack status (channel: external -- remote {url}):")
    print(f"  work dir      {paths.work_dir}")
    probe = f"{url}/models.ini"
    responds = port_responds(probe, timeout=3.0)
    suffix = f"OK {probe}" if responds else f"no response @ {probe}"
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

    # Channel decision is pinned at install time; status just reads it.
    # active-channel (set by `start`) takes precedence over default-channel
    # (set by `install`) so a `start --next` run is reflected immediately.
    default = read_marker(paths.default_marker)
    active = read_marker(paths.active_marker)
    persisted = active or default

    if persisted and persisted.channel == "external":
        url = (persisted.url or "").rstrip("/") or DEFAULT_REMOTE_URL
        return _print_remote_status(paths, url)

    if active:
        channel = active.channel
    elif default and default.channel in ("current", "next"):
        channel = f"{default.channel} (or stopped)"
    elif port_responds(f"http://127.0.0.1:{SWAP_PORT}/health"):
        channel = "(other) -- daemons running on :10102 are not ours"
    else:
        channel = "current (or stopped)"

    tiers = load_tiers()
    has_gguf = any(t.is_gguf for t in tiers.values())

    print(f"stack status (channel: {channel}):")
    print(f"  work dir      {paths.work_dir}")
    # Router has no /health route (dropped in v3.x); /v1/models always
    # 200s on a live router. llama-swap is a separate binary with its
    # own /health endpoint -- leave that one alone.
    _check_local("router", f"http://127.0.0.1:{ROUTER_PORT}/v1/models")
    if has_gguf:
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

    if has_gguf:
        _list_models(f"http://127.0.0.1:{ROUTER_PORT}")
    else:
        print()
        print("current models in /v1/models:")
        try:
            with urllib.request.urlopen(f"{f'http://127.0.0.1:{ROUTER_PORT}'}/v1/models", timeout=5) as resp:
                data = json.load(resp)
            for m in data.get("data", []):
                print(f"  - {m.get('id')}")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            print(f"  (no response @ http://127.0.0.1:{ROUTER_PORT}/v1/models)")

    if has_gguf:
        print()
        print("loaded llama-server processes:")
        pids = pgrep(r"llama-server.*--alias")
        if pids:
            _print_process_table(pids)
        else:
            print("  (none loaded)")

    if channel.split()[0] == "next" and has_gguf and paths.llama_swap_yaml.is_file():
        print()
        print(f"next-channel swaps (from {paths.llama_swap_yaml.name}):")
        try:
            cfg = yaml.safe_load(paths.llama_swap_yaml.read_text())
            for name, m in (cfg.get("models") or {}).items():
                md = m.get("metadata") or {}
                if md.get("channel") != "next":
                    continue
                hff = "?"
                for line in (m.get("cmd") or "").splitlines():
                    s = line.strip()
                    if s.startswith("-hff ") and not s.lstrip().startswith("#"):
                        hff = s[len("-hff "):].strip()
                        break
                print(f"  {name:<18}  -> {hff}  ({md.get('quant', '?')}, {md.get('size_gb', '?')} GB)")
        except (OSError, yaml.YAMLError):
            pass
    return 0
