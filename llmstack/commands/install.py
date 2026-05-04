"""``llmstack install`` -- regenerate llama-swap.yaml + opencode.json.

Mirrors the shell ``cmd_install``: render both configs atomically (tmp file
in target dir, validate, ``mv``), copy AGENTS.md alongside opencode.json,
pin the default channel, and print a passive llama-swap version check.
``--print`` writes both configs to stdout instead of files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from llmstack.download.binary import installed_version, latest_release_tag
from llmstack.generators import render_to
from llmstack.generators.llama_swap import render as render_yaml
from llmstack.generators.llama_swap import validate as validate_yaml
from llmstack.generators.opencode import render as render_opencode
from llmstack.generators.opencode import validate as validate_opencode
from llmstack.paths import (
    AGENTS_TEMPLATE,
    ChannelMark,
    ensure_models_ini,
    ensure_state_dirs,
    is_remote,
    remote_url,
    write_marker,
)


def _print_help() -> None:
    print("usage: llmstack install [--print] [--current | --next]")


def run(args: list[str]) -> int:
    print_only = False
    default_channel = "current"
    for arg in args:
        if arg in ("--print", "-n"):
            print_only = True
        elif arg == "--next":
            default_channel = "next"
        elif arg == "--current":
            default_channel = "current"
        elif arg in ("-h", "--help"):
            _print_help()
            return 0
        else:
            print(f"[!] unknown arg to install: {arg} (try --print, --current, --next, -h)")
            return 2

    ini_path, seeded = ensure_models_ini()
    if seeded:
        print(f"[*] no models.ini found -- seeded default at {ini_path}")
        print("    edit it to taste, then re-run `llmstack install`.")

    paths = ensure_state_dirs()
    remote = is_remote()

    if print_only:
        if remote:
            print(f"# remote mode (LLMSTACK_REMOTE_URL={remote_url()}); llama-swap.yaml not used.")
            print()
        else:
            print("----- llama-swap.yaml -----")
            print(render_yaml())
            print()
        print("----- opencode.json -----")
        print(render_opencode())
        return 0

    if remote:
        print(f"[1/1] remote mode (LLMSTACK_REMOTE_URL={remote_url()})")
        print("      skipping llama-swap.yaml -- the remote stack runs llama-swap.")
        print()
    else:
        print("[1/2] llama-swap.yaml")
        render_to(
            paths.llama_swap_yaml,
            render=lambda p: Path(p).write_text(render_yaml()),
            validate=validate_yaml,
        )
        print(f"[OK] installed {paths.llama_swap_yaml}")
        print()

    label = "[1/1]" if remote else "[2/2]"
    print(f"{label} opencode.json + AGENTS.md")
    if AGENTS_TEMPLATE.is_file():
        shutil.copyfile(AGENTS_TEMPLATE, paths.agents_local)
        os.chmod(paths.agents_local, 0o644)
        print(f"[OK] copied AGENTS.md -> {paths.agents_local}")
    else:
        print(f"[!] AGENTS.md template not found at {AGENTS_TEMPLATE}; skipping copy")

    prev = os.environ.get("OPENCODE_INSTRUCTIONS")
    os.environ["OPENCODE_INSTRUCTIONS"] = str(paths.agents_local)
    try:
        render_to(
            paths.opencode_json,
            render=lambda p: Path(p).write_text(render_opencode()),
            validate=validate_opencode,
        )
    finally:
        if prev is None:
            os.environ.pop("OPENCODE_INSTRUCTIONS", None)
        else:
            os.environ["OPENCODE_INSTRUCTIONS"] = prev
    print(f"[OK] installed {paths.opencode_json}")

    if remote:
        write_marker(paths.default_marker, ChannelMark("external", remote_url()))
        print(f"[OK] default channel: external (remote: {remote_url()})")
    else:
        write_marker(paths.default_marker, ChannelMark(default_channel))
        print(f"[OK] default channel: {default_channel}")

    if remote:
        print()
        print(f"[OK] configs generated from {paths.models_ini}.")
        print("     opencode.json points at the remote router; no llama-swap.yaml needed.")
        print()
        print(f"  config:       {paths.opencode_json}")
        print(f"  instructions: {paths.agents_local}")
        print(f"  remote:       {remote_url()}")
        print()
        print("Next:")
        print("  llmstack start     # verify remote + drop into the client subshell")
        return 0

    print()
    print("[*] checking llama-swap version...")
    if paths.llama_swap_bin.exists():
        installed = installed_version(paths.llama_swap_bin)
        if installed:
            print(f"    installed: v{installed}")
        latest = latest_release_tag()
        if latest:
            latest_num = latest.lstrip("v")
            if installed == latest_num:
                print(f"    latest:    {latest}  [up to date]")
            else:
                print(f"    latest:    {latest}  [update available -> run: llmstack setup to update]")
        else:
            print("    (could not reach GitHub to check for updates)")
    else:
        print("    llama-swap not installed yet -- run: llmstack setup")

    print()
    print(f"[OK] configs generated from {paths.models_ini}.")
    print("     re-run any time you edit models.ini.")
    print()
    print(f"  config:       {paths.opencode_json}")
    print(f"  instructions: {paths.agents_local}")
    print(f"  channel:      {default_channel}")
    print()
    print("Next:")
    print("  llmstack start     # bring up the stack + enter the shell")
    print("  llmstack check     # snapshot configured GGUFs + drift check")
    return 0
