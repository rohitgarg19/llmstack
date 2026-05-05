"""``llmstack install`` -- regenerate ``opencode.json`` (and AGENTS.md copy).

Renders the opencode config atomically (tmp file in target dir, validate,
``mv``), copies AGENTS.md alongside it, and pins the default channel for
``start`` to pick up. ``llama-swap.yaml`` is *not* generated here -- it's
a runtime-only artifact owned by ``llmstack start`` (which knows the
chosen channel and regenerates the yaml on each launch).

``--print`` writes the opencode config to stdout instead of files.

When this command seeds a fresh ``models.ini`` from the bundled template
and the ``bedrock`` extra is installed (i.e. ``import boto3`` succeeds),
any block fenced with ``; >>> AUTO-ENABLE-WHEN-BEDROCK-AVAILABLE >>>``
markers in the seeded file is uncommented in place. The auto-enable
runs only on the *initial* seed; subsequent ``install`` runs never
mutate the user's models.ini.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from llmstack.generators import render_to
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


_BEDROCK_BEGIN = "; >>> AUTO-ENABLE-WHEN-BEDROCK-AVAILABLE >>>"
_BEDROCK_END   = "; <<< AUTO-ENABLE-WHEN-BEDROCK-AVAILABLE <<<"


def _try_enable_bedrock_blocks(ini_path: Path) -> int:
    """Activate any ``AUTO-ENABLE-WHEN-BEDROCK-AVAILABLE`` block in
    ``ini_path`` when ``boto3`` is importable.

    For each fenced block we drop the BEGIN / END marker lines and
    strip a single leading ``"; "`` (or ``";\\t"``) from every line in
    between -- so a doubly-commented line like ``; ; aws_profile = ...``
    becomes a still-commented ``; aws_profile = ...`` in the active
    config (preserving the "uncomment to use" semantics of literal
    in-file comments). Returns the number of blocks rewritten; ``0``
    when boto3 is missing, no markers exist, or every block is already
    expanded.
    """
    try:
        import boto3  # noqa: F401  -- presence check only
    except ImportError:
        return 0

    text = ini_path.read_text()
    if _BEDROCK_BEGIN not in text or _BEDROCK_END not in text:
        return 0

    out: list[str] = []
    inside = False
    blocks = 0
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n").rstrip()
        if bare == _BEDROCK_BEGIN:
            inside = True
            blocks += 1
            continue
        if bare == _BEDROCK_END:
            inside = False
            continue
        if inside:
            if line.startswith("; ") or line.startswith(";\t"):
                out.append(line[2:])
            elif bare == ";":
                out.append(line[1:])
            else:
                out.append(line)
        else:
            out.append(line)

    if blocks == 0:
        return 0
    ini_path.write_text("".join(out))
    return blocks


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
        enabled = _try_enable_bedrock_blocks(ini_path)
        if enabled:
            print(
                f"[*] boto3 detected -- enabled {enabled} bedrock-backed "
                f"tier block(s) in {ini_path}"
            )
        print("    edit it to taste, then re-run `llmstack install`.")

    paths = ensure_state_dirs()
    remote = is_remote()

    if print_only:
        if remote:
            print(f"# remote mode (LLMSTACK_REMOTE_URL={remote_url()}); llama-swap.yaml not used.")
            print()
        print("----- opencode.json -----")
        print(render_opencode())
        return 0

    print("[1/2] AGENTS.md")
    if AGENTS_TEMPLATE.is_file():
        shutil.copyfile(AGENTS_TEMPLATE, paths.agents_local)
        os.chmod(paths.agents_local, 0o644)
        print(f"[OK] copied AGENTS.md -> {paths.agents_local}")
    else:
        print(f"[!] AGENTS.md template not found at {AGENTS_TEMPLATE}; skipping copy")

    print()
    print("[2/2] opencode.json")
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

    print()
    print(f"[OK] opencode config generated from {paths.models_ini}.")
    print()
    print(f"  config:       {paths.opencode_json}")
    print(f"  instructions: {paths.agents_local}")
    if remote:
        print(f"  remote:       {remote_url()}")
    else:
        print(f"  channel:      {default_channel}")
    print()
    print("Next:")
    if remote:
        print("  llmstack start     # verify remote + drop into the client subshell")
    else:
        print("  llmstack start     # generate llama-swap.yaml + bring up the stack")
        print("  llmstack check     # snapshot configured GGUFs + drift check")
    return 0
