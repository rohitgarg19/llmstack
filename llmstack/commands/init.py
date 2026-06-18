"""``llmstack init`` -- seed a fresh ``.llmstack/`` in the current directory.

``init`` does two things:

  1. **Decides the channel** (local ``current`` / ``next``, or
     ``external <url>``) and writes ``.llmstack/default-channel`` so
     every downstream command (``configure``, ``start``, ``status``,
     the activate hook) agrees on what kind of project this is without
     re-deriving it.

  2. **Copies the editable input files** into ``.llmstack/``:
     ``models.ini``, ``instructions.md``, the per-agent prompt markdown
     (``agents/*.md``) and ``litellm_config.yaml``. These are the knobs
     the user edits before running ``llmstack configure`` to generate
     the derived outputs (``opencode.json``, ``llama-swap.yaml``).

Crucially ``init`` always targets the **current working directory**,
never a parent project. The activate hook exports
``LLMSTACK_WORK_DIR`` pointing at the nearest ancestor ``.llmstack/``,
so a plain ``resolve()`` inside a hooked shell would seed the parent.
``init`` ignores that on purpose: running it in a subdirectory means
"make *this* directory its own project".

By default existing input files are left untouched (safe to re-run).
``--force`` resets the project completely: re-copies every input file
from the bundled templates *and* clears any previously generated
outputs (``opencode.json``, ``llama-swap.yaml``, ``default-channel``,
``active-channel``) so the next ``configure`` starts from a clean
slate. Use ``--force`` when switching a project between local and
external mode.

When a fresh ``models.ini`` is seeded and the ``litellm`` extra is
installed, any block fenced with
``; >>> AUTO-ENABLE-WHEN-LITELLM-AVAILABLE >>>`` markers is uncommented
in place (same behaviour the old ``install`` had on first seed).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from llmstack.paths import (
    AGENT_TEMPLATES,
    AGENTS_TEMPLATE,
    ChannelMark,
    DEFAULT_REMOTE_URL,
    LITELLM_CONFIG_TEMPLATE,
    MODELS_INI_TEMPLATE,
    env_remote_url,
    write_marker,
)

_LITELLM_BEGIN = "; >>> AUTO-ENABLE-WHEN-LITELLM-AVAILABLE >>>"
_LITELLM_END   = "; <<< AUTO-ENABLE-WHEN-LITELLM-AVAILABLE <<<"

# Derived outputs that --force clears so configure starts clean.
_DERIVED_FILES = [
    "opencode.json",
    "llama-swap.yaml",
    "default-channel",
    "active-channel",
]


def _try_enable_litellm_blocks(ini_path: Path) -> int:
    """Activate any ``AUTO-ENABLE-WHEN-LITELLM-AVAILABLE`` block in
    ``ini_path`` when ``litellm`` is importable.

    For each fenced block we drop the BEGIN / END marker lines and
    strip a single leading ``"; "`` (or ``";\\t"``) from every line in
    between -- so a doubly-commented line like ``; ; model = ...``
    becomes a still-commented ``; model = ...`` in the active config.
    Returns the number of blocks rewritten; ``0`` when litellm is
    missing, no markers exist, or every block is already expanded.
    """
    try:
        import litellm  # noqa: F401  -- presence check only
    except ImportError:
        return 0

    text = ini_path.read_text()
    if _LITELLM_BEGIN not in text or _LITELLM_END not in text:
        return 0

    out: list[str] = []
    inside = False
    blocks = 0
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n").rstrip()
        if bare == _LITELLM_BEGIN:
            inside = True
            blocks += 1
            continue
        if bare == _LITELLM_END:
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
    print(
        "usage: llmstack init [--force] "
        "[--current | --next | --external [URL]]"
    )


def _parse_args(args: list[str]) -> tuple[bool, str, str | None, bool]:
    """Parse ``init``'s flags.

    Returns ``(force, local_channel, external_url, want_external)``.
    ``--external`` accepts ``--external <url>`` or ``--external=<url>``
    and is mutually exclusive with ``--current`` / ``--next``.
    """
    force = False
    local_channel = "current"
    local_explicit = False
    external_url: str | None = None
    want_external = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--force":
            force = True
        elif arg == "--next":
            local_channel = "next"
            local_explicit = True
        elif arg == "--current":
            local_channel = "current"
            local_explicit = True
        elif arg == "--external":
            want_external = True
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                external_url = args[i + 1]
                i += 1
        elif arg.startswith("--external="):
            want_external = True
            external_url = arg[len("--external="):]
        elif arg in ("-h", "--help"):
            _print_help()
            raise SystemExit(0)
        else:
            print(
                f"[!] unknown arg to init: {arg} "
                "(try --force, --current, --next, --external, -h)"
            )
            raise SystemExit(2)
        i += 1

    if want_external and local_explicit:
        print(
            "[!] --external is mutually exclusive with --current / --next."
        )
        raise SystemExit(2)

    return force, local_channel, external_url, want_external


def _resolve_external_url(flag_url: str | None) -> str:
    """Pick the remote URL to bake into the channel marker.

    Precedence: explicit ``--external <url>`` arg > ``$LLMSTACK_REMOTE_URL``
    env var > :data:`DEFAULT_REMOTE_URL` (the local router).
    """
    if flag_url:
        return flag_url.rstrip("/")
    env = env_remote_url()
    if env:
        return env
    return DEFAULT_REMOTE_URL


def _copy(src: Path, dst: Path, *, force: bool) -> str:
    """Copy ``src`` -> ``dst``. Returns one of ``seeded`` / ``kept`` /
    ``forced`` / ``missing`` for reporting."""
    if not src.is_file():
        return "missing"
    existed = dst.is_file()
    if existed and not force:
        return "kept"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o644)
    return "forced" if existed else "seeded"


def run(args: list[str]) -> int:
    try:
        force, local_channel, external_url_arg, want_external = _parse_args(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0

    # Env-var fallback: ``LLMSTACK_REMOTE_URL`` set without ``--external``
    # still implies external mode (the activate hook re-exports it).
    if not want_external and env_remote_url() is not None:
        want_external = True

    if want_external:
        remote = _resolve_external_url(external_url_arg)
        channel: str = "external"
    else:
        remote = None
        channel = local_channel

    # Always target the *current* directory, never a parent project the
    # activate hook may have pointed LLMSTACK_WORK_DIR at.
    state = (Path.cwd() / ".llmstack").resolve()
    agents_dir = state / "agents"
    state.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] initializing project at {state}")
    if force:
        print("    (--force: overwriting input files + clearing derived outputs)")

    # --force: clear previously generated outputs so configure starts clean.
    if force:
        for name in _DERIVED_FILES:
            p = state / name
            if p.exists():
                p.unlink()
                print(f"[*] cleared {p.name}")

    # --- channel marker (written now, read by configure / start / status) ---
    marker_path = state / "default-channel"
    mark = ChannelMark("external", remote) if remote else ChannelMark(channel)
    write_marker(marker_path, mark)
    if remote:
        print(f"[OK] channel: external (remote: {remote})")
    else:
        print(f"[OK] channel: {channel}")

    seeded_models = False

    # models.ini -- the primary input for local mode.
    # External projects don't need a local models.ini (configure fetches
    # the router's live copy), but we seed it anyway so the user has a
    # reference and can switch back to local mode later.
    models_dst = state / "models.ini"
    result = _copy(MODELS_INI_TEMPLATE, models_dst, force=force)
    if result == "missing":
        print(f"[!] models.ini template not found at {MODELS_INI_TEMPLATE}; skipping")
    elif result == "kept":
        print(f"[=] models.ini exists -- kept {models_dst}")
    else:
        seeded_models = True
        print(f"[OK] models.ini -> {models_dst}")

    # instructions.md (agent instructions surfaced to opencode).
    instr_dst = state / "instructions.md"
    result = _copy(AGENTS_TEMPLATE, instr_dst, force=force)
    if result == "missing":
        print(f"[!] instructions.md template not found at {AGENTS_TEMPLATE}; skipping")
    elif result == "kept":
        print(f"[=] instructions.md exists -- kept {instr_dst}")
    else:
        print(f"[OK] instructions.md -> {instr_dst}")

    # agent prompts -- one md per agent, editable per project.
    for name, src in AGENT_TEMPLATES.items():
        dst = agents_dir / name
        result = _copy(src, dst, force=force)
        if result == "missing":
            print(f"[!] agent template not found at {src}; skipping")
        elif result == "kept":
            print(f"[=] agents/{name} exists -- kept")
        else:
            print(f"[OK] agents/{name} -> {dst}")

    # litellm_config.yaml -- editable proxy config.
    litellm_dst = state / "litellm_config.yaml"
    result = _copy(LITELLM_CONFIG_TEMPLATE, litellm_dst, force=force)
    if result == "missing":
        print(f"[!] litellm_config.yaml template not found at {LITELLM_CONFIG_TEMPLATE}; skipping")
    elif result == "kept":
        print(f"[=] litellm_config.yaml exists -- kept {litellm_dst}")
    else:
        print(f"[OK] litellm_config.yaml -> {litellm_dst}")

    # Auto-enable litellm tier blocks only on a fresh models.ini seed.
    if seeded_models and models_dst.is_file():
        enabled = _try_enable_litellm_blocks(models_dst)
        if enabled:
            print(
                f"[*] litellm detected -- enabled {enabled} litellm-backed "
                f"tier block(s) in {models_dst}"
            )

    print()
    print("[OK] project initialized.")
    print()
    if channel == "external":
        print(f"  mode:    external (remote: {remote})")
        print()
        print("Next:")
        print("  llmstack configure   # render opencode.json from the remote router's models.ini")
    else:
        print(f"  mode:    local ({channel} channel)")
        print()
        print("Next:")
        print("  edit .llmstack/models.ini to taste, then:")
        print("  llmstack configure   # generate opencode.json + llama-swap.yaml")
    return 0
