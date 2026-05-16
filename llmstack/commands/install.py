"""``llmstack install`` -- regenerate ``opencode.json`` (and AGENTS.md copy).

Renders the opencode config atomically (tmp file in target dir, validate,
``mv``), copies AGENTS.md alongside it, pins the default channel for
``start`` to pick up, and -- for local channels -- renders
``llama-swap.yaml`` for that channel. ``start`` / ``restart`` consume
the yaml as-is; channel selection (``--current`` / ``--next``) is an
install-time decision and is not honoured by ``start``.

This is also where the **channel** is decided -- everything downstream
(``start``, ``status``, the activate hook) reads the persisted choice
from ``.llmstack/default-channel`` and never re-derives it. Three
channels exist:

  * ``current``   -- local stack, canonical channel (default)
  * ``next``      -- local stack, queued-upgrade channel
  * ``external``  -- thin client; no daemons launched. Opt in via
                     ``--external [URL]`` (URL defaults to the local
                     router, ``http://127.0.0.1:10101``, so two
                     projects on one host can share daemons without
                     fighting for ports). ``LLMSTACK_REMOTE_URL`` in the
                     environment is honoured as an alternative way in.

``--print`` writes the opencode config to stdout instead of files.

When this command seeds a fresh ``models.ini`` from the bundled template
and the ``litellm`` extra is installed (i.e. ``import litellm`` succeeds),
any block fenced with ``; >>> AUTO-ENABLE-WHEN-LITELLM-AVAILABLE >>>``
markers in the seeded file is uncommented in place. The auto-enable
runs only on the *initial* seed; subsequent ``install`` runs never
mutate the user's models.ini.
"""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from llmstack.generators import render_to
from llmstack.generators.litellm_config import merge as merge_litellm_config
from llmstack.generators.llama_swap import render as render_yaml
from llmstack.generators.llama_swap import validate as validate_yaml
from llmstack.generators.opencode import render as render_opencode
from llmstack.generators.opencode import validate as validate_opencode
from llmstack.paths import (
    AGENTS_TEMPLATE,
    DEFAULT_REMOTE_URL,
    ChannelMark,
    ensure_litellm_config,
    ensure_models_ini,
    ensure_state_dirs,
    env_remote_url,
    write_marker,
)

_LITELLM_BEGIN = "; >>> AUTO-ENABLE-WHEN-LITELLM-AVAILABLE >>>"
_LITELLM_END   = "; <<< AUTO-ENABLE-WHEN-LITELLM-AVAILABLE <<<"


def _try_enable_litellm_blocks(ini_path: Path) -> int:
    """Activate any ``AUTO-ENABLE-WHEN-LITELLM-AVAILABLE`` block in
    ``ini_path`` when ``litellm`` is importable.

    For each fenced block we drop the BEGIN / END marker lines and
    strip a single leading ``"; "`` (or ``";\\t"``) from every line in
    between -- so a doubly-commented line like ``; ; model = ...``
    becomes a still-commented ``; model = ...`` in the active
    config (preserving the "uncomment to use" semantics of literal
    in-file comments). Returns the number of blocks rewritten; ``0``
    when litellm is missing, no markers exist, or every block is already
    expanded.
    """
    try:
        import litellm  # noqa: F401  -- presence check only
    except ImportError:
        return 0

    text = ini_path.read_text()
    if _LITELLM_BEGIN not in text or _LITELLM_END not in text:
        return 0
    yaml_path, seeded = ensure_litellm_config()
    if seeded:
        print(f"[*] no litellm_config.yaml found -- seeded default at {yaml_path}")

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
        "usage: llmstack install [--print] [--current | --next] "
        "[--external [URL]]"
    )


def _parse_args(args: list[str]) -> tuple[bool, str, str | None, bool]:
    """Parse ``install``'s flags.

    Returns ``(print_only, local_channel, external_url, want_external)``:

      * ``local_channel`` is ``current`` or ``next`` -- ignored when
        ``want_external`` is ``True``.
      * ``external_url`` is the explicit URL given to ``--external <url>``,
        if any. ``None`` when the flag was bare or absent.
      * ``want_external`` is ``True`` iff the user passed ``--external``
        (with or without a URL). The env-var fallback is layered in by
        the caller, not here, so this stays a pure CLI parse.

    ``--external`` accepts either ``--external <url>`` (separate arg) or
    ``--external=<url>``. Mutually exclusive with ``--current`` /
    ``--next`` -- mixing them raises ``SystemExit``.
    """
    print_only = False
    local_channel = "current"
    local_explicit = False
    external_url: str | None = None
    want_external = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--print", "-n"):
            print_only = True
        elif arg == "--next":
            local_channel = "next"
            local_explicit = True
        elif arg == "--current":
            local_channel = "current"
            local_explicit = True
        elif arg == "--external":
            want_external = True
            # Optional URL as next positional, but only if it looks like
            # a URL (not the next flag).
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
                f"[!] unknown arg to install: {arg} "
                "(try --print, --current, --next, --external, -h)"
            )
            raise SystemExit(2)
        i += 1

    if want_external and local_explicit:
        print(
            "[!] --external is mutually exclusive with --current / --next "
            "(external installs don't run local daemons).",
        )
        raise SystemExit(2)

    return print_only, local_channel, external_url, want_external


def _resolve_external_url(flag_url: str | None) -> str:
    """Pick the URL to bake into opencode.json + the channel marker.

    Precedence: explicit ``--external <url>`` arg > ``$LLMSTACK_REMOTE_URL``
    env var > :data:`DEFAULT_REMOTE_URL` (the local router). The default
    is what makes the "two projects on one host" workflow zero-config:
    ``llmstack install --external`` with nothing else set wires this
    project as a thin client of localhost so it can ride alongside
    whichever project actually owns the daemons.

    The ``$LLMSTACK_REMOTE_URL`` rung is what the activate hook
    populates when the user ``cd``-s into a project pinned to
    ``external`` -- so re-running ``llmstack install`` from an active
    shell inside an external project doesn't require the URL again.
    """
    if flag_url:
        return flag_url.rstrip("/")
    env = env_remote_url()
    if env:
        return env
    return DEFAULT_REMOTE_URL


def _fetch_remote_models_ini(url: str) -> str:
    """Pull the live ``models.ini`` from a remote llmstack router.

    External installs use the router as the source of truth for tier
    inventory: the same file the router parsed at startup is what the
    thin client renders ``opencode.json`` against, so tier names +
    descriptions agree with what the router actually serves. The fetch
    is also the canonical health check -- a 200 with parseable INI
    content proves both that the router is reachable and that the
    operator on the remote side has wired their config.

    Raises ``SystemExit`` (with a user-facing message) on any failure
    -- DNS, connection refused, non-2xx, empty body. The thin-client
    install is meaningless without the file, so we refuse to write a
    stale opencode.json from cached state. There is no client-side
    cache: every ``install`` re-fetches.
    """
    fetch_url = f"{url.rstrip('/')}/models.ini"
    req = urllib.request.Request(fetch_url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            if resp.status != 200:
                raise SystemExit(
                    f"[!] {fetch_url} returned HTTP {resp.status} -- "
                    "the remote router is up but doesn't have a "
                    "models.ini. Run `llmstack install` on the router "
                    "host to seed one, then retry here."
                )
            charset = resp.headers.get_content_charset() or "utf-8"
            text = resp.read().decode(charset)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"[!] {fetch_url} returned HTTP {e.code} {e.reason}.\n"
            "    is the remote running an llmstack version with "
            "GET /models.ini? (added in v3.x)"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SystemExit(
            f"[!] failed to reach {fetch_url}: {e}\n"
            "    check the URL, the network path, and that the remote "
            "router is up."
        ) from e

    if not text.strip():
        raise SystemExit(
            f"[!] {fetch_url} returned an empty body -- nothing to "
            "render opencode.json from."
        )
    return text


def run(args: list[str]) -> int:
    try:
        print_only, local_channel, external_url_arg, want_external = _parse_args(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0

    # Env-var fallback: ``LLMSTACK_REMOTE_URL`` set without ``--external``
    # still implies external mode. The activate hook re-exports this
    # var from the channel marker when the user ``cd``-s into an
    # external project, so re-running ``llmstack install`` from inside
    # an active shell doesn't need the URL or the flag again.
    if not want_external and env_remote_url() is not None:
        want_external = True

    if want_external:
        remote = _resolve_external_url(external_url_arg)
        channel: str = "external"
    else:
        remote = None
        channel = local_channel

    # Source of the INI is mode-dependent. Local mode reads (and
    # seeds-if-missing) the per-project file. External mode pulls the
    # router's live copy on every install -- the thin client never
    # keeps a local models.ini, since that would just be a stale
    # mirror of the router's truth.
    ini_text: str | None = None
    ini_source_label: str
    if remote is not None:
        # Flush so the "fetching" line lands before the network call;
        # otherwise an error written to stderr from inside
        # _fetch_remote_models_ini races ahead of buffered stdout and
        # the user sees the failure message before the "what we're
        # doing" message.
        print(f"[*] fetching models.ini from {remote}/models.ini ...", flush=True)
        ini_text = _fetch_remote_models_ini(remote)
        print(f"[OK] {len(ini_text.splitlines())} lines from {remote}")
        ini_source_label = f"{remote}/models.ini"
    else:
        ini_path, seeded = ensure_models_ini()
        if seeded:
            print(f"[*] no models.ini found -- seeded default at {ini_path}")
            enabled = _try_enable_litellm_blocks(ini_path)
            if enabled:
                print(
                    f"[*] litellm detected -- enabled {enabled} litellm-backed "
                    f"tier block(s) in {ini_path}"
                )
            print("    edit it to taste, then re-run `llmstack install`.")
        ini_source_label = str(ini_path)

        # Reconcile litellm_config.yaml against the litellm-backed
        # tiers. Non-destructive: existing model_list entries (which
        # the user may have customised) are preserved verbatim;
        # we only append stubs for tiers that don't yet have one.
        try:
            yaml_path, added, rewrote = merge_litellm_config()
        except SystemExit as exc:
            print(f"[!] litellm_config.yaml merge skipped: {exc}")
        else:
            if added:
                print(
                    f"[*] litellm_config.yaml: added {len(added)} model_list "
                    f"entr{'y' if len(added) == 1 else 'ies'}: {', '.join(added)}"
                )
                if rewrote:
                    print(
                        "    note: PyYAML rewrite stripped any comments "
                        "in the file. Future installs leave the file "
                        "untouched unless new tiers appear."
                    )

    paths = ensure_state_dirs()

    if print_only:
        if remote is not None:
            print(f"# external mode (channel: external, remote: {remote}); llama-swap.yaml not used.")
            print()
        print("----- opencode.json -----")
        print(render_opencode(ini_text=ini_text, remote=remote))
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
            render=lambda p: Path(p).write_text(
                render_opencode(ini_text=ini_text, remote=remote)
            ),
            validate=validate_opencode,
        )
    finally:
        if prev is None:
            os.environ.pop("OPENCODE_INSTRUCTIONS", None)
        else:
            os.environ["OPENCODE_INSTRUCTIONS"] = prev
    print(f"[OK] installed {paths.opencode_json}")

    if remote is not None:
        write_marker(paths.default_marker, ChannelMark("external", remote))
        print(f"[OK] default channel: external (remote: {remote})")
    else:
        write_marker(paths.default_marker, ChannelMark(channel))
        print(f"[OK] default channel: {channel}")

    if remote is None:
        # Render llama-swap.yaml for the pinned channel. ``start`` and
        # ``restart`` consume this file as-is and never regenerate it
        # -- channel selection is install-only.
        print()
        print(f"[*] generating llama-swap.yaml -> {paths.llama_swap_yaml}")
        use_next = channel == "next"
        render_to(
            paths.llama_swap_yaml,
            render=lambda p: Path(p).write_text(render_yaml(use_next=use_next)),
            validate=validate_yaml,
        )
        print(f"[OK] wrote {paths.llama_swap_yaml}")

    print()
    print(f"[OK] opencode config generated from {ini_source_label}.")
    print()
    print(f"  config:       {paths.opencode_json}")
    print(f"  instructions: {paths.agents_local}")
    if remote is not None:
        print(f"  remote:       {remote}")
    else:
        print(f"  channel:      {channel}")
    print()
    print("Next:")
    if remote is not None:
        print("  llmstack start     # re-fetch /models.ini + drop into the client subshell")
    else:
        print("  llmstack start     # bring up the stack (uses the llama-swap.yaml just written)")
        print("  llmstack check     # snapshot configured GGUFs + drift check")
    return 0
