"""``llmstack configure`` -- generate the derived configs for a project.

``configure`` is the *output* half of project setup (``llmstack init``
is the input half). It reads the channel marker and editable files that
``init`` wrote and renders everything downstream:

  * ``.llmstack/opencode.json``       -- atomic render (tmp, validate, mv)
  * ``.llmstack/llama-swap.yaml``     -- local channels only
  * ``.llmstack/litellm_config.yaml`` -- reconciled (stubs appended for
    any litellm tier that lacks a ``model_list`` entry; local only)

The **channel** (``current`` / ``next`` / ``external``) is decided by
``llmstack init`` and persisted in ``.llmstack/default-channel``.
``configure`` reads that marker and never changes it. To switch
channels (e.g. local -> external or current -> next) re-run
``llmstack init [--force] [--current|--next|--external]``.

In **external** mode ``configure`` fetches ``models.ini`` live from the
remote router (``GET <url>/models.ini``) and renders ``opencode.json``
against that -- no local ``models.ini`` is needed and no
``llama-swap.yaml`` is written (the router owns the daemons). Re-run
``configure`` any time to pick up router-side edits.

In **local** mode ``configure`` reads ``.llmstack/models.ini`` (seeded
by ``init``). A missing ``models.ini`` is a hard error pointing back at
``init``.

``--print`` writes the rendered opencode.json to stdout instead of
files (still fetches the remote in external mode).
"""

from __future__ import annotations

import os
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
    ensure_state_dirs,
    models_ini_path,
    read_marker,
    resolve,
)


def _print_help() -> None:
    print("usage: llmstack configure [--print]")


def _fetch_remote_models_ini(url: str) -> str:
    """Pull the live ``models.ini`` from a remote llmstack router.

    Raises ``SystemExit`` (with a user-facing message) on any failure.
    """
    fetch_url = f"{url.rstrip('/')}/models.ini"
    req = urllib.request.Request(fetch_url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            if resp.status != 200:
                raise SystemExit(
                    f"[!] {fetch_url} returned HTTP {resp.status} -- "
                    "the remote router is up but doesn't have a models.ini. "
                    "Run `llmstack init && llmstack configure` on the router "
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
    print_only = False
    for arg in args:
        if arg in ("--print", "-n"):
            print_only = True
        elif arg in ("-h", "--help"):
            _print_help()
            return 0
        else:
            print(
                f"[!] unknown arg to configure: {arg} "
                "(try --print, -h)\n"
                "    to change channel or mode, re-run: "
                "llmstack init [--force] [--current|--next|--external]"
            )
            return 2

    # Read the channel marker written by `init`. Fail clearly if it's
    # missing -- configure has no channel flags of its own.
    paths_snap = resolve()
    mark = read_marker(paths_snap.default_marker)
    if mark is None:
        print(
            "[!] no channel marker found -- run `llmstack init` first "
            "to set up this project, then re-run `llmstack configure`."
        )
        return 2

    channel = mark.channel
    remote: str | None = mark.url.rstrip("/") if (channel == "external" and mark.url) else None
    if channel == "external" and not remote:
        remote = DEFAULT_REMOTE_URL

    # Source of the INI is mode-dependent.
    ini_text: str | None = None
    ini_source_label: str
    if channel == "external":
        assert remote is not None
        print(f"[*] fetching models.ini from {remote}/models.ini ...", flush=True)
        ini_text = _fetch_remote_models_ini(remote)
        print(f"[OK] {len(ini_text.splitlines())} lines from {remote}")
        ini_source_label = f"{remote}/models.ini"
    else:
        ini_path = models_ini_path()
        if not ini_path.is_file():
            print(
                f"[!] models.ini not found at {ini_path}\n"
                "    run `llmstack init` first to seed the project's "
                "input files, then re-run `llmstack configure`."
            )
            return 2
        ini_source_label = str(ini_path)

        # Reconcile litellm_config.yaml against the litellm-backed
        # tiers. Non-destructive: existing model_list entries are
        # preserved verbatim; we only append stubs for new tiers.
        try:
            _yaml_path, added, rewrote = merge_litellm_config()
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
                        "in the file. Future runs leave the file "
                        "untouched unless new tiers appear."
                    )

    paths = ensure_state_dirs()

    if print_only:
        if channel == "external":
            print(f"# external mode (remote: {remote}); llama-swap.yaml not used.")
            print()
        print("----- opencode.json -----")
        print(render_opencode(ini_text=ini_text, remote=remote))
        return 0

    print("[1/2] opencode.json")
    # Point opencode at the per-project instructions.md when `init`
    # seeded one; otherwise fall back to the bundled template.
    instructions = paths.agents_local if paths.agents_local.is_file() else AGENTS_TEMPLATE
    prev = os.environ.get("OPENCODE_INSTRUCTIONS")
    os.environ["OPENCODE_INSTRUCTIONS"] = str(instructions)
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
    print(f"[OK] wrote {paths.opencode_json}")

    if channel != "external":
        # Render llama-swap.yaml for the pinned channel. ``start`` /
        # ``restart`` consume this file as-is and never regenerate it.
        print()
        print(f"[2/2] llama-swap.yaml -> {paths.llama_swap_yaml}")
        use_next = channel == "next"
        render_to(
            paths.llama_swap_yaml,
            render=lambda p: Path(p).write_text(render_yaml(use_next=use_next)),
            validate=validate_yaml,
        )
        print(f"[OK] wrote {paths.llama_swap_yaml}")
    else:
        print("[2/2] llama-swap.yaml -- skipped (external mode, router owns daemons)")

    print()
    print(f"[OK] opencode config generated from {ini_source_label}.")
    print()
    print(f"  config:       {paths.opencode_json}")
    print(f"  instructions: {instructions}")
    if channel == "external":
        print(f"  remote:       {remote}")
    else:
        print(f"  channel:      {channel}")
    print()
    print("Next:")
    if channel == "external":
        print("  llmstack start     # verify remote + drop into the client subshell")
    else:
        print("  llmstack start     # bring up the stack (uses the llama-swap.yaml just written)")
        print("  llmstack check     # snapshot configured GGUFs + drift check")
    return 0
