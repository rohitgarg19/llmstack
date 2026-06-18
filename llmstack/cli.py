"""``llmstack`` console-script entry point.

This is the Python replacement for ``llmstack.sh``. It does only one
thing: parse the action word, look up the matching ``commands.<action>``
module, and call its ``run(args)`` function. Every action implements its
own flag parsing so help text and error messages stay close to the
behaviour they describe.

  llmstack <action> [args...]

For machine readers / shell completions, ``llmstack help`` prints the
full action table; ``llmstack <action> -h`` prints per-action help.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from llmstack import __version__

USAGE = """\
llmstack - multi-tier local LLM stack (llama-swap + auto-router + opencode wiring)

Does NOT touch ~/.config/opencode/opencode.json. Instead, the generated
opencode config lives at <work-dir>/.llmstack/opencode.json, and the
activate hook (`llmstack activate <shell>`) auto-exports OPENCODE_CONFIG
whenever you cd into a project that has a `.llmstack/`. Inside that
hooked shell, `opencode` picks up our config; in any other terminal,
opencode keeps using your global setup unchanged.

Usage:
  llmstack <action> [options]

Actions:
  setup [--skip-download] [--skip-wait]
      First-time walkthrough: kick off GGUF downloads, wait for them, install
      the llama-swap binary, print the shell activation hook, check opencode.

  init [--force] [--current | --next | --external [URL]]
      Seed a fresh .llmstack/ in the CURRENT directory (never a parent
      project, even inside a hook-active shell), decide the channel, and
      copy the editable input files into it.

      Channel flags (written to .llmstack/default-channel; read by
      configure / start / status / the activate hook):
        --current (default)  local stack, canonical channel
        --next               local stack, queued-upgrade channel
        --external [URL]     thin client of a remote router. URL
                             precedence: flag arg > $LLMSTACK_REMOTE_URL
                             > http://127.0.0.1:10101 (local router,
                             for "two projects, one host" zero-config).

      Input files copied: models.ini, instructions.md, agents/*.md,
      litellm_config.yaml. Existing files are kept as-is by default.

      --force resets the project completely: re-copies every input file
      from the bundled templates AND clears previously generated outputs
      (opencode.json, llama-swap.yaml, channel markers) so the next
      `configure` starts from a clean slate. Use --force when switching
      a project between local and external mode.

      $LLMSTACK_REMOTE_URL set without --external still implies
      --external (the activate hook re-exports it when you cd into an
      external project, so re-running init from inside an active shell
      doesn't need the URL or the flag again).

  configure [--print]
      Generate .llmstack/opencode.json (and llama-swap.yaml for local
      channels) from the inputs `init` seeded. Reads the channel from
      .llmstack/default-channel (written by `init`) -- configure has no
      channel flags of its own. To change channel or mode, re-run:
        llmstack init [--force] [--current|--next|--external]

      In external mode, fetches models.ini live from the remote router
      on every run; no local models.ini is needed. In local mode, reads
      .llmstack/models.ini (seeded by init).

      `--print` writes the rendered opencode.json to stdout instead of
      files (still fetches the remote in external mode).

  install-llama-swap [--force]
      (Re-)download the llama-swap Go binary into $LLMSTACK_BIN_DIR (default
      $XDG_DATA_HOME/llmstack/bin/). Setup runs this for you.

  download
      Download every GGUF named in models.ini (current + queued next) to
      the standard llama.cpp cache, in parallel, in the background.

  start [--detach] [--host HOST] [--port PORT]
      Bring up llama-swap (:10102) + auto-router (:10101) using the
      .llmstack/llama-swap.yaml that `configure` wrote. Channel is
      whatever `configure` pinned (else `current`); selection is a
      configure-time decision -- `start` does not accept --current /
      --next and does not regenerate the yaml. To change channels or
      pick up models.ini edits: `llmstack configure [--current|--next]`
      then `llmstack restart`.

      --host HOST overrides the address the router listens on (default
      from models.ini, typically 127.0.0.1). Use 0.0.0.0 to expose on
      all interfaces.

      --port PORT overrides the router port (default from models.ini,
      typically 10101).

      Both --host and --port are persisted to models.ini so subsequent
      starts reuse them without re-specifying.

      Subshell behaviour: if LLMSTACK_ACTIVE is already set (i.e. the
      activate hook has wired this shell up) `start` just brings up
      daemons and returns. Only when the env is not set does `start`
      drop you into a subshell with OPENCODE_CONFIG exported -- as a
      fallback for users who haven't run the activate hook yet.
      `--detach` skips the subshell unconditionally.

      When the project is configured with channel=external (see
      `configure --external`), no daemons are launched: this just
      verifies the pinned remote `GET /models.ini` (which doubles as
      the router's health check -- there's no separate /health route).

  activate <zsh|bash|powershell>
      Write the auto-activation hook to ~/.<shell>_llmstack_hook and
      print a `source` line to stdout, so

          eval "$(llmstack activate zsh)"

      both regenerates the file and turns the hook on in the current
      shell. Paste the same line into your shell rc to make it stick:
          # ~/.zshrc
          eval "$(llmstack activate zsh)"
      The hook walks up from $PWD on every prompt, finds the nearest
      .llmstack/opencode.json, and exports OPENCODE_CONFIG +
      LLMSTACK_WORK_DIR + LLMSTACK_CHANNEL accordingly. Walks back out
      when you cd away. There is no separate `shell` action -- this is
      the shell action.

  stop
      Stop the router + llama-swap (and any orphaned llama-server children).

  restart [--detach]
      stop + configure + start. Re-reads models.ini and agent prompts
      on every restart, so edits land without a separate `configure`
      step. Channel is whatever `init` pinned in default-channel.
      Flags (--detach, --host, --port) are forwarded to `start`.

  reload
      Emit shell commands that re-export LLMSTACK_CHANNEL +
      OPENCODE_CONFIG and re-render the [llmstack:<project>] prompt
      prefix for the current channel marker. Pipe through eval to
      apply in-place (no nested subshell):
          eval "$(llmstack reload)"
      Useful after `configure --next && restart` switches channels in an
      already-active shell -- the activate hook only refreshes on
      chpwd, so without this the prompt would lag until your next cd.

  status
      Show channel, pids, /v1/models, loaded llama-server processes.

  check [args]
      Snapshot configured GGUFs + flag drift between models.ini and
      llama-swap.yaml.

  help | -h | --help
      This message.

  version | --version
      Print the package version and exit.

Environment overrides:
  LLMSTACK_REMOTE_URL     base URL of a *remote* llmstack router (e.g.
                          `http://10.0.0.5:10101`). Picked up by
                          `configure` as an alternative to passing
                          `--external <url>`; once `configure` runs,
                          the channel + URL are persisted in
                          .llmstack/default-channel and that file is
                          the source of truth (the env var is only
                          re-exported by the activate hook for
                          downstream callers).
  LLMSTACK_MODELS_INI     path to models.ini (default:
                          <work-dir>/.llmstack/models.ini).
  LLMSTACK_WORK_DIR       where .llmstack/ + logs/ live (default: $PWD
                          when invoked). Auto-exported by the activate
                          hook (`llmstack activate <shell>`) and by the
                          subshell `start` spawns, set to the project
                          root -- so commands work from any subdirectory
                          of a configured project. Without the hook,
                          run from the project root (or set this var).
                          Local daemons are singleton (ports 10101/10102);
                          to consume them from a second project on the
                          same host, configure that project --external.
  LLMSTACK_DATA_DIR       persistent user-data root (default:
                          $XDG_DATA_HOME/llmstack). Where the binary lives.
  LLMSTACK_BIN_DIR        override just the binary location.
  OPENCODE_CONFIG_DIR     where to write opencode.json (default: .llmstack/).
  LLAMA_SWAP_VERSION      pin a specific llama-swap release (e.g. v211).
  HF_TOKEN                authenticate model downloads (faster rate limits).
  LLMSTACK_SHELL          shell to spawn from `start` when no active env
                          is detected (default: $SHELL).

Channel labels (LLMSTACK_CHANNEL):
  current    local stack, canonical channel (steel-blue prompt prefix)
  next       local stack, queued-upgrade channel (orange prompt prefix)
  external   thin client of an llmstack router (medium-purple prompt
             prefix; the URL is shown alongside the project name in the
              prompt: `[llmstack:<project> <url>]`). The URL is pinned at
              configure time -- typically a remote host, but defaults to
             the local router so two projects on one host can share a
             single set of daemons cleanly.

Channel markers on disk (.llmstack/active-channel, .llmstack/default-channel):
  one line, format `<channel>[ <url>]`. The URL is only present for
  channel=external; the activate hook re-exports it as
  LLMSTACK_REMOTE_URL when you cd into the project, so you don't have to
  put the URL in your shell rc.

Variables exported by the activate hook (and the start fallback subshell):
  OPENCODE_CONFIG         path to the generated .llmstack/opencode.json
  LLMSTACK_WORK_DIR       absolute path to the project root (auto-detected
                          by walking up from $PWD looking for .llmstack/)
  LLMSTACK_CHANNEL        current | next | external
  LLMSTACK_ACTIVE         "1" while the env is wired up
  LLMSTACK_REMOTE_URL     set when channel == external, from the marker file
  LLMSTACK_ROOT           absolute path to the llmstack package (start only)
"""


def _print_help() -> None:
    sys.stdout.write(USAGE)


def _print_version() -> None:
    print(f"llmstack {__version__}")


def _load_action(action: str) -> Callable[[list[str]], int]:
    """Resolve ``action`` to a ``run(args)`` callable, lazy-importing the module."""
    aliases = {
        "download-models":     "download",
        "check-models":        "check",
    }
    name = aliases.get(action, action)
    name = name.replace("-", "_")
    target = f"llmstack.commands.{name}"

    from importlib import import_module

    try:
        module = import_module(target)
    except ModuleNotFoundError as e:
        # Only swallow the error when the *action module itself* is missing.
        # Transitive ImportErrors (e.g. an uninstalled third-party dep) must
        # surface, otherwise we mislead the user with "unknown action".
        if e.name == target:
            raise SystemExit(f"[!] unknown action: {action}\n\nrun: llmstack help") from None
        raise SystemExit(
            f"[!] action '{action}' failed to load: missing dependency '{e.name}'\n"
            f"    hint: pip install -e . (or pipx install .) to install opencode-llmstack's deps"
        ) from e

    run = getattr(module, "run", None)
    if not callable(run):
        raise SystemExit(f"[!] action '{action}' is missing run() -- bug in llmstack")
    return run


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)

    if not args or args[0] in ("help", "-h", "--help"):
        _print_help()
        return 0
    if args[0] in ("version", "-V", "--version"):
        _print_version()
        return 0

    action, rest = args[0], args[1:]
    run = _load_action(action)
    try:
        rc = run(rest)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n[!] interrupted", file=sys.stderr)
        return 130
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
