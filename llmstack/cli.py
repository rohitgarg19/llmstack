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

  install [--print] [--current | --next]
      Regenerate .llmstack/opencode.json (+ AGENTS.md copy) from models.ini
      and pin the default channel for the next `start`. Seeds a default
      models.ini in the work-dir on first run if none exists. --print
      writes the opencode config to stdout instead of files. Note:
      llama-swap.yaml is NOT touched here -- `start` owns that and
      regenerates it for the chosen channel on each launch.

  install-llama-swap [--force]
      (Re-)download the llama-swap Go binary into $LLMSTACK_BIN_DIR (default
      $XDG_DATA_HOME/llmstack/bin/). Setup runs this for you.

  download
      Download every GGUF named in models.ini (current + queued next) to
      the standard llama.cpp cache, in parallel, in the background.

  start [--current | --next] [--detach]
      Generate .llmstack/llama-swap.yaml for the chosen channel, bring up
      llama-swap (:10102) + auto-router (:10101). Default channel =
      whatever `install` pinned, else `current`. `--next` swaps any tier
      with hf_file_next. The yaml is regenerated on each fresh launch
      so it always matches the live models.ini; if the daemons are
      already up the running yaml is left alone.

      Subshell behaviour: if LLMSTACK_ACTIVE is already set (i.e. the
      activate hook has wired this shell up) `start` just brings up
      daemons and returns. Only when the env is not set does `start`
      drop you into a subshell with OPENCODE_CONFIG exported -- as a
      fallback for users who haven't run the activate hook yet.
      `--detach` skips the subshell unconditionally.

      When LLMSTACK_REMOTE_URL is set, no daemons are launched: this
      just verifies the remote /health endpoint (channel: external).

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

  restart [--current | --next] [--detach]
      stop + start. Convenient for cycling channels.

  reload
      Emit shell commands that re-export LLMSTACK_CHANNEL +
      OPENCODE_CONFIG and re-render the [llmstack:<project>] prompt
      prefix for the current channel marker. Pipe through eval to
      apply in-place (no nested subshell):
          eval "$(llmstack reload)"
      Useful after `start --next` switches channels in an
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
                          `http://10.0.0.5:10101`). When set, this host
                          becomes a thin client: no daemons, no llama-swap
                          binary, no GGUFs. `install` writes opencode.json
                          with baseURL pointing at the remote, and `start`
                          just verifies the remote (channel: external).
                          `setup`, `download`, `install-llama-swap` are
                          local-only and refuse when this is set.
  LLMSTACK_MODELS_INI     path to models.ini (default:
                          <work-dir>/.llmstack/models.ini).
  LLMSTACK_WORK_DIR       where .llmstack/ + logs/ live (default: $PWD
                          when invoked). Auto-exported by the activate
                          hook (`llmstack activate <shell>`) and by the
                          subshell `start` spawns, set to the project
                          root -- so commands work from any subdirectory
                          of an installed project. Without the hook,
                          run from the project root (or set this var).
                          Local daemons are singleton (ports 10101/10102)
                          and shared across projects.
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
  external   thin client of a remote llmstack via LLMSTACK_REMOTE_URL
             (medium-purple prompt prefix; the URL is shown alongside the
             project name in the prompt: `[llmstack:<project> <url>]`)
  shared     local daemons started by another project on this host
             (yellow prompt prefix)

Channel markers on disk (.llmstack/active-channel, .llmstack/default-channel):
  one line, format `<channel>[ <url>]`. The URL is only present for
  channel=external; the activate hook re-exports it as
  LLMSTACK_REMOTE_URL when you cd into the project, so you don't have to
  put the URL in your shell rc.

Variables exported by the activate hook (and the start fallback subshell):
  OPENCODE_CONFIG         path to the generated .llmstack/opencode.json
  LLMSTACK_WORK_DIR       absolute path to the project root (auto-detected
                          by walking up from $PWD looking for .llmstack/)
  LLMSTACK_CHANNEL        current | next | external | shared
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
            f"    hint: pip install -e . (or pipx install .) to install llmstack's deps"
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
