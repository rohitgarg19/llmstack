#!/usr/bin/env bash
# llmstack.sh - single entry point for all stack operations.
#
# Usage:
#   bash llmstack.sh <action> [options]
#
# This stack does NOT touch ~/.config/opencode/opencode.json. Instead, the
# generated opencode config lives at llmstack/.llmstack/opencode.json, and the
# `start` and `shell` actions drop you into a subshell with OPENCODE_CONFIG
# pointed at it. Inside that subshell, `opencode` picks up our config; in
# any other terminal, opencode keeps using your global setup unchanged.
#
# Actions:
#   setup [--skip-download] [--skip-wait]
#       First-time install: download every GGUF in models.ini, wait for them
#       to finish, then generate + install both config files. Does NOT start
#       the stack.
#   install [--print]
#       Generate llama-swap.yaml + .llmstack/opencode.json from models.ini, and
#       ensure bin/llama-swap is up to date. --print writes both configs to
#       stdout instead of installing them.
#   install-llama-swap [--force]
#       Just (re)download the llama-swap binary for this OS/arch.
#   download
#       Download every GGUF named in models.ini (current + queued next) to
#       the standard llama.cpp cache, in parallel, in the background.
#   start [--current | --next] [--detach]
#       Bring up llama-swap (:10102) + auto-router (:10101) AND drop into a
#       subshell with OPENCODE_CONFIG set. Default channel is `current`
#       (the canonical llama-swap.yaml). `--next` swaps any tier with
#       hf_file_next set in models.ini. `--detach` skips the subshell
#       (daemons only; old behavior).
#   shell
#       Drop into the env-prepared subshell without (re)starting daemons.
#       Useful for opening more terminals into the same stack. Refuses if
#       the daemons aren't running.
#   activate <zsh|bash>
#       Print shell hook code that auto-activates llmstack whenever you
#       `cd` into a project that has a `.llmstack/` dir (or any subdir of
#       one). Drop the eval into your rc file once and forget about
#       `llmstack.sh shell` forever:
#           # ~/.zshrc
#           eval "$(bash /abs/path/to/llmstack/llmstack.sh activate zsh)"
#       The hook walks up from $PWD on each prompt, exports OPENCODE_CONFIG
#       + LLMSTACK_PROJECT + LLMSTACK_CHANNEL, and prefixes your prompt
#       with [llmstack:<project>]. Stepping out of the project directory
#       reverses everything.
#   stop
#       Stop the router + llama-swap (and any orphaned llama-server children).
#   restart [--current | --next] [--detach]
#       stop + start. Convenient for cycling channels.
#   status
#       Show channel, pids, /v1/models, loaded llama-server processes.
#   check [args passed through to src/check_models.py]
#       Snapshot configured GGUFs + flag drift between models.ini and
#       llama-swap.yaml.
#   help | -h | --help
#       This message.
#
# Examples:
#   bash llmstack.sh setup                   # first-time install
#   bash llmstack.sh start                   # daemons + drop into subshell
#   bash llmstack.sh start --next            # try queued upgrades, same UX
#   bash llmstack.sh start --detach          # daemons only, no subshell
#   bash llmstack.sh shell                   # second terminal into the stack
#   bash llmstack.sh restart --next          # cycle into the next channel
#   bash llmstack.sh status                  # what's up
#
# Environment overrides:
#   OPENCODE_CONFIG_DIR     where to write opencode.json (default: .llmstack/)
#   LLAMA_SWAP_VERSION      pin a specific llama-swap release (e.g. v211)
#   HF_TOKEN                authenticate model downloads (faster rate limits)
#   LLMSTACK_SHELL          shell to spawn in `start`/`shell` (default: $SHELL)
#   LLMSTACK_WORK_DIR       where .llmstack/ + logs/ go (default: $PWD when invoked).
#                           This means you can `cd` into any project and run
#                           `bash /abs/path/to/llmstack.sh start` -- the per-
#                           project .llmstack/ (opencode.json + AGENTS.md + pids)
#                           is created right there. Daemons are singleton
#                           (ports 10101/10102) and shared across projects.
#
# Variables exported into the spawned subshell:
#   OPENCODE_CONFIG         path to the generated .llmstack/opencode.json
#   LLMSTACK_CHANNEL        current | next
#   LLMSTACK_ACTIVE         "1" while inside the subshell
#   LLMSTACK_ROOT           absolute path to the llmstack/ directory

set -euo pipefail

# --- common paths and helpers -----------------------------------------------

# --- source-anchored: where the script + assets live ----------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # llmstack/
ROOT="$HERE"
PROJ="$(cd "$ROOT/.." && pwd)"                                 # repo root

BIN_DIR="$ROOT/bin"
SRC_DIR="$ROOT/src"
PYTHON="$ROOT/.venv/bin/python"
LLAMA_SWAP="$BIN_DIR/llama-swap"

GEN_YAML="$SRC_DIR/gen_llama_swap_yaml.py"
GEN_JSON="$SRC_DIR/gen_opencode_config.py"
ROUTER="$SRC_DIR/router.py"
TIERS="$SRC_DIR/tiers.py"
CHECK_MODELS="$SRC_DIR/check_models.py"
INI="$PROJ/models.ini"

AGENTS_TEMPLATE="$ROOT/AGENTS.md"
DEFAULT_CONFIG="$ROOT/llama-swap.yaml"

REPO_LLAMA_SWAP="mostlygeek/llama-swap"

# --- work-anchored: per-cwd state (this is the portable bit) ---------------
# WORK_DIR defaults to $PWD at the time llmstack.sh is invoked, so you can
# `cd` into any project and get a project-local .llmstack/. Override with
# LLMSTACK_WORK_DIR to pin it somewhere stable (e.g. when called from cron
# or from inside an editor that has a weird cwd).
WORK_DIR="${LLMSTACK_WORK_DIR:-$PWD}"
LLMSTACK_DIR="$WORK_DIR/.llmstack"
LOG_DIR="$LLMSTACK_DIR/logs"

# Two channel markers, intentionally separate:
#   ACTIVE_MARKER  - the channel the daemons are CURRENTLY running.
#                    Written by start, removed by stop. Reflects live state.
#   DEFAULT_MARKER - the channel install pinned as the default.
#                    Persistent across stop/start; consulted by `start` when
#                    no --next/--current flag is given.
ACTIVE_MARKER="$LLMSTACK_DIR/active-channel"
DEFAULT_MARKER="$LLMSTACK_DIR/default-channel"
AGENTS_LOCAL="$LLMSTACK_DIR/AGENTS.md"

# opencode config: written into .llmstack/ by `install`, exported via OPENCODE_CONFIG
# in the subshell that `start`/`shell` spawn. The user's global
# ~/.config/opencode/opencode.json is never modified.
OPENCODE_JSON_DIR="${OPENCODE_CONFIG_DIR:-$LLMSTACK_DIR}"
OPENCODE_JSON="$OPENCODE_JSON_DIR/opencode.json"

# Actions that touch state (install, start, etc.) call _ensure_state_dirs before
# writing. We deliberately do NOT mkdir at script-load time -- otherwise read-only
# actions like `help` and `activate` would pollute every directory the user
# happens to run them from with an empty .llmstack/.
_ensure_state_dirs() {
    mkdir -p "$LOG_DIR" "$LLMSTACK_DIR"
}

require_python() {
    [[ -x "$PYTHON" ]] || die "missing venv at $PYTHON (see UPGRADING.md > 'Upgrading the Python toolchain')"
}

require_ini() {
    [[ -f "$INI" ]] || die "missing $INI"
}

die() { echo "[!] $*" >&2; exit 1; }

is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

# Probe a port for a /health-style 200 response. Used to detect daemons
# launched from a different project's .llmstack/ (since LLMSTACK_DIR is per-cwd, the
# local pid file may be missing even when the daemons are alive).
port_responds() {
    local url="$1"
    curl -fsS --max-time 2 "$url" >/dev/null 2>&1
}

# --- action: help -----------------------------------------------------------

cmd_help() {
    # Print the file's leading comment block (everything between the shebang
    # and the first non-comment line), stripping the leading "# ".
    awk '
        NR==1 { next }                            # skip shebang
        /^[[:space:]]*$/ && !comment_started { next }
        /^#/ { comment_started=1; sub(/^# ?/, ""); print; next }
        { exit }                                  # stop at first non-comment line
    ' "$0"
}

# --- action: install-llama-swap --------------------------------------------

cmd_install_llama_swap() {
    local force=0
    for arg in "$@"; do
        case "$arg" in
            --force|-f) force=1 ;;
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to install-llama-swap: $arg (try --force, -h)" ;;
        esac
    done

    local target="$BIN_DIR/llama-swap"
    case "$(uname -s)" in
        Darwin)  os="darwin"  ;;
        Linux)   os="linux"   ;;
        FreeBSD) os="freebsd" ;;
        *) die "unsupported OS: $(uname -s) (need Darwin/Linux/FreeBSD)" ;;
    esac
    case "$(uname -m)" in
        arm64|aarch64) arch="arm64" ;;
        x86_64|amd64)  arch="amd64" ;;
        *) die "unsupported arch: $(uname -m) (need arm64 or x86_64)" ;;
    esac
    [[ "$os" != "freebsd" || "$arch" == "amd64" ]] || die "no llama-swap release for $os/$arch"

    local tag
    if [[ -n "${LLAMA_SWAP_VERSION:-}" ]]; then
        tag="$LLAMA_SWAP_VERSION"
        echo "[*] version: $tag (from \$LLAMA_SWAP_VERSION)"
    else
        echo "[*] resolving latest release tag from github.com/$REPO_LLAMA_SWAP..."
        tag="$(curl -fsSL "https://api.github.com/repos/$REPO_LLAMA_SWAP/releases/latest" \
                | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null \
                || true)"
        [[ -n "$tag" ]] || die "could not resolve latest release tag"
        echo "[*] latest release: $tag"
    fi

    local num="${tag#v}"
    local asset="llama-swap_${num}_${os}_${arch}.tar.gz"
    local url="https://github.com/$REPO_LLAMA_SWAP/releases/download/$tag/$asset"

    if [[ -x "$target" && "$force" -eq 0 ]]; then
        local current
        if current="$("$target" --version 2>/dev/null | head -1 || true)"; then
            if echo "$current" | grep -qE "version: $num\b"; then
                echo "[=] already installed: $target"
                echo "         $current"
                echo "    (re-run with --force to redownload)"
                return 0
            fi
            echo "[*] currently installed: $current"
            echo "    upgrading to $tag"
        fi
    fi

    mkdir -p "$BIN_DIR"
    local tmp
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN

    echo "[*] downloading $asset"
    echo "    from $url"
    curl -fL --progress-bar "$url" -o "$tmp/$asset"

    echo "[*] extracting"
    tar -xzf "$tmp/$asset" -C "$tmp" llama-swap
    [[ -f "$tmp/llama-swap" ]] || {
        echo "[!] tarball did not contain a top-level 'llama-swap' file" >&2
        tar -tzf "$tmp/$asset" | head
        return 1
    }

    mv "$tmp/llama-swap" "$target.new"
    chmod +x "$target.new"
    mv "$target.new" "$target"

    echo "[OK] installed $target ($os/$arch)"
    "$target" --version 2>&1 | sed 's/^/     /'
}

# --- action: install (configs + binary) ------------------------------------

_render_install() {
    # Generate -> validate -> atomic mv.
    local label="$1" generator="$2" target="$3" validator="$4"
    local target_dir
    target_dir="$(dirname "$target")"
    mkdir -p "$target_dir"
    local tmp
    tmp="$(mktemp "$target_dir/.${label}.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -f '$tmp'" RETURN
    "$PYTHON" "$generator" "$tmp"
    "$PYTHON" -c "$validator" "$tmp"
    mv "$tmp" "$target"
    chmod 644 "$target"
    echo "[OK] installed $target"
}

cmd_install() {
    require_python; require_ini
    [[ -f "$GEN_YAML" ]] || die "missing $GEN_YAML"
    [[ -f "$GEN_JSON" ]] || die "missing $GEN_JSON"
    _ensure_state_dirs

    local print_only=0
    local default_channel="current"
    for arg in "$@"; do
        case "$arg" in
            --print|-n)        print_only=1 ;;
            --next)            default_channel="next" ;;
            --current)         default_channel="current" ;;
            -h|--help)         cmd_help; return 0 ;;
            *) die "unknown arg to install: $arg (try --print, --current, --next, -h)" ;;
        esac
    done

    if [[ "$print_only" -eq 1 ]]; then
        echo "----- llama-swap.yaml -----"
        "$PYTHON" "$GEN_YAML" -
        echo
        echo "----- opencode.json -----"
        "$PYTHON" "$GEN_JSON" -
        return 0
    fi

    echo "[1/3] llama-swap binary"
    cmd_install_llama_swap
    echo

    echo "[2/3] llama-swap.yaml"
    _render_install "llama-swap.yaml" "$GEN_YAML" "$ROOT/llama-swap.yaml" \
        "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"
    echo

    echo "[3/3] opencode.json + AGENTS.md"
    # Copy the AGENTS.md template into .llmstack/ first so the generated
    # opencode.json can reference it by absolute path. Keeping AGENTS.md
    # alongside opencode.json makes the .llmstack/ folder self-contained --
    # users can hand-edit .llmstack/AGENTS.md for one-off, project-specific
    # instructions without polluting the source-of-truth template.
    if [[ -f "$AGENTS_TEMPLATE" ]]; then
        cp "$AGENTS_TEMPLATE" "$AGENTS_LOCAL"
        chmod 644 "$AGENTS_LOCAL"
        echo "[OK] copied $AGENTS_TEMPLATE"
        echo "         -> $AGENTS_LOCAL"
    else
        echo "[!] AGENTS.md template not found at $AGENTS_TEMPLATE; skipping copy"
    fi

    # Point the generator at the .llmstack/ copy of AGENTS.md (absolute path)
    # so opencode loads it regardless of where the user invokes from.
    OPENCODE_INSTRUCTIONS="$AGENTS_LOCAL" \
    _render_install "opencode.json" "$GEN_JSON" "$OPENCODE_JSON" \
        "import json,sys; json.load(open(sys.argv[1]))"

    # Persist the channel choice so a bare `bash llmstack.sh start` later on
    # picks it up automatically. `start --current/--next` still overrides at
    # runtime; install just sets the default.
    echo "$default_channel" > "$DEFAULT_MARKER"
    echo "[OK] default channel pinned to '$default_channel' (in $DEFAULT_MARKER)"

    cat <<EOF

[OK] both configs derived from $INI.
     re-run any time you edit models.ini.

opencode wiring:
  config:        $OPENCODE_JSON
  instructions:  $AGENTS_LOCAL
  default chan:  $default_channel  (pinned by this install; override with start --next/--current)
  Picked up via OPENCODE_CONFIG; either via the subshell spawned by
    bash llmstack.sh start
    bash llmstack.sh shell
  or automatically via the activate hook (one-time setup):
    eval "\$(bash $0 activate zsh)"   # add to ~/.zshrc
    eval "\$(bash $0 activate bash)"  # add to ~/.bashrc
  Your ~/.config/opencode/opencode.json is intentionally NOT modified.

Per-project layout:
  Everything above lives under \$PWD/.llmstack/, so invoking llmstack.sh from
  a different project gives you a separate opencode config there.
  Daemons are singleton (one set at a time, on ports 10101/10102).
  Only one channel runs at any moment.

Next:
  bash llmstack.sh start            # bring up the pinned channel + drop into shell
  bash llmstack.sh check            # snapshot configured GGUFs + drift check
EOF
}

# --- action: download -------------------------------------------------------

cmd_download() {
    require_python; require_ini
    [[ -f "$TIERS" ]] || die "missing $TIERS"
    _ensure_state_dirs

    for arg in "$@"; do
        case "$arg" in
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to download: $arg" ;;
        esac
    done

    # Prefer llama-completion (modern split: chat=llama-cli, one-shot=llama-completion).
    # Fall back to llama-cli on older llama.cpp installs.
    local llama_bin=""
    local candidate path
    for candidate in llama-completion llama-cli; do
        if path="$(command -v "$candidate" 2>/dev/null)"; then
            llama_bin="$path"
            break
        fi
    done
    [[ -n "$llama_bin" ]] || die "neither llama-completion nor llama-cli found in PATH (brew install llama.cpp)"

    echo "[*] downloader: $llama_bin"
    echo "[*] cache:      ~/.cache/huggingface/hub  (default for llama.cpp)"
    echo "[*] inventory:  $INI  (via $TIERS --downloads)"
    local hf_flag=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "[*] HF_TOKEN set (faster rate limits)"
        hf_flag=(--hf-token "$HF_TOKEN")
    else
        echo "[*] no HF_TOKEN (rate-limited unauthenticated downloads)"
    fi
    echo

    _run_dl() {
        local tag="$1" repo="$2" file="$3" label="$4"
        local log="$LOG_DIR/dl-${tag}.log"
        printf "[*] %-32s %-7s %s / %s\n" "$tag" "($label)" "$repo" "$file"
        echo "    log -> $log"
        local -a argv=(
            -hf "$repo" -hff "$file"
            --no-warmup -ngl 0 -c 256 -p ok -n 1
        )
        if [[ "${#hf_flag[@]}" -gt 0 ]]; then
            argv+=("${hf_flag[@]}")
        fi
        nohup "$llama_bin" "${argv[@]}" > "$log" 2>&1 < /dev/null &
        echo "    pid -> $!"
    }

    local count=0 tag repo file label
    while IFS=$'\t' read -r tag repo file label; do
        [[ -z "$tag" ]] && continue
        _run_dl "$tag" "$repo" "$file" "$label"
        count=$((count + 1))
    done < <("$PYTHON" "$TIERS" --downloads)

    [[ "$count" -gt 0 ]] || die "no download targets found in $INI"

    cat <<EOF

$count download(s) queued in the background.

Watch progress:
    tail -f $LOG_DIR/dl-*.log
    llama-cli -cl                        # lists completed cache entries

When you want to try queued upgrade targets without committing:
    bash llmstack.sh stop && bash llmstack.sh start --next
EOF
}

# --- action: setup ----------------------------------------------------------

cmd_setup() {
    local skip_download=0 skip_wait=0
    for arg in "$@"; do
        case "$arg" in
            --skip-download) skip_download=1 ;;
            --skip-wait)     skip_wait=1     ;;
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to setup: $arg (try --skip-download, --skip-wait, -h)" ;;
        esac
    done

    if [[ "$skip_download" -eq 0 ]]; then
        echo "[1/3] downloading required GGUFs..."
        cmd_download
        echo
    else
        echo "[1/3] (skipped) downloads"
        echo
    fi

    if [[ "$skip_download" -eq 0 && "$skip_wait" -eq 0 ]]; then
        echo "[2/3] waiting for downloads to finish..."
        echo "      (logs: $LOG_DIR/dl-*.log)"
        sleep 2
        while pgrep -f 'llama-(completion|cli) .*-hf ' >/dev/null 2>&1; do
            local n
            n=$(pgrep -f 'llama-(completion|cli) .*-hf ' | wc -l | tr -d ' ')
            echo "      $n download(s) still running..."
            sleep 10
        done
        echo "[OK] all downloads complete."
        echo
    else
        echo "[2/3] (skipped) wait"
        echo
    fi

    echo "[3/3] generating + installing configs..."
    cmd_install

    cat <<EOF

[OK] setup complete.

To start the stack:
  bash llmstack.sh start

To check what's configured + drift between models.ini and llama-swap.yaml:
  bash llmstack.sh check
EOF
}

# --- action: start ----------------------------------------------------------

cmd_start() {
    # Channel resolution order, highest priority first:
    #   1. explicit --current / --next on the command line
    #   2. .llmstack/default-channel pinned by the last `install`
    #   3. hard-coded fallback: "current"
    local channel=""
    local detach=0
    for arg in "$@"; do
        case "$arg" in
            --next)    channel="next"    ;;
            --current) channel="current" ;;
            --detach|--no-shell) detach=1 ;;
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to start: $arg (try --next, --current, --detach, -h)" ;;
        esac
    done
    if [[ -z "$channel" && -f "$DEFAULT_MARKER" ]]; then
        channel="$(<"$DEFAULT_MARKER")"
    fi
    [[ -z "$channel" ]] && channel="current"

    require_python
    [[ -x "$LLAMA_SWAP" ]] || die "missing $LLAMA_SWAP (run: bash llmstack.sh install)"
    [[ -f "$DEFAULT_CONFIG" || "$channel" == "next" ]] || die "missing $DEFAULT_CONFIG (run: bash llmstack.sh install)"
    [[ -f "$OPENCODE_JSON" ]] || die "no .llmstack/opencode.json in $WORK_DIR -- run: llmstack install"
    _ensure_state_dirs

    # Three states to detect:
    #   (a) local pid file says daemons are up      -> idempotent, channel-checked
    #   (b) port 10102 responds but no local pid    -> daemons started elsewhere
    #                                                  (e.g. another project's
    #                                                  .llmstack/); reuse them
    #   (c) nothing                                 -> launch fresh
    local launch_daemons=1
    local external_daemons=0
    if is_running "$LLMSTACK_DIR/llama-swap.pid"; then
        launch_daemons=0
        local live
        live="$(cat "$ACTIVE_MARKER" 2>/dev/null || echo current)"
        if [[ "$live" != "$channel" ]]; then
            cat >&2 <<EOF
[!] llama-swap is already running in '$live' channel; refusing to also
    start '$channel'. Stop the stack first:

      bash llmstack.sh stop
      bash llmstack.sh start --$channel
EOF
            return 1
        fi
    elif port_responds "http://127.0.0.1:10102/health"; then
        launch_daemons=0
        external_daemons=1
        echo "[*] daemons already up on :10101/:10102 (started elsewhere)"
        echo "    will reuse them. Use 'bash llmstack.sh stop' from any project to stop."
    fi

    local config="$DEFAULT_CONFIG"
    if [[ "$channel" == "next" && "$launch_daemons" -eq 1 ]]; then
        local queued
        queued="$("$PYTHON" "$TIERS" --downloads | awk -F'\t' '$4=="next" {print $1}' | sed 's/-next$//' | tr '\n' ' ')"
        if [[ -z "${queued// }" ]]; then
            echo "[!] no tiers have hf_file_next set in models.ini -- nothing to do." >&2
            echo "    add an hf_file_next line to a tier and re-run, or use --current." >&2
            return 1
        fi
        echo "[*] generating next-channel yaml -> $DEFAULT_CONFIG"
        echo "    queued upgrade tiers: $queued"
        "$PYTHON" "$GEN_YAML" --use-next "$DEFAULT_CONFIG"
        "$PYTHON" -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$DEFAULT_CONFIG"
    fi

    if [[ "$external_daemons" -eq 1 ]]; then
        echo "[*] channel: (external -- whatever the running daemons were started with)"
    else
        echo "[*] channel: $channel  ($(basename "$config"))"
    fi

    if [[ "$launch_daemons" -eq 1 ]]; then
        echo "[*] starting llama-swap on :10102"
        nohup "$LLAMA_SWAP" --config "$config" --listen 127.0.0.1:10102 \
            >"$LOG_DIR/llama-swap.log" 2>&1 &
        echo $! >"$LLMSTACK_DIR/llama-swap.pid"
        echo "$channel" >"$ACTIVE_MARKER"
        sleep 1
        if ! is_running "$LLMSTACK_DIR/llama-swap.pid"; then
            echo "[!] llama-swap failed to start. Check $LOG_DIR/llama-swap.log"
            rm -f "$LLMSTACK_DIR/llama-swap.pid" "$ACTIVE_MARKER"
            return 1
        fi
        echo "    pid $(cat "$LLMSTACK_DIR/llama-swap.pid")"

        echo "[*] starting router on :10101"
        LLAMA_SWAP_URL="http://127.0.0.1:10102" \
        ROUTER_HOST="127.0.0.1" ROUTER_PORT="10101" \
        nohup "$PYTHON" "$ROUTER" >"$LOG_DIR/router.log" 2>&1 &
        echo $! >"$LLMSTACK_DIR/router.pid"
        sleep 1
        if ! is_running "$LLMSTACK_DIR/router.pid"; then
            echo "[!] router failed to start. Check $LOG_DIR/router.log"
            rm -f "$LLMSTACK_DIR/router.pid"
            return 1
        fi
        echo "    pid $(cat "$LLMSTACK_DIR/router.pid")"
    elif [[ "$external_daemons" -eq 0 ]]; then
        echo "[=] llama-swap already running (pid $(cat "$LLMSTACK_DIR/llama-swap.pid"), channel $channel)"
        if is_running "$LLMSTACK_DIR/router.pid"; then
            echo "[=] router already running (pid $(cat "$LLMSTACK_DIR/router.pid"))"
        fi
    fi

    local other channel_label
    other=$([[ "$channel" == "current" ]] && echo next || echo current)
    if [[ "$external_daemons" -eq 1 ]]; then
        channel_label="external"
    else
        channel_label="$channel"
    fi
    cat <<EOF

[OK] stack is up (channel: $channel_label).

  router       http://127.0.0.1:10101     (OpenAI-compatible, "auto" routing)
  llama-swap   http://127.0.0.1:10102     (raw model endpoints + UI)

Try:
  curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'
  curl -s http://127.0.0.1:10101/health | jq '.tiers'

Logs:
  tail -f $LOG_DIR/llama-swap.log
  tail -f $LOG_DIR/router.log

Switch channel (requires stop first):
  bash llmstack.sh restart --$other

Stop:
  bash llmstack.sh stop
EOF

    if [[ "$detach" -eq 1 ]]; then
        return 0
    fi

    # If we're already inside an llmstack shell, reload rather than nest:
    #   - same channel: nothing to do, env is already correct
    #   - different channel: re-exec the shell so LLMSTACK_CHANNEL + prompt update
    if [[ "${LLMSTACK_ACTIVE:-}" == "1" ]]; then
        if [[ "${LLMSTACK_CHANNEL:-}" == "$channel" ]]; then
            echo "[=] already in the '$channel' llmstack shell -- no reload needed."
            return 0
        fi
        echo "[*] channel changed ($LLMSTACK_CHANNEL -> $channel); reloading shell..."
    fi

    _spawn_shell "$channel"
}

# --- action: shell ----------------------------------------------------------

cmd_shell() {
    for arg in "$@"; do
        case "$arg" in
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to shell: $arg" ;;
        esac
    done
    if ! is_running "$LLMSTACK_DIR/llama-swap.pid" && ! port_responds "http://127.0.0.1:10102/health"; then
        die "stack is not running. start it first: bash llmstack.sh start"
    fi
    local channel
    if [[ -f "$ACTIVE_MARKER" ]]; then
        channel="$(cat "$ACTIVE_MARKER")"
    else
        channel="external"
    fi
    _spawn_shell "$channel"
}

# Drop into an interactive subshell with OPENCODE_CONFIG and friends exported.
# Bash and zsh get a wrapper rcfile that sources the user's normal rc and
# then prefixes the prompt with [llmstack:<channel>]. Other shells get the
# env vars set but no prompt customization.
_spawn_shell() {
    local channel="$1"
    [[ -f "$OPENCODE_JSON" ]] || die "missing $OPENCODE_JSON (run: bash llmstack.sh install)"

    if [[ "${LLMSTACK_ACTIVE:-}" == "1" ]]; then
        echo "[!] already inside an llmstack shell (channel: ${LLMSTACK_CHANNEL:-?})." >&2
        echo "    exit this shell first if you want to start a new one." >&2
        return 1
    fi

    local user_shell="${LLMSTACK_SHELL:-${SHELL:-/bin/bash}}"
    local shell_name
    shell_name="$(basename "$user_shell")"

    cat <<EOF

[*] entering llmstack subshell (channel: $channel, shell: $shell_name)
    OPENCODE_CONFIG -> $OPENCODE_JSON
    alias 'llmstack' -> $0
    daemons keep running on exit; stop them with: llmstack stop
    second window?: llmstack shell

EOF

    local -a exec_argv=()
    case "$shell_name" in
        bash)
            local rcfile="$LLMSTACK_DIR/llmstack.bashrc"
            cat > "$rcfile" <<RC
# AUTO-GENERATED by llmstack.sh; sourced by the spawned subshell.
[ -f "\$HOME/.bashrc" ] && source "\$HOME/.bashrc"
alias llmstack='bash "$0"'
PS1="[llmstack:$channel] \${PS1:-\\\$ }"
RC
            exec_argv=("$user_shell" --rcfile "$rcfile" -i)
            ;;
        zsh)
            local zdotdir="$LLMSTACK_DIR/zdotdir"
            mkdir -p "$zdotdir"
            cat > "$zdotdir/.zshrc" <<RC
# AUTO-GENERATED by llmstack.sh; sourced by the spawned subshell.
[ -f "\$HOME/.zshrc" ] && source "\$HOME/.zshrc"
alias llmstack='bash "$0"'
PROMPT="[llmstack:$channel] \${PROMPT:-%# }"
RC
            exec_argv=(env "ZDOTDIR=$zdotdir" "$user_shell" -i)
            ;;
        *)
            exec_argv=("$user_shell" -i)
            ;;
    esac

    exec env \
        OPENCODE_CONFIG="$OPENCODE_JSON" \
        LLMSTACK_CHANNEL="$channel" \
        LLMSTACK_ACTIVE="1" \
        LLMSTACK_ROOT="$ROOT" \
        "${exec_argv[@]}"
}

# --- action: stop -----------------------------------------------------------

cmd_stop() {
    for arg in "$@"; do
        case "$arg" in
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to stop: $arg" ;;
        esac
    done

    local name pid pid_file _i
    for name in router llama-swap; do
        pid_file="$LLMSTACK_DIR/$name.pid"
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "[*] stopping $name (pid $pid)"
                kill "$pid" 2>/dev/null || true
                for _i in 1 2 3 4 5 6 7 8 9 10; do
                    kill -0 "$pid" 2>/dev/null || break
                    sleep 0.5
                done
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done

    # Cross-project safety net: if the daemons were started from a different
    # project's .llmstack/ (and so we have no local pid files), fall back to
    # killing by process name. This means `llmstack.sh stop` from any
    # project tears down the singleton daemons, regardless of where they
    # were started from.
    local stragglers
    stragglers="$(pgrep -fa 'llama-swap --config|src/router.py' 2>/dev/null || true)"
    if [[ -n "$stragglers" ]]; then
        echo "[*] stopping daemons by name (no local pid files, started elsewhere):"
        echo "$stragglers" | sed 's/^/    /'
        pkill -f 'llama-swap --config' 2>/dev/null || true
        pkill -f 'src/router.py'       2>/dev/null || true
        sleep 1
        pkill -9 -f 'llama-swap --config' 2>/dev/null || true
        pkill -9 -f 'src/router.py'       2>/dev/null || true
    fi

    # Defensive: also kill any orphaned llama-server children of llama-swap.
    pkill -f 'llama-server.*--alias (code-fast|code-smart|plan|plan-uncensored)' 2>/dev/null || true

    rm -f "$ACTIVE_MARKER"
    echo "[OK] stopped."
}

# --- action: restart --------------------------------------------------------

cmd_restart() {
    cmd_stop
    cmd_start "$@"
}

# --- action: status ---------------------------------------------------------

cmd_status() {
    for arg in "$@"; do
        case "$arg" in
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to status: $arg" ;;
        esac
    done

    local channel
    if [[ -f "$ACTIVE_MARKER" ]]; then
        channel="$(cat "$ACTIVE_MARKER")"
    elif port_responds "http://127.0.0.1:10102/health"; then
        channel="external (started from another project)"
    else
        channel="current (or stopped)"
    fi
    echo "stack status (channel: $channel):"
    echo "  work dir      $WORK_DIR"
    _check router     "http://127.0.0.1:10101/health"
    _check llama-swap "http://127.0.0.1:10102/health"

    echo
    if [[ -f "$OPENCODE_JSON" ]]; then
        echo "  opencode      $OPENCODE_JSON"
        if [[ -f "$AGENTS_LOCAL" ]]; then
            echo "  instructions  $AGENTS_LOCAL"
        fi
    else
        echo "  opencode      (not generated for this work dir; run: llmstack install)"
    fi

    if [[ "${LLMSTACK_ACTIVE:-}" == "1" ]]; then
        echo "  in-shell      OPENCODE_CONFIG=${OPENCODE_CONFIG:-?}, LLMSTACK_CHANNEL=${LLMSTACK_CHANNEL:-?}"
    fi

    echo
    echo "current models in /v1/models:"
    if curl -fsS --max-time 5 http://127.0.0.1:10101/v1/models 2>/dev/null \
        | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); [print(' -', m['id']) for m in d.get('data',[])]" \
        2>/dev/null; then
        :
    else
        echo "  (router not responding)"
    fi

    echo
    echo "loaded llama-server processes:"
    # Tolerate `set -o pipefail`: if pgrep finds nothing the whole pipeline
    # would exit 1 and abort the function.
    local pids=""
    pids="$(pgrep -f 'llama-server.*--alias' 2>/dev/null || true)"
    pids="$(echo "$pids" | tr '\n' ',' | sed 's/,$//')"
    if [[ -n "$pids" ]]; then
        ps -o pid,rss,command -p "$pids" 2>/dev/null \
            | awk 'NR==1 || NR>1 {if(NR>1){$2=int($2/1024)" MB"} print}'
    else
        echo "  (none loaded)"
    fi

    if [[ "${channel%% *}" == "next" && -f "$DEFAULT_CONFIG" ]]; then
        echo
        echo "next-channel swaps (from $(basename "$DEFAULT_CONFIG")):"
        "$PYTHON" -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
for name, m in cfg.get('models', {}).items():
    md = m.get('metadata') or {}
    if md.get('channel') == 'next':
        active = next(
            (l.strip()[5:] for l in m.get('cmd','').splitlines()
             if l.strip().startswith('-hff ') and not l.lstrip().startswith('#')),
            '?',
        )
        print(f'  {name:18}  -> {active}  ({md.get(\"quant\",\"?\")}, {md.get(\"size_gb\",\"?\")} GB)')
" "$DEFAULT_CONFIG" 2>/dev/null || true
    fi
}

_check() {
    local name="$1" url="$2" pid_file="$LLMSTACK_DIR/$1.pid"
    local responds=0
    curl -fsS --max-time 3 "$url" >/dev/null 2>&1 && responds=1

    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        printf "  %-12s pid %-7s  " "$name" "$(cat "$pid_file")"
    elif [[ "$responds" -eq 1 ]]; then
        # Daemons are alive but their pid file lives in another project's
        # .llmstack/ -- still mark as up so cross-project status looks sane.
        printf "  %-12s %-11s  " "$name" "external"
    else
        printf "  %-12s %-11s  " "$name" "DOWN"
    fi
    if [[ "$responds" -eq 1 ]]; then
        echo "OK $url"
    else
        echo "no response @ $url"
    fi
}

# --- action: check ----------------------------------------------------------

cmd_check() {
    require_python
    [[ -f "$CHECK_MODELS" ]] || die "missing $CHECK_MODELS"
    exec "$PYTHON" "$CHECK_MODELS" "$@"
}

# --- action: activate -------------------------------------------------------
#
# Emit shell-specific hook code that auto-activates whenever the user `cd`s
# into a project containing .llmstack/. Drop the eval into ~/.zshrc once:
#
#     eval "$(bash /abs/path/to/llmstack.sh activate zsh)"
#
# The emitted code is self-contained (no further calls back into this
# script). At each prompt it walks up from $PWD to find the nearest
# .llmstack/opencode.json, and toggles the env vars + prompt prefix
# accordingly. That keeps the hook fast (just stat()s + one read) and free
# of subprocess overhead.

cmd_activate() {
    local shell="${1:-}"
    [[ -z "$shell" || "$shell" == "-h" || "$shell" == "--help" ]] && {
        cmd_help
        return 0
    }
    case "$shell" in
        zsh)  _emit_activate_zsh  ;;
        bash) _emit_activate_bash ;;
        *)    die "activate: unknown shell '$shell' (supported: zsh, bash)" ;;
    esac
}

_emit_activate_zsh() {
    cat <<'ZSH_HOOK'
# --- llmstack auto-activation hook (zsh) -----------------------------------
# Generated by `llmstack.sh activate zsh`. Walks up from $PWD on each
# directory change to find the nearest .llmstack/opencode.json. Sets
# OPENCODE_CONFIG + LLMSTACK_PROJECT and prefixes the prompt; clears
# everything when you step out.

_llmstack_find_root() {
    local dir="${1:-$PWD}"
    while [[ "$dir" != "/" && -n "$dir" ]]; do
        if [[ -f "$dir/.llmstack/opencode.json" ]]; then
            print -r -- "$dir"
            return 0
        fi
        dir="${dir:h}"
    done
    return 1
}

_llmstack_activate() {
    local found
    found="$(_llmstack_find_root)" || found=""

    if [[ -n "$found" ]]; then
        # entering or moving within a project
        if [[ "${LLMSTACK_PROJECT:-}" != "$found" ]]; then
            export OPENCODE_CONFIG="$found/.llmstack/opencode.json"
            export LLMSTACK_PROJECT="$found"
            export LLMSTACK_ACTIVE="1"
            if [[ -f "$found/.llmstack/active-channel" ]]; then
                export LLMSTACK_CHANNEL="$(<"$found/.llmstack/active-channel")"
            else
                export LLMSTACK_CHANNEL="current"
            fi
            : "${_LLMSTACK_PS1_BACKUP:=$PROMPT}"
            export _LLMSTACK_PS1_BACKUP
            local label="${found:t}"
            PROMPT="%F{magenta}[llmstack:${label}]%f $_LLMSTACK_PS1_BACKUP"
        fi
    else
        # leaving any project
        if [[ -n "${LLMSTACK_PROJECT:-}" ]]; then
            unset OPENCODE_CONFIG LLMSTACK_PROJECT LLMSTACK_ACTIVE LLMSTACK_CHANNEL
            if [[ -n "${_LLMSTACK_PS1_BACKUP:-}" ]]; then
                PROMPT="$_LLMSTACK_PS1_BACKUP"
                unset _LLMSTACK_PS1_BACKUP
            fi
        fi
    fi
}

# Hook on every directory change AND once for the current shell.
autoload -U add-zsh-hook
add-zsh-hook chpwd _llmstack_activate
_llmstack_activate
# --- end llmstack hook -----------------------------------------------------
ZSH_HOOK
}

_emit_activate_bash() {
    cat <<'BASH_HOOK'
# --- llmstack auto-activation hook (bash) ----------------------------------
# Generated by `llmstack.sh activate bash`. Walks up from $PWD on each
# prompt to find the nearest .llmstack/opencode.json. Sets OPENCODE_CONFIG
# + LLMSTACK_PROJECT and prefixes the prompt; clears everything when you
# step out.

_llmstack_find_root() {
    local dir="${1:-$PWD}"
    while [[ "$dir" != "/" && -n "$dir" ]]; do
        if [[ -f "$dir/.llmstack/opencode.json" ]]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

_llmstack_activate() {
    local found
    found="$(_llmstack_find_root)" || found=""

    if [[ -n "$found" ]]; then
        if [[ "${LLMSTACK_PROJECT:-}" != "$found" ]]; then
            export OPENCODE_CONFIG="$found/.llmstack/opencode.json"
            export LLMSTACK_PROJECT="$found"
            export LLMSTACK_ACTIVE="1"
            if [[ -f "$found/.llmstack/active-channel" ]]; then
                export LLMSTACK_CHANNEL="$(<"$found/.llmstack/active-channel")"
            else
                export LLMSTACK_CHANNEL="current"
            fi
            : "${_LLMSTACK_PS1_BACKUP:=$PS1}"
            export _LLMSTACK_PS1_BACKUP
            local label
            label="$(basename "$found")"
            PS1="\[\033[35m\][llmstack:${label}]\[\033[0m\] $_LLMSTACK_PS1_BACKUP"
        fi
    else
        if [[ -n "${LLMSTACK_PROJECT:-}" ]]; then
            unset OPENCODE_CONFIG LLMSTACK_PROJECT LLMSTACK_ACTIVE LLMSTACK_CHANNEL
            if [[ -n "${_LLMSTACK_PS1_BACKUP:-}" ]]; then
                PS1="$_LLMSTACK_PS1_BACKUP"
                unset _LLMSTACK_PS1_BACKUP
            fi
        fi
    fi
}

# Bash has no chpwd hook, so we run on every prompt. Idempotent: if the
# project hasn't changed, we no-op.
case ";${PROMPT_COMMAND:-};" in
    *";_llmstack_activate;"*) ;;
    *) PROMPT_COMMAND="_llmstack_activate;${PROMPT_COMMAND:-}" ;;
esac
_llmstack_activate
# --- end llmstack hook -----------------------------------------------------
BASH_HOOK
}

# --- dispatch ---------------------------------------------------------------

action="${1:-help}"
[[ $# -gt 0 ]] && shift || true

case "$action" in
    help|-h|--help)        cmd_help                       ;;
    setup)                 cmd_setup              "$@"    ;;
    install)               cmd_install            "$@"    ;;
    install-llama-swap)    cmd_install_llama_swap "$@"    ;;
    download|download-models) cmd_download        "$@"    ;;
    start)                 cmd_start              "$@"    ;;
    shell)                 cmd_shell              "$@"    ;;
    stop)                  cmd_stop               "$@"    ;;
    restart)               cmd_restart            "$@"    ;;
    status)                cmd_status             "$@"    ;;
    check|check-models)    cmd_check              "$@"    ;;
    activate)              cmd_activate           "$@"    ;;
    *)
        echo "[!] unknown action: $action" >&2
        echo >&2
        cmd_help >&2
        exit 2
        ;;
esac
