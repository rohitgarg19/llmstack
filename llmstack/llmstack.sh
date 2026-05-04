#!/usr/bin/env bash
# llmstack.sh - single entry point for all stack operations.
#
# Usage:
#   bash llmstack.sh <action> [options]
#
# This stack does NOT touch ~/.config/opencode/opencode.json. Instead, the
# generated opencode config lives at llmstack/.run/opencode.json, and the
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
#       Generate llama-swap.yaml + .run/opencode.json from models.ini, and
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
#       hf_file_next set in models.ini to that file via an ephemeral sidecar
#       yaml. `--detach` skips the subshell (daemons only; old behavior).
#   shell
#       Drop into the env-prepared subshell without (re)starting daemons.
#       Useful for opening more terminals into the same stack. Refuses if
#       the daemons aren't running.
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
#   OPENCODE_CONFIG_DIR     where to write opencode.json (default: .run/)
#   LLAMA_SWAP_VERSION      pin a specific llama-swap release (e.g. v211)
#   HF_TOKEN                authenticate model downloads (faster rate limits)
#   LLMSTACK_USE_NEXT=1     same as `start --next` (env path)
#   LLMSTACK_SHELL          shell to spawn in `start`/`shell` (default: $SHELL)
#
# Variables exported into the spawned subshell:
#   OPENCODE_CONFIG         path to the generated .run/opencode.json
#   LLMSTACK_CHANNEL        current | next
#   LLMSTACK_ACTIVE         "1" while inside the subshell
#   LLMSTACK_ROOT           absolute path to the llmstack/ directory

set -euo pipefail

# --- common paths and helpers -----------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # llmstack/
ROOT="$HERE"
PROJ="$(cd "$ROOT/.." && pwd)"                                 # project root

LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/.run"
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

DEFAULT_CONFIG="$ROOT/llama-swap.yaml"
NEXT_CONFIG="$RUN_DIR/llama-swap.next.yaml"
ACTIVE_MARKER="$RUN_DIR/active-channel"

# opencode config: written into .run/ by `install`, exported via OPENCODE_CONFIG
# in the subshell that `start`/`shell` spawn. The user's global
# ~/.config/opencode/opencode.json is never modified.
OPENCODE_JSON_DIR="${OPENCODE_CONFIG_DIR:-$RUN_DIR}"
OPENCODE_JSON="$OPENCODE_JSON_DIR/opencode.json"

REPO_LLAMA_SWAP="mostlygeek/llama-swap"

mkdir -p "$LOG_DIR" "$RUN_DIR"

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

# --- action: help -----------------------------------------------------------

cmd_help() {
    sed -n '2,49p' "$0" | sed 's/^# \{0,1\}//'
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

    local print_only=0
    for arg in "$@"; do
        case "$arg" in
            --print|-n) print_only=1 ;;
            -h|--help) cmd_help; return 0 ;;
            *) die "unknown arg to install: $arg (try --print, -h)" ;;
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

    echo "[3/3] opencode.json"
    _render_install "opencode.json" "$GEN_JSON" "$OPENCODE_JSON" \
        "import json,sys; json.load(open(sys.argv[1]))"

    cat <<EOF

[OK] both configs derived from $INI.
     re-run any time you edit models.ini.

opencode wiring:
  $OPENCODE_JSON
  Picked up via OPENCODE_CONFIG in the subshell spawned by:
    bash llmstack.sh start
    bash llmstack.sh shell
  Your ~/.config/opencode/opencode.json is intentionally NOT modified.

Next:
  bash llmstack.sh start            # daemons + drop into env-prepared shell
  bash llmstack.sh check            # snapshot configured GGUFs + drift check
EOF
}

# --- action: download -------------------------------------------------------

cmd_download() {
    require_python; require_ini
    [[ -f "$TIERS" ]] || die "missing $TIERS"

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
    local channel="current"
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
    # Env-var path: same as --next.
    if [[ "${LLMSTACK_USE_NEXT:-}" =~ ^(1|true|yes|on)$ && "$channel" == "current" ]]; then
        channel="next"
    fi

    require_python
    [[ -x "$LLAMA_SWAP" ]] || die "missing $LLAMA_SWAP (run: bash llmstack.sh install)"
    [[ -f "$DEFAULT_CONFIG" || "$channel" == "next" ]] || die "missing $DEFAULT_CONFIG (run: bash llmstack.sh install)"

    if is_running "$RUN_DIR/llama-swap.pid"; then
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
    fi

    local config="$DEFAULT_CONFIG"
    if [[ "$channel" == "next" ]]; then
        local queued
        queued="$("$PYTHON" "$TIERS" --downloads | awk -F'\t' '$4=="next" {print $1}' | sed 's/-next$//' | tr '\n' ' ')"
        if [[ -z "${queued// }" ]]; then
            echo "[!] no tiers have hf_file_next set in models.ini -- nothing to do." >&2
            echo "    add an hf_file_next line to a tier and re-run, or use --current." >&2
            return 1
        fi
        echo "[*] generating sidecar yaml: $NEXT_CONFIG"
        echo "    queued upgrade tiers: $queued"
        "$PYTHON" "$GEN_YAML" --use-next "$NEXT_CONFIG"
        "$PYTHON" -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$NEXT_CONFIG"
        config="$NEXT_CONFIG"
    fi

    echo "[*] channel: $channel  ($(basename "$config"))"

    if is_running "$RUN_DIR/llama-swap.pid"; then
        echo "[=] llama-swap already running (pid $(cat "$RUN_DIR/llama-swap.pid"), channel $channel)"
    else
        echo "[*] starting llama-swap on :10102"
        nohup "$LLAMA_SWAP" --config "$config" --listen 127.0.0.1:10102 \
            >"$LOG_DIR/llama-swap.log" 2>&1 &
        echo $! >"$RUN_DIR/llama-swap.pid"
        echo "$channel" >"$ACTIVE_MARKER"
        sleep 1
        if ! is_running "$RUN_DIR/llama-swap.pid"; then
            echo "[!] llama-swap failed to start. Check $LOG_DIR/llama-swap.log"
            rm -f "$RUN_DIR/llama-swap.pid" "$ACTIVE_MARKER"
            return 1
        fi
        echo "    pid $(cat "$RUN_DIR/llama-swap.pid")"
    fi

    if is_running "$RUN_DIR/router.pid"; then
        echo "[=] router already running (pid $(cat "$RUN_DIR/router.pid"))"
    else
        echo "[*] starting router on :10101"
        cd "$SRC_DIR"
        LLAMA_SWAP_URL="http://127.0.0.1:10102" \
        ROUTER_HOST="127.0.0.1" ROUTER_PORT="10101" \
        nohup "$PYTHON" "$ROUTER" >"$LOG_DIR/router.log" 2>&1 &
        echo $! >"$RUN_DIR/router.pid"
        sleep 1
        if ! is_running "$RUN_DIR/router.pid"; then
            echo "[!] router failed to start. Check $LOG_DIR/router.log"
            rm -f "$RUN_DIR/router.pid"
            return 1
        fi
        echo "    pid $(cat "$RUN_DIR/router.pid")"
    fi

    local other
    other=$([[ "$channel" == "current" ]] && echo next || echo current)
    cat <<EOF

[OK] stack is up (channel: $channel).

  router       http://127.0.0.1:10101     (OpenAI-compatible, "auto" routing)
  llama-swap   http://127.0.0.1:10102     (raw model endpoints + UI)

Try:
  curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'
  curl -sN http://127.0.0.1:10101/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'

Logs:
  tail -f $LOG_DIR/llama-swap.log
  tail -f $LOG_DIR/router.log

Switch channel (requires a stop first):
  bash llmstack.sh restart --$other

Stop:
  bash llmstack.sh stop
EOF

    if [[ "$detach" -eq 1 ]]; then
        return 0
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
    is_running "$RUN_DIR/llama-swap.pid" || die "stack is not running. start it first: bash llmstack.sh start"
    local channel
    channel="$(cat "$ACTIVE_MARKER" 2>/dev/null || echo current)"
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
    daemons keep running on exit; stop them with: bash llmstack.sh stop
    second window?: bash llmstack.sh shell

EOF

    local -a exec_argv=()
    case "$shell_name" in
        bash)
            local rcfile="$RUN_DIR/llmstack.bashrc"
            cat > "$rcfile" <<RC
# AUTO-GENERATED by llmstack.sh; sourced by the spawned subshell.
[ -f "\$HOME/.bashrc" ] && source "\$HOME/.bashrc"
PS1="[llmstack:$channel] \${PS1:-\\\$ }"
RC
            exec_argv=("$user_shell" --rcfile "$rcfile" -i)
            ;;
        zsh)
            local zdotdir="$RUN_DIR/zdotdir"
            mkdir -p "$zdotdir"
            cat > "$zdotdir/.zshrc" <<RC
# AUTO-GENERATED by llmstack.sh; sourced by the spawned subshell.
[ -f "\$HOME/.zshrc" ] && source "\$HOME/.zshrc"
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
        pid_file="$RUN_DIR/$name.pid"
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
        else
            echo "[=] $name not running"
        fi
    done

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
    else
        channel="current (or stopped)"
    fi
    echo "stack status (channel: $channel):"
    _check router     "http://127.0.0.1:10101/health"
    _check llama-swap "http://127.0.0.1:10102/health"

    echo
    if [[ -f "$OPENCODE_JSON" ]]; then
        echo "  opencode      $OPENCODE_JSON"
    else
        echo "  opencode      (not generated; run: bash llmstack.sh install)"
    fi
    if [[ "${LLMSTACK_ACTIVE:-}" == "1" ]]; then
        echo "                in-shell: OPENCODE_CONFIG=${OPENCODE_CONFIG:-?}, LLMSTACK_CHANNEL=${LLMSTACK_CHANNEL:-?}"
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

    if [[ "${channel%% *}" == "next" && -f "$NEXT_CONFIG" ]]; then
        echo
        echo "next-channel swaps (from $(basename "$NEXT_CONFIG")):"
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
" "$NEXT_CONFIG" 2>/dev/null || true
    fi
}

_check() {
    local name="$1" url="$2" pid_file="$RUN_DIR/$1.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        printf "  %-12s pid %-7s  " "$name" "$(cat "$pid_file")"
    else
        printf "  %-12s %-11s  " "$name" "DOWN"
    fi
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
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
    *)
        echo "[!] unknown action: $action" >&2
        echo >&2
        cmd_help >&2
        exit 2
        ;;
esac
