#!/usr/bin/env bash
# Start llama-swap (port 10102) and the auto-router proxy (port 10101).
# Logs go to llmstack/logs/.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/.run"
mkdir -p "$LOG_DIR" "$RUN_DIR"

LLAMA_SWAP="$ROOT/bin/llama-swap"
CONFIG="$ROOT/llama-swap.yaml"
PYTHON="$ROOT/.venv/bin/python"
ROUTER="$ROOT/src/router.py"

[[ -x "$LLAMA_SWAP" ]] || { echo "missing $LLAMA_SWAP"; exit 1; }
[[ -f "$CONFIG"     ]] || { echo "missing $CONFIG";     exit 1; }
[[ -x "$PYTHON"     ]] || { echo "missing venv at $PYTHON"; exit 1; }

is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

# ----- llama-swap on :10102 --------------------------------------------------
if is_running "$RUN_DIR/llama-swap.pid"; then
    echo "[=] llama-swap already running (pid $(cat "$RUN_DIR/llama-swap.pid"))"
else
    echo "[*] starting llama-swap on :10102"
    nohup "$LLAMA_SWAP" --config "$CONFIG" --listen 127.0.0.1:10102 \
        >"$LOG_DIR/llama-swap.log" 2>&1 &
    echo $! >"$RUN_DIR/llama-swap.pid"
    sleep 1
    if ! is_running "$RUN_DIR/llama-swap.pid"; then
        echo "[!] llama-swap failed to start. Check $LOG_DIR/llama-swap.log"
        rm -f "$RUN_DIR/llama-swap.pid"
        exit 1
    fi
    echo "    pid $(cat "$RUN_DIR/llama-swap.pid")"
fi

# ----- router on :10101 ------------------------------------------------------
if is_running "$RUN_DIR/router.pid"; then
    echo "[=] router already running (pid $(cat "$RUN_DIR/router.pid"))"
else
    echo "[*] starting router on :10101"
    cd "$ROOT/src"
    LLAMA_SWAP_URL="http://127.0.0.1:10102" \
    ROUTER_HOST="127.0.0.1" ROUTER_PORT="10101" \
    nohup "$PYTHON" "$ROUTER" >"$LOG_DIR/router.log" 2>&1 &
    echo $! >"$RUN_DIR/router.pid"
    sleep 1
    if ! is_running "$RUN_DIR/router.pid"; then
        echo "[!] router failed to start. Check $LOG_DIR/router.log"
        rm -f "$RUN_DIR/router.pid"
        exit 1
    fi
    echo "    pid $(cat "$RUN_DIR/router.pid")"
fi

cat <<EOF

[OK] stack is up.

  router       http://127.0.0.1:10101     (OpenAI-compatible, "auto" routing)
  llama-swap   http://127.0.0.1:10102     (raw model endpoints + UI)

Try:
  curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'
  curl -sN http://127.0.0.1:10101/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'

Logs:
  tail -f $LOG_DIR/llama-swap.log
  tail -f $LOG_DIR/router.log

Stop:
  $HERE/stop.sh
EOF
