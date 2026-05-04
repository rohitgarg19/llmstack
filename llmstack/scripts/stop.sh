#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RUN_DIR="$ROOT/.run"

for name in router llama-swap; do
    pid_file="$RUN_DIR/$name.pid"
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[*] stopping $name (pid $pid)"
            kill "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10; do
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

echo "[OK] stopped."
