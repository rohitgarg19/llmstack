#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RUN_DIR="$ROOT/.run"

check() {
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

echo "stack status:"
check router     "http://127.0.0.1:10101/health"
check llama-swap "http://127.0.0.1:10102/health"

echo
echo "current models in /v1/models:"
curl -fsS --max-time 5 http://127.0.0.1:10101/v1/models 2>/dev/null \
    | "$ROOT/.venv/bin/python" -c "import sys,json; d=json.load(sys.stdin); [print(' -', m['id']) for m in d.get('data',[])]" \
    2>/dev/null || echo "  (router not responding)"

echo
echo "loaded llama-server processes:"
ps -o pid,rss,command -p $(pgrep -f 'llama-server.*--alias' 2>/dev/null | tr '\n' ',' | sed 's/,$//') 2>/dev/null \
    | awk 'NR==1 || NR>1 {if(NR>1){$2=int($2/1024)" MB"} print}' \
    || echo "  (none loaded)"
