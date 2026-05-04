#!/usr/bin/env bash
# setup.sh - first-time / full install orchestrator. Three steps:
#
#   1. download every GGUF listed in ../models.ini
#      (delegates to scripts/download-models.sh which fires llama-cli in
#      the background and writes per-target logs to logs/dl-*.log)
#
#   2. wait for downloads to finish
#      (polls for any running `llama-cli ... -hf ...` processes)
#
#   3. generate + install both config files
#      (delegates to scripts/install.sh, which writes ../llama-swap.yaml
#      and ~/.config/opencode/opencode.json with timestamped backups)
#
# Does NOT start the stack. Run `bash scripts/start.sh` when you're ready.
#
# Skip any step:
#   bash setup.sh --skip-download   # configs only (same as scripts/install.sh)
#   bash setup.sh --skip-wait       # kick off downloads in background, install
#                                   # configs immediately, return (downloads
#                                   # keep running)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

SKIP_DOWNLOAD=0
SKIP_WAIT=0
for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --skip-wait)     SKIP_WAIT=1     ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "[!] unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    echo "[1/3] downloading required GGUFs..."
    bash "$HERE/download-models.sh"
    echo
else
    echo "[1/3] (skipped) downloads"
    echo
fi

if [[ "$SKIP_DOWNLOAD" -eq 0 && "$SKIP_WAIT" -eq 0 ]]; then
    echo "[2/3] waiting for downloads to finish..."
    echo "      (logs: $ROOT/logs/dl-*.log)"
    # Give nohup'd processes a moment to register before the first poll.
    sleep 2
    # Match both llama-completion (modern) and llama-cli (legacy fallback).
    while pgrep -f 'llama-(completion|cli) .*-hf ' >/dev/null 2>&1; do
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
bash "$HERE/install.sh"

cat <<EOF

[OK] setup complete.

To start the stack:
  bash $HERE/start.sh

To check what's configured + drift between models.ini and llama-swap.yaml:
  bash $HERE/check-models.sh

EOF
