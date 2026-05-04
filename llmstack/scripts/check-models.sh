#!/usr/bin/env bash
# Print a snapshot of the configured GGUFs:
#   - what each tier points at (current + queued upgrade target if any)
#   - upstream HF repo + file size + last-modified date
#   - direct URLs you can open to look for newer/better quants
#   - DRIFT markers when models.ini and llama-swap.yaml disagree
#
# Read-only. Use this before and after an upgrade so you can diff, and any
# time you want to check what's running.
#
# Logic lives in src/check_models.py; this script is just a venv-aware launcher.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

[[ -x "$PYTHON" ]] || { echo "[!] venv missing at $PYTHON (see UPGRADING.md > 'Upgrading the Python toolchain')"; exit 1; }

exec "$PYTHON" "$ROOT/src/check_models.py" "$@"
