#!/usr/bin/env bash
# install.sh - prepare the runtime: download the llama-swap binary for this
# OS/arch and generate both config files from ../models.ini.
#
# Step 1: ensure ../bin/llama-swap exists at the right architecture
#         (delegates to install-llama-swap.sh - skips if already up to date)
# Step 2: render ../llama-swap.yaml from src/gen_llama_swap_yaml.py
# Step 3: render ~/.config/opencode/opencode.json from src/gen_opencode_config.py
#
# Does NOT download GGUFs (use scripts/download-models.sh) and does NOT start
# the stack (use scripts/start.sh). For the full first-time install (binary +
# configs + GGUFs), see scripts/setup.sh.
#
# Usage:
#   bash llmstack/scripts/install.sh             # generate + install both
#   bash llmstack/scripts/install.sh --print     # print to stdout, install nothing
#
# Override the opencode install location with $OPENCODE_CONFIG_DIR if needed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PROJ="$(cd "$ROOT/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
GEN_YAML="$ROOT/src/gen_llama_swap_yaml.py"
GEN_JSON="$ROOT/src/gen_opencode_config.py"
INI="$PROJ/models.ini"

[[ -f "$INI"      ]] || { echo "[!] missing $INI"; exit 1; }
[[ -x "$PYTHON"   ]] || { echo "[!] missing venv at $PYTHON (see UPGRADING.md > 'Upgrading the Python toolchain')"; exit 1; }
[[ -f "$GEN_YAML" ]] || { echo "[!] missing $GEN_YAML"; exit 1; }
[[ -f "$GEN_JSON" ]] || { echo "[!] missing $GEN_JSON"; exit 1; }

if [[ "${1:-}" == "--print" || "${1:-}" == "-n" ]]; then
    echo "----- llama-swap.yaml -----"
    "$PYTHON" "$GEN_YAML" -
    echo
    echo "----- opencode.json -----"
    "$PYTHON" "$GEN_JSON" -
    exit 0
fi

# Step 1: llama-swap binary (skips if already at latest version)
echo "[1/3] llama-swap binary"
bash "$HERE/install-llama-swap.sh"
echo

# Helper: render -> validate -> backup -> atomic mv -> chmod
render_install() {
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

    if [[ -f "$target" ]]; then
        local bak="$target.bak.$(date +%Y%m%d-%H%M%S)"
        cp -p "$target" "$bak"
        echo "[*] backed up $target"
        echo "         -> $bak"
    fi

    mv "$tmp" "$target"
    chmod 644 "$target"
    echo "[OK] installed $target"
}

# Step 2: llama-swap.yaml (project-local)
echo "[2/3] llama-swap.yaml"
render_install "llama-swap.yaml" "$GEN_YAML" "$ROOT/llama-swap.yaml" \
    "import yaml,sys; yaml.safe_load(open(sys.argv[1]))"

echo

# Step 3: opencode.json (~/.config/opencode/opencode.json by default)
echo "[3/3] opencode.json"
JSON_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"
render_install "opencode.json" "$GEN_JSON" "$JSON_DIR/opencode.json" \
    "import json,sys; json.load(open(sys.argv[1]))"

cat <<EOF

[OK] both configs derived from $INI.
     re-run any time you edit models.ini.

Next:
  bash $HERE/start.sh           # bring up llama-swap + router
  bash $HERE/check-models.sh    # snapshot configured GGUFs + drift check
EOF
