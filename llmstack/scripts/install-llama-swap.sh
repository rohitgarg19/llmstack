#!/usr/bin/env bash
# install-llama-swap.sh - download the llama-swap binary for this OS/arch
# from https://github.com/mostlygeek/llama-swap/releases.
#
# Asset naming: llama-swap_<num>_<os>_<arch>.tar.gz
# Resolves the latest tag via the GitHub API; override with $LLAMA_SWAP_VERSION
# (e.g. v211).
#
# Idempotent: skips download if ../bin/llama-swap is already at the target
# version. Pass --force / -f to redownload anyway.
#
# Supported: darwin/arm64, darwin/amd64, linux/arm64, linux/amd64, freebsd/amd64.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BIN_DIR="$ROOT/bin"
TARGET="$BIN_DIR/llama-swap"
REPO="mostlygeek/llama-swap"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "[!] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# --- detect platform -------------------------------------------------------
case "$(uname -s)" in
    Darwin)  OS="darwin"  ;;
    Linux)   OS="linux"   ;;
    FreeBSD) OS="freebsd" ;;
    *) echo "[!] unsupported OS: $(uname -s) (need Darwin/Linux/FreeBSD)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
    arm64|aarch64) ARCH="arm64" ;;
    x86_64|amd64)  ARCH="amd64" ;;
    *) echo "[!] unsupported arch: $(uname -m) (need arm64 or x86_64)" >&2; exit 1 ;;
esac

# freebsd only has amd64 builds
if [[ "$OS" == "freebsd" && "$ARCH" != "amd64" ]]; then
    echo "[!] no llama-swap release for freebsd/$ARCH" >&2
    exit 1
fi

# --- resolve version -------------------------------------------------------
if [[ -n "${LLAMA_SWAP_VERSION:-}" ]]; then
    TAG="$LLAMA_SWAP_VERSION"
    echo "[*] version: $TAG (from \$LLAMA_SWAP_VERSION)"
else
    echo "[*] resolving latest release tag from github.com/$REPO..."
    TAG="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
            | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null \
            || true)"
    [[ -n "$TAG" ]] || { echo "[!] could not resolve latest release tag" >&2; exit 1; }
    echo "[*] latest release: $TAG"
fi

NUM="${TAG#v}"
ASSET="llama-swap_${NUM}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

# --- skip if already at target version -------------------------------------
if [[ -x "$TARGET" && "$FORCE" -eq 0 ]]; then
    if current="$("$TARGET" --version 2>/dev/null | head -1 || true)"; then
        if echo "$current" | grep -qE "version: $NUM\b"; then
            echo "[=] already installed: $TARGET"
            echo "         $current"
            echo "    (re-run with --force to redownload)"
            exit 0
        fi
        echo "[*] currently installed: $current"
        echo "    upgrading to $TAG"
    fi
fi

# --- download + extract + atomic install -----------------------------------
mkdir -p "$BIN_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[*] downloading $ASSET"
echo "    from $URL"
curl -fL --progress-bar "$URL" -o "$TMP/$ASSET"

echo "[*] extracting"
tar -xzf "$TMP/$ASSET" -C "$TMP" llama-swap
[[ -f "$TMP/llama-swap" ]] || {
    echo "[!] tarball did not contain a top-level 'llama-swap' file"
    tar -tzf "$TMP/$ASSET" | head
    exit 1
}

# Atomic-replace via mv on the same filesystem.
mv "$TMP/llama-swap" "$TARGET.new"
chmod +x "$TARGET.new"
mv "$TARGET.new" "$TARGET"

echo "[OK] installed $TARGET ($OS/$ARCH)"
"$TARGET" --version 2>&1 | sed 's/^/     /'
