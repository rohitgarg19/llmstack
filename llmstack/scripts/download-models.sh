#!/usr/bin/env bash
# Download every GGUF this stack wants into llama.cpp's default cache
# (~/.cache/huggingface/hub).
#
# WHAT THIS DOWNLOADS ---------------------------------------------------------
# The (repo, file, tag) tuples are NOT hardcoded here. They come from
# `python src/tiers.py --downloads`, which reads ../models.ini and yields
# one row per (tier, label) pair where label is "current" or "next".
#
# That makes models.ini the single source of truth for what we cache.
# To add/remove a download target, edit the tier in models.ini (set or
# clear the hf_file / hf_file_next field) - this script picks it up
# automatically on the next run.
#
# CACHE OWNERSHIP -------------------------------------------------------------
# This script uses llama-cli as the ONLY writer to the model cache. Why:
#
#   * `llama-cli`, `llama-server`, and `llama-swap` all read from the same
#     cache directory: ~/.cache/huggingface/hub. (Verify with `llama-cli -cl`.)
#   * Although `huggingface_hub.hf_hub_download()` writes to the same
#     directory and produces interchangeable *completed* blobs, it uses a
#     different temp-file suffix (.incomplete vs .downloadInProgress), so
#     in-flight files cannot be resumed across tools. Sticking to llama-cli
#     keeps the cache state coherent and resumable.
#
# WHAT EACH DOWNLOAD DOES -----------------------------------------------------
# `llama-cli -hf <repo> -hff <file>` downloads <file> into the standard cache
# (resuming from any prior partial download), mmap-loads the model with
# minimal context, generates one CPU token, and exits. The mmap is virtual
# so RAM cost is tiny; the slow part is the network download.
#
# RUNNING ---------------------------------------------------------------------
#   bash scripts/download-models.sh        # queue every target in parallel
#   tail -f logs/dl-*.log                  # watch progress
#   llama-cli -cl                          # see what's complete
#
# Re-running this script is safe: completed files are detected and skipped.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LOG_DIR="$ROOT/logs"
PYTHON="$ROOT/.venv/bin/python"
TIERS="$ROOT/src/tiers.py"

mkdir -p "$LOG_DIR"

[[ -x "$PYTHON" ]] || { echo "[!] venv missing at $PYTHON (see UPGRADING.md > 'Upgrading the Python toolchain')" >&2; exit 1; }
[[ -f "$TIERS"  ]] || { echo "[!] missing $TIERS" >&2; exit 1; }

# Pick the canonical "download + 1-token validate then exit" tool. Prefer
# llama-completion (newer split: chat=llama-cli, one-shot=llama-completion);
# fall back to llama-cli on older llama.cpp installs.
LLAMA_BIN=""
for candidate in llama-completion llama-cli; do
    if path="$(command -v "$candidate" 2>/dev/null)"; then
        LLAMA_BIN="$path"
        break
    fi
done
[[ -n "$LLAMA_BIN" ]] || { echo "[!] neither llama-completion nor llama-cli found in PATH (brew install llama.cpp)" >&2; exit 1; }

echo "[*] downloader: $LLAMA_BIN"
echo "[*] cache:      ~/.cache/huggingface/hub  (default for llama.cpp)"
echo "[*] inventory:  $ROOT/../models.ini  (via $TIERS --downloads)"
# Optional: set HF_TOKEN in the environment for higher rate limits + parallelism.
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[*] HF_TOKEN set (faster rate limits)"
    HF_FLAG=(--hf-token "$HF_TOKEN")
else
    echo "[*] no HF_TOKEN (rate-limited unauthenticated downloads)"
    HF_FLAG=()
fi
echo

# Bash strict-mode quirk: ${arr[@]} on an empty array errors under `set -u`.
# This expansion form ('use ${arr[@]} if set, else empty') sidesteps that
# while still expanding correctly when HF_FLAG has elements.
expand_hf_flag() {
    if [[ "${#HF_FLAG[@]}" -gt 0 ]]; then
        printf '%s\n' "${HF_FLAG[@]}"
    fi
}

run_dl() {
    local tag="$1" repo="$2" file="$3" label="$4"
    local log="$LOG_DIR/dl-${tag}.log"
    printf "[*] %-32s %-7s %s / %s\n" "$tag" "($label)" "$repo" "$file"
    echo "    log -> $log"
    # IMPORTANT: must use llama-completion (or llama-cli on old installs).
    # Modern llama-cli is chat-only and ignores -no-cnv with the message
    # "--no-conversation is not supported by llama-cli, please use
    # llama-completion instead", then falls back to interactive mode
    # printing "> " prompts forever (~1.5 MB/s of log noise). Even with
    # -p/-n set and stdin closed via /dev/null. llama-completion doesn't
    # have this bug - it generates n tokens and exits.
    #
    # --no-warmup     skip the post-load warmup
    # -ngl 0          keep weights on CPU (mmap'd, no Metal alloc)
    # -c 256          tiny KV cache (just for the 1-token validation)
    # -p ok -n 1      generate exactly one token then exit
    # < /dev/null     close stdin so any prompt-loop sees EOF and exits
    # Build the argv as an array so we can safely include zero or more
    # --hf-token flags without tripping `set -u` on empty array expansion.
    local -a argv=(
        -hf "$repo" -hff "$file"
        --no-warmup -ngl 0 -c 256 -p ok -n 1
    )
    if [[ "${#HF_FLAG[@]}" -gt 0 ]]; then
        argv+=("${HF_FLAG[@]}")
    fi
    nohup "$LLAMA_BIN" "${argv[@]}" \
        > "$log" 2>&1 < /dev/null &
    echo "    pid -> $!"
}

count=0
while IFS=$'\t' read -r tag repo file label; do
    [[ -z "$tag" ]] && continue
    run_dl "$tag" "$repo" "$file" "$label"
    count=$((count + 1))
done < <("$PYTHON" "$TIERS" --downloads)

[[ "$count" -gt 0 ]] || { echo "[!] no download targets found in models.ini" >&2; exit 1; }

cat <<EOF

$count download(s) queued in the background.

Watch progress:
    tail -f $LOG_DIR/dl-*.log
    llama-cli -cl                        # lists completed cache entries

When an upgrade-target file finishes, edit the matching -hff line in
$ROOT/llama-swap.yaml then:
    bash $HERE/stop.sh && bash $HERE/start.sh

EOF
