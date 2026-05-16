"""Snapshot the currently-configured GGUFs and what's recommended.

For every tier in ``models.ini``, prints a row per file (current + upgrade
target if defined) with:

  - filename
  - HuggingFace size + last-modified
  - direct URL to the GGUF on HF
  - DRIFT marker when ``models.ini`` and ``llama-swap.yaml`` disagree about
    the currently-configured file

Read-only -- no side effects. Invoked by ``llmstack check``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from llmstack.paths import resolve
from llmstack.tiers import load_tiers

HF_RE = re.compile(r"-hf\s+(\S+/\S+)")
HFF_RE = re.compile(r"-hff\s+(\S+\.gguf)")


def parse_yaml(yaml_path: Path) -> dict[str, tuple[str, str]]:
    """Map tier-name -> (repo, file) by reading ``llama-swap.yaml``.

    Strips comment lines from each ``cmd:`` block before regex-matching so
    commented-out ``-hf`` examples don't pollute the result.
    """
    if not yaml_path.exists():
        return {}
    cfg = yaml.safe_load(yaml_path.read_text())
    out: dict[str, tuple[str, str]] = {}
    for tier, m in (cfg.get("models") or {}).items():
        cmd_lines = [
            line for line in (m.get("cmd") or "").splitlines()
            if not line.strip().startswith("#")
        ]
        cmd = "\n".join(cmd_lines)
        repo_m = HF_RE.search(cmd)
        file_m = HFF_RE.search(cmd)
        if repo_m and file_m:
            out[tier] = (repo_m.group(1), file_m.group(1))
    return out


def hf_meta(api: HfApi, repo: str, fname: str) -> tuple[str, str]:
    """Fetch (size_human, last_modified_iso) from HF for a single file."""
    try:
        info = api.model_info(repo, files_metadata=True)
        size = next((s.size for s in info.siblings if s.rfilename == fname), None)
        size_s = f"{size / 1024 / 1024 / 1024:.1f} GB" if size else "?"
        mod = info.last_modified.strftime("%Y-%m-%d") if info.last_modified else "?"
        return size_s, mod
    except Exception as e:  # network / 404 / auth - keep going for the next row
        return "ERR", str(e)[:24]


def main(argv: list[str] | None = None) -> int:
    api = HfApi()
    tiers = load_tiers()
    yaml_cfg = parse_yaml(resolve().llama_swap_yaml)

    fmt = "{:<18} {:<8} {:<70} {:>10} {:>12}  {}"
    print(fmt.format("tier", "label", "file / model-id", "size", "updated", "url / region"))
    print("-" * 165)

    drift = []
    for tier in tiers.values():
        if tier.is_litellm and tier.litellm is not None:
            b = tier.litellm
            scope_parts = [p for p in (b.region, b.profile) if p]
            scope = " / ".join(scope_parts) if scope_parts else "(default chain)"
            print(fmt.format(tier.name, "litellm", b.model_id, "-", "-", scope))
            if b.has_next:
                next_scope_parts = [p for p in (b.region_next or b.region, b.profile) if p]
                next_scope = " / ".join(next_scope_parts) if next_scope_parts else "(default chain)"
                print(fmt.format(tier.name, "next", b.model_id_next or "", "-", "-", next_scope))
            continue

        for tf in tier.files():
            size_s, mod = hf_meta(api, tf.repo, tf.file)
            url = f"https://huggingface.co/{tf.repo}/blob/main/{tf.file}"
            label = tf.label
            if tf.label == "current":
                actual = yaml_cfg.get(tier.name)
                if actual and actual != (tf.repo, tf.file):
                    label = "DRIFT!"
                    drift.append((tier.name, (tf.repo, tf.file), actual))
            print(fmt.format(tier.name, label, tf.file, size_s, mod, url))

    if drift:
        print()
        print("[!] DRIFT detected between models.ini (recommended) and llama-swap.yaml (active):")
        for tier, want, got in drift:
            print(f"    [{tier}]  ini wants  {want[0]} / {want[1]}")
            print(f"             yaml has   {got[0]} / {got[1]}")
        print("    Reconcile by editing one of the two so they match.")

    print()
    print("To look for upgrades, browse:")
    print("  https://huggingface.co/models?library=gguf&sort=trending")
    print("  https://huggingface.co/bartowski        (general GGUF maintainer)")
    print("  https://huggingface.co/unsloth          (Qwen + UD dynamic quants)")
    print("  https://huggingface.co/mradermacher     (i1 + abliterated/heretic)")
    print()
    print("Then: see UPGRADING.md for the workflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
