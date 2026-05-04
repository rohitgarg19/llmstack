"""Install (or update) the ``llama-swap`` binary.

Replaces the shell ``_install_llama_swap`` helper. Resolves the latest
GitHub release tag (or honours ``$LLAMA_SWAP_VERSION``), downloads the
asset for the current OS+arch, extracts the single ``llama-swap``
executable, and atomically renames it into place under
:func:`llmstack.paths.bin_dir`.

A second call short-circuits when the installed version already matches
the resolved tag, unless ``force=True`` is passed.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from llmstack.paths import REPO_LLAMA_SWAP, ensure_data_dirs

GH_API = "https://api.github.com"
GH_DL = "https://github.com"
VERSION_RE = re.compile(r"version:\s*v?([0-9][\w.-]*)", re.IGNORECASE)


def _detect_os_arch() -> tuple[str, str]:
    sysname = platform.system()
    os_map = {"Darwin": "darwin", "Linux": "linux", "FreeBSD": "freebsd"}
    if sysname not in os_map:
        raise SystemExit(f"unsupported OS: {sysname} (need Darwin/Linux/FreeBSD)")
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "amd64"
    else:
        raise SystemExit(f"unsupported arch: {machine} (need arm64 or x86_64)")
    if os_map[sysname] == "freebsd" and arch != "amd64":
        raise SystemExit(f"no llama-swap release for {os_map[sysname]}/{arch}")
    return os_map[sysname], arch


def _resolve_latest_tag() -> str:
    print(f"[*] resolving latest release tag from github.com/{REPO_LLAMA_SWAP}...")
    url = f"{GH_API}/repos/{REPO_LLAMA_SWAP}/releases/latest"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            import json as _json
            tag = _json.load(resp).get("tag_name") or ""
    except Exception as e:
        raise SystemExit(f"could not resolve latest release tag: {e}") from None
    if not tag:
        raise SystemExit("could not resolve latest release tag (empty response)")
    print(f"[*] latest release: {tag}")
    return tag


def _installed_version_line(target: Path) -> str:
    """Return the first line of ``llama-swap --version`` (or empty on error)."""
    if not target.exists():
        return ""
    try:
        proc = subprocess.run(
            [str(target), "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "").splitlines()[0] if proc.stdout else ""


def latest_release_tag() -> str | None:
    """Best-effort lookup; returns ``None`` instead of raising."""
    try:
        url = f"{GH_API}/repos/{REPO_LLAMA_SWAP}/releases/latest"
        with urllib.request.urlopen(url, timeout=5) as resp:
            import json as _json
            tag = _json.load(resp).get("tag_name") or ""
        return tag or None
    except Exception:
        return None


def installed_version(target: Path) -> str | None:
    """Parse the version number out of ``--version`` output, e.g. ``"211"``."""
    line = _installed_version_line(target)
    m = VERSION_RE.search(line)
    return m.group(1) if m else None


def install_llama_swap(*, force: bool = False) -> Path:
    """Download/refresh the ``llama-swap`` binary.

    Returns the absolute path to the installed binary. ``force=True``
    re-downloads even when the version matches.
    """
    paths = ensure_data_dirs()
    target = paths.llama_swap_bin

    os_name, arch = _detect_os_arch()
    tag = os.environ.get("LLAMA_SWAP_VERSION", "").strip()
    if tag:
        print(f"[*] version: {tag} (from $LLAMA_SWAP_VERSION)")
    else:
        tag = _resolve_latest_tag()

    num = tag.lstrip("v")
    asset = f"llama-swap_{num}_{os_name}_{arch}.tar.gz"
    url = f"{GH_DL}/{REPO_LLAMA_SWAP}/releases/download/{tag}/{asset}"

    if target.exists() and not force:
        line = _installed_version_line(target)
        if line and re.search(rf"version:\s*v?{re.escape(num)}\b", line, re.IGNORECASE):
            print(f"[=] already installed: {target}")
            print(f"         {line}")
            print("    (re-run with --force to redownload)")
            return target
        if line:
            print(f"[*] currently installed: {line}")
            print(f"    upgrading to {tag}")

    paths.bin_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llmstack-llama-swap-") as tmp_dir:
        tmp = Path(tmp_dir)
        archive = tmp / asset

        print(f"[*] downloading {asset}")
        print(f"    from {url}")
        try:
            urllib.request.urlretrieve(url, archive)
        except Exception as e:
            raise SystemExit(f"download failed: {e}") from None

        print("[*] extracting")
        try:
            with tarfile.open(archive, "r:gz") as tf:
                member = next((m for m in tf.getmembers() if m.name == "llama-swap"), None)
                if member is None:
                    raise SystemExit("[!] tarball did not contain a top-level 'llama-swap' file")
                tf.extract(member, tmp)
        except tarfile.TarError as e:
            raise SystemExit(f"extract failed: {e}") from None

        extracted = tmp / "llama-swap"
        if not extracted.is_file():
            raise SystemExit("[!] tarball did not contain a top-level 'llama-swap' file")

        staged = target.with_suffix(".new")
        shutil.move(str(extracted), staged)
        staged.chmod(0o755)
        os.replace(staged, target)

    print(f"[OK] installed {target} ({os_name}/{arch})")
    line = _installed_version_line(target)
    if line:
        print(f"     {line}")
    return target
