"""Install (or update) the ``llama-swap`` binary.

Replaces the shell ``_install_llama_swap`` helper. Resolves the latest
GitHub release tag (or honours ``$LLAMA_SWAP_VERSION``), downloads the
asset for the current OS+arch, extracts the single ``llama-swap``
executable (``llama-swap.exe`` on Windows), and atomically renames it
into place under :func:`llmstack.paths.bin_dir`.

A second call short-circuits when the installed version already matches
the resolved tag, unless ``force=True`` is passed.

Asset naming on the upstream release matches goreleaser's convention:

  * POSIX:  ``llama-swap_<num>_<os>_<arch>.tar.gz``
  * Windows: ``llama-swap_<num>_windows_amd64.zip`` (only amd64 is published)
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
import zipfile
from pathlib import Path

from llmstack._platform import EXE_SUFFIX, IS_WINDOWS, make_executable
from llmstack.paths import REPO_LLAMA_SWAP, ensure_data_dirs

GH_API = "https://api.github.com"
GH_DL = "https://github.com"
VERSION_RE = re.compile(r"version:\s*v?([0-9][\w.-]*)", re.IGNORECASE)

BINARY_NAME = f"llama-swap{EXE_SUFFIX}"


def _detect_os_arch() -> tuple[str, str, str]:
    """Return ``(os_label, arch_label, archive_ext)`` for the current host.

    The third element drives the asset name suffix: ``"tar.gz"`` for the
    POSIX builds, ``"zip"`` for the Windows build. Goreleaser's defaults.
    """
    sysname = platform.system()
    os_map = {"Darwin": "darwin", "Linux": "linux", "FreeBSD": "freebsd", "Windows": "windows"}
    if sysname not in os_map:
        raise SystemExit(f"unsupported OS: {sysname} (need Darwin/Linux/FreeBSD/Windows)")
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "amd64"
    else:
        raise SystemExit(f"unsupported arch: {machine} (need arm64 or x86_64)")

    os_label = os_map[sysname]
    if os_label == "freebsd" and arch != "amd64":
        raise SystemExit(f"no llama-swap release for {os_label}/{arch}")
    if os_label == "windows":
        if arch != "amd64":
            raise SystemExit(
                f"no llama-swap windows release for {arch} -- "
                "only windows_amd64 is published upstream."
            )
        return os_label, arch, "zip"
    return os_label, arch, "tar.gz"


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


def _extract_binary(archive: Path, dest_dir: Path, *, archive_ext: str) -> Path:
    """Pull the ``llama-swap[.exe]`` file out of ``archive`` into ``dest_dir``.

    Returns the path to the extracted executable. We deliberately ignore
    the rest of the archive contents (READMEs, sample configs) -- the
    package only consumes the binary itself.
    """
    if archive_ext == "zip":
        try:
            with zipfile.ZipFile(archive) as zf:
                member = next((m for m in zf.namelist() if Path(m).name == BINARY_NAME), None)
                if member is None:
                    raise SystemExit(
                        f"[!] zip did not contain a top-level '{BINARY_NAME}' file"
                    )
                zf.extract(member, dest_dir)
                extracted = dest_dir / member
        except zipfile.BadZipFile as e:
            raise SystemExit(f"extract failed: {e}") from None
    else:
        try:
            with tarfile.open(archive, "r:gz") as tf:
                member = next((m for m in tf.getmembers() if Path(m.name).name == BINARY_NAME), None)
                if member is None:
                    raise SystemExit(
                        f"[!] tarball did not contain a top-level '{BINARY_NAME}' file"
                    )
                tf.extract(member, dest_dir)
                extracted = dest_dir / member.name
        except tarfile.TarError as e:
            raise SystemExit(f"extract failed: {e}") from None

    if not extracted.is_file():
        raise SystemExit(f"[!] archive did not yield a '{BINARY_NAME}' file")
    return extracted


def install_llama_swap(*, force: bool = False) -> Path:
    """Download/refresh the ``llama-swap`` binary.

    Returns the absolute path to the installed binary. ``force=True``
    re-downloads even when the version matches.
    """
    paths = ensure_data_dirs()
    target = paths.llama_swap_bin

    os_name, arch, archive_ext = _detect_os_arch()
    tag = os.environ.get("LLAMA_SWAP_VERSION", "").strip()
    if tag:
        print(f"[*] version: {tag} (from $LLAMA_SWAP_VERSION)")
    else:
        tag = _resolve_latest_tag()

    num = tag.lstrip("v")
    asset = f"llama-swap_{num}_{os_name}_{arch}.{archive_ext}"
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
        extracted = _extract_binary(archive, tmp, archive_ext=archive_ext)

        # Stage with a sibling name (NOT ``with_suffix(".new")`` -- on
        # Windows that would replace ".exe" with ".new" and lose the
        # executable extension).
        staged = target.with_name(target.name + ".new")
        if staged.exists():
            staged.unlink()
        shutil.move(str(extracted), staged)
        make_executable(staged)
        # Windows ``os.replace`` on an open / running binary fails with
        # ERROR_ACCESS_DENIED; the daemon must be stopped before
        # upgrading. We don't try to be clever about it.
        if IS_WINDOWS and target.exists():
            try:
                target.unlink()
            except OSError as e:
                staged.unlink(missing_ok=True)
                raise SystemExit(
                    f"[!] could not replace {target}: {e}\n"
                    "    is llama-swap still running? stop the stack first: "
                    "llmstack stop"
                ) from None
        os.replace(staged, target)

    print(f"[OK] installed {target} ({os_name}/{arch})")
    line = _installed_version_line(target)
    if line:
        print(f"     {line}")
    return target
