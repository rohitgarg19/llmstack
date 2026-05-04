"""Background GGUF downloader.

Replaces the shell ``cmd_download`` action. We shell out to
``llama-completion`` (preferred; modern llama.cpp split: chat=llama-cli,
one-shot=llama-completion) or legacy ``llama-cli`` because the standard
llama.cpp HF cache uses a resumable partial-file convention
(``.downloadInProgress``) that ``huggingface_hub.hf_hub_download`` does
not understand. Co-mixing the two would leave un-resumable partial blobs
on disk -- see ``UPGRADING.md`` "Cache management".

Every download is launched as a backgrounded subprocess with its own
log file at ``<state>/logs/dl-<tier>-<label>.log``. We do **not** wait
for them to finish; the caller decides whether to poll
:func:`running_downloads`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from llmstack.paths import ensure_state_dirs, resolve, require_models_ini
from llmstack.tiers import iter_download_targets

LLAMA_BINS = ("llama-completion", "llama-cli")


@dataclass(frozen=True)
class DownloadJob:
    """A single backgrounded ``llama-*`` invocation."""

    tag: str
    repo: str
    file: str
    label: str
    log: Path
    pid: int


def _find_llama_bin() -> str:
    for candidate in LLAMA_BINS:
        path = shutil.which(candidate)
        if path:
            return path
    raise SystemExit(
        "[!] neither llama-completion nor llama-cli found in PATH "
        "(brew install llama.cpp)"
    )


def _spawn(llama_bin: str, repo: str, file: str, log: Path, hf_token: str | None) -> int:
    """Launch a backgrounded one-shot completion that downloads ``repo/file``."""
    argv: list[str] = [
        llama_bin,
        "-hf", repo,
        "-hff", file,
        "--no-warmup",
        "-ngl", "0",
        "-c", "256",
        "-p", "ok",
        "-n", "1",
    ]
    if hf_token:
        argv += ["--hf-token", hf_token]

    log.parent.mkdir(parents=True, exist_ok=True)
    fp = log.open("wb")
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    fp.close()
    return proc.pid


def download_all() -> list[DownloadJob]:
    """Kick off downloads for every tier file declared in models.ini.

    Returns the list of launched jobs. Always non-empty: raises
    :exc:`SystemExit` if the ini has no download targets.
    """
    require_models_ini()
    paths = ensure_state_dirs()
    llama_bin = _find_llama_bin()
    hf_token = os.environ.get("HF_TOKEN") or None

    print(f"[*] downloader: {llama_bin}")
    print("[*] cache:      ~/.cache/huggingface/hub  (default for llama.cpp)")
    print(f"[*] inventory:  {paths.models_ini}")
    if hf_token:
        print("[*] HF_TOKEN set (faster rate limits)")
    else:
        print("[*] no HF_TOKEN (rate-limited unauthenticated downloads)")
    print()

    jobs: list[DownloadJob] = []
    for tf in iter_download_targets():
        log = paths.log_dir / f"dl-{tf.tag}.log"
        print(f"[*] {tf.tag:<32} ({tf.label:<7}) {tf.repo} / {tf.file}")
        print(f"    log -> {log}")
        pid = _spawn(llama_bin, tf.repo, tf.file, log, hf_token)
        print(f"    pid -> {pid}")
        jobs.append(DownloadJob(
            tag=tf.tag, repo=tf.repo, file=tf.file, label=tf.label,
            log=log, pid=pid,
        ))

    if not jobs:
        raise SystemExit(f"no download targets found in {paths.models_ini}")

    print()
    print(f"{len(jobs)} download(s) queued in the background.")
    print()
    print("Watch progress:")
    print(f"    tail -f {paths.log_dir}/dl-*.log")
    print("    llama-cli -cl                        # lists completed cache entries")
    print()
    print("When you want to try queued upgrade targets without committing:")
    print("    llmstack stop && llmstack start --next")
    return jobs


def running_downloads() -> int:
    """Return the count of in-flight ``llama-{completion,cli}`` HF downloads.

    Uses ``pgrep -f`` because we don't want to add a ``psutil`` dependency
    just for one ad-hoc check. Returns 0 when ``pgrep`` is missing.
    """
    if not shutil.which("pgrep"):
        return 0
    pattern = r"llama-(completion|cli) .*-hf "
    try:
        proc = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return 0
    if proc.returncode not in (0, 1):
        return 0
    pids = [line for line in proc.stdout.splitlines() if line.strip()]
    return len(pids)


def wait_for_downloads(poll_seconds: float = 10.0, *, log_dir: Path | None = None) -> None:
    """Block until no ``llama-*`` HF download subprocesses remain.

    Prints a one-liner every ``poll_seconds`` so the user can see we're
    not hung. Honours Ctrl-C politely.
    """
    log = log_dir or resolve().log_dir
    print(f"      (logs: {log}/dl-*.log)")
    time.sleep(2)
    try:
        while True:
            n = running_downloads()
            if n == 0:
                break
            print(f"      {n} download(s) still running...")
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n[!] interrupted -- downloads continue in the background.", file=sys.stderr)
        raise SystemExit(130) from None
    print("[OK] all downloads complete.")
