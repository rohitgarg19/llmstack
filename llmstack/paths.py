"""Centralised path & state-dir resolution.

Three classes of path are kept apart on purpose:

  *Source* paths
    Things that ship with the package (``AGENTS.md`` template). These are
    read-only after ``pip install`` and live next to the package modules.

  *User-data* paths (writable, persistent, shared across projects)
    The ``llama-swap`` binary, mostly. Lives under
    ``$XDG_DATA_HOME/llmstack/`` (defaults to ``~/.local/share/llmstack/``)
    so a single download is reused regardless of which project the CLI
    was invoked from. Override with ``$LLMSTACK_BIN_DIR`` or
    ``$LLMSTACK_DATA_DIR``.

  *Work-dir* paths (writable, per-project)
    Generated configs, pid files, channel markers, prompt rcfiles and
    daemon logs. Lives at ``<work-dir>/.llmstack/`` so each project gets
    its own ``opencode.json`` + ``llama-swap.yaml``. ``work-dir`` is
    ``$LLMSTACK_WORK_DIR`` if set, else ``$PWD`` at invocation time.

The state-dir resolution uses a frozen :class:`Paths` instance constructed
once per CLI invocation so we don't accidentally pick up a different
``$PWD`` mid-command.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parent

AGENTS_TEMPLATE = PACKAGE_DIR / "AGENTS.md"
MODELS_INI_TEMPLATE = PACKAGE_DIR / "models.ini"

REPO_LLAMA_SWAP = "mostlygeek/llama-swap"
ROUTER_HOST = "127.0.0.1"
ROUTER_PORT = 10101
SWAP_PORT = 10102


def remote_url() -> str | None:
    """Base URL of a *remote* llmstack router (router endpoint, no ``/v1``).

    When set via ``LLMSTACK_REMOTE_URL`` the stack runs in **client mode**:

      - no daemons are launched (no llama-swap, no router)
      - no GGUFs / binaries need to live on this host
      - ``opencode.json`` is generated with ``baseURL`` pointing at the
        remote router instead of ``http://127.0.0.1:10101/v1``
      - ``LLMSTACK_CHANNEL`` is exported as ``external``

    Returns ``None`` (= local mode) when the env var is unset or empty.
    Trailing slashes are stripped so callers can build URLs by simple
    concatenation (``f"{remote_url()}/health"``).
    """
    raw = (os.environ.get("LLMSTACK_REMOTE_URL") or "").strip().rstrip("/")
    return raw or None


def is_remote() -> bool:
    """Convenience wrapper around :func:`remote_url`."""
    return remote_url() is not None


def router_health_url() -> str:
    """Where to probe for ``/health`` -- remote URL when set, else localhost."""
    return f"{remote_url() or f'http://{ROUTER_HOST}:{ROUTER_PORT}'}/health"


def _xdg_data_home() -> Path:
    """``$XDG_DATA_HOME`` with the spec-defined fallback."""
    raw = os.environ.get("XDG_DATA_HOME") or ""
    return Path(raw) if raw else Path.home() / ".local" / "share"


def models_ini_path() -> Path:
    """Locate ``models.ini``.

    Canonical location is ``<work-dir>/.llmstack/models.ini`` (per-project,
    sits next to the rest of the generated state but is itself the *input*
    to ``install``). ``$LLMSTACK_MODELS_INI`` overrides this with an
    explicit absolute or relative path. Returns the resolved path even
    when the file doesn't exist; callers decide whether that's an error
    or a seed opportunity.
    """
    explicit = os.environ.get("LLMSTACK_MODELS_INI")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (work_dir() / ".llmstack" / "models.ini").resolve()


def work_dir() -> Path:
    """Where per-project state (``.llmstack/``) lives."""
    raw = os.environ.get("LLMSTACK_WORK_DIR")
    return Path(raw).expanduser().resolve() if raw else Path.cwd().resolve()


def data_dir() -> Path:
    """Persistent user-data root for the package (binary, etc.)."""
    raw = os.environ.get("LLMSTACK_DATA_DIR")
    return Path(raw).expanduser().resolve() if raw else (_xdg_data_home() / "llmstack").resolve()


def bin_dir() -> Path:
    """Where ``llama-swap`` is installed. Falls back under ``data_dir()``."""
    raw = os.environ.get("LLMSTACK_BIN_DIR")
    return Path(raw).expanduser().resolve() if raw else (data_dir() / "bin").resolve()


@dataclass(frozen=True)
class Paths:
    """Snapshot of the resolved paths for a single CLI invocation."""

    work_dir: Path
    data_dir: Path
    bin_dir: Path
    state_dir: Path           # <work>/.llmstack
    log_dir: Path             # <state>/logs
    llama_swap_bin: Path      # <bin>/llama-swap
    llama_swap_yaml: Path     # <state>/llama-swap.yaml  (was llmstack/llama-swap.yaml)
    opencode_json: Path       # <state>/opencode.json
    agents_local: Path        # <state>/AGENTS.md (copy of template)
    active_marker: Path       # <state>/active-channel
    default_marker: Path      # <state>/default-channel
    router_pid: Path          # <state>/router.pid
    swap_pid: Path            # <state>/llama-swap.pid

    @property
    def models_ini(self) -> Path:
        return models_ini_path()


@lru_cache(maxsize=1)
def resolve() -> Paths:
    """Build (and cache) the path snapshot for this process."""
    work = work_dir()
    state = work / ".llmstack"
    data = data_dir()
    bind = bin_dir()
    opencode_dir = Path(os.environ["OPENCODE_CONFIG_DIR"]).expanduser().resolve() \
        if "OPENCODE_CONFIG_DIR" in os.environ else state
    return Paths(
        work_dir=work,
        data_dir=data,
        bin_dir=bind,
        state_dir=state,
        log_dir=state / "logs",
        llama_swap_bin=bind / "llama-swap",
        llama_swap_yaml=state / "llama-swap.yaml",
        opencode_json=opencode_dir / "opencode.json",
        agents_local=state / "AGENTS.md",
        active_marker=state / "active-channel",
        default_marker=state / "default-channel",
        router_pid=state / "router.pid",
        swap_pid=state / "llama-swap.pid",
    )


def ensure_state_dirs() -> Paths:
    """Create the per-project state dirs lazily, return the Paths snapshot.

    Read-only commands (``help``, ``activate``) deliberately don't call
    this -- otherwise running them from any directory would litter the
    filesystem with empty ``.llmstack/`` folders.
    """
    p = resolve()
    p.state_dir.mkdir(parents=True, exist_ok=True)
    p.log_dir.mkdir(parents=True, exist_ok=True)
    return p


def ensure_data_dirs() -> Paths:
    """Create the user-data dirs (for the binary install)."""
    p = resolve()
    p.data_dir.mkdir(parents=True, exist_ok=True)
    p.bin_dir.mkdir(parents=True, exist_ok=True)
    return p


def require_models_ini() -> Path:
    p = models_ini_path()
    if not p.is_file():
        raise SystemExit(
            f"[!] models.ini not found at {p}\n"
            "    set $LLMSTACK_MODELS_INI or run `llmstack install` to seed one"
        )
    return p


def ensure_models_ini() -> tuple[Path, bool]:
    """Resolve ``models.ini``, seeding the canonical location from the
    bundled template when nothing exists yet. Returns ``(path, seeded)``
    where ``seeded`` is ``True`` only when we just wrote the file.

    Lookup follows :func:`models_ini_path`. When neither the canonical
    nor the legacy location has a file, the seed is written to the
    canonical path: ``<work-dir>/.llmstack/models.ini`` (or wherever
    ``$LLMSTACK_MODELS_INI`` points).
    """
    p = models_ini_path()
    if p.is_file():
        return p, False
    if not MODELS_INI_TEMPLATE.is_file():
        raise SystemExit(
            f"[!] models.ini not found at {p} and no bundled template at "
            f"{MODELS_INI_TEMPLATE}\n"
            "    reinstall the llmstack package or set $LLMSTACK_MODELS_INI"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    _shutil.copyfile(MODELS_INI_TEMPLATE, p)
    return p, True


# ---------------------------------------------------------------------------
# channel-marker on-disk format
#
# Both ``.llmstack/active-channel`` (live) and ``.llmstack/default-channel``
# (intent, written by install) use the same one-line format:
#
#     <channel>[ <url>]\n
#
# For local channels (``current`` / ``next`` / ``shared``) the line is just
# the channel name. For ``external`` we append the remote llmstack URL so
# the activate hook can re-export ``LLMSTACK_REMOTE_URL`` without the user
# having to set it in their shell rc -- entering a project is enough to
# wire the env back up.
#
# The format is deliberately whitespace-separated (not JSON / TSV) so a
# shell can parse it with ``read -r channel url < marker`` -- no jq, no
# python, just a builtin.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelMark:
    """One channel-marker file's worth of state."""

    channel: str
    url: Optional[str] = None

    def serialize(self) -> str:
        if self.url:
            return f"{self.channel} {self.url}\n"
        return f"{self.channel}\n"

    @classmethod
    def parse(cls, text: str) -> "ChannelMark | None":
        line = text.strip()
        if not line:
            return None
        parts = line.split(maxsplit=1)
        return cls(parts[0], parts[1] if len(parts) > 1 else None)


def read_marker(path: Path) -> ChannelMark | None:
    """Return the parsed marker, or ``None`` when the file is missing/empty."""
    if not path.is_file():
        return None
    try:
        return ChannelMark.parse(path.read_text())
    except OSError:
        return None


def write_marker(path: Path, mark: ChannelMark) -> None:
    """Atomically write ``mark`` to ``path`` (creates parents as needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(mark.serialize())
    os.replace(tmp, path)
