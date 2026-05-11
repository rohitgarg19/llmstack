#!/usr/bin/env python3
"""Bump the llmstack version across all files that declare it.

Usage:
    python scripts/bump_version.py <new_version>
    python scripts/bump_version.py --bump patch|minor|major [--rc <n>]

Examples:
    python scripts/bump_version.py 1.0.0
    python scripts/bump_version.py 0.10.0-rc.1
    python scripts/bump_version.py --bump minor --rc 1   # 0.9.4 -> 0.10.0-rc.1
    python scripts/bump_version.py --bump patch          # 0.9.4 -> 0.9.5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILES = [
    (ROOT / "pyproject.toml", re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)),
    (ROOT / "llmstack" / "__init__.py", re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.MULTILINE)),
]


def read_current_version() -> str:
    init_file = ROOT / "llmstack" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init_file.read_text(encoding="utf-8"))
    if not match:
        sys.exit("ERROR: cannot read current version from llmstack/__init__.py")
    return match.group(1)


def parse_version(v: str) -> tuple[int, int, int, str | None]:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$", v)
    if not m:
        sys.exit(f"ERROR: invalid version format: {v!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def compute_bumped(current: str, bump: str, rc: int | None) -> str:
    major, minor, patch, _ = parse_version(current)
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        major, minor, patch = major, minor + 1, 0
    elif bump == "patch":
        major, minor, patch = major, minor, patch + 1
    else:
        sys.exit(f"ERROR: unknown bump type: {bump!r}")
    version = f"{major}.{minor}.{patch}"
    if rc is not None:
        version += f"-rc.{rc}"
    return version


def apply_version(new_version: str) -> None:
    for filepath, pattern in VERSION_FILES:
        if not filepath.exists():
            print(f"  SKIP (not found): {filepath}")
            continue
        text = filepath.read_text(encoding="utf-8")
        new_text, count = pattern.subn(rf"\g<1>{new_version}\g<3>", text)
        if count == 0:
            print(f"  WARN (no match): {filepath}")
            continue
        filepath.write_text(new_text, encoding="utf-8")
        print(f"  updated: {filepath.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump llmstack version.")
    parser.add_argument("version", nargs="?", help="Explicit version string (e.g. 1.0.0-rc.1)")
    parser.add_argument("--bump", choices=["patch", "minor", "major"], help="Bump type")
    parser.add_argument("--rc", type=int, default=None, help="RC number (appends -rc.N suffix)")
    args = parser.parse_args()

    if args.version and args.bump:
        sys.exit("ERROR: provide either an explicit version OR --bump, not both.")
    if not args.version and not args.bump:
        sys.exit("ERROR: provide a version string or --bump {patch|minor|major}.")

    current = read_current_version()

    if args.version:
        new_version = args.version
    else:
        new_version = compute_bumped(current, args.bump, args.rc)

    parse_version(new_version.split("-")[0])

    print(f"Bumping version: {current} -> {new_version}")
    apply_version(new_version)
    print("Done.")


if __name__ == "__main__":
    main()
