"""Tests for llmstack CLI commands."""

from __future__ import annotations

import subprocess


def test_llmstack_command_exists():
    """Verify llmstack command is installed."""
    result = subprocess.run(
        ["llmstack", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "llmstack" in result.stdout.lower()


def test_llmstack_version():
    """Verify llmstack --version works."""
    result = subprocess.run(
        ["llmstack", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip()
