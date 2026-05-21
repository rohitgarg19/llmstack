"""Tests for llmstack.commands.start -- argument parsing and _persist_ini_defaults."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from llmstack.commands.start import _persist_ini_defaults


@pytest.fixture()
def ini_file(tmp_path: Path) -> Path:
    p = tmp_path / "models.ini"
    p.write_text(
        textwrap.dedent("""\
            [DEFAULT]
            router_host  = 127.0.0.1
            router_port  = 10101
            n_gpu_layers = 999

            [code-fast]
            role = fast
        """),
        encoding="utf-8",
    )
    return p


class TestPersistIniDefaults:
    def test_updates_host(self, ini_file: Path):
        with patch("llmstack.commands.start.models_ini_path", return_value=ini_file):
            _persist_ini_defaults("0.0.0.0", None)
        text = ini_file.read_text(encoding="utf-8")
        assert "router_host  = 0.0.0.0" in text
        assert "router_port  = 10101" in text

    def test_updates_port(self, ini_file: Path):
        with patch("llmstack.commands.start.models_ini_path", return_value=ini_file):
            _persist_ini_defaults(None, 9999)
        text = ini_file.read_text(encoding="utf-8")
        assert "router_host  = 127.0.0.1" in text
        assert "router_port  = 9999" in text

    def test_updates_both(self, ini_file: Path):
        with patch("llmstack.commands.start.models_ini_path", return_value=ini_file):
            _persist_ini_defaults("192.168.1.1", 8080)
        text = ini_file.read_text(encoding="utf-8")
        assert "router_host  = 192.168.1.1" in text
        assert "router_port  = 8080" in text

    def test_preserves_other_keys(self, ini_file: Path):
        with patch("llmstack.commands.start.models_ini_path", return_value=ini_file):
            _persist_ini_defaults("0.0.0.0", 8080)
        text = ini_file.read_text(encoding="utf-8")
        assert "n_gpu_layers = 999" in text
        assert "[code-fast]" in text

    def test_noop_when_file_missing(self, tmp_path: Path):
        missing = tmp_path / "no-such-file.ini"
        with patch("llmstack.commands.start.models_ini_path", return_value=missing):
            _persist_ini_defaults("0.0.0.0", 8080)
        assert not missing.exists()


class TestStartArgParsing:
    """Test that run() parses --host / --port correctly (without launching daemons)."""

    def test_host_parsed(self, tmp_path: Path):
        from llmstack.commands.start import run

        with (
            patch("llmstack.commands.start.ensure_state_dirs") as mock_paths,
            patch("llmstack.commands.start.read_marker", return_value=None),
            patch("llmstack.commands.start.models_ini_path", return_value=tmp_path / "m.ini"),
        ):
            mock_paths.return_value.llama_swap_bin.exists.return_value = False
            with pytest.raises(SystemExit):
                run(["--host", "0.0.0.0"])

    def test_port_parsed(self, tmp_path: Path):
        from llmstack.commands.start import run

        with (
            patch("llmstack.commands.start.ensure_state_dirs") as mock_paths,
            patch("llmstack.commands.start.read_marker", return_value=None),
            patch("llmstack.commands.start.models_ini_path", return_value=tmp_path / "m.ini"),
        ):
            mock_paths.return_value.llama_swap_bin.exists.return_value = False
            with pytest.raises(SystemExit):
                run(["--port", "9999"])

    def test_invalid_port_rejected(self):
        from llmstack.commands.start import run

        rc = run(["--port", "abc"])
        assert rc == 2

    def test_unknown_arg_rejected(self):
        from llmstack.commands.start import run

        rc = run(["--bogus"])
        assert rc == 2
