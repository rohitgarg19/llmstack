"""Tests for llmstack path utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmstack.paths import EXAMPLES_DIR, models_ini_path, require_models_ini


@pytest.fixture(autouse=True)
def use_bundled_models_ini(monkeypatch):
    monkeypatch.setenv("LLMSTACK_MODELS_INI", str(EXAMPLES_DIR / "gguf" / "models.ini"))


class TestModelsIniPath:
    """Tests for models_ini_path()."""

    def test_returns_path_object(self):
        result = models_ini_path()
        assert isinstance(result, Path)

    def test_path_exists(self):
        result = models_ini_path()
        assert result.exists()

    def test_path_has_correct_name(self):
        result = models_ini_path()
        assert result.name == "models.ini"


class TestRequireModelsIni:
    """Tests for require_models_ini()."""

    def test_returns_path_when_exists(self):
        result = require_models_ini()
        assert isinstance(result, Path)
        assert result.exists()
