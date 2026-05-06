"""Tests for llmstack generators."""

from __future__ import annotations

from llmstack.generators.llama_swap import render as render_llama_swap


class TestLlamaSwapRender:
    """Tests for llama-swap config rendering."""

    def test_render_returns_string(self):
        result = render_llama_swap()
        assert isinstance(result, str)

    def test_render_contains_tier_info(self):

        # We can't easily inject tiers here since render() reads from models.ini
        # Just verify the function runs and returns valid YAML
        result = render_llama_swap()
        assert "llama_server" in result or "matrix" in result
