"""Tests for llmstack package."""

from __future__ import annotations

from llmstack.tiers import parse_sampler


class TestParseSampler:
    """Tests for parse_sampler()."""

    def test_empty_string(self):
        assert parse_sampler("") == {}

    def test_none(self):
        assert parse_sampler(None) == {}

    def test_single_key_value(self):
        result = parse_sampler("temp=0.5")
        assert result == {"temp": 0.5}

    def test_multiple_key_values(self):
        result = parse_sampler("temp=0.5, top_p=0.85, top_k=20")
        assert result == {"temp": 0.5, "top_p": 0.85, "top_k": 20}

    def test_spaces_around_equals(self):
        result = parse_sampler("temp = 0.5, top_p = 0.85")
        assert result == {"temp": 0.5, "top_p": 0.85}

    def test_integer_values(self):
        result = parse_sampler("top_k=20, min_p=1")
        assert result == {"top_k": 20.0, "min_p": 1.0}

    def test_preserves_order(self):
        result = parse_sampler("rep_pen=1.1, temp=0.2, top_p=0.9")
        assert list(result.keys()) == ["rep_pen", "temp", "top_p"]
