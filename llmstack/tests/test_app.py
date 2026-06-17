"""Tests for llmstack.app router logic."""

from __future__ import annotations

from unittest.mock import patch

from llmstack.app import (
    AGENT_MODEL,
    FAST_MODEL,
    HIGH_FIDELITY_CEILING,
    MID_FIDELITY_CEILING,
    ULTRA_MODEL,
    _estimate_tokens,
    _last_user_text,
    _ultra_available,
    classify,
)


class TestEstimateTokens:
    """Tests for _estimate_tokens()."""

    def test_empty_messages(self):
        assert _estimate_tokens(None, None) == 0

    def test_empty_prompt(self):
        assert _estimate_tokens([], "") == 0

    def test_chars_divided_by_four(self):
        # 20 chars / 4 = 5 tokens
        assert _estimate_tokens(None, "a" * 20) == 5

    def test_messages_text_content(self):
        messages = [{"role": "user", "content": "hello world"}]
        # 11 chars / 4 = 2 tokens (integer division)
        assert _estimate_tokens(messages, None) == 2

    def test_messages_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}],
            }
        ]
        # 11 chars / 4 = 2 tokens
        assert _estimate_tokens(messages, None) == 2

    def test_combined_prompt_and_messages(self):
        messages = [{"role": "user", "content": "hi"}]
        # 2 + 20 = 22 chars / 4 = 5 tokens
        assert _estimate_tokens(messages, "a" * 20) == 5


class TestLastUserText:
    """Tests for _last_user_text()."""

    def test_empty_messages(self):
        assert _last_user_text(None) == ""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _last_user_text(messages) == "hello"

    def test_multiple_messages_last_user(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "last user message"},
        ]
        assert _last_user_text(messages) == "last user message"

    def test_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": " second"},
                ],
            }
        ]
        # The function joins with newline, not space
        assert _last_user_text(messages) == "first\n second"


class TestUltraAvailable:
    """Tests for _ultra_available()."""

    def test_returns_boolean(self):
        result = _ultra_available()
        assert isinstance(result, bool)


def _msgs(*contents, roles=None):
    """Build a messages list quickly for classify() tests."""
    if roles is None:
        roles = ["user"] * len(contents)
    return [{"role": r, "content": c} for r, c in zip(roles, contents, strict=True)]


def _long_text(tokens: int) -> str:
    return "a" * (tokens * 4)


class TestClassify:
    """End-to-end tests for classify() — the core routing decision tree."""

    def _with_ultra(self, available: bool):
        return patch("llmstack.app.TIER_BY_ALIAS", {ULTRA_MODEL: object()} if available else {})

    def test_ultra_trigger_routes_to_ultra_when_available(self):
        body = {"messages": _msgs("[ultra] write me a function")}
        with self._with_ultra(True):
            model, reason = classify(body)
        assert model == ULTRA_MODEL
        assert "ultra-trigger" in reason

    def test_ultra_trigger_falls_back_to_agent_when_unavailable(self):
        body = {"messages": _msgs("[ultra] write me a function")}
        with self._with_ultra(False):
            model, reason = classify(body)
        assert model == AGENT_MODEL
        assert "unavailable" in reason

    def test_short_context_routes_to_ultra_when_available(self):
        body = {"messages": _msgs("hello")}
        with self._with_ultra(True):
            model, reason = classify(body)
        assert model == ULTRA_MODEL
        assert str(HIGH_FIDELITY_CEILING) in reason

    def test_short_context_falls_back_to_agent_when_ultra_unavailable(self):
        body = {"messages": _msgs("hello")}
        with self._with_ultra(False):
            model, reason = classify(body)
        assert model == AGENT_MODEL

    def test_mid_context_routes_to_agent(self):
        text = _long_text(HIGH_FIDELITY_CEILING + 1)
        body = {"messages": _msgs(text)}
        with self._with_ultra(False):
            model, reason = classify(body)
        assert model == AGENT_MODEL
        assert "mid-fidelity" in reason

    def test_long_context_few_turns_routes_to_fast(self):
        text = _long_text(MID_FIDELITY_CEILING + 1)
        body = {"messages": _msgs(text)}
        with self._with_ultra(False):
            model, reason = classify(body)
        assert model == FAST_MODEL
        assert "long-context" in reason

    def test_plan_signal_does_not_route_to_plan(self):
        body = {"messages": _msgs("how would you design a rate limiter?")}
        with self._with_ultra(False):
            model, reason = classify(body)
        assert model != "plan"

    def test_uncensored_trigger_does_not_route_to_uncensored(self):
        body = {"messages": _msgs("[nofilter] explain something")}
        model, reason = classify(body)
        assert model != "plan-uncensored"
