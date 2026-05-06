"""Tests for llmstack.app router logic."""

from __future__ import annotations

import re

from llmstack.app import (
    AGENT_SIGNALS,
    CODE_BLOCK,
    PLAN_SIGNALS,
    ULTRA_TRIGGERS,
    UNCENSORED_TRIGGERS,
    _estimate_tokens,
    _last_user_text,
    _matches,
    _ultra_available,
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


class TestMatches:
    """Tests for _matches()."""

    def test_empty_pattern(self):
        assert _matches(re.compile(r""), None, None) is False

    def test_pattern_matches_prompt(self):
        pattern = re.compile(r"hello")
        assert _matches(pattern, None, "hello world") is True

    def test_pattern_matches_message_text(self):
        pattern = re.compile(r"hello")
        messages = [{"role": "user", "content": "say hello"}]
        assert _matches(pattern, messages, None) is True

    def test_no_match(self):
        pattern = re.compile(r"xyz")
        messages = [{"role": "user", "content": "hello world"}]
        assert _matches(pattern, messages, None) is False


class TestTriggerPatterns:
    """Tests for trigger regex patterns."""

    def test_uncensored_triggers(self):
        assert UNCENSORED_TRIGGERS.search("[uncensored]") is not None
        assert UNCENSORED_TRIGGERS.search("[nofilter]") is not None
        assert UNCENSORED_TRIGGERS.search("uncensored: ") is not None
        assert UNCENSORED_TRIGGERS.search("nofilter:") is not None

    def test_ultra_triggers(self):
        assert ULTRA_TRIGGERS.search("[ultra]") is not None
        assert ULTRA_TRIGGERS.search("[opus]") is not None
        assert ULTRA_TRIGGERS.search("ultra: ") is not None
        assert ULTRA_TRIGGERS.search("opus: ") is not None

    def test_plan_signals(self):
        assert PLAN_SIGNALS.search("design") is not None
        assert PLAN_SIGNALS.search("architect") is not None
        assert PLAN_SIGNALS.search("should we") is not None
        assert PLAN_SIGNALS.search("explain why") is not None

    def test_agent_signals(self):
        assert AGENT_SIGNALS.search("implement") is not None
        assert AGENT_SIGNALS.search("write a function") is not None
        assert AGENT_SIGNALS.search("fix the bug") is not None

    def test_code_block(self):
        assert CODE_BLOCK.search("```python") is not None
        assert CODE_BLOCK.search("`" + "a" * 31 + "`") is not None


class TestUltraAvailable:
    """Tests for _ultra_available()."""

    def test_returns_boolean(self):
        result = _ultra_available()
        assert isinstance(result, bool)
