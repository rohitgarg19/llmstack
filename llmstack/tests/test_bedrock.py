"""Tests for llmstack.backends.bedrock message conversion."""

from __future__ import annotations

from llmstack.backends.bedrock import (
    _converse_messages,
    _messages_reference_tools,
    _stub_tool_config,
)


class TestConverseMessages:
    """Tests for _converse_messages()."""

    def test_empty_messages(self):
        assert _converse_messages([]) == []

    def test_system_message_excluded(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = _converse_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _converse_messages(messages)
        assert result == [{"role": "user", "content": [{"text": "Hello"}]}]

    def test_assistant_message(self):
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = _converse_messages(messages)
        assert result == [{"role": "assistant", "content": [{"text": "Hi there"}]}]

    def test_tool_result_in_user_turn(self):
        messages = [
            {"role": "user", "content": "Run tool"},
            {
                "role": "tool",
                "tool_call_id": "tool_123",
                "content": "result value",
            },
        ]
        result = _converse_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1] == {
            "toolResult": {
                "toolUseId": "tool_123",
                "content": [{"text": "result value"}],
            }
        }

    def test_tool_use_from_assistant(self):
        messages = [
            {
                "role": "assistant",
                "content": "I'll run a tool",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
            }
        ]
        result = _converse_messages(messages)
        # The function injects a toolResult stub for orphan toolUse blocks
        assert len(result) == 2
        # First is the assistant turn with text + toolUse
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1] == {
            "toolUse": {
                "toolUseId": "call_123",
                "name": "get_weather",
                "input": {"city": "NYC"},
            }
        }
        # Second is the injected toolResult stub user turn
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0] == {
            "toolResult": {
                "toolUseId": "call_123",
                "content": [{"text": "(no result; tool call was cancelled or interrupted -- treat as failed)"}],
                "status": "error",
            }
        }


class TestMessagesReferenceTools:
    """Tests for _messages_reference_tools()."""

    def test_empty_messages(self):
        assert _messages_reference_tools([]) == set()

    def test_no_tool_use(self):
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi"}]},
        ]
        assert _messages_reference_tools(messages) == set()

    def test_single_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_1",
                            "name": "get_weather",
                        }
                    }
                ],
            }
        ]
        assert _messages_reference_tools(messages) == {"get_weather"}

    def test_multiple_tool_uses(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "t1", "name": "tool_a"}},
                    {"toolUse": {"toolUseId": "t2", "name": "tool_b"}},
                ],
            }
        ]
        assert _messages_reference_tools(messages) == {"tool_a", "tool_b"}


class TestStubToolConfig:
    """Tests for _stub_tool_config()."""

    def test_empty_names(self):
        result = _stub_tool_config(set())
        assert result == {"tools": []}

    def test_single_name(self):
        result = _stub_tool_config({"get_weather"})
        assert result == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_weather",
                        "description": "(replayed from history; schema unavailable)",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }

    def test_sorted_names(self):
        result = _stub_tool_config({"zebra", "apple", "banana"})
        names = [t["toolSpec"]["name"] for t in result["tools"]]
        assert names == ["apple", "banana", "zebra"]
