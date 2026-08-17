"""Tests for llmstack tier data structures."""

from __future__ import annotations

from llmstack.tiers import (
    BACKEND_GGUF,
    BACKEND_LITELLM,
    LiteLLMConfig,
    Tier,
)


class TestLiteLLMConfig:
    """Tests for LiteLLMConfig dataclass."""

    def test_basic_config(self):
        config = LiteLLMConfig(model="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert config.model == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_has_next_false_when_no_next(self):
        config = LiteLLMConfig(model="model-id")
        assert config.has_next is False

    def test_has_next_true_with_next(self):
        config = LiteLLMConfig(
            model="model-id",
            model_next="model-id-v2",
        )
        assert config.has_next is True

    def test_resolved_returns_self_when_no_next(self):
        config = LiteLLMConfig(model="model-id")
        result = config.resolved(use_next=True)
        assert result is config

    def test_resolved_swaps_when_next_available(self):
        config = LiteLLMConfig(
            model="model-id-v1",
            model_next="model-id-v2",
        )
        result = config.resolved(use_next=True)
        assert result.model == "model-id-v2"


class TestTier:
    """Tests for Tier dataclass."""

    def test_gguf_tier(self):
        tier = Tier(
            name="code-fast",
            tier="subagent",
            role="agent",
            backend=BACKEND_GGUF,
            description="Fast coder",
            ctx_size=128000,
            repo="Qwen/Qwen2.5-Coder",
            file="qwen.Q5_K_M.gguf",
        )
        assert tier.is_gguf is True
        assert tier.is_litellm is False

    def test_litellm_tier(self):
        litellm_config = LiteLLMConfig(
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        tier = Tier(
            name="plan",
            tier="agent",
            role="chat",
            backend=BACKEND_LITELLM,
            description="Plan mode",
            ctx_size=64000,
            litellm=litellm_config,
        )
        assert tier.is_gguf is False
        assert tier.is_litellm is True

    def test_has_next_gguf_with_file_next(self):
        tier = Tier(
            name="code-smart",
            role="build",
            tier="agent",
            backend=BACKEND_GGUF,
            description="Smart coder",
            ctx_size=64000,
            repo="Qwen/Qwen3-Coder",
            file="qwen.Q4_K_M.gguf",
            file_next="qwen.Q6_K.gguf",
        )
        assert tier.has_next is True

    def test_has_next_litellm_with_model_next(self):
        litellm_config = LiteLLMConfig(
            model="model-v1",
            model_next="model-v2",
        )
        tier = Tier(
            name="plan",
            role="chat",
            tier="agent",
            backend=BACKEND_LITELLM,
            description="Plan mode",
            ctx_size=64000,
            litellm=litellm_config,
        )
        assert tier.has_next is True

    def test_files_returns_empty_for_litellm(self):
        litellm_config = LiteLLMConfig(model="model-id")
        tier = Tier(
            name="plan",
            role="chat",
            tier="agent",
            backend=BACKEND_LITELLM,
            description="Plan mode",
            ctx_size=64000,
            litellm=litellm_config,
        )
        assert tier.files() == []
