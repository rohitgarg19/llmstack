"""Tests for llmstack tier data structures."""

from __future__ import annotations

from llmstack.tiers import (
    BACKEND_BEDROCK,
    BACKEND_GGUF,
    BedrockConfig,
    Tier,
)


class TestBedrockConfig:
    """Tests for BedrockConfig dataclass."""

    def test_basic_config(self):
        config = BedrockConfig(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert config.model_id == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert config.region is None
        assert config.profile is None

    def test_config_with_region(self):
        config = BedrockConfig(
            model_id="model-id",
            region="us-east-1",
        )
        assert config.region == "us-east-1"

    def test_has_next_false_when_no_next(self):
        config = BedrockConfig(model_id="model-id")
        assert config.has_next is False

    def test_has_next_true_with_next(self):
        config = BedrockConfig(
            model_id="model-id",
            model_id_next="model-id-v2",
        )
        assert config.has_next is True

    def test_resolved_returns_self_when_no_next(self):
        config = BedrockConfig(model_id="model-id")
        result = config.resolved(use_next=True)
        assert result is config

    def test_resolved_swaps_when_next_available(self):
        config = BedrockConfig(
            model_id="model-id-v1",
            model_id_next="model-id-v2",
            region="us-east-1",
            region_next="us-west-2",
        )
        result = config.resolved(use_next=True)
        assert result.model_id == "model-id-v2"
        assert result.region == "us-west-2"


class TestTier:
    """Tests for Tier dataclass."""

    def test_gguf_tier(self):
        tier = Tier(
            name="code-fast",
            role="agent",
            backend=BACKEND_GGUF,
            description="Fast coder",
            ctx_size=128000,
            repo="Qwen/Qwen2.5-Coder",
            file="qwen.Q5_K_M.gguf",
        )
        assert tier.is_gguf is True
        assert tier.is_bedrock is False

    def test_bedrock_tier(self):
        bedrock_config = BedrockConfig(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        tier = Tier(
            name="plan",
            role="chat",
            backend=BACKEND_BEDROCK,
            description="Plan mode",
            ctx_size=64000,
            bedrock=bedrock_config,
        )
        assert tier.is_gguf is False
        assert tier.is_bedrock is True

    def test_has_next_gguf_with_file_next(self):
        tier = Tier(
            name="code-smart",
            role="agent",
            backend=BACKEND_GGUF,
            description="Smart coder",
            ctx_size=64000,
            repo="Qwen/Qwen3-Coder",
            file="qwen.Q4_K_M.gguf",
            file_next="qwen.Q6_K.gguf",
        )
        assert tier.has_next is True

    def test_has_next_bedrock_with_model_id_next(self):
        bedrock_config = BedrockConfig(
            model_id="model-v1",
            model_id_next="model-v2",
        )
        tier = Tier(
            name="plan",
            role="chat",
            backend=BACKEND_BEDROCK,
            description="Plan mode",
            ctx_size=64000,
            bedrock=bedrock_config,
        )
        assert tier.has_next is True

    def test_files_returns_empty_for_bedrock(self):
        bedrock_config = BedrockConfig(model_id="model-id")
        tier = Tier(
            name="plan",
            role="chat",
            backend=BACKEND_BEDROCK,
            description="Plan mode",
            ctx_size=64000,
            bedrock=bedrock_config,
        )
        assert tier.files() == []
