"""Tests for llmstack generators."""

from __future__ import annotations

import pytest

from llmstack.generators.opencode import build_config
from llmstack.paths import EXAMPLES_DIR

GGUF_INI = """
[DEFAULT]
router_host  = 127.0.0.1
router_port  = 10101

[code-fast]
tier     = subagent
role     = build
hf_repo  = bartowski/Qwen2.5-Coder-3B-Instruct-GGUF
hf_file  = Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf
ctx_size = 131072
max_output_tokens = 8192
sampler  = temp=0.2
description = Qwen2.5-Coder 3B fast

[code-smart]
tier     = agent
role     = build
hf_repo  = unsloth/Qwen3-Coder-Next-GGUF
hf_file  = Qwen3-Coder-Next-Q4_K_M.gguf
ctx_size = 64000
max_output_tokens = 32768
sampler  = temp=0.5
description = Qwen3-Coder-Next agent

[plan]
tier     = agent
role     = chat
hf_repo  = Jackrong/Qwopus-GLM-18B-Merged-GGUF
hf_file  = Qwopus-GLM-18B-Healed-Q4_K_M.gguf
ctx_size = 65536
max_output_tokens = 16384
sampler  = temp=0.7
description = Qwopus plan

[plan-uncensored]
tier     = agent
role     = nofilter-chat
hf_repo  = mradermacher/Mistral-Small-3.2-24B-Instruct-GGUF
hf_file  = Mistral-Small.gguf
ctx_size = 131072
max_output_tokens = 16384
sampler  = temp=0.85
description = Mistral uncensored

[ROUTING]
high_fidelity_ceiling = 12000
mid_fidelity_ceiling  = 32000
"""

LITELLM_INI = """
[DEFAULT]
router_host = 127.0.0.1
router_port = 10101

[code-fast]
tier         = subagent
role         = build
backend      = litellm
model        = anthropic/claude-haiku-4-5-20251001
ctx_size     = 200000
max_output_tokens = 4096
description  = Haiku fast

[code-smart]
tier         = agent
role         = build
backend      = litellm
model        = anthropic/claude-sonnet-4-20250514
ctx_size     = 200000
max_output_tokens = 16384
description  = Sonnet agent

[ROUTING]
high_fidelity_ceiling = 12000
mid_fidelity_ceiling  = 32000
"""


class TestBuildConfigGguf:
    def setup_method(self):
        self.cfg = build_config(ini_text=GGUF_INI)

    def test_schema_present(self):
        assert "$schema" in self.cfg

    def test_provider_key(self):
        assert "llama-swap" in self.cfg["provider"]

    def test_base_url_uses_ini_host_port(self):
        url = self.cfg["provider"]["llama-swap"]["options"]["baseURL"]
        assert "127.0.0.1:10101" in url

    def test_auto_model_present(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert "auto" in models

    def test_auto_ctx_equals_fast_ctx(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["auto"]["limit"]["context"] == 131072

    def test_auto_output_equals_fast_output(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["auto"]["limit"]["output"] == 8192

    def test_tier_output_limits_from_ini(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["code-fast"]["limit"]["output"] == 8192
        assert models["code-smart"]["limit"]["output"] == 32768
        assert models["plan"]["limit"]["output"] == 16384
        assert models["plan-uncensored"]["limit"]["output"] == 16384

    def test_small_model_wired_to_fast(self):
        assert self.cfg["small_model"] == "llama-swap/code-fast"

    def test_build_agent_wired_to_auto(self):
        assert self.cfg["agent"]["build"]["model"] == "llama-swap/auto"

    def test_plan_agent_wired_to_plan_tier(self):
        assert self.cfg["agent"]["plan"]["model"] == "llama-swap/plan"

    def test_plan_agent_is_read_only(self):
        perm = self.cfg["agent"]["plan"]["permission"]
        assert perm["bash"] == "deny"

    def test_all_tiers_in_models(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        for name in ("code-fast", "code-smart", "plan", "plan-uncensored"):
            assert name in models

    def test_no_sampler_params_in_agents(self):
        for agent in self.cfg.get("agent", {}).values():
            assert "temperature" not in agent
            assert "top_p" not in agent

    def test_commands_present(self):
        assert "review" in self.cfg["command"]
        assert "nofilter" in self.cfg["command"]


class TestBuildConfigLiteLLM:
    def setup_method(self):
        self.cfg = build_config(ini_text=LITELLM_INI)

    def test_auto_ctx_equals_fast_ctx(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["auto"]["limit"]["context"] == 200000

    def test_auto_output_equals_fast_output(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["auto"]["limit"]["output"] == 4096

    def test_tier_output_limits_from_ini(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["code-fast"]["limit"]["output"] == 4096
        assert models["code-smart"]["limit"]["output"] == 16384

    def test_small_model_wired_to_fast(self):
        assert self.cfg["small_model"] == "llama-swap/code-fast"

    def test_build_agent_wired_to_auto(self):
        assert self.cfg["agent"]["build"]["model"] == "llama-swap/auto"

    def test_litellm_mcp_entry_present(self):
        # Any backend=litellm tier means llama-swap will start the
        # litellm proxy; opencode should get an mcp.litellm entry
        # pointed at the proxy's /mcp gateway.
        mcp = self.cfg.get("mcp", {})
        assert "litellm" in mcp
        assert mcp["litellm"]["type"] == "remote"
        assert mcp["litellm"]["url"].endswith("/mcp")
        assert mcp["litellm"]["enabled"] is True

    def test_litellm_mcp_uses_local_proxy_port(self):
        url = self.cfg["mcp"]["litellm"]["url"]
        assert url == "http://127.0.0.1:10103/mcp"


class TestLiteLLMMcpRemote:
    def test_litellm_mcp_uses_remote_base(self):
        cfg = build_config(ini_text=LITELLM_INI, remote="http://10.0.0.5:10101")
        assert cfg["mcp"]["litellm"]["url"] == "http://10.0.0.5:10101/mcp"


class TestNoLiteLLMNoMcpEntry:
    def test_litellm_mcp_absent_for_pure_gguf(self):
        cfg = build_config(ini_text=GGUF_INI)
        # Pure-gguf install: no litellm proxy will run, so no
        # mcp.litellm entry should be emitted. opencode.json may
        # legitimately have no `mcp` key at all.
        assert "litellm" not in cfg.get("mcp", {})


class TestBuildConfigRemote:
    def test_remote_url_overrides_base_url(self):
        cfg = build_config(ini_text=GGUF_INI, remote="http://10.0.0.5:10101")
        url = cfg["provider"]["llama-swap"]["options"]["baseURL"]
        assert url == "http://10.0.0.5:10101/v1"

    def test_local_url_used_when_no_remote(self):
        cfg = build_config(ini_text=GGUF_INI, remote=None)
        url = cfg["provider"]["llama-swap"]["options"]["baseURL"]
        assert "127.0.0.1:10101" in url


_NO_PLAN_UNCENSORED_INI = """
[DEFAULT]
router_host = 127.0.0.1
router_port = 10101

[code-fast]
tier     = subagent
role     = build
hf_repo  = owner/repo
hf_file  = model.gguf
ctx_size = 32768
description = fast tier

[code-smart]
tier     = agent
role     = build
hf_repo  = owner/repo
hf_file  = model.gguf
ctx_size = 65536
description = agent tier

[plan]
tier     = agent
role     = chat
hf_repo  = owner/repo
hf_file  = model.gguf
ctx_size = 65536
description = plan tier
"""


class TestNoPlanUncensoredTier:
    def setup_method(self):
        self.cfg = build_config(ini_text=_NO_PLAN_UNCENSORED_INI)

    def test_plan_nofilter_agent_absent(self):
        assert "plan-nofilter" not in self.cfg.get("agent", {})

    def test_nofilter_command_absent(self):
        assert "nofilter" not in self.cfg["command"]

    def test_review_command_present(self):
        assert "review" in self.cfg["command"]


_NO_OUTPUT_TOKENS_INI = """
[DEFAULT]
router_host = 127.0.0.1
router_port = 10101

[code-fast]
tier     = subagent
role     = build
hf_repo  = owner/repo
hf_file  = model.gguf
ctx_size = 32768
description = fast tier

[code-smart]
tier     = agent
role     = build
hf_repo  = owner/repo
hf_file  = model.gguf
ctx_size = 65536
description = agent tier

[plan]
tier     = agent
role     = chat
hf_repo  = owner/repo
hf_file  = model.gguf
ctx_size = 65536
description = plan tier

[plan-uncensored]
tier     = agent
role     = nofilter-chat
hf_repo  = owner/repo
hf_file  = model.gguf
ctx_size = 65536
description = uncensored tier
"""


class TestOutputLimitFallbacks:
    def setup_method(self):
        self.cfg = build_config(ini_text=_NO_OUTPUT_TOKENS_INI)

    def test_auto_output_falls_back_to_16384(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["auto"]["limit"]["output"] == 16384

    def test_agent_output_falls_back_to_32768(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["code-smart"]["limit"]["output"] == 32768

    def test_fast_output_falls_back_to_8192(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["code-fast"]["limit"]["output"] == 8192

    def test_plan_uncensored_output_falls_back_to_32768(self):
        models = self.cfg["provider"]["llama-swap"]["models"]
        assert models["plan-uncensored"]["limit"]["output"] == 32768


class TestLlamaSwapRender:
    @pytest.fixture(autouse=True)
    def use_bundled_models_ini(self, monkeypatch):
        monkeypatch.setenv("LLMSTACK_MODELS_INI", str(EXAMPLES_DIR / "gguf" / "models.ini"))

    def test_render_returns_string(self):
        from llmstack.generators.llama_swap import render as render_llama_swap
        result = render_llama_swap()
        assert isinstance(result, str)

    def test_render_contains_tier_info(self):
        from llmstack.generators.llama_swap import render as render_llama_swap
        result = render_llama_swap()
        assert "llama_server" in result or "matrix" in result
