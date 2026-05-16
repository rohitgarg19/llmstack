"""LiteLLM tier descriptors for the auto-router's ``/v1/models`` endpoint.

Routing of litellm-backed requests is handled entirely by llama-swap:
each tier with ``backend = litellm`` is registered as an alias of the
``litellm_proxy`` model in ``llama-swap.yaml``, so a chat-completions
request for ``<tier>`` is dispatched to the litellm proxy process
(started by llama-swap on a fixed port) without the router needing a
separate HTTP path. The router merely rewrites ``body["model"]`` to
``<tier>`` (or ``<tier>_next`` when the ``--next`` channel is active)
and forwards through the same llama-swap proxy used for gguf tiers.
See :func:`llmstack.app._handle_completion`.

This module survives only to format the descriptor row that
``/v1/models`` returns for litellm tiers.
"""

from __future__ import annotations

from typing import Any

from llmstack.tiers import Tier


def model_descriptor(tier: Tier) -> dict[str, Any]:
    """Return an OpenAI-format model descriptor for a litellm tier."""
    return {
        "id": tier.name,
        "object": "model",
        "created": 0,
        "owned_by": "litellm",
        "name": tier.description,
        "description": f"{tier.description} (via litellm proxy: {tier.litellm.model})",
        "tier": tier.name,
    }
