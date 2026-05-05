"""Pluggable request backends.

The router (:mod:`llmstack.app`) classifies a request and picks a tier
name. Each tier's :attr:`Tier.backend` selects how the request actually
gets fulfilled:

  ``gguf``     reverse-proxy to the local llama-swap (the default; no
               module needed -- :mod:`llmstack.app` does the proxying
               itself).
  ``bedrock``  hand off to :mod:`llmstack.backends.bedrock` which
               translates OpenAI chat/completions to AWS Bedrock
               Converse and streams the response back as OpenAI SSE.

Each backend module is loaded lazily so the optional cloud SDKs are
only imported when the operator has actually configured a tier that
needs them (and only when they're invoked).
"""

from __future__ import annotations
