"""Pluggable request backends.

The router (:mod:`llmstack.app`) classifies a request and picks a tier
name. Each tier's :attr:`Tier.backend` selects how the request actually
gets fulfilled:

  ``gguf``     reverse-proxy to the local llama-swap (the default; no
               module needed -- :mod:`llmstack.app` does the proxying
               itself).
  ``litellm``  hand off to :mod:`llmstack.backends.litellm_backend` which
               calls litellm.acompletion() and streams the response back
               as OpenAI SSE (litellm already returns OpenAI format, so
               no translation layer is needed).

Each backend module is loaded lazily so the optional LLM SDKs are
only imported when the operator has actually configured a tier that
needs them (and only when they're invoked).
"""

from __future__ import annotations
