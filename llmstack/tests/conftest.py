"""Pin tests to the bundled ``models.ini``.

``llmstack.app`` evaluates ``FAST_MODEL`` / ``HIGH_FIDELITY_CEILING`` /
upstream URL etc. at import time from whatever ``models_ini_path()``
resolves to. Without this hook, a dev who has run ``llmstack install``
in the repo root sees their per-project ``.llmstack/models.ini``
(custom ceilings, custom tiers) leak into the test run, which both
masks and causes spurious failures.

Set ``LLMSTACK_MODELS_INI`` to the canonical bundled file before any
``llmstack.app`` / ``llmstack.tiers`` import so every test sees the
shipped defaults regardless of the working directory's state.
"""

from __future__ import annotations

import os

from llmstack.paths import EXAMPLES_DIR

os.environ["LLMSTACK_MODELS_INI"] = str(EXAMPLES_DIR / "gguf" / "models.ini")
