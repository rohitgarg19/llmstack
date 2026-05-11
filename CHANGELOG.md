# Changelog

All notable changes to `opencode-llmstack` are documented here.

---

## [0.9.4] — 2026-05-11

### Fixed
- `classify()` now scopes `has_code_signal` to the **last user message only**
  (was scanning the full conversation history). Previously, any prior coding
  exchange in the session (code blocks, agent verbs) would permanently block
  plan routing for the rest of the conversation — e.g. "explain why these
  changes are important?" after a refactor request would never reach `plan`.
- Added regression test:
  `test_plan_signal_after_prior_coding_exchange_routes_to_plan`.
- `__version__` corrected from `"0.9.2"` to `"0.9.4"` (was skewed vs
  `pyproject.toml` since 0.9.3).

---

## [0.9.2] — 2026-05-11

### Fixed
- `classify()` now counts only **user-role messages** when evaluating the
  multi-turn floor (`n_turns`). Previously `len(messages)` counted system
  prompts, assistant turns, and tool-result messages, causing the floor to
  fire after just a few real exchanges and permanently blocking `code-fast`
  routing for the rest of any session.
- Multi-turn floor threshold raised from **6 → 10** user turns. `code-fast`
  is now a hosted Bedrock model (Haiku 4.5) that tool-calls reliably, so
  the old 3B-model rationale no longer applies. Sessions with fewer than 10
  user turns will now correctly step down to `code-fast` past 32k tokens.
- Log label corrected: `(tools floor)` → `(user-turns=N>=10 floor)`.
- `__version__` corrected from `"0.1.0"` to `"0.9.2"`.
- CI release pipeline now runs lint (`ruff`) + `pytest` across Python
  3.11/3.12/3.13 before building the wheel. Previously the `test` job was
  documented in comments but never implemented.
- Added `LICENSE` (MIT) file to the repository root.
- README routing table updated: high-fidelity ceiling corrected (8k → 12k),
  tools-floor condition updated to reflect user-turn counting, `ROUTER_MULTI_TURN`
  default corrected (6 → 10).
- UPGRADING.md corrected: `llmstack install` does **not** regenerate
  `llama-swap.yaml` — that is `llmstack restart`'s job. Three places in the
  doc had this wrong.
- README layout tree: repo root label corrected (`opencode/` → `llmstack/`),
  `models.ini` moved to its correct location inside the package, `shell.py`
  (deleted) removed, `reload.py` and `LICENSE` added.
- `iter_downloads` reference in UPGRADING.md corrected to `iter_download_targets`.
- Bundled `llmstack/models.ini` header comment paths updated from legacy
  locations to current `.llmstack/` state-dir layout.
- `assert` statements in production code (`app.py`, `bedrock.py`) replaced
  with explicit `RuntimeError` / `TypeError` raises so `-O` optimisation
  does not silently swallow them.
- `UPGRADING.md` and `LICENSE` added to the sdist via `pyproject.toml`
  `package-data` / `tool.setuptools` config.
- `[tool.pytest.ini_options]` added to `pyproject.toml` with `testpaths`
  and `addopts`.
- Python 3.14 classifier added to `pyproject.toml`.

### Added
- `classify()` end-to-end test coverage: step-down ladder (short/mid/long
  context), multi-turn floor, plan-signal routing, ultra-trigger routing,
  uncensored-trigger routing, plan ctx-size overflow fall-through.
- Generator tests: `build_config()` coverage for gguf tiers, bedrock tiers,
  `use_next`, `small_model` wiring, agent wiring, `auto` ctx derivation.
- `X-LLMStack-Tokens` response header on every `/v1/chat/completions` and
  `/v1/completions` response so opencode (and curl) can see the estimated
  token count the router used to make its routing decision.

---

## [0.9.1] — 2026-05-11

### Fixed
- `classify()` multi-turn floor: count only `role == "user"` messages
  (not all messages). This was the primary fix preventing `code-fast`
  from ever being reached in long sessions.
- Multi-turn threshold raised 6 → 10 (see 0.9.2 for full rationale).
- Log label `(tools floor)` corrected to `(user-turns=N>=10 floor)`.

---

## [0.9.0] — 2026-05-08

### Changed
- Plan tiers now strip `tools` from the request body before dispatch.
  Previously a plan-routed request carrying a `tools` array would fail
  on Bedrock (Converse rejects tool configs on non-agent models).
- Long-context fall-through to `code-fast` is now allowed even when
  `tools[]` is present in the request body. The tools-presence check
  was removed from the floor condition; only turn count matters now.
- `plan` tier ctx-size overflow: when estimated tokens exceed the
  planner's `ctx_size`, the request falls through to the coding ladder
  instead of being sent to a planner whose window can't hold it.
- `HIGH_FIDELITY_CEILING` raised to 12 000 (was 8 000).

---

## [0.8.0] — 2026-05-07

### Changed
- Fidelity-ceiling overhaul: each ceiling is now exactly half of the
  corresponding tier's `ctx_size` (the "comfortable headroom" invariant).
- `code-ultra.ctx_size` set to 24 000 (2× high ceiling of 12 000).
- `code-smart.ctx_size` set to 64 000 (2× mid ceiling of 32 000).
- `code-fast.ctx_size` set to 128 000 (YaRN ×4 from native 32k).
- `HIGH_FIDELITY_CEILING` env var added; overrides the 12 000 default.
- `MID_FIDELITY_CEILING` env var added; overrides the 32 000 default.

---

## [0.7.3] — 2026-05-06

### Added
- Per-tier Bedrock alternatives in `models.ini`: every tier now ships a
  commented-out Bedrock block directly beneath its GGUF block.
- All Bedrock tiers anchored to `eu-west-3`; `plan-uncensored` pinned to
  `us-west-2` (Llama 405B has no EU deployment).
- `aws_model_id_next` / `aws_region_next` support for Bedrock upgrade
  pre-staging (mirrors gguf `hf_file_next`).

### Fixed
- `models.ini` comment cleanup: removed stale references to old model names.

---

## [0.7.2] — earlier

### Fixed
- Soft-fail when `llama-server` binary is missing at startup.
- PowerShell activation hook: fixed `Invoke-Expression` quoting.
