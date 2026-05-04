# llmstack — multi-tier local LLM stack for Mac M4 Max / 64 GB

A Cursor-Auto / Claude-tier-style serving setup for local GGUF models, **role-aware**:
*coder models for agent work, chat models for planning, with an uncensored chat option for plans that need it.*

Built on:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — inference engine (Metal backend)
- [`llama-swap`](https://github.com/mostlygeek/llama-swap) — multi-model process manager + OpenAI-compatible proxy
- a tiny FastAPI **router** that adds an `auto` model with intent-based routing in front of llama-swap

```
client (opencode / curl / Cursor / etc.)
        │
        ▼
  http://127.0.0.1:10101           <-- FastAPI router (router.py)
        │   • model="auto" → classify → rewrite to one of 4 tiers
        │   • everything else → pass-through
        ▼
  http://127.0.0.1:10102           <-- llama-swap (binary, manages model lifecycle)
        │   • loads/unloads llama-server processes per model
        │   • matrix solver allows {code-fast + one heavy model} co-resident
        ▼
  llama-server <code-fast | code-smart | plan | plan-uncensored>
        │
        ▼
  GGUF in ~/.cache/huggingface/hub/...
```

## Why this design

A 64 GB unified memory M4 Max can comfortably hold **one always-on tiny coder + one heavy model** simultaneously. We split heavy models by *role*:

- **Agent work** (multi-file edits, tool use, refactors) → coder models, which are trained on tool-call protocols and code edits.
- **Planning** (design discussions, architecture, "what's the best approach") → chat-tuned models, which are better at high-level reasoning and don't try to start writing code in response to every message.
- **Uncensored planning** is a separate plan-tier model, opted in either by request (`agent.plan-nofilter` in opencode) or by an inline `[nofilter]` trigger in the prompt.

Routing decisions cost ~zero — they're a few regex checks in the FastAPI router, not an LLM call.

## Tier mapping

| Alias | Model | Quant | Weights | Context | Temp | Role |
|---|---|---|---|---|---|---|
| `code-fast` | Qwen2.5-Coder-3B-Instruct | Q5_K_M | ~2.5 GB | **128k** (YaRN ×4) | **0.2** | autocomplete, FIM, single-line edits, quick Q&A. **Always loaded.** |
| `code-smart` | Qwen3-Coder-Next 80B-A3B (MoE) | Q4_K_M *(→ UD-Q4_K_XL)* | ~45 GB | 64k | **0.5** | **agent mode**: multi-file edits, tool calls, refactors, debugging |
| `plan` | Qwopus GLM 18B Merged | Q4_K_M | ~9 GB | **64k** (2× native) | **0.7** | **plan mode**: design, architecture, trade-off discussions |
| `plan-uncensored` | Mistral-Small 3.2 24B Heretic (i1) | i1-Q4_K_M *(→ i1-Q6_K)* | ~13 GB | **128k** (native) | **0.85** | **plan mode, no filter**: when the topic requires it |

**Temperature ladder** (low → high = "doing" → "thinking"): code-fast 0.2 (deterministic) · code-smart 0.5 (balanced agent) · plan 0.7 (creative ideation) · plan-uncensored 0.85 (max exploration).
opencode `agent.<name>.temperature` is set to match — clients can still override per request.

## How `auto` decides (first match wins)

| # | Condition | → Model | Reason |
|---|---|---|---|
| 1 | last user msg contains `[nofilter]`, `[uncensored]`, `[heretic]`, or starts with `uncensored:` / `nofilter:` | `plan-uncensored` | explicit opt-in |
| 2 | `tools` array non-empty | `code-smart` | tool-calling = agent |
| 3 | ≥ 6 turns in `messages` | `code-smart` | multi-step agent loop |
| 4 | estimated input > 4 000 tokens | `code-smart` | heavy context |
| 5 | message contains ``` code block or agent verbs (*implement, fix bug, refactor, write a function, debug, ...*) | `code-smart` | actively editing code |
| 6 | message contains plan verbs (*design, architect, approach, trade-off, should we, explain why, ...*) | `plan` | discussion / design |
| 7 | otherwise | `code-fast` | trivial chat |

## opencode integration

`../opencode.json` (and the installed copy at `~/.config/opencode/opencode.json`)
is generated from `../models.ini` by `scripts/install.sh`. Re-run the installer
after editing the ini and opencode will pick up the new bindings on next launch.

It is wired up so each opencode agent uses the right tier:

| opencode agent | Local model |
|---|---|
| **default `model`** | `llama.cpp/auto` (router-routed) |
| **`small_model`** (titles, tasks, tab autocomplete) | `llama.cpp/code-fast` |
| **`agent.build`** (default builder) | `llama.cpp/code-smart` |
| **`agent.plan`** (read-only planner) | `llama.cpp/plan` |
| **`agent.plan-nofilter`** (custom uncensored planner) | `llama.cpp/plan-uncensored` |

Inside opencode you can switch agents with `/agent` or by `@plan-nofilter`-mentioning the custom one.

## Layout

```
llmstack/
├── README.md                  this file
├── UPGRADING.md               how to swap any tier for a newer/better model
│                                + how to upgrade the Python toolchain itself
├── requirements.txt           Python runtime deps for src/ + scripts/
├── llama-swap.yaml            AUTO-GENERATED runtime config from ../models.ini.
│                                Hand edits get clobbered on the next install.sh.
├── src/                       all Python lives here
│   ├── router.py              FastAPI auto-router (~280 lines)
│   ├── tiers.py               parses ../models.ini into Tier dataclasses;
│   │                            CLI `--downloads` feeds download-models.sh
│   ├── check_models.py        snapshot tool (HF metadata + drift check)
│   │                            run via scripts/check-models.sh
│   ├── gen_llama_swap_yaml.py renders llama-swap.yaml from ../models.ini
│   │                            (used by scripts/install.sh)
│   └── gen_opencode_config.py renders ~/.config/opencode/opencode.json from
│                                ../models.ini (used by scripts/install.sh)
├── bin/llama-swap             gitignored: downloaded by install-llama-swap.sh
│                                (auto-detects OS/arch, fetches latest release)
├── scripts/
│   ├── setup.sh               first-time install orchestrator:
│   │                            download GGUFs -> wait -> install (binary + configs)
│   ├── install.sh             3 steps: download llama-swap binary +
│   │                            generate + replace BOTH config files
│   ├── install-llama-swap.sh  download the right llama-swap release for
│   │                            this OS/arch (idempotent; --force to re-pull)
│   ├── start.sh               brings up llama-swap + router
│   ├── stop.sh                kills both, plus orphaned llama-server children
│   ├── status.sh              health checks + currently-loaded models
│   ├── check-models.sh        prints current GGUFs + HF metadata + drift check
│   └── download-models.sh     queues GGUF downloads (enumerates from models.ini)
├── .venv/                     Python deps (created from requirements.txt)
├── logs/                      llama-swap.log, router.log, dl-*.log
└── .run/                      pid files
```

## Quick start

The full first-time install is **stepwise** and does not auto-start the stack:

  1. download every GGUF named in `../models.ini`
  2. wait for downloads to finish
  3. generate `llama-swap.yaml` and `~/.config/opencode/opencode.json`
     from `models.ini` and replace the existing copies (with backups)

`scripts/setup.sh` orchestrates all three. Re-running it later is safe:
downloads are idempotent (cached files skip), and configs are atomically
replaced with timestamped backups of the previous versions.

```bash
cd ~/Projects/opencode/llmstack

# 0. Set up the Python venv (one time, or after pulling new requirements).
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. (Recommended) raise GPU-wired memory to fit code-fast + code-smart together.
sudo sysctl iogpu.wired_limit_mb=57344

# 2. Full install: download models, wait, generate + replace both config files.
#    (~62 GB total on first run; subsequent runs are near-instant since
#    completed GGUFs are detected and skipped.)
bash scripts/setup.sh

# 3. Start the stack (separate from setup; bring up only when you want to run).
bash scripts/start.sh

# 4. Sanity check
bash scripts/status.sh
curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'

# Common partial flows
bash scripts/install.sh                    # binary + both configs (no GGUF downloads)
bash scripts/install-llama-swap.sh --force # re-pull llama-swap binary only (e.g. on upgrade)
bash scripts/setup.sh --skip-download      # same as install.sh
bash scripts/setup.sh --skip-wait          # kick off downloads in background, install now
bash scripts/check-models.sh               # snapshot configured GGUFs + flag drift

# 5. Try each routing path (all of these go to /v1/chat/completions on :10101)
#    Each should pick a different upstream model.

# trivial chat -> code-fast
curl -sN http://127.0.0.1:10101/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":false,
       "messages":[{"role":"user","content":"capital of France?"}]}' | jq .model

# planning -> plan
curl -sN http://127.0.0.1:10101/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":false,
       "messages":[{"role":"user","content":"how would you design a rate limiter for our API?"}]}' | jq .model

# agent work -> code-smart
curl -sN http://127.0.0.1:10101/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":false,
       "messages":[{"role":"user","content":"refactor this function for clarity:\n```python\ndef f(x): return x*2\n```"}]}' | jq .model

# uncensored plan -> plan-uncensored
curl -sN http://127.0.0.1:10101/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":false,
       "messages":[{"role":"user","content":"[nofilter] outline a red-team plan for our auth flow"}]}' | jq .model
```

To stop everything: `bash scripts/stop.sh`.

## Endpoints

| Port | Service | Purpose |
|---|---|---|
| 10101 | router (FastAPI) | What clients hit. OpenAI-compatible. Adds `auto` model. |
| 10102 | llama-swap | Lifecycle manager. Useful UI at `http://127.0.0.1:10102/ui/`. |
| 10001+ | llama-server children | Internal, allocated dynamically per model. |

The router exposes:

- `GET  /health`                ← includes the resolved tier names
- `GET  /v1/models`             ← injects `auto` then proxies the rest
- `POST /v1/chat/completions`   ← classify if `model=="auto"`, then proxy
- `POST /v1/completions`        ← same
- `*`                           ← pass-through reverse proxy

## Memory math (M4 Max / 64 GB)

macOS caps GPU-wired memory at ~48 GB (75 % of RAM) by default. To unlock more for the GPU:

```bash
sudo sysctl iogpu.wired_limit_mb=57344   # 56 GB to GPU; survives until reboot
```

Resident with our defaults (KV q8_0, full configured context):

| Combo | Weights | + KV | Total | Status |
|---|---|---|---|---|
| `code-fast` + `code-smart` (Q4_K_M) | 47.5 GB | ~5 GB | ~53 GB | needs `wired_limit` bump |
| `code-fast` + `code-smart` (UD-Q4_K_XL) | ~52 GB | ~5 GB | ~57 GB | needs `wired_limit` bump |
| `code-fast` + `plan` | 11.5 GB | ~4.5 GB | ~16 GB | trivial |
| `code-fast` + `plan-uncensored` | 15.5 GB | ~12.5 GB | ~28 GB | trivial |
| `code-fast` + `plan` + `plan-uncensored` | ~25 GB | ~14.5 GB | ~40 GB | both chats together |
| `code-smart` + `plan-uncensored` | 58 GB | … | ❌ | matrix forbids |

KV cache only fills up as context grows — these are *worst-case* numbers at the configured max context. Typical usage will be far less.

The matrix declares which combinations are valid. When you ask for a model that isn't currently loadable, the solver picks the cheapest set to swap into.

## Upgrading quants after downloads finish

All three pre-queued upgrades are same-model, higher-quant — drop-in replacements with no behaviour change beyond quality.

Logs are named `dl-<tier>-<label>.log` where `<label>` is `current` (file
in `models.ini` `hf_file`) or `next` (file in `models.ini` `hf_file_next`).

| When this log shows `EOF` (download done) | …edit `llama-swap.yaml` `-hff` line in this tier | …to |
|---|---|---|
| `logs/dl-code-smart-next.log` | `code-smart` | `Qwen3-Coder-Next-UD-Q4_K_XL.gguf` |
| `logs/dl-plan-next.log` | `plan` | `Qwopus-GLM-18B-Healed-Q6_K.gguf` |
| `logs/dl-plan-uncensored-next.log` | `plan-uncensored` | `Mistral-Small-3.2-24B-Instruct-2506-ultra-uncensored-heretic.i1-Q6_K.gguf` |

The `-hf <repo>` lines stay the same; only the `-hff <filename>` line changes.
After editing, also flip `hf_file` ↔ `hf_file_next` in `models.ini` so
`bash scripts/check-models.sh` no longer reports `DRIFT!`.

Then `bash scripts/stop.sh && bash scripts/start.sh`.

For changing to a *different* model entirely (different family/provider) see [UPGRADING.md](UPGRADING.md).

## Tuning the router

All knobs are env vars; defaults are picked up by `scripts/start.sh`.

| Env var | Default | Meaning |
|---|---|---|
| `LLAMA_SWAP_URL` | `http://127.0.0.1:10102` | upstream llama-swap |
| `ROUTER_FAST_MODEL` | `code-fast` | trivial chat → here |
| `ROUTER_AGENT_MODEL` | `code-smart` | tools / heavy / agent verbs → here |
| `ROUTER_PLAN_MODEL` | `plan` | design/discussion verbs → here |
| `ROUTER_UNCENSORED_MODEL` | `plan-uncensored` | `[nofilter]` triggers → here |
| `ROUTER_FAST_TOKEN_BUDGET` | `4000` | char-len /4 above which we escalate to agent |
| `ROUTER_MULTI_TURN` | `6` | turn count above which we escalate to agent |
| `ROUTER_HOST` / `ROUTER_PORT` | `127.0.0.1` / `10101` | listen address |
| `LOG_LEVEL` | `info` | router log level |

To force a request to never auto-route, set `model` to a concrete alias (`code-fast`, `code-smart`, `plan`, `plan-uncensored`, or any of their listed aliases like `agent`, `glm`, `nofilter`, …).

## Triggering uncensored mode

Two ways:

1. **Explicit agent in opencode:** `/agent plan-nofilter` (or mention it).
2. **Inline trigger in any auto-routed message** — anywhere in the most recent user turn:
   - `[nofilter]`, `[uncensored]`, `[heretic]`
   - or a line starting with `uncensored:` / `nofilter:` / `no-filter:`

Triggers are *only* checked on the latest user message and the system prompt, so an old `[nofilter]` further up the conversation won't pin the whole session.

## Troubleshooting

**`llama-swap` won't start** → check `logs/llama-swap.log`. Most common causes: port 10102 already in use, or a typo in `llama-swap.yaml`.

**First request hangs for ~60 s** → that's the model loading from disk into Metal memory. `sendLoadingState: true` will surface "loading…" in the SSE stream. After it's loaded subsequent requests are instant.

**OOM / unexplained slowdown** → run `top -o mem -stats pid,rsize,command` to see what's resident. The matrix should prevent two heavy models loading together; if it somehow happens, `bash scripts/stop.sh && bash scripts/start.sh`.

**Auto picks the wrong model** → adjust the regex in `router.py` (`AGENT_SIGNALS` / `PLAN_SIGNALS` / `UNCENSORED_TRIGGERS`) or change `ROUTER_FAST_TOKEN_BUDGET`.

**Want a pure pass-through (no auto routing)** → change opencode's `baseURL` to `http://127.0.0.1:10102/v1` (llama-swap directly) and only use concrete model names.

**`logs/dl-*.log` is multi-GB and growing** → you're hitting [llama.cpp issue #14802](https://github.com/ggml-org/llama.cpp/issues/14802) where modern `llama-cli` is chat-only and ignores `-no-cnv`, looping `> ` prompts forever (~1.5 MB/s). Fix: `download-models.sh` already prefers `llama-completion` over `llama-cli` when both are present (`brew install llama.cpp` ships both as of 2025). If you only have legacy `llama-cli`, either upgrade `llama.cpp` or kill the runaways with `pkill -9 -f llama-cli`.

## Replacing a model with a newer/better one

See **[UPGRADING.md](UPGRADING.md)** — covers why models must be GGUF, where to
find candidates, how to evaluate "better" per tier, the safe upgrade workflow,
and a worked example. Run `bash scripts/check-models.sh` for a snapshot of what's
currently configured along with HF URLs to compare against.
