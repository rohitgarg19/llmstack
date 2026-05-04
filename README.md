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
  http://127.0.0.1:10101           <-- FastAPI router (llmstack.app)
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

The whole thing is a pure Python package distributed via standard Python tooling
(`pip install llmstack`, or `pip install -e .` from this repo). Once installed
you get a single `llmstack` console-script.

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

`llmstack install` generates an opencode config at
`<work-dir>/.llmstack/opencode.json` (derived from `models.ini`), where
`<work-dir>` is whatever directory you ran `llmstack` from (or
`$LLMSTACK_WORK_DIR`). You can `cd` into any project and run
`llmstack install` to get a project-local config there. The script also
copies `AGENTS.md` next to the generated JSON, so the `.llmstack/` folder
is a self-contained opencode bundle. Your global
`~/.config/opencode/opencode.json` is **never modified** by this stack.

opencode picks up our config because `llmstack start` (and `llmstack
shell`) drop you into a subshell with these env vars exported:

| Env var | Value |
|---|---|
| `OPENCODE_CONFIG` | `<work-dir>/.llmstack/opencode.json` (overrides global, sits below project configs) |
| `LLMSTACK_CHANNEL` | `current`, `next`, `external` (thin client of a remote llmstack, see below), or `shared` (local daemons started by another project) |
| `LLMSTACK_ACTIVE` | `1` (used to refuse recursive entry) |
| `LLMSTACK_ROOT` | absolute path to the installed `llmstack` package |

The llama-swap and router daemons are singleton on ports 10101/10102 and
**shared across projects**: `start` from a second project notices the
running daemons and reuses them rather than fighting for the ports;
`stop` from any project tears them down.

The shell's prompt is prefixed with `[llmstack:<channel>]` so you always
know whether you're in the env or not. Bash and zsh source your normal
rc first, then add the prefix; other shells just get the env vars.

Inside the subshell, run `opencode` and it will pick up the wiring
below. Outside the subshell (any other terminal), opencode keeps using
your global setup unchanged.

| opencode agent | Local model |
|---|---|
| **default `model`** | `llama.cpp/auto` (router-routed) |
| **`small_model`** (titles, tasks, tab autocomplete) | `llama.cpp/code-fast` |
| **`agent.build`** (default builder) | `llama.cpp/code-smart` |
| **`agent.plan`** (read-only planner) | `llama.cpp/plan` |
| **`agent.plan-nofilter`** (custom uncensored planner) | `llama.cpp/plan-uncensored` |

Inside opencode you can switch agents with `/agent` or by `@plan-nofilter`-mentioning
a custom one. Slash-commands `/review`, `/nofilter` are also available.

Want a second terminal into the same stack? `llmstack shell` spawns
another env-prepared subshell without touching the daemons. Want to run
opencode without the subshell? `OPENCODE_CONFIG=$PWD/.llmstack/opencode.json opencode`
from any directory you previously ran `install` in.

## Layout

```
opencode/                       # repo root
├── pyproject.toml              # package metadata + `llmstack` console script
├── README.md                   # this file
├── UPGRADING.md                # how to swap any tier for a newer/better model
│                                  + how to upgrade the Python toolchain itself
├── models.ini                  # SINGLE SOURCE OF TRUTH for tiers + sampler
└── llmstack/                   # the python package (importable, installable)
    ├── __init__.py
    ├── __main__.py             # `python -m llmstack`
    ├── cli.py                  # arg dispatch (the `llmstack` console-script)
    ├── paths.py                # state / bin / work dir resolution + env overrides
    ├── shell_env.py            # spawn the env-prepared subshell + activate hooks
    ├── app.py                  # FastAPI auto-router (~280 lines)
    ├── tiers.py                # parse models.ini -> Tier dataclasses
    ├── check_models.py         # snapshot tool (HF metadata + drift check)
    ├── AGENTS.md               # opencode agent template (shipped as package data)
    ├── generators/
    │   ├── llama_swap.py       # render llama-swap.yaml from models.ini
    │   └── opencode.py         # render opencode.json from models.ini
    ├── download/
    │   ├── ggufs.py            # background GGUF downloader
    │   └── binary.py           # llama-swap release downloader
    └── commands/               # one module per CLI action
        ├── setup.py            # first-time walkthrough
        ├── install.py          # generate configs
        ├── install_llama_swap.py
        ├── download.py
        ├── start.py
        ├── shell.py
        ├── stop.py
        ├── restart.py
        ├── status.py
        ├── check.py
        └── activate.py
```

Per-project state (gitignored) is created lazily under `<work-dir>/.llmstack/`:

```
.llmstack/
├── opencode.json          consumed via OPENCODE_CONFIG
├── AGENTS.md              copy of the package template
├── llama-swap.yaml        generated runtime config
├── default-channel        pinned by `llmstack install`
├── active-channel         written by `llmstack start`, removed by `stop`
├── llama-swap.pid         daemon pid files
├── router.pid
├── llmstack.bashrc        prompt-prefix rcfile (bash)
├── zdotdir/               prompt-prefix rcfile (zsh)
└── logs/
    ├── llama-swap.log
    ├── router.log
    └── dl-*.log
```

The `llama-swap` binary lives outside any project at
`$XDG_DATA_HOME/llmstack/bin/llama-swap` (override with
`LLMSTACK_BIN_DIR`). One download is reused across all projects.

## Quick start

Everything runs through one entry point: `llmstack <action>`.
Run `llmstack help` to see all actions and options.

```bash
# 0. Install the package (editable, from this repo).
python3 -m venv .venv
.venv/bin/pip install -e .

# 1. (Recommended) raise GPU-wired memory to fit code-fast + code-smart together.
sudo sysctl iogpu.wired_limit_mb=57344

# 2. Full setup: download GGUFs, wait, install the llama-swap binary, print
#    the activation hook, check opencode is on PATH. Stepwise & idempotent;
#    re-running it later is safe.
llmstack setup

# 3. Generate this project's .llmstack/opencode.json + .llmstack/llama-swap.yaml
llmstack install

# 4. Bring up llama-swap + router AND drop into a subshell with
#    OPENCODE_CONFIG pointed at .llmstack/opencode.json. Prompt is
#    prefixed with [llmstack:current]. Run `opencode` from this subshell.
#    Daemons keep running when you exit; stop them with `llmstack stop`.
llmstack start

# 4a. Same thing but no subshell (daemons only, return immediately).
llmstack start --detach

# 4b. Already running? Open another terminal into the same env:
llmstack shell

# 5. Sanity check (works from any terminal; doesn't need the subshell)
llmstack status
curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'
```

To stop everything: `llmstack stop`.

### Thin-client mode (connect to a remote llmstack)

Set `LLMSTACK_REMOTE_URL` to the router URL of another machine running
llmstack and this host stops launching anything locally — no llama-swap,
no router, no GGUFs needed. `install` generates an `opencode.json` whose
`baseURL` points at the remote, and `start` just verifies `/health` and
drops you into the client subshell:

```bash
# laptop -> desktop running llmstack on 10.0.0.5
export LLMSTACK_REMOTE_URL=http://10.0.0.5:10101

llmstack install      # writes .llmstack/opencode.json (baseURL = remote/v1)
                      # and .llmstack/default-channel = "external <url>"
llmstack start        # verifies http://10.0.0.5:10101/health, enters subshell
                      # prompt is medium-purple and shows the URL:
                      #   [llmstack:opencode http://10.0.0.5:10101]
opencode              # talks straight to the remote router
```

The URL is persisted into the channel marker, so any new terminal you
open with the activate hook installed (`eval "$(llmstack activate zsh)"`)
will re-export `LLMSTACK_REMOTE_URL` automatically when you `cd` into
the project — no need to repeat the `export` in every shell.

The local commands that manage local resources (`setup`, `download`,
`install-llama-swap`) refuse when `LLMSTACK_REMOTE_URL` is set.
`stop` is a no-op (nothing local to tear down) — to stop the daemons
themselves, run `llmstack stop` on the host that started them.

You typically also want a copy of the same `models.ini` the remote was
configured with, so the generated tier names + agent wiring match what
the remote actually serves. (The router decides which tier handles a
request; the client just provides hints.)

### Auto-activate per project

Once you have a `.llmstack/` in a project, you can have your shell
auto-export `OPENCODE_CONFIG` and friends whenever you `cd` into that
tree. Drop the eval line into your rc once and forget about
`llmstack shell` forever:

```bash
# ~/.zshrc (zsh)
eval "$(llmstack activate zsh)"

# or ~/.bashrc (bash)
eval "$(llmstack activate bash)"
```

### Common partial flows

```bash
llmstack install                       # binary + both configs (no GGUF downloads)
llmstack install-llama-swap --force    # re-pull llama-swap binary only
llmstack setup --skip-download         # same as install
llmstack setup --skip-wait             # kick off downloads in background, install now
llmstack check                         # snapshot configured GGUFs + flag drift
llmstack start --next                  # try queued hf_file_next upgrades (reversible)
llmstack restart --next                # cycle into the next channel
```

### Try each routing path

All of these go to `/v1/chat/completions` on `:10101`. Each should pick a different upstream model:

```bash
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
`llmstack check` no longer reports `DRIFT!`.

Then `llmstack restart`.

For changing to a *different* model entirely (different family/provider) see [UPGRADING.md](UPGRADING.md).

## Tuning the router

All knobs are env vars; defaults are picked up by `llmstack start`.

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

**`llama-swap` won't start** → check `.llmstack/logs/llama-swap.log`. Most common causes: port 10102 already in use, or a typo in `llama-swap.yaml`.

**First request hangs for ~60 s** → that's the model loading from disk into Metal memory. `sendLoadingState: true` will surface "loading…" in the SSE stream. After it's loaded subsequent requests are instant.

**OOM / unexplained slowdown** → run `top -o mem -stats pid,rsize,command` to see what's resident. The matrix should prevent two heavy models loading together; if it somehow happens, `llmstack restart`.

**Auto picks the wrong model** → adjust the regex in `llmstack/app.py` (`AGENT_SIGNALS` / `PLAN_SIGNALS` / `UNCENSORED_TRIGGERS`) or change `ROUTER_FAST_TOKEN_BUDGET`.

**Want a pure pass-through (no auto routing)** → change opencode's `baseURL` to `http://127.0.0.1:10102/v1` (llama-swap directly) and only use concrete model names.

**`logs/dl-*.log` is multi-GB and growing** → you're hitting [llama.cpp issue #14802](https://github.com/ggml-org/llama.cpp/issues/14802) where modern `llama-cli` is chat-only and ignores `-no-cnv`, looping `> ` prompts forever (~1.5 MB/s). Fix: `llmstack download` already prefers `llama-completion` over `llama-cli` when both are present (`brew install llama.cpp` ships both as of 2025). If you only have legacy `llama-cli`, either upgrade `llama.cpp` or kill the runaways with `pkill -9 -f llama-cli`.

## Replacing a model with a newer/better one

See **[UPGRADING.md](UPGRADING.md)** — covers why models must be GGUF, where to
find candidates, how to evaluate "better" per tier, the safe upgrade workflow,
and a worked example. Run `llmstack check` for a snapshot of what's
currently configured along with HF URLs to compare against.
