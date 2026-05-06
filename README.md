# llmstack — multi-tier local LLM stack for Mac M4 Max / 64 GB

A Cursor-Auto / Claude-tier-style serving setup for local GGUF models, **role-aware**:
*coder models for agent work, chat models for planning, with an uncensored chat option for plans that need it.*

Each tier can be served by either a **local GGUF** (default) or a **hosted AWS
Bedrock model** — useful for the top-tier weights that don't fit on a laptop.
Both backends share the same `auto` router, so opencode/curl/Cursor never need
to know which one a tier resolves to.

Built on:

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — inference engine (Metal backend)
- [`llama-swap`](https://github.com/mostlygeek/llama-swap) — multi-model process manager + OpenAI-compatible proxy
- a tiny FastAPI **router** that adds an `auto` model with intent-based routing in front of llama-swap (and AWS Bedrock)

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

## How `auto` decides

The router runs a **step-down fidelity ladder**: start at the top tier
for new / short conversations, drop down as the context grows. This
inverts the classic "escalate when input gets big" pattern, and it
matches how these models actually behave on this stack:

- **Top-tier hosted** (Claude Opus/Sonnet on Bedrock) — fastest *and*
  most accurate on short prompts, but per-request latency and $cost
  scale with input tokens, and long-context behaviour degrades faster
  than headline benchmarks suggest.
- **`code-smart`** (Qwen3-Coder 80B) — 64k window. Sweet spot is the
  middle of that range; saturates near the top.
- **`code-fast`** (Qwen2.5-Coder 3B + YaRN ×4) — **128k** window,
  always-resident, free. Smaller models lean on explicit context rather
  than priors, so they tend to *improve* relative to top-tier as the
  conversation grows.

First match wins:

| # | Condition | → Model | Reason |
|---|---|---|---|
| 1 | last user msg contains `[nofilter]`, `[uncensored]`, `[heretic]`, or starts with `uncensored:` / `nofilter:` | `plan-uncensored` | explicit opt-in |
| 2 | `[ultra]` / `[opus]` / `ultra:` trigger AND `code-ultra` tier configured | `code-ultra` | explicit top-tier opt-in |
| 3 | plan verbs (*design, architect, approach, trade-off, should we, explain why, …*) AND no code blocks / agent verbs / tools | `plan` | pure design discussion (orthogonal track) |
| 4 | estimated input ≤ 8 000 tokens | `code-ultra` *(or `code-smart` if ultra unwired)* | top tier — context still being built, latency/$ are best here |
| 5 | estimated input ≤ 32 000 tokens | `code-smart` | mid-context, local heavy coder is at its sweet spot |
| 6 | otherwise (long context) AND (`tools[]` OR ≥ 6 turns) | `code-smart` | floor: 3B model tool-calls unreliably |
| 7 | otherwise (long context) | `code-fast` | 128k YaRN window + always-resident + free |

Token estimates are `chars / 4` over all message text + `prompt`. The
`code-ultra` rungs (2 and 4) are gated on availability: when no
`[code-ultra]` section is loaded from `models.ini`, both silently fall
back to `code-smart` so vanilla installs don't 404.

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

Want a second terminal into the same stack? Install the activate hook
once (`eval "$(llmstack activate zsh)"`) and any new shell that `cd`s
into the project picks up `OPENCODE_CONFIG` automatically. Want to run
opencode without the hook? `OPENCODE_CONFIG=$PWD/.llmstack/opencode.json opencode`
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
        ├── install.py          # generate opencode.json (+ AGENTS.md copy)
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
├── opencode.json          consumed via OPENCODE_CONFIG (written by `install`)
├── AGENTS.md              copy of the package template (written by `install`)
├── llama-swap.yaml        generated runtime config (written by `start`)
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

# 3. Generate this project's .llmstack/opencode.json (+ AGENTS.md copy).
#    `install` does NOT touch llama-swap.yaml -- that's regenerated
#    fresh by `start` for the channel you're booting into.
llmstack install

# 4. Generate .llmstack/llama-swap.yaml for the chosen channel, bring up
#    llama-swap + router. With the activate hook installed (see below),
#    your prompt is already wired to .llmstack/opencode.json -- just run
#    `opencode`. Without the hook, `start` falls back to spawning a
#    subshell with OPENCODE_CONFIG set, prefixed with [llmstack:current].
#    Daemons keep running when you exit; stop them with `llmstack stop`.
llmstack start

# 4a. Daemons only (no fallback subshell, return immediately).
llmstack start --detach

# 4b. Want auto-activation in any new terminal you cd into? Install once:
eval "$(llmstack activate zsh)"
# add the same line to ~/.zshrc to make it stick.

# 5. Sanity check (works from any terminal)
llmstack status
curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'
```

To stop everything: `llmstack stop`.

### Windows

The CLI runs the same way on Windows (PowerShell or `cmd.exe`); the only
moving parts that differ are the binary asset and the activation hook.

```powershell
# 0. Install the package (editable, from this repo).
py -3 -m venv .venv
.venv\Scripts\pip install -e .

# 1. Pull GGUFs + the windows_amd64 llama-swap binary (lives under
#    %LOCALAPPDATA%\llmstack\bin\llama-swap.exe).
.venv\Scripts\llmstack setup

# 2. Generate this project's .llmstack\opencode.json (+ AGENTS.md copy).
.venv\Scripts\llmstack install

# 3. Generate .llmstack\llama-swap.yaml for the chosen channel, bring up
#    the stack. If you've installed the activate hook (step 4) the
#    current shell is already wired to .llmstack\opencode.json; otherwise
#    `start` falls back to spawning a PowerShell subshell.
.venv\Scripts\llmstack start

# 4. Auto-activate per project from any new PowerShell window:
Invoke-Expression (& llmstack activate powershell | Out-String)
# or persist (writes ~/.powershell_llmstack_hook + sources it on every shell):
"Invoke-Expression (& llmstack activate powershell | Out-String)" | Add-Content $PROFILE
```

Notes:

- Only `windows_amd64` llama-swap binaries are published upstream; arm64
  Windows is not supported. GPU acceleration uses whatever backend
  `llama-server` was built with (CUDA / Vulkan / CPU) -- get
  `llama-server.exe` from the [llama.cpp Windows releases](https://github.com/ggml-org/llama.cpp/releases)
  or a package like `winget install ggml.llama-cpp` and put it on
  `PATH` (or set `$env:LLAMA_SERVER_BIN`). The Mac-only
  `iogpu.wired_limit_mb` step does not apply.
- The `[llmstack:<channel>]` prompt prefix shows up in PowerShell too;
  `cmd.exe` gets a simpler `[llmstack:<channel>]` prompt via `doskey`.
- Stopping daemons uses `taskkill /T /F` under the hood, so the
  llama-server children get cleaned up as well.

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
                      # (no llama-swap.yaml -- the remote owns that)
llmstack start        # verifies http://10.0.0.5:10101/health
                      # (with the activate hook installed, your prompt
                      # is already medium-purple with the URL:
                      #   [llmstack:opencode http://10.0.0.5:10101])
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

`llmstack activate <shell>` writes the hook to
`~/.<shell>_llmstack_hook` and prints a `source` line to stdout, so a
single `eval` both regenerates the file and turns the hook on in your
current shell. Pasting the same `eval` into your rc keeps it on for
every new shell:

```bash
# ~/.zshrc (zsh)
eval "$(llmstack activate zsh)"

# or ~/.bashrc (bash)
eval "$(llmstack activate bash)"
```

With the hook installed, `cd` into any project that has a `.llmstack/`
and your shell is wired up automatically — `OPENCODE_CONFIG`,
`LLMSTACK_WORK_DIR`, `LLMSTACK_CHANNEL` (and `LLMSTACK_REMOTE_URL` for
external projects) all toggle on/off as you walk in and out. There is
no separate `llmstack shell` command — this is the shell command.

### Common partial flows

```bash
llmstack install                       # opencode.json + AGENTS.md (no GGUF downloads)
llmstack install-llama-swap --force    # re-pull llama-swap binary only
llmstack setup --skip-download         # full setup minus the GGUF pull
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
| `ROUTER_FAST_MODEL` | `code-fast` | long-context (>= mid ceiling) → here |
| `ROUTER_AGENT_MODEL` | `code-smart` | mid-context + tools/loop floor → here |
| `ROUTER_ULTRA_MODEL` | `code-ultra` | short-context top tier → here (gated on availability) |
| `ROUTER_PLAN_MODEL` | `plan` | design/discussion verbs → here |
| `ROUTER_UNCENSORED_MODEL` | `plan-uncensored` | `[nofilter]` triggers → here |
| `ROUTER_HIGH_FIDELITY_CEILING` | `8000` | tokens; at or below this, route to top tier (ultra → smart fallback) |
| `ROUTER_MID_FIDELITY_CEILING` | `32000` | tokens; at or below this, route to `code-smart`; beyond, step down to `code-fast` |
| `ROUTER_MULTI_TURN` | `6` | turn count that floors the long-context rung at `code-smart` |
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

**Auto picks the wrong model** → adjust the regex in `llmstack/app.py` (`AGENT_SIGNALS` / `PLAN_SIGNALS` / `UNCENSORED_TRIGGERS`) or move the ladder ceilings via `ROUTER_HIGH_FIDELITY_CEILING` / `ROUTER_MID_FIDELITY_CEILING`. To force a request to never auto-route, pass an explicit `model` (e.g. `code-smart`) instead of `auto`.

**Want a pure pass-through (no auto routing)** → change opencode's `baseURL` to `http://127.0.0.1:10102/v1` (llama-swap directly) and only use concrete model names. (Note: this skips the bedrock dispatcher; only GGUF tiers will be reachable.)

## Hosted tiers via AWS Bedrock

Any tier in `models.ini` that declares `aws_model_id = ...` is served from
AWS Bedrock instead of llama-swap. The same tier names + auto-routing apply,
so swapping `code-smart` from a local GGUF to Claude on Bedrock is a
`models.ini` edit + `llmstack install` + `llmstack restart` away — clients
don't change.

```ini
[code-smart]
role         = agent
aws_model_id = anthropic.claude-sonnet-4-5-20250929-v1:0
aws_region   = us-west-2
aws_profile  = bedrock-prod          ; named profile in ~/.aws/config
ctx_size     = 200000
sampler      = temp=0.5    ; Sonnet 4.5 accepts ONE of temp / top_p
description  = Claude Sonnet 4.5 on Bedrock - heavy coder for agent loops
```

> **Sampler is per-tier, declared in `models.ini`, applied per backend.**
> `opencode.json` is intentionally sampler-free in both cases — clients
> just specify a model. How the sampler reaches the actual inference
> engine depends on the backend:
>
> - **gguf tiers** — the llama-swap generator bakes each tier's
>   `sampler = …` keys into its `llama-server` startup command line as
>   `--temp` / `--top-p` / `--top-k` / `--min-p` / `--repeat-penalty`
>   flags. llama-server applies them as its defaults for every request.
>   The router doesn't touch the body.
> - **Bedrock tiers** — Bedrock has no server-side defaults mechanism,
>   so the router injects the sampler keys into each outbound request
>   body (mapping `temp` → `temperature`, `top_p` → `topP`; the other
>   llama.cpp-extension keys `top_k`/`min_p`/`rep_pen` are silently
>   dropped because Converse doesn't accept them). Caller-supplied
>   values in the request body still win for per-call overrides.
>
> Per-Bedrock-family rules (declare only what your Bedrock model
> accepts):
>
> | Bedrock model family | What `sampler` may contain |
> |---|---|
> | Claude Opus 4.7+ | (omit `sampler =` entirely — Opus 4.7 rejects all sampler params) |
> | Claude Sonnet 4.5 / Haiku 4.5 | `temp` **or** `top_p`, never both |
> | Claude Opus 4.x (4.1, 4.5, 4.6) | `temp` and/or `top_p` |
> | Llama / Titan / Cohere / etc. | `temp` and/or `top_p` (check the model card) |
>
> Local gguf tiers accept the full set (`temp`, `top_p`, `top_k`,
> `min_p`, `rep_pen`) — llama-server honours all of them as startup
> defaults.

`models.ini` is meant to be committable, so it **only names a profile**.
Credentials, SSO, role chaining, MFA — everything boto3 normally
handles — live in the standard AWS shared config:

```bash
aws configure --profile bedrock-prod        # static keys
aws configure sso --profile bedrock-prod    # SSO

# role chaining: edit ~/.aws/config, add a profile like
# [profile bedrock-planning]
# role_arn       = arn:aws:iam::123456789012:role/llmstack-bedrock
# source_profile = bedrock-prod
# region         = us-east-1
```

Then reference the profile by name from each tier. Different tiers can
point at different profiles, so two tiers can live in different
accounts/regions cleanly:

| Key (in `models.ini`) | Meaning |
|---|---|
| `aws_model_id` | Bedrock model ID (`anthropic.claude-...`, `meta.llama3-1-...`, etc.). Required. |
| `aws_region` | Region the tier lives in. Falls back to the profile's region / `AWS_REGION` / default chain. |
| `aws_profile` | Named profile in `~/.aws/config` / `~/.aws/credentials`. Omit for boto3's default chain (env vars, default profile, instance role). |
| `aws_endpoint_url` | Custom Bedrock endpoint (VPC endpoint, FedRAMP, etc.). |
| `aws_model_id_next` (+ optional `aws_region_next`) | Queued upgrade target. Mirrors gguf `hf_file_next`: `llmstack start --next` swaps the tier to this model id (and region, if set) until you switch back; permanent promotion is `aws_model_id` edit + `llmstack install`. |
| `backend = bedrock` | Optional explicit override; auto-detected from `aws_model_id`. |

Banned in `models.ini` (parse-time error): `aws_access_key_id`,
`aws_secret_access_key`, `aws_session_token`, `aws_role_arn`,
`aws_role_session_name`. Put them in `~/.aws/credentials` or
`~/.aws/config` under a named profile and reference the profile.

Internally the router builds one `bedrock-runtime` client per
distinct (profile, region, endpoint) tuple, cached for the life of the
process. Credential refresh (SSO token rotation, role re-assumption,
IMDS) is handled by boto3 transparently.

Install the AWS SDK (it's an opt-in extra so the local-only path stays
small):

```bash
pip install -e '.[bedrock]'
```

The router translates OpenAI chat/completions to [Bedrock Converse]
(text + tool calls; streaming and non-streaming both supported) and
streams the response back as standard OpenAI SSE. Multimodal inputs are
text-only for now.

[Bedrock Converse]: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

Hosted tiers are skipped by `llmstack download` (nothing to fetch) and by
the `llama-swap.yaml` matrix (nothing to load). They show up in
`llmstack check` with the model id + region (and a `next` row when
`aws_model_id_next` is set) instead of HF metadata, and in `/v1/models`
alongside the local GGUF tiers — including a `channel: current|next`
metadata field so clients can tell which model id they're actually
talking to.

`llmstack start --next` flips both backends in lock-step: gguf tiers
swap to `hf_file_next` and bedrock tiers swap to `aws_model_id_next`
(the router subprocess is launched with `LLMSTACK_USE_NEXT=1`). Either
backend having a queued upgrade is enough to satisfy `--next`.

**`logs/dl-*.log` is multi-GB and growing** → you're hitting [llama.cpp issue #14802](https://github.com/ggml-org/llama.cpp/issues/14802) where modern `llama-cli` is chat-only and ignores `-no-cnv`, looping `> ` prompts forever (~1.5 MB/s). Fix: `llmstack download` already prefers `llama-completion` over `llama-cli` when both are present (`brew install llama.cpp` ships both as of 2025). If you only have legacy `llama-cli`, either upgrade `llama.cpp` or kill the runaways with `pkill -9 -f llama-cli`.

## Replacing a model with a newer/better one

See **[UPGRADING.md](UPGRADING.md)** — covers why models must be GGUF, where to
find candidates, how to evaluate "better" per tier, the safe upgrade workflow,
and a worked example. Run `llmstack check` for a snapshot of what's
currently configured along with HF URLs to compare against.
