# UPGRADING — replacing models with newer / better ones

This stack is built around **swapping individual model files in/out**. There is no
lock-in to provider, model family, or name — you can replace any tier with any
other GGUF that fits the role and your hardware budget.

This doc explains:

1. [Why GGUF specifically](#why-gguf)
2. [Where each model is referenced (single source of truth + IDE wiring)](#where-the-models-live)
3. [The upgrade workflow](#upgrade-workflow)
4. [How to judge "holistically better" per tier](#judging-better-per-tier)
5. [Where to look for candidates](#where-to-find-candidates)
6. [Worked example](#worked-example-replacing-the-plan-tier)
7. [After-upgrade housekeeping](#after-upgrade-housekeeping)
8. [Upgrading the Python toolchain](#upgrading-the-python-toolchain)

---

## Why GGUF

Every model the stack runs **must** be a GGUF (`.gguf`) file because:

- **`llama.cpp` only loads GGUF natively.** The whole stack — `llama-server`,
  `llama-swap`, the Metal backend on Apple Silicon — is built on llama.cpp.
- **Self-contained format.** A single `.gguf` includes weights, tokenizer,
  chat template (Jinja), and metadata. No `transformers`, no separate
  tokenizer files, no Python deps at runtime.
- **First-class quantisation.** GGUF is the format quantised models ship in
  (Q4_K_M, Q5_K_M, Q8_0, i1, UD, …). Quants are how a 24 B model fits in
  13 GB of RAM with little quality loss.
- **Instant `-hf` resolution.** `llama-server -hf <repo> -hff <file>.gguf`
  downloads on demand into the standard HF cache and starts serving — no
  conversion step, no extra tooling.

If a model you want is **not** published as a GGUF, you have two options:

1. Wait for one of the trusted converters (`bartowski`, `unsloth`,
   `mradermacher`, `lmstudio-community`, `MaziyarPanahi`, …) to publish one —
   usually within hours of the original release.
2. Convert it yourself with `llama.cpp/convert_hf_to_gguf.py` then quantise
   with `llama-quantize`. Doable but slow and adds maintenance burden — only
   worth it for niche models.

**Rule of thumb:** if you can't find `<model-name>-GGUF` on HF or via the
maintainers above, the model is not ready for this stack. Pick another.

## Cache management

There is **one** writer to the model cache: `llama-cli`. Everything reads from
the same place: `~/.cache/huggingface/hub/`.

```
~/.cache/huggingface/hub/
└── models--<owner>--<repo>/
    ├── blobs/
    │   ├── <sha>                       ← completed file
    │   └── <sha>.downloadInProgress    ← llama.cpp's resumable partial
    ├── refs/main                       ← commit hash
    └── snapshots/<commit>/
        └── <filename>.gguf             ← symlink to the blob
```

Key points:

- **`llama-cli -cl`** lists everything currently complete in the cache.
- **`llmstack download` uses `llama-cli`** so the cache stays coherent. It
  accepts `HF_TOKEN` from the environment if you have one.
- **Do NOT use `huggingface-cli download` or `pip install huggingface_hub`'s
  `hf_hub_download()`** to write to the same cache — they use a different
  partial-file convention (`.incomplete` vs `.downloadInProgress`) and
  cannot resume each other's partials. Completed files are interchangeable;
  in-flight downloads are not.
- **Cache-dir override:** all llama.cpp tools accept `-cd <dir>` to point at
  a different cache. We don't use it; the default is correct.
- **Cache cleanup:** `llama-cli -cr <repo>:<quant>` removes a single entry,
  or `rm -rf ~/.cache/huggingface/hub/models--<owner>--<repo>/` removes a
  whole repo's blobs.

## `-hf` / `-hff` syntax (gotcha)

```
-hf  <user>/<repo>             repo on HuggingFace
-hf  <user>/<repo>:<quant>     repo + quant TAG  (e.g. :Q4_K_M, :Q6_K) — auto-resolves a file
-hff <filename>.gguf           explicit file inside the repo
```

`-hf <repo>:<full-filename>.gguf` **does not work** — the `:suffix` part is
parsed as a quant tag, so a full filename ending in `.gguf` is rejected with
`get_hf_plan: no GGUF files found in repository`. Always pair `-hf <repo>` with
`-hff <filename>.gguf` for explicit selection. This is what the generated
`llama-swap.yaml` uses everywhere.

---

## Where the models live

There is exactly **one source of truth** that determines what runs in a
given project: `.llmstack/models.ini` in the work-dir. Everything else is
generated from it. The first time you run `llmstack install` in a fresh
project, this file is auto-seeded from the bundled template at
`llmstack/models.ini` inside the package; from then on it's per-project.

| File | What it does | Required when changing a model? |
|---|---|---|
| **`<work-dir>/.llmstack/models.ini`** | The single source of truth: tier definitions (repo, file, ctx, sampler, role, …). All other artefacts derive from it. Auto-seeded from the bundled template on first `install`; gitignored after that — yours to edit per project. | **Yes — primary edit.** |
| `llmstack/models.ini` (in the package) | Bundled template, version-controlled, ships with `pip install -e .`. Only consulted as the seed source when a project has no `.llmstack/models.ini` yet. Edit if you want to change the *factory defaults* every new project starts from. | Only when changing factory defaults. |
| `<work-dir>/.llmstack/llama-swap.yaml` | AUTO-GENERATED by `llmstack.generators.llama_swap`. Determines what `llama-server` actually loads. Regenerated fresh by `llmstack start` (or `llmstack restart`) for the channel you're booting into. Do not regenerate with `llmstack install` — `install` only writes `opencode.json` + `AGENTS.md`. | Auto — don't hand-edit. |
| `<work-dir>/.llmstack/opencode.json` | AUTO-GENERATED by `llmstack.generators.opencode`. `<work-dir>` is whatever directory the CLI was invoked from — `cd` into any project to get a project-local config. Wires opencode's `model`, `small_model`, and `agent.*` to the right tiers. Loaded by opencode via the `OPENCODE_CONFIG` env var that the activate hook (or `llmstack start`'s fallback subshell) exports — your global `~/.config/opencode/opencode.json` is intentionally left alone. Regenerate via `llmstack install`. | Auto — don't hand-edit. |
| `<work-dir>/.llmstack/AGENTS.md` | Copy of `llmstack/AGENTS.md` (the bundled template), made by `install`. Loaded by opencode via the `instructions` field of `opencode.json`. Edit the source template to change defaults; edit the `.llmstack/` copy for one-off project tweaks (will be overwritten on next `install`). | Auto — copy of template. |
| `llmstack download` | Enumerates download targets at runtime via `llmstack.tiers.iter_download_targets` (which reads `models.ini`). | No edits — it reflects `models.ini` automatically. |

`llmstack check` flags any DRIFT between `models.ini` (the recommended file
per tier) and `llama-swap.yaml` (the file actually configured). If you ever
see DRIFT, run `llmstack restart` to regenerate the yaml and cycle the stack.

### Where the download list comes from

`llmstack download` does **not** hardcode the (repo, file) tuples. It
enumerates them at runtime by reading `.llmstack/models.ini` (via
`llmstack.tiers`), yielding one row per `(tier, label)` where `label` is
`current` (the active `hf_file`) or `next` (the queued `hf_file_next`).

To add a new file to the download set, edit the matching tier in
`models.ini`. To stop pre-fetching an upgrade target, blank out its
`hf_file_next` line. No edits to Python code needed.

### Trying the queued upgrade without committing to it

Once an `hf_file_next` finishes downloading you can run the whole stack
against the **next** file for every tier that has one, without touching
`models.ini` permanently:

```bash
llmstack stop
llmstack start --next
llmstack status           # shows "channel: next" + per-tier swap list
```

Under the hood, `start --next` calls
`llmstack.generators.llama_swap.render(use_next=True)` and writes the
result to the same `<work-dir>/.llmstack/llama-swap.yaml` (with a header
banner reminding you it's the next-channel build), then points llama-swap
at it. Tiers with no `hf_file_next` are unchanged. To revert, `llmstack
stop && llmstack start` (no `--next`) regenerates the canonical yaml.
To make the upgrade permanent, swap `hf_file_next` into `hf_file` in
`models.ini` and re-run `llmstack install`.

### Anchors to find on each upgrade

Inside `<work-dir>/.llmstack/llama-swap.yaml`, each tier has a clearly-marked
model block:

```yaml
models:
  code-fast:                     # ← tier alias (do not rename without updating opencode.json)
    cmd: |
      ${llama_server} ${metal_defaults}
      # >>> UPGRADE-POINT (code-fast): swap the -hf/-hff pair below to change this tier.
      -hf  bartowski/Qwen2.5-Coder-3B-Instruct-GGUF        # <-- REPLACE: HF repo
      -hff Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf           # <-- REPLACE: filename in that repo
      --alias code-fast
      -c 131072
      ...
```

But don't edit the YAML — it's regenerated. Find the same lines in
`.llmstack/models.ini` (search for the tier name, e.g. `[code-fast]`) and
edit `hf_repo` / `hf_file` there. Then re-run `llmstack install`.

The download list comes from the same ini file: the four tiers map 1:1 to
four download jobs, each with `(repo, current-file, next-file?)`.

---

## Upgrade workflow

Apply this **per tier**, not all at once. Test one model at a time so you
can roll back cleanly.

```
0. (Optional, fast) llmstack check
   → prints current model + repo + last-modified + HF URL for each tier

1. Identify a candidate.
   - Hugging Face search:  https://huggingface.co/models?search=<keywords>+GGUF&sort=trending
   - Filter:               GGUF format, last 30 days, downloads > 1k
   - Cross-check:          a leaderboard appropriate to the tier (see below)

2. Verify the candidate is suitable for the tier.
   - GGUF available?              must exist as a .gguf file
   - Size in budget?              see "tier sizes" table below
   - Tool calls supported?        only matters for code-smart. Check that the model
                                   page lists "function calling" or its chat template
                                   handles tool-call blocks.
   - Native context length?       must be ≥ what you set with -c, or have YaRN config.

3. Smoke-test it before wiring it in. First, prefetch + validate via llama-cli:

       llama-cli -hf <new-repo> -hff <new-file>.gguf \
                 --no-warmup -ngl 0 -c 256 -p ok -n 1

       # then bring up llama-server on a throwaway port to test inference:
       llama-server -hf <new-repo> -hff <new-file>.gguf \
                    --port 9999 -ngl 999 -fa on \
                    --cache-type-k q8_0 --cache-type-v q8_0 -c 32768

       # in another terminal:
       curl -sN http://127.0.0.1:9999/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{"model":"x","messages":[{"role":"user","content":"hello"}]}'

   Confirm:
     - it loads without errors                     (look for "main: model loaded")
     - it streams a sensible response
     - it accepts your typical request shape       (tools, long context, …)

4. Edit ONE file: `.llmstack/models.ini`. In the matching `[tier]` section,
   update:

       hf_repo      = <new-repo>
       hf_file      = <new-file>.gguf
       ctx_size     = <new-context>     ; only if changing
       sampler      = temp=..., top_p=..., ...   ; only if changing
       size_gb, quant, status            ; for documentation accuracy

   No other files need editing — `llama-swap.yaml`, `opencode.json`, and
   the download list are all auto-derived from this.

5. Regenerate everything from the ini:

       llmstack install     # rewrites opencode.json + AGENTS.md
       llmstack download    # picks up the new (repo, file) pair from models.ini

6. Cycle the stack:

       llmstack stop
       llmstack start
       llmstack status
       curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'

7. Sanity-check via opencode (or curl) — fire a few characteristic prompts at
   the upgraded tier. If the new model misbehaves, the rollback is just a
   one-line revert in models.ini + `llmstack install` + `llmstack restart`.
```

---

## Judging "better" per tier

A model that scores +5 % on MMLU is not automatically better for *your*
workflow. Score against the role the tier plays.

### `code-fast` — autocomplete / FIM / quick Q&A

What matters:
- **Tokens/sec on M4 Max** (must clear ~60 tok/s for tab-feel responsiveness)
- **FIM (fill-in-the-middle) support** — the chat template must include a FIM
  format, and the model must be trained for it
- **Tool calling** — *not* required, this tier never calls tools

How to evaluate:
- Run `llama-bench -m <new>.gguf -p 512 -n 128 -ngl 999` for raw speed
- Sniff test with a typical autocomplete prompt; latency should feel like
  the cursor is barely ahead of you
- [Aider leaderboard](https://aider.chat/docs/leaderboards/) "edit format" column — proxy for FIM quality

Size budget: **~2–6 GB** weights (we want this resident permanently while
sharing memory with the heavy tier).

Good candidates to track:
- `bartowski/Qwen2.5-Coder-*-Instruct-GGUF` (any of 3B/7B)
- `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF`
- `unsloth/Qwen3-Coder-*-GGUF` (smaller variants)
- Any new "tab" / "FIM" coder ≤ 7B that drops

### `code-smart` — agent: tool calls, multi-file edits, refactors

What matters:
- **Tool / function-calling correctness** (the model must reliably emit
  well-formed tool-call blocks)
- **Long-context recall** (≥ 64 k usable, ideally 128 k+)
- **Code editing benchmarks** (aider edit success, SWE-Bench Verified)
- **Speed at full context** (MoE models win here on Apple Silicon)

How to evaluate:
- [Aider's LLM Leaderboard](https://aider.chat/docs/leaderboards/) — most
  honest signal for agentic coding
- [LiveCodeBench](https://livecodebench.github.io/leaderboard.html) scores
- [SWE-Bench Verified](https://www.swebench.com/) (the "real PRs" benchmark)
- Run an actual opencode session in `build` mode against your repo

Size budget: **~30–55 GB** weights (must fit alongside `code-fast` ≈ 5 GB
in your wired-mem cap).

Good candidates to track:
- `unsloth/Qwen3-Coder-*-GGUF` (incl. the Next 80B-A3B you have)
- `bartowski/DeepSeek-Coder-V*-Instruct-GGUF`
- `bartowski/Codestral-*-GGUF` future versions
- New "Coder" or "Devstral" releases as they appear

### `plan` — design discussions, architecture, trade-offs

What matters:
- **Reasoning quality** (MMLU-Pro, GPQA-Diamond)
- **Instruction-following on multi-step prompts** (IFEval)
- **Discussion style** — should propose alternatives, not jump to code
- **Refusals on edge cases** — fine to refuse weird stuff in plain plan mode

How to evaluate:
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) (filter to chat/instruct, your size class)
- [Chatbot Arena](https://lmarena.ai/) — vibes-based but useful proxy
- Hand-roll a "design this rate limiter" prompt and compare outputs

Size budget: **~7–25 GB** weights — this tier shouldn't dominate memory.

Good candidates to track:
- `bartowski/Qwen3-*-Instruct-GGUF`
- `bartowski/Mistral-Small-*-Instruct-GGUF` (non-uncensored)
- `bartowski/gemma-*-it-GGUF`
- `bartowski/glm-*-chat-GGUF`
- Reasoning-tuned variants: QwQ, DeepSeek-R1-Distill, Qwen3-Thinking

### `plan-uncensored` — no-filter planning

What matters:
- Same metrics as `plan`
- **Plus** demonstrably reduced refusal rate (look for "abliterated",
  "uncensored", "heretic", "dolphin", "neural-chat" branding)

Good candidates to track:
- `mradermacher/<base-model>-uncensored-*-GGUF`
- `mradermacher/<base-model>-heretic-i1-GGUF`
- `bartowski/<base-model>-abliterated-*-GGUF`
- `cognitivecomputations/dolphin-*` (then look for GGUF re-uploads)

Same size budget as `plan`.

---

## Where to find candidates

**HuggingFace search** (fastest):

- All recent GGUFs sorted by trending:
  https://huggingface.co/models?library=gguf&sort=trending
- New coder GGUFs in the last 30 days:
  https://huggingface.co/models?other=code&library=gguf&sort=created
- Specific maintainer feeds (subscribe / bookmark):
  - https://huggingface.co/bartowski (broad coverage, fast turnaround)
  - https://huggingface.co/unsloth (Qwen, Llama, with Dynamic UD quants)
  - https://huggingface.co/mradermacher (i1 / abliterated / heretic variants)
  - https://huggingface.co/lmstudio-community (curated, conservative quants)
  - https://huggingface.co/MaziyarPanahi (broad chat + coder)

**Leaderboards** (signal):

| Tier | Leaderboard |
|---|---|
| `code-fast` / `code-smart` | [Aider LLM Leaderboard](https://aider.chat/docs/leaderboards/) |
|                            | [LiveCodeBench](https://livecodebench.github.io/leaderboard.html) |
|                            | [SWE-Bench Verified](https://www.swebench.com/) |
| `plan` / `plan-uncensored` | [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) |
|                            | [Chatbot Arena](https://lmarena.ai/) |
|                            | [LiveBench](https://livebench.ai/) |

**Community signal** (qualitative but valuable):

- r/LocalLLaMA — daily threads on what's good
- HF model card discussions — real-world report-backs
- Each maintainer's repo READMEs often link to benchmarks they ran themselves

---

## Worked example: replacing the `plan` tier

Suppose tomorrow `bartowski` ships `Qwen3-Next-32B-Thinking-GGUF` and you
think it'd plan better than your current Qwopus GLM 18B.

```bash
# 0. Snapshot the current configuration
cd ~/Projects/opencode
llmstack check > /tmp/before.txt

# 1. Pre-pull via llama-cli (writes to the standard cache)
llama-cli -hf bartowski/Qwen3-Next-32B-Thinking-GGUF \
          -hff Qwen3-Next-32B-Thinking-Q5_K_M.gguf \
          --no-warmup -ngl 0 -c 256 -p ok -n 1

# 2. Smoke-test inference on a throwaway port
llama-server -hf bartowski/Qwen3-Next-32B-Thinking-GGUF \
             -hff Qwen3-Next-32B-Thinking-Q5_K_M.gguf \
             --port 9999 -ngl 999 -fa on --jinja \
             --cache-type-k q8_0 --cache-type-v q8_0 -c 65536 &

curl -sN http://127.0.0.1:9999/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"x","messages":[{"role":"user","content":"How would you architect a rate limiter for our API?"}]}'

kill %1   # stop the test server

# 3. Edit ONE file: .llmstack/models.ini  -> the [plan] section.
#      OLD:
#        hf_repo = Jackrong/Qwopus-GLM-18B-Merged-GGUF
#        hf_file = Qwopus-GLM-18B-Healed-Q4_K_M.gguf
#      NEW:
#        hf_repo = bartowski/Qwen3-Next-32B-Thinking-GGUF
#        hf_file = Qwen3-Next-32B-Thinking-Q5_K_M.gguf
#      Also bump size_gb / quant for documentation accuracy.

# 4. Apply: regenerate llama-swap.yaml + opencode.json, restart, verify.
llmstack install
llmstack restart
llmstack status
llmstack check          # confirms no DRIFT vs models.ini
```

Roll back? Revert the two lines in `.llmstack/models.ini` and re-run
`llmstack install && llmstack restart`. The old GGUF is still in the HF
cache so loading it costs nothing extra.

---

## After-upgrade housekeeping

When you're confident in the new model, reclaim disk space:

```bash
# What does llama.cpp see in its cache?
llama-cli -cl

# Sizes on disk (sorted)
du -h ~/.cache/huggingface/hub/* | sort -h

# Drop a single quant
llama-cli -cr <user>/<repo>:<quant>          # e.g. -cr unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M

# Drop a whole repo
rm -rf ~/.cache/huggingface/hub/models--<owner>--<repo>/
```

The old `Q4_K_M` for a tier you've upgraded is also a candidate for deletion
once the new quant is verified working.

---

## Upgrading the Python toolchain

The stack is a single Python package (`llmstack`) installed via
`pyproject.toml`. The runtime needs:

- `llmstack/app.py` — `fastapi`, `uvicorn[standard]`, `httpx` (router)
- `llmstack/check_models.py` — `huggingface_hub`, `PyYAML`
- `llmstack/tiers.py`, `llmstack/generators/*.py` — stdlib only

`hf_transfer` is a declared dep and gets used automatically when
`HF_HUB_ENABLE_HF_TRANSFER=1` is set, for faster multi-GB GGUF pulls.

Dependency versions live in `pyproject.toml`'s `[project.dependencies]`,
not in a `requirements.txt`. There is no checked-in venv either.

### First-time setup

The cleanest install puts the CLI on your PATH in an isolated env:

```bash
cd ~/Projects/opencode
pipx install -e .            # editable install + isolated venv
llmstack --version
```

Or, if you prefer a managed venv (no pipx):

```bash
cd ~/Projects/opencode
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/llmstack --version
```

Pin a specific Python (e.g. 3.13) by passing `python3.13 -m venv .venv` or
`pipx install --python python3.13 -e .`. Anything ≥ 3.11 works (we use
3.11+ syntax: PEP 604 `X | None`, `list[Tier]`, etc.).

### Routine upgrade (latest patch versions of declared deps)

```bash
cd ~/Projects/opencode
pipx upgrade llmstack             # if installed via pipx
# or, in your venv:
.venv/bin/pip install -U -e .

llmstack stop && llmstack start   # bounce the router
llmstack check                    # smoke-test PyYAML + huggingface_hub
```

### Bumping a major version (e.g. fastapi 0 → 1)

1. Edit the version constraint in `pyproject.toml`'s
   `[project.dependencies]` (e.g. `"fastapi>=1.0,<2.0"`).
2. Reinstall: `pipx upgrade llmstack` or `.venv/bin/pip install -U -e .`.
3. Test the router:

       llmstack stop && llmstack start
       curl -s http://127.0.0.1:10101/v1/models | jq '.data[].id'
       curl -s http://127.0.0.1:10101/models.ini | head
       curl -sN http://127.0.0.1:10101/v1/chat/completions -H 'Content-Type: application/json' \
            -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'

4. Read the upstream changelog for any breaking imports the router relies
   on (`fastapi.FastAPI`, `Request`, `Response`, `JSONResponse`,
   `StreamingResponse`, lifespan handlers, etc.). Update `llmstack/app.py`
   to match if needed.

### Rebuilding the install from scratch

If anything gets weird (corrupt install, Python upgrade, dependency
conflicts), nuke and reinstall — it's cheap:

```bash
# pipx install:
pipx uninstall llmstack
pipx install -e ~/Projects/opencode

# venv install:
cd ~/Projects/opencode
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install -e .
```

Nothing in the venv / pipx env is editable — all source lives under
`llmstack/` in the repo and is installed in editable mode (`-e`), so edits
land instantly.

### Upgrading the llama-swap binary

The binary lives at `$LLMSTACK_BIN_DIR/llama-swap`, which defaults to
`$XDG_DATA_HOME/llmstack/bin/llama-swap` (i.e.
`~/.local/share/llmstack/bin/llama-swap` on macOS). It is **not** in
version control — `llmstack setup` downloads it as part of the first-time
walkthrough, and `llmstack install-llama-swap` lets you re-fetch it on
demand.

```bash
llmstack install-llama-swap           # latest, idempotent (no-op if up-to-date)
llmstack install-llama-swap --force   # redownload even if up-to-date
LLAMA_SWAP_VERSION=v211 \
  llmstack install-llama-swap         # pin a specific tag
```

It auto-detects OS (Darwin / Linux / FreeBSD) and arch (arm64 / amd64) and
downloads the matching tarball from
https://github.com/mostlygeek/llama-swap/releases.

To check what's installed, look at the version line printed by
`llmstack install-llama-swap` (or run the binary directly):

```bash
"$(llmstack status 2>/dev/null | awk '/llama-swap binary/ {print $NF}')" --version
# or simply:
~/.local/share/llmstack/bin/llama-swap --version
```

After bumping the binary, re-run `llmstack install` only if the new
version of llama-swap requires a config-schema change (it usually doesn't —
the YAML schema is stable).

### Snapshotting the exact installed set

For reproducibility (e.g. before a risky upgrade) capture the full lock:

```bash
# pipx:
pipx runpip llmstack freeze > requirements.lock.txt

# venv:
.venv/bin/pip freeze > requirements.lock.txt

# …upgrade…
# if it goes wrong:
.venv/bin/pip install -r requirements.lock.txt
```

We don't commit `requirements.lock.txt` by default — the floor pins in
`pyproject.toml` are sufficient for this stack's blast radius.

---

## Quick reference — tier sizes & natural model classes

| Tier | Weights budget | Typical params (dense) | Typical params (MoE) | Examples |
|---|---|---|---|---|
| `code-fast` | 2–6 GB | 1.5–7 B | — | Qwen2.5-Coder 3B/7B, DeepSeek-Coder-V2-Lite |
| `code-smart` | 25–55 GB | 30–70 B | 30B-A3B / 80B-A3B | Qwen3-Coder-Next, DeepSeek-Coder-V2, Codestral |
| `plan` | 7–25 GB | 14–32 B | — | Qwen3, Mistral-Small, Gemma-3, GLM |
| `plan-uncensored` | 7–25 GB | 14–32 B | — | abliterated/heretic/dolphin variants of the above |

Anything outside these brackets either won't fit, or wastes hardware.
