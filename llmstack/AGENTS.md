# Local llmstack — agent operating notes

You are running inside opencode against a local llama-swap stack served by a
FastAPI auto-router. You have access to four model tiers, but you do NOT
choose which one runs — that's the router's job. Behave consistently
regardless of which tier picked up this turn.

## How routing works (FYI, you don't need to act on this)

Requests with `model: auto` go through the FastAPI router (`llmstack.app`),
which classifies the request body and rewrites `model` to one of:

- `code-fast` — Qwen2.5-Coder 3B, default for trivial chat
- `code-smart` — Qwen3-Coder-Next 80B MoE, picked when tools, multi-turn,
  large context, code blocks, or "implement / fix / refactor" verbs appear
- `plan` — Qwopus GLM 18B, picked when "design / architect / trade-off"
  verbs appear
- `plan-uncensored` — Mistral-Small 3.2 24B Heretic, only when the user
  explicitly opts in via `[nofilter]` / `[uncensored]` / `nofilter:` /
  `uncensored:` triggers

Slash-commands are available as shortcuts:

- `/fast`     — pin the small coder for a one-shot answer
- `/review`   — kick the planning model for a review pass
- `/nofilter` — route to the no-filter model

## House style

- Be concise. Local models are slower per token than hosted ones;
  every redundant paragraph costs the user real wall-clock time.
- Do not narrate file edits in prose — let the diff speak.
- When unsure between two approaches, pick one and explain the trade-off
  in one sentence rather than asking the user to choose.
- Code comments should explain non-obvious intent, not restate the code.
- Prefer editing existing files to creating new ones.

## Do NOT scan `.llmstack/`

`.llmstack/` is per-project runtime state — auto-generated, gitignored,
not source. Skip it during search, exploration, and bulk reads:

- Do not `grep` / `rg` / `glob` into `.llmstack/` to answer questions
  about the project. The answers live in `llmstack/` (the package),
  in `llmstack/generators/*.py`, or in the bundled
  `llmstack/models.ini` template.
- Do not read `.llmstack/llama-swap.yaml` or `.llmstack/opencode.json`
  to understand intent — they are regenerated on every `llmstack
  install` and will silently lie about the design. Read the generators
  and the models.ini they consume instead.
- Do not read anything under `.llmstack/logs/` unless the user is
  actively debugging a runtime issue and points you there. Log files
  are large, noisy, and burn context.
- `.llmstack/models.ini` is fair game when the user asks "which tiers
  is *this* project wired up for" — it's the project-local copy of
  the template — but treat the bundled `llmstack/models.ini` as the
  canonical reference for defaults.

If a tool call returns matches inside `.llmstack/`, treat them as noise
and re-scope the search to exclude that directory.

## Environment facts

- macOS on Apple Silicon, Metal acceleration via llama.cpp.
- Local-only stack: no outbound network calls, no telemetry, no sharing.
- The `plan` and `plan-nofilter` agents are read-only by configuration —
  do not propose `bash`/`edit`/`write` actions in those modes; describe
  what should change and let the user run `/build` to apply it.
- Title generation, summaries, and other "small_model" calls go to the
  3B coder. Keep titles under 6 words.

## When the user mentions the stack itself

The stack is the `llmstack` Python package, exposed as the `llmstack` CLI
(`pip install -e .` from this repo). Source of truth is
`.llmstack/models.ini` in the project (or `$LLMSTACK_MODELS_INI`); the
per-project `.llmstack/llama-swap.yaml` and `.llmstack/opencode.json` are
auto-generated
by `llmstack.generators.llama_swap` and `llmstack.generators.opencode`,
overwritten on every `llmstack install`. Never hand-edit the generated files;
edit `models.ini` or the generators. The user's *global* opencode config at
`~/.config/opencode/opencode.json` is intentionally NOT touched by this
stack — opencode picks up our config because `OPENCODE_CONFIG` is exported
in the subshell that `llmstack start` drops the user into.

`opencode.json` (and this `AGENTS.md` next to it) lives in `$PWD/.llmstack/`
of wherever `llmstack` was invoked. The user can `cd` into any project,
run `llmstack start`, and get a project-local opencode config there.
The llama-swap and router daemons are singleton on ports 10101/10102 and
shared across projects; `start` from a second project detects the running
daemons and reuses them rather than fighting for the ports. If you see
`LLMSTACK_CHANNEL=external` it means the daemons were launched from a
different project's `.llmstack/`.

`LLMSTACK_CHANNEL` values you may see:

- `current` — local stack, canonical channel.
- `next` — local stack, but the underlying GGUF files have been swapped
  to the queued upgrade target for any tier with `hf_file_next` set in
  `models.ini`. Semantics otherwise unchanged.
- `external` — **the user is a thin client of a remote llmstack**
  (`LLMSTACK_REMOTE_URL` is set; the prompt is medium-purple and shows
  the URL: `[llmstack:<project> <url>]`). No llama-swap or router runs
  on this host; opencode talks to the remote router directly. The user
  can't inspect logs, reload daemons, or change channels from this side
  — all of that has to happen on the remote host.
- `shared` — local daemons are running, but they were started by a
  different project on this host (no pid file in this project's
  `.llmstack/`). Stopping or restarting the stack from here will affect
  every project on this host that's currently using those daemons.
