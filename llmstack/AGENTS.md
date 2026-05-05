# Local llmstack — agent operating notes

You are running inside opencode against a local llama-swap stack. There are
four model tiers; you do **not** pick which one runs. Behave consistently
regardless of which tier picked up this turn.

## The four tiers

| Tier              | Role               | Used for                                      |
| ----------------- | ------------------ | --------------------------------------------- |
| `code-fast`       | tiny resident coder | autocomplete, titles, trivial Q&A             |
| `code-smart`      | heavy coder        | tool calls, multi-file edits, agent loops     |
| `plan`            | chat / planner     | design discussions, architecture, trade-offs  |
| `plan-uncensored` | unfiltered planner | only when the user explicitly opts in         |

The `plan` and `plan-uncensored` tiers back read-only agents — describe
what should change rather than proposing `bash` / `edit` / `write` actions
in those modes; the user runs `/build` to apply.

## `models.ini` — the source of truth

Tier inventory lives in `llmstack/models.ini` (bundled template) and
`.llmstack/models.ini` (per-project copy). When the user asks "what model
is X" or "which tiers are wired up", read **`models.ini`**. Nothing else
is canonical.

Each tier section declares one of two backends:

- **`gguf`** (default, auto-detected from `hf_repo` + `hf_file`) — local
  weights served by llama-swap.
- **`bedrock`** (auto-detected from `aws_model_id`) — hosted AWS Bedrock
  model called directly by the router. Credentials are scoped per tier
  (different tiers can live in different accounts/regions). See the
  `[BEDROCK EXAMPLES]` block at the bottom of the file.

Keys you'll see commonly:

- `hf_repo` / `hf_file` — current GGUF.
- `hf_file_next` (and optional `hf_repo_next`) — queued upgrade target;
  swapped in when `LLMSTACK_CHANNEL=next`.
- `ctx_size`, `sampler`, `quant`, `size_gb` — runtime knobs / metadata.
- `[ROUTING]` — tunables for the auto-router (signal words, token budgets,
  uncensored triggers). Edit here, not in the router code.

If the user wants to change model selection, add a tier, or swap to
Bedrock, they edit `models.ini` and re-run `llmstack install`.

## When you edit `models.ini` on the user's behalf

Edit **`.llmstack/models.ini`** (the per-project live copy) — *not* the
bundled template at `llmstack/models.ini`, which only seeds the project
copy on first install. Then apply the change:

```bash
llmstack install     # regenerate opencode.json from models.ini
llmstack restart     # regenerate llama-swap.yaml + cycle daemons so it's live
```

Notes:

- `install` only writes `opencode.json` (+ AGENTS.md copy). The
  llama-swap config is owned by `start`/`restart` -- they regenerate
  `llama-swap.yaml` for the resolved channel on each fresh launch.
- If you only changed routing-relevant keys that opencode cares about
  (model lists, agent wiring), `install` alone is enough -- opencode
  re-reads its config per request.
- For tier-runtime changes (sampler, ctx, GGUF file, Bedrock creds),
  the daemons need to come down: `llmstack restart` regenerates the
  yaml and cycles them.
- `restart` accepts the same channel flags as `start` (`--current` /
  `--next`); pass through whatever channel the user is on.
- Don't run `restart` if the only thing you changed was the `description`
  or a comment -- those don't reach the runtime; `install` is fine.
- If `llmstack install` or the yaml regen during `start` fails validation,
  fix `models.ini` and re-run; the generated files are written atomically
  so a failure leaves the previous good config in place.

## Do NOT scan `.llmstack/`

`.llmstack/` is per-project runtime state — auto-generated, gitignored,
regenerated on every `llmstack install` (opencode.json) or `llmstack
start` (llama-swap.yaml). Skip it during search, exploration, and bulk
reads:

- Don't `grep` / `rg` / glob into `.llmstack/` to answer questions about
  the project. Answers live in the `llmstack/` package and `models.ini`.
- Don't read `.llmstack/llama-swap.yaml` or `.llmstack/opencode.json`
  to understand intent — they're generated outputs and will silently
  mislead. Read the generators or `models.ini` instead.
- Don't read `.llmstack/logs/**` unless the user is actively debugging a
  runtime issue and points you there. Logs are large and burn context.
- `.llmstack/models.ini` is fair game when the user asks "which tiers is
  *this* project wired up for" — it's the project-local copy.

If a tool call returns matches inside `.llmstack/`, treat them as noise
and re-scope the search to exclude that directory.

## House style

- Be concise. Local models are slower per token than hosted ones; every
  redundant paragraph costs the user real wall-clock time.
- Don't narrate file edits in prose — let the diff speak.
- When unsure between two approaches, pick one and explain the trade-off
  in one sentence rather than asking the user to choose.
- Comments should explain non-obvious intent, not restate the code.
- Prefer editing existing files to creating new ones.
- Title generation goes to the 3B coder. Keep titles under 6 words.
