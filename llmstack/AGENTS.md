# llmstack — agent notes

Agent modes: `plan` / `plan-uncensored` are read-only (describe changes;
the user runs `/build` to apply). `build` mode edits files directly.

Skip `.llmstack/` during search and reads — it's generated runtime state.
Only look there if the user explicitly points you at it.

To adjust model parameters on the user's request, edit `.llmstack/models.ini`,
then run `llmstack install` (add `llmstack restart` if you changed
`sampler`, `ctx_size`, GGUF file). Don't ever change the model names or any creds.

Be concise. Don't narrate edits.
