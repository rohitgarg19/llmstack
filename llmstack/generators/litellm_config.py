"""Non-destructive merge of ``litellm_config.yaml`` against ``models.ini``.

Goal: ensure every tier in ``models.ini`` declared with
``backend = litellm`` has a matching ``model_list`` entry in the
per-project ``litellm_config.yaml`` -- so the litellm proxy (started
by llama-swap on a fixed port) can route requests for
``<tier>_<role>`` to the actual provider model.

Constraints:

* The user owns ``litellm_config.yaml``. They may have edited
  ``litellm_settings``, ``mcp_servers``, ``general_settings``, or any
  ``model_list`` entry. We must not lose those edits.
* PyYAML round-tripping is lossy for comments. We mitigate that by
  *only rewriting* the file when there are actually new or changed tier
  entries. Steady-state runs of ``llmstack install`` never touch the
  file. First-install seeds from the bundled template (whose comments
  are then lost only once, on the first rewrite that adds tier stubs).

Algorithm:

1. Parse the yaml. If the file is missing, seed from the template
   via :func:`ensure_litellm_config`.
2. Compute desired ``model_name`` values: ``tier.name`` (matching the
   ``[section]`` heading in ``models.ini``) for each litellm-backed
   tier, plus a parallel ``..._next`` when ``model_next`` is declared.
3. For each desired entry:
   a. If the ``model_name`` is absent from ``model_list``, append a
      new stub.
   b. If the ``model_name`` is present but its ``litellm_params.model``
      has drifted from ``models.ini``, update the model string
      unconditionally.  All other keys in the entry are preserved.
4. If nothing changed, return without writing.

This module is read by :mod:`llmstack.commands.install`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmstack.paths import ensure_litellm_config
from llmstack.tiers import load_tiers



def _desired_entries() -> list[dict[str, Any]]:
    """Walk :func:`load_tiers` and emit a stub per litellm tier (+ _next)."""
    out: list[dict[str, Any]] = []
    for tier in load_tiers().values():
        if not tier.is_litellm or tier.litellm is None:
            continue
        base = tier.name
        out.append({
            "model_name": base,
            "litellm_params": {"model": tier.litellm.model},
        })
        if tier.litellm.model_next:
            out.append({
                "model_name": f"{base}_next",
                "litellm_params": {"model": tier.litellm.model_next},
            })
    return out


def merge() -> tuple[Path, list[str], bool]:
    """Reconcile ``litellm_config.yaml`` against the litellm tiers.

    Returns ``(path, changed_names, rewrote)``. ``rewrote`` is ``True``
    only when the file content was rewritten (steady state: ``False``,
    no comments lost).

    Two classes of change are handled:

    * **New entries** -- a tier in models.ini has no matching
      ``model_name`` in ``model_list`` yet.  The stub is appended.
    * **Stale model strings** -- a tier's ``model`` key in models.ini
      has changed since the entry was first written.  The ``model``
      value inside ``litellm_params`` is always updated to match.
    """
    path, _ = ensure_litellm_config()
    text = path.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    desired = _desired_entries()
    if not desired:
        return path, [], False

    model_list = parsed.get("model_list")
    if not isinstance(model_list, list):
        model_list = []

    # Build a lookup of existing entries by model_name for O(1) access.
    existing_by_name: dict[str, dict[str, Any]] = {}
    for entry in model_list:
        if isinstance(entry, dict):
            n = entry.get("model_name")
            if isinstance(n, str):
                existing_by_name[n] = entry

    changed: list[str] = []

    for desired_entry in desired:
        name = desired_entry["model_name"]
        desired_model = desired_entry["litellm_params"]["model"]

        if name not in existing_by_name:
            # New tier -- append the stub.
            model_list.append(desired_entry)
            changed.append(name)
        else:
            existing_entry = existing_by_name[name]
            existing_model = (
                existing_entry.get("litellm_params") or {}
            ).get("model")
            if existing_model != desired_model:
                # Model string drifted -- update it unconditionally.
                if not isinstance(existing_entry.get("litellm_params"), dict):
                    existing_entry["litellm_params"] = {}
                existing_entry["litellm_params"]["model"] = desired_model
                changed.append(name)

    if not changed:
        return path, [], False

    parsed["model_list"] = model_list

    new_text = yaml.safe_dump(
        parsed,
        sort_keys=False,
        default_flow_style=False,
        width=200,
    )
    path.write_text(new_text, encoding="utf-8")
    return path, changed, True
