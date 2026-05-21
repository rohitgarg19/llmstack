"""llmstack — multi-tier local LLM stack.

Public surface is the CLI (``llmstack <action>``). The package modules are
organised by concern:

  llmstack.app          FastAPI auto-router (uvicorn entry-point).
  llmstack.tiers        models.ini -> Tier dataclasses (data layer).
  llmstack.paths        state-dir / bin-dir / work-dir resolution.
  llmstack.shell_env    spawn the env-prepared subshell + activate hooks.
  llmstack.generators   render llama-swap.yaml + opencode.json from models.ini.
  llmstack.download     fetch GGUFs (via llama-completion) and the
                        llama-swap binary.
  llmstack.commands     one module per CLI action (setup / install / start ...).
  llmstack.cli          argparse dispatch (the `llmstack` console-script).
"""

from __future__ import annotations

__version__ = "0.12.0"
__all__ = ["__version__"]
