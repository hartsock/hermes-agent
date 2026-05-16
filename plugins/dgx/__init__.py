"""dgx plugin — manage NVIDIA DGX Spark inference endpoints from Hermes Agent.

Adds the ``hermes dgx`` subcommand group:
  setup     — interactive wizard to configure DGX host, endpoints, default model
  status    — show GPU memory, running models, endpoint health
  models    — list available models across Ollama and vLLM
  use       — switch the active model
  endpoint  — switch between ollama / vllm / litellm endpoints
"""

from __future__ import annotations

from plugins.dgx.cli import dgx_command, register_cli as _register_dgx_cli


def register(ctx) -> None:
    ctx.register_cli_command(
        name="dgx",
        help="NVIDIA DGX Spark endpoint management",
        setup_fn=_register_dgx_cli,
        handler_fn=dgx_command,
        description=(
            "Manage local GPU inference endpoints on a DGX Spark. "
            "See: hermes dgx setup"
        ),
    )
