"""DGX config helpers — read/write the ``dgx:`` block in config.yaml
and the ``model:`` block that points at DGX endpoints.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "host": "192.168.0.103",
    "ssh_user": "hartsock",
    "ollama_port": 11434,
    "vllm_port": 30800,
    "litellm_host": "192.168.0.104",
    "litellm_port": 4000,
    "active_endpoint": "ollama",
    "default_model": "qwen2.5-coder:latest",
}

ENDPOINT_LABELS = {
    "ollama": "Ollama (direct, no auth)",
    "vllm": "vLLM (direct, no auth)",
    "litellm": "LiteLLM proxy (HA pool, requires API key)",
}


# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------

def load_dgx_config() -> Dict[str, Any]:
    from hermes_cli.config import load_config
    cfg = load_config()
    dgx = dict(DEFAULTS)
    dgx.update(cfg.get("dgx") or {})
    return dgx


def save_dgx_config(dgx: Dict[str, Any]) -> None:
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg["dgx"] = dgx
    save_config(cfg)


def apply_endpoint(dgx: Dict[str, Any], endpoint: str | None = None) -> None:
    """Write model.provider + model.base_url to point at the given endpoint.

    If *endpoint* is None, uses dgx["active_endpoint"].
    Also updates dgx["active_endpoint"] and saves both dgx and model sections.
    """
    from hermes_cli.config import load_config, save_config

    ep = endpoint or dgx.get("active_endpoint", "ollama")
    host = dgx["host"]

    if ep == "ollama":
        base_url = f"http://{host}:{dgx['ollama_port']}/v1"
        provider = "ollama"
    elif ep == "vllm":
        base_url = f"http://{host}:{dgx['vllm_port']}/v1"
        provider = "custom"
    elif ep == "litellm":
        lh = dgx.get("litellm_host", "192.168.0.104")
        lp = dgx.get("litellm_port", 4000)
        base_url = f"http://{lh}:{lp}/v1"
        provider = "custom"
    else:
        raise ValueError(f"Unknown endpoint: {ep!r}")

    dgx["active_endpoint"] = ep

    cfg = load_config()
    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    cfg["model"]["provider"] = provider
    cfg["model"]["base_url"] = base_url
    cfg["dgx"] = dgx
    save_config(cfg)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def ollama_base(dgx: Dict[str, Any]) -> str:
    return f"http://{dgx['host']}:{dgx['ollama_port']}"


def vllm_base(dgx: Dict[str, Any]) -> str:
    return f"http://{dgx['host']}:{dgx['vllm_port']}"


def litellm_base(dgx: Dict[str, Any]) -> str:
    lh = dgx.get("litellm_host", "192.168.0.104")
    lp = dgx.get("litellm_port", 4000)
    return f"http://{lh}:{lp}"
