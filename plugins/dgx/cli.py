"""CLI commands for the dgx plugin.

Wires ``hermes dgx <subcommand>``:
  setup     — interactive wizard: configure host, ports, default model/endpoint
  status    — GPU memory, running models, endpoint health
  models    — list models available on Ollama and vLLM
  use       — switch active model (updates config.yaml model.default)
  endpoint  — switch active endpoint (ollama / vllm / litellm)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from plugins.dgx._dgx_config import (
    DEFAULTS,
    ENDPOINT_LABELS,
    apply_endpoint,
    litellm_base,
    load_dgx_config,
    ollama_base,
    save_dgx_config,
    vllm_base,
)


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="dgx_command")

    subs.add_parser(
        "setup",
        help="Interactive wizard: configure DGX host, endpoints, default model",
    )

    subs.add_parser(
        "status",
        help="Show GPU memory usage, running models, and endpoint health",
    )

    subs.add_parser(
        "models",
        help="List models available on Ollama and vLLM",
    )

    use_p = subs.add_parser("use", help="Switch the active model")
    use_p.add_argument("model", help="Model name (e.g. qwen2.5-coder:latest)")
    use_p.add_argument(
        "--endpoint",
        choices=["ollama", "vllm"],
        default=None,
        help="Endpoint to use for this model (auto-detected if omitted)",
    )

    ep_p = subs.add_parser(
        "endpoint",
        help="Switch between ollama / vllm / litellm endpoints",
    )
    ep_p.add_argument(
        "name",
        choices=["ollama", "vllm", "litellm"],
        help="Endpoint to activate",
    )

    subparser.set_defaults(func=dgx_command)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dgx_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "dgx_command", None)
    if not sub:
        print("usage: hermes dgx {setup,status,models,use,endpoint}")
        return 2
    if sub == "setup":
        return _cmd_setup()
    if sub == "status":
        return _cmd_status()
    if sub == "models":
        return _cmd_models()
    if sub == "use":
        return _cmd_use(model=args.model, endpoint=getattr(args, "endpoint", None))
    if sub == "endpoint":
        return _cmd_endpoint(name=args.name)
    print(f"unknown subcommand: {sub}")
    return 2


# ---------------------------------------------------------------------------
# HTTP helpers (no extra deps — stdlib only)
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 5) -> Tuple[Optional[Dict], Optional[str]]:
    """GET *url*, return (parsed_json, None) or (None, error_string)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read()), None
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except Exception as e:
        return None, str(e)


def _check_endpoint(url: str, timeout: int = 4) -> Tuple[bool, str]:
    """Return (reachable, status_string)."""
    _, err = _get_json(url, timeout=timeout)
    return (err is None), (err or "ok")


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_run(user: str, host: str, cmd: str, timeout: int = 10) -> Tuple[bool, str]:
    """Run *cmd* on host via SSH. Returns (ok, output_or_error)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes",
             f"{user}@{host}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr.strip() or f"exit {result.returncode}")
    except subprocess.TimeoutExpired:
        return False, "ssh timed out"
    except FileNotFoundError:
        return False, "ssh not found"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# hermes dgx status
# ---------------------------------------------------------------------------

def _cmd_status() -> int:
    dgx = load_dgx_config()
    host = dgx["host"]
    user = dgx["ssh_user"]
    active = dgx.get("active_endpoint", "ollama")

    print(f"DGX Spark  {host}  (active endpoint: {active})")
    print()

    # --- GPU via nvidia-smi over SSH ---
    print("GPU memory")
    ok, out = _ssh_run(
        user, host,
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits",
    )
    if ok and out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, used, total, util = parts[0], parts[1], parts[2], parts[3], parts[4]
                bar_pct = int(used) / max(int(total), 1)
                bar = "█" * int(bar_pct * 20) + "░" * (20 - int(bar_pct * 20))
                print(f"  GPU {idx}  {name}")
                print(f"  [{bar}] {used}/{total} MiB  ({util}% util)")
    else:
        print(f"  (SSH unavailable: {out})")
    print()

    # --- Ollama ---
    obase = ollama_base(dgx)
    data, err = _get_json(f"{obase}/api/tags")
    if data is not None:
        models = [m["name"] for m in data.get("models", [])]
        print(f"Ollama  {obase}  ✓  ({len(models)} models loaded)")
        for m in models:
            marker = " ◀ active" if (active == "ollama" and m.startswith(
                dgx.get("default_model", "").split(":")[0])) else ""
            print(f"  {m}{marker}")
    else:
        print(f"Ollama  {obase}  ✗  ({err})")
    print()

    # --- vLLM ---
    vbase = vllm_base(dgx)
    data, err = _get_json(f"{vbase}/v1/models")
    if data is not None:
        models = [m["id"] for m in data.get("data", [])]
        print(f"vLLM    {vbase}  ✓  ({len(models)} models loaded)")
        for m in models:
            marker = " ◀ active" if active == "vllm" else ""
            print(f"  {m}{marker}")
    else:
        print(f"vLLM    {vbase}  ✗  ({err})")
    print()

    # --- LiteLLM ---
    lbase = litellm_base(dgx)
    ok, msg = _check_endpoint(f"{lbase}/health")
    sym = "✓" if ok else "✗"
    marker = " ◀ active" if active == "litellm" else ""
    print(f"LiteLLM {lbase}  {sym}  ({msg}){marker}")

    return 0


# ---------------------------------------------------------------------------
# hermes dgx models
# ---------------------------------------------------------------------------

def _cmd_models() -> int:
    dgx = load_dgx_config()
    found_any = False

    # Ollama
    obase = ollama_base(dgx)
    data, err = _get_json(f"{obase}/api/tags")
    if data is not None:
        models = data.get("models", [])
        print(f"Ollama  {obase}")
        for m in models:
            name = m["name"]
            size_gb = m.get("size", 0) / 1e9
            print(f"  {name:<40} {size_gb:.1f} GB")
        found_any = True
    else:
        print(f"Ollama  {obase}  (unreachable: {err})")
    print()

    # vLLM
    vbase = vllm_base(dgx)
    data, err = _get_json(f"{vbase}/v1/models")
    if data is not None:
        models = data.get("data", [])
        print(f"vLLM    {vbase}")
        for m in models:
            print(f"  {m['id']}")
        found_any = True
    else:
        print(f"vLLM    {vbase}  (unreachable: {err})")

    return 0 if found_any else 1


# ---------------------------------------------------------------------------
# hermes dgx use <model>
# ---------------------------------------------------------------------------

def _cmd_use(model: str, endpoint: Optional[str] = None) -> int:
    from hermes_cli.config import load_config, save_config

    dgx = load_dgx_config()

    # Auto-detect endpoint from which service has the model, if not specified
    if endpoint is None:
        endpoint = dgx.get("active_endpoint", "ollama")
        obase = ollama_base(dgx)
        data, _ = _get_json(f"{obase}/api/tags")
        if data is not None:
            ollama_models = [m["name"] for m in data.get("models", [])]
            model_root = model.split(":")[0]
            if any(m.startswith(model_root) for m in ollama_models):
                endpoint = "ollama"

    dgx["default_model"] = model
    apply_endpoint(dgx, endpoint)

    # Also update model.default
    cfg = load_config()
    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    cfg["model"]["default"] = model
    save_config(cfg)

    print(f"Active model : {model}")
    print(f"Endpoint     : {endpoint}  ({ENDPOINT_LABELS.get(endpoint, endpoint)})")
    return 0


# ---------------------------------------------------------------------------
# hermes dgx endpoint <name>
# ---------------------------------------------------------------------------

def _cmd_endpoint(name: str) -> int:
    dgx = load_dgx_config()
    apply_endpoint(dgx, name)
    label = ENDPOINT_LABELS.get(name, name)
    print(f"Switched to {name}  ({label})")
    _print_model_summary(dgx, name)
    return 0


def _print_model_summary(dgx: Dict[str, Any], endpoint: str) -> None:
    from hermes_cli.config import load_config
    cfg = load_config()
    model = cfg.get("model", {})
    print(f"  base_url : {model.get('base_url', '(not set)')}")
    print(f"  provider : {model.get('provider', '(not set)')}")
    print(f"  model    : {model.get('default', '(not set)')}")


# ---------------------------------------------------------------------------
# hermes dgx setup
# ---------------------------------------------------------------------------

def _prompt(label: str, default: Any) -> str:
    val = input(f"  {label} [{default}]: ").strip()
    return val if val else str(default)


def _probe_ollama(host: str, port: int) -> Tuple[bool, List[str]]:
    url = f"http://{host}:{port}/api/tags"
    data, err = _get_json(url, timeout=4)
    if data is None:
        return False, []
    models = [m["name"] for m in data.get("models", [])]
    return True, models


def _probe_vllm(host: str, port: int) -> Tuple[bool, List[str]]:
    url = f"http://{host}:{port}/v1/models"
    data, err = _get_json(url, timeout=4)
    if data is None:
        return False, []
    models = [m["id"] for m in data.get("data", [])]
    return True, models


def _probe_litellm(host: str, port: int) -> bool:
    url = f"http://{host}:{port}/health"
    ok, _ = _check_endpoint(url, timeout=4)
    return ok


def _cmd_setup() -> int:
    current = load_dgx_config()

    print()
    print("hermes dgx setup")
    print("────────────────")
    print("Configure your DGX Spark inference endpoints.")
    print("Press Enter to accept the current value shown in brackets.")
    print()

    # --- Host / SSH ---
    print("DGX Spark host")
    host = _prompt("IP address", current.get("host", DEFAULTS["host"]))
    ssh_user = _prompt("SSH user", current.get("ssh_user", DEFAULTS["ssh_user"]))
    print()

    # --- Ports ---
    print("Endpoint ports")
    ollama_port = int(_prompt("Ollama port", current.get("ollama_port", DEFAULTS["ollama_port"])))
    vllm_port = int(_prompt("vLLM port", current.get("vllm_port", DEFAULTS["vllm_port"])))
    print()

    # --- Probe endpoints ---
    print("Probing endpoints...")
    ollama_ok, ollama_models = _probe_ollama(host, ollama_port)
    vllm_ok, vllm_models = _probe_vllm(host, vllm_port)

    print(f"  Ollama  http://{host}:{ollama_port}  {'✓' if ollama_ok else '✗'}", end="")
    if ollama_ok:
        print(f"  ({len(ollama_models)} models: {', '.join(ollama_models[:3])}{'...' if len(ollama_models) > 3 else ''})")
    else:
        print("  (unreachable — check host/port or VPN)")

    print(f"  vLLM    http://{host}:{vllm_port}   {'✓' if vllm_ok else '✗'}", end="")
    if vllm_ok:
        print(f"  ({', '.join(vllm_models)})")
    else:
        print("  (unreachable)")
    print()

    # --- LiteLLM (optional) ---
    print("LiteLLM proxy (optional — HA pool across all GPU nodes)")
    litellm_host = _prompt("LiteLLM host", current.get("litellm_host", DEFAULTS["litellm_host"]))
    litellm_port = int(_prompt("LiteLLM port", current.get("litellm_port", DEFAULTS["litellm_port"])))
    litellm_ok = _probe_litellm(litellm_host, litellm_port)
    print(f"  LiteLLM http://{litellm_host}:{litellm_port}  {'✓' if litellm_ok else '✗'}", end="")
    if not litellm_ok:
        print("  (unreachable — key required; set OPENAI_API_KEY in ~/.hermes/.env)")
    else:
        print()
    print()

    # --- Choose default endpoint ---
    available = []
    if ollama_ok:
        available.append("ollama")
    if vllm_ok:
        available.append("vllm")
    if litellm_ok:
        available.append("litellm")

    current_ep = current.get("active_endpoint", "ollama")
    if not available:
        print("WARNING: no endpoints are reachable. Saving config anyway.")
        print("         Check your VPN (WireGuard) or DGX host/port settings.")
        chosen_ep = current_ep
    else:
        print("Default endpoint")
        for i, ep in enumerate(available, 1):
            marker = " (current)" if ep == current_ep else ""
            print(f"  {i}) {ep:<8} — {ENDPOINT_LABELS[ep]}{marker}")
        while True:
            raw = input(f"  Choice [{'1' if available else '?'}]: ").strip()
            if not raw and available:
                chosen_ep = available[0]
                break
            try:
                chosen_ep = available[int(raw) - 1]
                break
            except (ValueError, IndexError):
                print("  Enter a number from the list above.")
    print()

    # --- Choose default model ---
    all_models: List[str] = []
    if chosen_ep == "ollama":
        all_models = ollama_models
    elif chosen_ep == "vllm":
        all_models = vllm_models

    current_model = current.get("default_model", DEFAULTS["default_model"])
    if all_models:
        print("Default model")
        for i, m in enumerate(all_models, 1):
            marker = " (current)" if m == current_model else ""
            print(f"  {i}) {m}{marker}")
        print(f"  Or type a model name directly.")
        raw = input(f"  Choice [{current_model}]: ").strip()
        if not raw:
            chosen_model = current_model
        elif raw.isdigit() and 1 <= int(raw) <= len(all_models):
            chosen_model = all_models[int(raw) - 1]
        else:
            chosen_model = raw
    else:
        chosen_model = _prompt("Default model", current_model)
    print()

    # --- Save ---
    dgx = {
        "host": host,
        "ssh_user": ssh_user,
        "ollama_port": ollama_port,
        "vllm_port": vllm_port,
        "litellm_host": litellm_host,
        "litellm_port": litellm_port,
        "active_endpoint": chosen_ep,
        "default_model": chosen_model,
    }
    save_dgx_config(dgx)
    apply_endpoint(dgx, chosen_ep)

    # Update model.default too
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    cfg["model"]["default"] = chosen_model
    save_config(cfg)

    print("Configuration saved.")
    print()
    print(f"  DGX host  : {host}")
    print(f"  Endpoint  : {chosen_ep}  ({ENDPOINT_LABELS.get(chosen_ep, chosen_ep)})")
    print(f"  Model     : {chosen_model}")
    print()
    print("Next steps:")
    print("  hermes dgx status    — verify GPU and endpoint health")
    print("  hermes dgx models    — browse available models")
    print("  hermes               — start chatting")
    return 0


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="hermes dgx")
    register_cli(parser)
    ns = parser.parse_args()
    sys.exit(dgx_command(ns))
