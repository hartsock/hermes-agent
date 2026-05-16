"""Tests for plugins/dgx/_dgx_config.py

Covers config load/save, endpoint application, and URL helpers.
All tests use an isolated HERMES_HOME (from conftest) so they never
touch the developer's real ~/.hermes/config.yaml.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(monkeypatch, initial: dict):
    """Patch load_config / save_config with a simple in-memory store."""
    store = dict(initial)

    def _load():
        return dict(store)

    def _save(cfg):
        store.clear()
        store.update(cfg)

    import plugins.dgx._dgx_config as dc
    monkeypatch.setattr("hermes_cli.config.load_config", _load)
    monkeypatch.setattr("hermes_cli.config.save_config", _save)
    monkeypatch.setattr(dc, "load_config", _load, raising=False)
    monkeypatch.setattr(dc, "save_config", _save, raising=False)
    return store


# ---------------------------------------------------------------------------
# load_dgx_config
# ---------------------------------------------------------------------------

class TestLoadDgxConfig:
    def test_returns_defaults_when_no_dgx_section(self, monkeypatch):
        from plugins.dgx._dgx_config import DEFAULTS, load_dgx_config
        _make_config(monkeypatch, {})
        cfg = load_dgx_config()
        assert cfg["host"] == DEFAULTS["host"]
        assert cfg["ollama_port"] == DEFAULTS["ollama_port"]
        assert cfg["vllm_port"] == DEFAULTS["vllm_port"]
        assert cfg["active_endpoint"] == DEFAULTS["active_endpoint"]

    def test_merges_user_values_over_defaults(self, monkeypatch):
        from plugins.dgx._dgx_config import load_dgx_config
        _make_config(monkeypatch, {"dgx": {"host": "10.0.0.5", "ssh_user": "admin"}})
        cfg = load_dgx_config()
        assert cfg["host"] == "10.0.0.5"
        assert cfg["ssh_user"] == "admin"
        assert cfg["ollama_port"] == 11434  # default preserved

    def test_partial_override_preserves_remaining_defaults(self, monkeypatch):
        from plugins.dgx._dgx_config import load_dgx_config
        _make_config(monkeypatch, {"dgx": {"vllm_port": 9000}})
        cfg = load_dgx_config()
        assert cfg["vllm_port"] == 9000
        assert cfg["ollama_port"] == 11434  # untouched


# ---------------------------------------------------------------------------
# save_dgx_config
# ---------------------------------------------------------------------------

class TestSaveDgxConfig:
    def test_writes_dgx_key(self, monkeypatch):
        from plugins.dgx._dgx_config import save_dgx_config
        store = _make_config(monkeypatch, {})
        save_dgx_config({"host": "192.168.1.1", "active_endpoint": "vllm"})
        assert store.get("dgx", {}).get("host") == "192.168.1.1"
        assert store.get("dgx", {}).get("active_endpoint") == "vllm"

    def test_preserves_existing_non_dgx_keys(self, monkeypatch):
        from plugins.dgx._dgx_config import save_dgx_config
        store = _make_config(monkeypatch, {"model": {"default": "some-model"}})
        save_dgx_config({"host": "192.168.1.1"})
        assert store.get("model", {}).get("default") == "some-model"


# ---------------------------------------------------------------------------
# apply_endpoint
# ---------------------------------------------------------------------------

class TestApplyEndpoint:
    def _run(self, monkeypatch, endpoint: str, dgx: dict | None = None):
        from plugins.dgx._dgx_config import DEFAULTS, apply_endpoint
        base_dgx = dict(DEFAULTS)
        if dgx:
            base_dgx.update(dgx)
        store = _make_config(monkeypatch, {})
        apply_endpoint(base_dgx, endpoint)
        return store

    def test_ollama_sets_correct_provider_and_url(self, monkeypatch):
        store = self._run(monkeypatch, "ollama")
        model = store.get("model", {})
        assert model["provider"] == "ollama"
        assert "192.168.0.103" in model["base_url"]
        assert "11434" in model["base_url"]

    def test_vllm_sets_correct_provider_and_url(self, monkeypatch):
        store = self._run(monkeypatch, "vllm")
        model = store.get("model", {})
        assert model["provider"] == "custom"
        assert "30800" in model["base_url"]

    def test_litellm_sets_correct_provider_and_url(self, monkeypatch):
        store = self._run(monkeypatch, "litellm")
        model = store.get("model", {})
        assert model["provider"] == "custom"
        assert "192.168.0.104" in model["base_url"]
        assert "4000" in model["base_url"]

    def test_updates_active_endpoint_in_dgx_block(self, monkeypatch):
        store = self._run(monkeypatch, "vllm")
        assert store.get("dgx", {}).get("active_endpoint") == "vllm"

    def test_unknown_endpoint_raises(self, monkeypatch):
        from plugins.dgx._dgx_config import DEFAULTS, apply_endpoint
        _make_config(monkeypatch, {})
        with pytest.raises(ValueError, match="Unknown endpoint"):
            apply_endpoint(dict(DEFAULTS), "bogus")

    def test_custom_host_reflected_in_url(self, monkeypatch):
        store = self._run(monkeypatch, "ollama", dgx={"host": "10.0.0.99"})
        assert "10.0.0.99" in store["model"]["base_url"]


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

class TestUrlHelpers:
    def _dgx(self, **overrides):
        from plugins.dgx._dgx_config import DEFAULTS
        d = dict(DEFAULTS)
        d.update(overrides)
        return d

    def test_ollama_base_default(self):
        from plugins.dgx._dgx_config import ollama_base
        assert ollama_base(self._dgx()) == "http://192.168.0.103:11434"

    def test_vllm_base_default(self):
        from plugins.dgx._dgx_config import vllm_base
        assert vllm_base(self._dgx()) == "http://192.168.0.103:30800"

    def test_litellm_base_default(self):
        from plugins.dgx._dgx_config import litellm_base
        assert litellm_base(self._dgx()) == "http://192.168.0.104:4000"

    def test_custom_host_and_port(self):
        from plugins.dgx._dgx_config import ollama_base
        assert ollama_base(self._dgx(host="10.0.0.5", ollama_port=9999)) == "http://10.0.0.5:9999"
