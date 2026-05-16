"""Tests for plugins/dgx/cli.py

Current functionality (green): HTTP helpers, SSH helpers, probe functions,
argparse wiring, dispatch, endpoint/use/models/status commands.

Planned functionality (red — TDD work items): pull, rm, ps, run, push,
doctor, watch, agent tools. These are marked xfail with the tier they
belong to. A passing xfail means the feature shipped; flip to a real test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dgx_defaults():
    from plugins.dgx._dgx_config import DEFAULTS
    return dict(DEFAULTS)


@pytest.fixture
def mock_config(monkeypatch, dgx_defaults):
    """Patch load_dgx_config and apply_endpoint to avoid real file I/O."""
    stored = {"dgx": dict(dgx_defaults), "model": {}}

    def _load_dgx():
        return dict(dgx_defaults)

    def _load():
        return dict(stored)

    def _save(cfg):
        stored.clear()
        stored.update(cfg)

    import plugins.dgx.cli as cli_mod
    import plugins.dgx._dgx_config as cfg_mod

    monkeypatch.setattr(cli_mod, "load_dgx_config", _load_dgx)
    monkeypatch.setattr(cfg_mod, "load_config", _load, raising=False)
    monkeypatch.setattr(cfg_mod, "save_config", _save, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", _load)
    monkeypatch.setattr("hermes_cli.config.save_config", _save)
    return stored


# ---------------------------------------------------------------------------
# _get_json
# ---------------------------------------------------------------------------

class TestGetJson:
    def test_returns_parsed_json_on_success(self):
        import urllib.request
        from plugins.dgx.cli import _get_json
        payload = json.dumps({"models": []}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            data, err = _get_json("http://localhost:11434/api/tags")
        assert data == {"models": []}
        assert err is None

    def test_returns_none_and_error_on_connection_refused(self):
        import urllib.error
        from plugins.dgx.cli import _get_json
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            data, err = _get_json("http://localhost:11434/api/tags")
        assert data is None
        assert err is not None

    def test_returns_none_on_timeout(self):
        import urllib.error
        from plugins.dgx.cli import _get_json
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            data, err = _get_json("http://localhost:11434/api/tags", timeout=1)
        assert data is None


# ---------------------------------------------------------------------------
# _check_endpoint
# ---------------------------------------------------------------------------

class TestCheckEndpoint:
    def test_returns_true_when_reachable(self):
        from plugins.dgx.cli import _check_endpoint
        with patch("plugins.dgx.cli._get_json", return_value=({"ok": True}, None)):
            ok, msg = _check_endpoint("http://localhost:4000/health")
        assert ok is True

    def test_returns_false_when_not_reachable(self):
        from plugins.dgx.cli import _check_endpoint
        with patch("plugins.dgx.cli._get_json", return_value=(None, "Connection refused")):
            ok, msg = _check_endpoint("http://localhost:4000/health")
        assert ok is False
        assert "Connection refused" in msg


# ---------------------------------------------------------------------------
# _ssh_run
# ---------------------------------------------------------------------------

class TestSshRun:
    def test_returns_stdout_on_success(self):
        from plugins.dgx.cli import _ssh_run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU 0, A100, 10000, 40960, 25\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, out = _ssh_run("hartsock", "192.168.0.103", "nvidia-smi")
        assert ok is True
        assert "GPU 0" in out

    def test_returns_false_on_nonzero_exit(self):
        from plugins.dgx.cli import _ssh_run
        mock_result = MagicMock()
        mock_result.returncode = 255
        mock_result.stdout = ""
        mock_result.stderr = "Connection refused"
        with patch("subprocess.run", return_value=mock_result):
            ok, out = _ssh_run("hartsock", "192.168.0.103", "nvidia-smi")
        assert ok is False

    def test_returns_false_on_timeout(self):
        from plugins.dgx.cli import _ssh_run
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 10)):
            ok, out = _ssh_run("hartsock", "192.168.0.103", "nvidia-smi")
        assert ok is False
        assert "timed out" in out

    def test_returns_false_when_ssh_not_found(self):
        from plugins.dgx.cli import _ssh_run
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ok, out = _ssh_run("hartsock", "192.168.0.103", "nvidia-smi")
        assert ok is False
        assert "not found" in out


# ---------------------------------------------------------------------------
# _probe_ollama / _probe_vllm / _probe_litellm
# ---------------------------------------------------------------------------

class TestProbes:
    def test_probe_ollama_returns_model_names_on_success(self):
        from plugins.dgx.cli import _probe_ollama
        payload = {"models": [{"name": "nemotron3:33b"}, {"name": "qwen2.5-coder:14b"}]}
        with patch("plugins.dgx.cli._get_json", return_value=(payload, None)):
            ok, models = _probe_ollama("192.168.0.103", 11434)
        assert ok is True
        assert "nemotron3:33b" in models
        assert len(models) == 2

    def test_probe_ollama_returns_false_on_failure(self):
        from plugins.dgx.cli import _probe_ollama
        with patch("plugins.dgx.cli._get_json", return_value=(None, "refused")):
            ok, models = _probe_ollama("192.168.0.103", 11434)
        assert ok is False
        assert models == []

    def test_probe_vllm_returns_model_ids_on_success(self):
        from plugins.dgx.cli import _probe_vllm
        payload = {"data": [{"id": "qwen2.5-coder-3b"}]}
        with patch("plugins.dgx.cli._get_json", return_value=(payload, None)):
            ok, models = _probe_vllm("192.168.0.103", 30800)
        assert ok is True
        assert "qwen2.5-coder-3b" in models

    def test_probe_vllm_returns_false_on_failure(self):
        from plugins.dgx.cli import _probe_vllm
        with patch("plugins.dgx.cli._get_json", return_value=(None, "refused")):
            ok, models = _probe_vllm("192.168.0.103", 30800)
        assert ok is False

    def test_probe_litellm_returns_true_when_healthy(self):
        from plugins.dgx.cli import _probe_litellm
        with patch("plugins.dgx.cli._check_endpoint", return_value=(True, "ok")):
            assert _probe_litellm("192.168.0.104", 4000) is True

    def test_probe_litellm_returns_false_when_down(self):
        from plugins.dgx.cli import _probe_litellm
        with patch("plugins.dgx.cli._check_endpoint", return_value=(False, "refused")):
            assert _probe_litellm("192.168.0.104", 4000) is False


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

class TestArgparseWiring:
    def _parser(self):
        from plugins.dgx.cli import register_cli
        p = argparse.ArgumentParser(prog="hermes dgx")
        register_cli(p)
        return p

    def test_no_subcommand_dispatches_to_dgx_command(self):
        from plugins.dgx.cli import dgx_command
        p = self._parser()
        ns = p.parse_args([])
        assert ns.func is dgx_command

    def test_setup_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["setup"])
        assert ns.dgx_command == "setup"

    def test_status_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["status"])
        assert ns.dgx_command == "status"

    def test_models_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["models"])
        assert ns.dgx_command == "models"

    def test_use_subcommand_parses_model_arg(self):
        p = self._parser()
        ns = p.parse_args(["use", "nemotron3:33b"])
        assert ns.dgx_command == "use"
        assert ns.model == "nemotron3:33b"

    def test_use_subcommand_parses_endpoint_flag(self):
        p = self._parser()
        ns = p.parse_args(["use", "qwen2.5-coder:14b", "--endpoint", "vllm"])
        assert ns.endpoint == "vllm"

    def test_endpoint_subcommand_parses_name(self):
        p = self._parser()
        for ep in ("ollama", "vllm", "litellm"):
            ns = p.parse_args(["endpoint", ep])
            assert ns.name == ep

    def test_endpoint_rejects_unknown_name(self):
        p = self._parser()
        with pytest.raises(SystemExit):
            p.parse_args(["endpoint", "bogus"])

    # --- Tier 1: implemented ---

    def test_pull_subcommand_parses_model_arg(self):
        p = self._parser()
        ns = p.parse_args(["pull", "nemotron3:70b"])
        assert ns.dgx_command == "pull"
        assert ns.model == "nemotron3:70b"

    def test_rm_subcommand_parses_model_arg(self):
        p = self._parser()
        ns = p.parse_args(["rm", "old-model:latest"])
        assert ns.dgx_command == "rm"

    def test_rm_force_flag(self):
        p = self._parser()
        ns = p.parse_args(["rm", "old-model:latest", "--force"])
        assert ns.force is True

    def test_ps_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["ps"])
        assert ns.dgx_command == "ps"

    # --- Tier 2: planned subcommands (red) ---

    @pytest.mark.xfail(reason="T2: hermes dgx run not yet implemented", strict=True)
    def test_run_subcommand_parses_cmd_arg(self):
        p = self._parser()
        ns = p.parse_args(["run", "nvidia-smi"])
        assert ns.dgx_command == "run"

    @pytest.mark.xfail(reason="T2: hermes dgx push not yet implemented", strict=True)
    def test_push_subcommand_parses_path_arg(self):
        p = self._parser()
        ns = p.parse_args(["push", "./myproject"])
        assert ns.dgx_command == "push"

    @pytest.mark.xfail(reason="T2: hermes dgx doctor not yet implemented", strict=True)
    def test_doctor_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["doctor"])
        assert ns.dgx_command == "doctor"

    @pytest.mark.xfail(reason="T2: hermes dgx watch not yet implemented", strict=True)
    def test_watch_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["watch"])
        assert ns.dgx_command == "watch"


# ---------------------------------------------------------------------------
# dgx_command dispatch
# ---------------------------------------------------------------------------

class TestDgxCommandDispatch:
    def test_no_subcommand_prints_usage_and_returns_2(self, capsys):
        from plugins.dgx.cli import dgx_command
        ret = dgx_command(SimpleNamespace(dgx_command=None))
        assert ret == 2
        assert "usage" in capsys.readouterr().out.lower()

    def test_unknown_subcommand_returns_2(self, capsys):
        from plugins.dgx.cli import dgx_command
        ret = dgx_command(SimpleNamespace(dgx_command="bogus"))
        assert ret == 2


# ---------------------------------------------------------------------------
# _cmd_status: GPU [N/A] handling
# ---------------------------------------------------------------------------

class TestStatusNvidiaNA:
    """Regression: DGX Spark (aarch64/GB10) returns [N/A] for some smi fields."""

    def test_status_handles_na_memory_fields_without_crash(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_status

        na_line = "[N/A], NVIDIA GH200 120GB, [N/A], 98304, [N/A]"
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, na_line))
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda url, **k: ({"models": [], "data": []}, None))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "unreachable"))

        ret = _cmd_status()
        out = capsys.readouterr().out
        assert "GH200" in out
        assert ret == 0

    def test_status_renders_bar_for_numeric_fields(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_status

        line = "0, NVIDIA A100, 20480, 40960, 50"
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, line))
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda url, **k: ({"models": [], "data": []}, None))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "unreachable"))

        _cmd_status()
        out = capsys.readouterr().out
        assert "█" in out
        assert "20480/40960" in out


# ---------------------------------------------------------------------------
# _cmd_endpoint
# ---------------------------------------------------------------------------

class TestCmdEndpoint:
    def test_switches_to_ollama(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_endpoint
        import plugins.dgx._dgx_config as cfg_mod
        monkeypatch.setattr(cfg_mod, "apply_endpoint", lambda dgx, ep: None)
        ret = _cmd_endpoint("ollama")
        assert ret == 0
        out = capsys.readouterr().out
        assert "ollama" in out.lower()

    def test_switches_to_vllm(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_endpoint
        import plugins.dgx._dgx_config as cfg_mod
        monkeypatch.setattr(cfg_mod, "apply_endpoint", lambda dgx, ep: None)
        ret = _cmd_endpoint("vllm")
        assert ret == 0


# ---------------------------------------------------------------------------
# _cmd_models
# ---------------------------------------------------------------------------

class TestCmdModels:
    def test_prints_ollama_models(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_models
        ollama_payload = {
            "models": [
                {"name": "nemotron3:33b", "size": 20_000_000_000},
                {"name": "qwen2.5-coder:14b", "size": 9_000_000_000},
            ]
        }
        vllm_payload = {"data": []}

        def _fake_get_json(url, **k):
            if "api/tags" in url:
                return ollama_payload, None
            return vllm_payload, None

        monkeypatch.setattr("plugins.dgx.cli._get_json", _fake_get_json)
        ret = _cmd_models()
        out = capsys.readouterr().out
        assert "nemotron3:33b" in out
        assert "qwen2.5-coder:14b" in out
        assert ret == 0

    def test_returns_1_when_all_endpoints_unreachable(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_models
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda *a, **k: (None, "refused"))
        ret = _cmd_models()
        assert ret == 1


# ---------------------------------------------------------------------------
# Tier 1 feature tests (red — implement these next)
# ---------------------------------------------------------------------------

class TestTier1Pull:
    def test_pull_runs_ollama_pull_via_ssh(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_pull
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_stream", lambda u, h, cmd, **k: calls.append(cmd) or 0)
        _cmd_pull("nemotron3:70b")
        assert any("ollama pull nemotron3:70b" in c for c in calls)

    def test_pull_returns_nonzero_on_ssh_failure(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_pull
        monkeypatch.setattr("plugins.dgx.cli._ssh_stream", lambda *a, **k: 1)
        ret = _cmd_pull("some-model:latest")
        assert ret != 0

    def test_pull_prints_host_before_streaming(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_pull
        monkeypatch.setattr("plugins.dgx.cli._ssh_stream", lambda *a, **k: 0)
        _cmd_pull("gemma4:26b")
        out = capsys.readouterr().out
        assert "gemma4:26b" in out


class TestTier1Rm:
    def test_rm_runs_ollama_rm_via_ssh(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_rm
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda u, h, cmd, **k: calls.append(cmd) or (True, ""))
        monkeypatch.setattr("builtins.input", lambda _: "y")
        _cmd_rm("old-model:latest")
        assert any("ollama rm old-model:latest" in c for c in calls)

    def test_rm_aborts_on_no_confirmation(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_rm
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: calls.append(True) or (True, ""))
        monkeypatch.setattr("builtins.input", lambda _: "n")
        _cmd_rm("old-model:latest")
        assert len(calls) == 0

    def test_rm_force_skips_prompt(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_rm
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda u, h, cmd, **k: calls.append(cmd) or (True, ""))
        _cmd_rm("old-model:latest", force=True)
        assert any("ollama rm" in c for c in calls)

    def test_rm_returns_nonzero_on_ssh_failure(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_rm
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "error"))
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert _cmd_rm("bad-model:latest") != 0


class TestTier1Ps:
    def test_ps_shows_loaded_models(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_ps
        ollama_ps_output = "NAME\t\tID\t\tSIZE\tPROCESSOR\nnemotron3:33b\tabc123\t20 GB\tgpu"
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, ollama_ps_output))
        ret = _cmd_ps()
        out = capsys.readouterr().out
        assert "nemotron3:33b" in out
        assert ret == 0

    def test_ps_graceful_when_nothing_loaded(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_ps
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, "NAME\tID\tSIZE\tPROCESSOR"))
        ret = _cmd_ps()
        assert ret == 0

    def test_ps_returns_1_when_ssh_fails(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_ps
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "refused"))
        assert _cmd_ps() == 1


# ---------------------------------------------------------------------------
# Tier 2 feature tests (red)
# ---------------------------------------------------------------------------

class TestTier2Run:
    @pytest.mark.xfail(reason="T2: run not implemented", strict=True)
    def test_run_executes_command_via_ssh(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_run
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda u, h, cmd, **k: (True, "hello from dgx"))
        ret = _cmd_run("echo hello")
        assert "hello from dgx" in capsys.readouterr().out
        assert ret == 0

    @pytest.mark.xfail(reason="T2: run not implemented", strict=True)
    def test_run_returns_nonzero_on_failure(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_run
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "error"))
        assert _cmd_run("bad-cmd") != 0


class TestTier2Push:
    @pytest.mark.xfail(reason="T2: push not implemented", strict=True)
    def test_push_calls_rsync(self, mock_config, monkeypatch, tmp_path):
        from plugins.dgx.cli import _cmd_push
        calls = []
        monkeypatch.setattr("subprocess.run", lambda cmd, **k: calls.append(cmd) or MagicMock(returncode=0))
        (tmp_path / "file.py").write_text("x = 1")
        ret = _cmd_push(str(tmp_path / "file.py"), None)
        assert any("rsync" in str(c) for c in calls)
        assert ret == 0


class TestTier2Doctor:
    @pytest.mark.xfail(reason="T2: doctor not implemented", strict=True)
    def test_doctor_reports_all_checks(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_doctor
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, "ok"))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (True, "ok"))
        ret = _cmd_doctor()
        out = capsys.readouterr().out
        for check in ("ssh", "ollama", "vllm", "gpu"):
            assert check.lower() in out.lower(), f"missing check: {check}"
        assert ret == 0

    @pytest.mark.xfail(reason="T2: doctor not implemented", strict=True)
    def test_doctor_returns_nonzero_when_ssh_unreachable(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_doctor
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "refused"))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "refused"))
        ret = _cmd_doctor()
        assert ret != 0


# ---------------------------------------------------------------------------
# Tier 3 feature tests (red)
# ---------------------------------------------------------------------------

class TestTier3AgentTools:
    @pytest.mark.xfail(reason="T3: agent tools not implemented", strict=True)
    def test_dgx_gpu_status_tool_registered(self):
        """Plugin registration should include dgx_gpu_status as an agent tool."""
        from plugins.dgx import register
        registered_tools = []

        class FakeCtx:
            def register_cli_command(self, **k): pass
            def register_tool(self, name, **k): registered_tools.append(name)

        register(FakeCtx())
        assert "dgx_gpu_status" in registered_tools

    @pytest.mark.xfail(reason="T3: agent tools not implemented", strict=True)
    def test_dgx_run_tool_registered(self):
        from plugins.dgx import register
        registered_tools = []

        class FakeCtx:
            def register_cli_command(self, **k): pass
            def register_tool(self, name, **k): registered_tools.append(name)

        register(FakeCtx())
        assert "dgx_run" in registered_tools

    @pytest.mark.xfail(reason="T3: agent tools not implemented", strict=True)
    def test_dgx_pull_model_tool_registered(self):
        from plugins.dgx import register
        registered_tools = []

        class FakeCtx:
            def register_cli_command(self, **k): pass
            def register_tool(self, name, **k): registered_tools.append(name)

        register(FakeCtx())
        assert "dgx_pull_model" in registered_tools


class TestTier3Formations:
    @pytest.mark.xfail(reason="T3: formations not implemented", strict=True)
    def test_formation_subcommand_parses(self):
        from plugins.dgx.cli import register_cli
        p = argparse.ArgumentParser()
        register_cli(p)
        ns = p.parse_args(["formation", "coding"])
        assert ns.dgx_command == "formation"

    @pytest.mark.xfail(reason="T3: formations not implemented", strict=True)
    def test_formation_switches_model_and_endpoint(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_formation
        monkeypatch.setattr("plugins.dgx.cli.apply_endpoint", lambda *a, **k: None)
        ret = _cmd_formation("coding")
        out = capsys.readouterr().out
        assert ret == 0
        assert "coding" in out.lower()


class TestTier3NIM:
    @pytest.mark.xfail(reason="T3: NIM integration not implemented", strict=True)
    def test_nim_list_subcommand_parses(self):
        from plugins.dgx.cli import register_cli
        p = argparse.ArgumentParser()
        register_cli(p)
        ns = p.parse_args(["nim", "list"])
        assert ns.dgx_command == "nim"

    @pytest.mark.xfail(reason="T3: NIM integration not implemented", strict=True)
    def test_nim_deploy_subcommand_parses_model(self):
        from plugins.dgx.cli import register_cli
        p = argparse.ArgumentParser()
        register_cli(p)
        ns = p.parse_args(["nim", "deploy", "nvidia/nemotron-3-super-120b-a12b"])
        assert hasattr(ns, "model")
