"""Tests for ACP lifecycle, readiness, and routing.

Verifies:
- acp_lifecycle: Readiness checks for Codex/OpenCode ACP servers
- acp_runtime: build_acp_agent_command, run_acp_job
- agent_start integration: Codex/OpenCode ACP routing, effort routing, preflight
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest import mock

import pytest

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────


class FakeSubprocessRunner:
    """Injectable runner for readiness tests."""

    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict[str, Any]] = []

    def run(self, args, *, timeout=None, cwd=None, env=None):
        self.calls.append({"args": list(args), "timeout": timeout, "cwd": cwd, "env": env})
        if not self.results:
            raise AssertionError(f"Unexpected subprocess call: {args}")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture(autouse=True)
def _explicit_model_for_acp_routing_tests(monkeypatch):
    """These routing tests always call the required-model API explicitly."""
    from agent_crossbar import server

    original = server.agent_start

    def with_model(*args, **kwargs):
        profile = kwargs.get("profile") or (args[0] if args else None)
        kwargs.setdefault(
            "model",
            {
                "codex": "gpt-5.6-sol",
                "opencode": "opencode/deepseek-v4-flash-free",
                "reasonix": "deepseek-v4-flash",
            }.get(profile, "sonnet"),
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "agent_start", with_model)


def _fake_job_store(tmp_path):
    """Minimal job store that works for testing."""
    from agent_crossbar.jobs import JobStore

    return JobStore(tmp_path)


# ──────────────────────────────────────────────────────────────────────
#  acp_lifecycle module tests  (READINESS — Codex / OpenCode ACP servers)
# ──────────────────────────────────────────────────────────────────────


class TestCodexAcpReadiness:
    """Readiness checks for the Codex ACP server."""

    def test_module_exists_and_exports_check(self) -> None:
        """acp_lifecycle must export check_codex_acp_readiness."""
        from agent_crossbar import acp_lifecycle

        assert hasattr(acp_lifecycle, "check_codex_acp_readiness")
        assert callable(acp_lifecycle.check_codex_acp_readiness)

    def test_codex_binary_missing_returns_actionable_error(self) -> None:
        """When pnpm binary is missing, return actionable preflight error."""
        from agent_crossbar.acp_lifecycle import check_codex_acp_readiness

        runner = FakeSubprocessRunner([FileNotFoundError("pnpm")])
        result = check_codex_acp_readiness(runner)
        assert result["state"] == "missing_binary"
        assert result["error_code"] is not None
        assert result["remediation"] is not None
        assert result["ready"] is False

    def test_codex_acp_version_too_low_returns_actionable_error(self) -> None:
        """codex-acp < 1.1.7 must return actionable preflight error."""
        from agent_crossbar.acp_lifecycle import check_codex_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "9.0.0\n", ""),  # pnpm --version OK
                _completed(0, "codex 1.3.0\n", ""),  # codex --version OK
                _completed(0, "@agentclientprotocol/codex-acp 1.1.6\n", ""),  # too old
            ]
        )
        result = check_codex_acp_readiness(runner)
        assert result["state"] == "version_too_low", f"Expected version_too_low, got: {result}"
        assert "1.1.7" in result.get("remediation", "")
        assert result["ready"] is False

    def test_codex_acp_ready_when_version_ok(self) -> None:
        """codex-acp >= 1.1.7 must return ready."""
        from agent_crossbar.acp_lifecycle import check_codex_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "9.0.0\n", ""),  # pnpm --version OK
                _completed(0, "codex 1.3.0\n", ""),  # codex --version OK
                _completed(0, "@agentclientprotocol/codex-acp 1.1.7\n", ""),  # OK
            ]
        )
        result = check_codex_acp_readiness(runner)
        assert result["state"] == "ready", f"Expected ready, got: {result}"
        assert result["ready"] is True

    def test_codex_acp_higher_version_ready(self) -> None:
        """codex-acp > 1.1.7 (e.g. 1.2.0) must return ready."""
        from agent_crossbar.acp_lifecycle import check_codex_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "9.0.0\n", ""),  # pnpm --version OK
                _completed(0, "codex 1.3.0\n", ""),  # codex --version OK
                _completed(0, "@agentclientprotocol/codex-acp 1.2.0\n", ""),  # OK
            ]
        )
        result = check_codex_acp_readiness(runner)
        assert result["state"] == "ready", f"Expected ready, got: {result}"
        assert result["ready"] is True

    def test_codex_readiness_probes_codex_acp_not_acpx(self) -> None:
        """Codex readiness must probe @agentclientprotocol/codex-acp@1.1.7 --version, not acpx."""
        from agent_crossbar.acp_lifecycle import check_codex_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "9.0.0\n", ""),  # pnpm --version
                _completed(0, "codex 1.3.0\n", ""),  # codex --version
                _completed(0, "@agentclientprotocol/codex-acp 1.1.7\n", ""),  # codex-acp version
            ]
        )
        check_codex_acp_readiness(runner)
        # Three probes: pnpm --version, codex --version, codex-acp --version
        assert len(runner.calls) == 3
        version_call = runner.calls[2]["args"]
        assert "@agentclientprotocol/codex-acp@1.1.7" in " ".join(version_call), (
            f"Must probe codex-acp pinned package, got: {version_call}"
        )
        assert "--version" in version_call


class TestOpencodeAcpReadiness:
    """Readiness checks for the OpenCode ACP server via native ``opencode acp``."""

    def test_module_exports_opencode_check(self) -> None:
        """acp_lifecycle must export check_opencode_acp_readiness."""
        from agent_crossbar import acp_lifecycle

        assert hasattr(acp_lifecycle, "check_opencode_acp_readiness")
        assert callable(acp_lifecycle.check_opencode_acp_readiness)

    def test_opencode_binary_missing_returns_actionable_error(self) -> None:
        """When opencode binary is missing, return actionable preflight error."""
        from agent_crossbar.acp_lifecycle import check_opencode_acp_readiness

        runner = FakeSubprocessRunner([FileNotFoundError("opencode")])
        result = check_opencode_acp_readiness(runner)
        assert result["state"] == "missing_binary"
        assert result["error_code"] == "opencode_missing"
        assert result["remediation"] is not None
        assert result["ready"] is False

    def test_opencode_acp_ready(self) -> None:
        """When opencode --version and opencode acp --help succeed, return ready."""
        from agent_crossbar.acp_lifecycle import check_opencode_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "opencode 1.0.0\n", ""),  # opencode --version OK
                _completed(0, "Usage: opencode acp ...\n", ""),  # opencode acp --help OK
            ]
        )
        result = check_opencode_acp_readiness(runner)
        assert result["state"] == "ready", f"Expected ready, got: {result}"
        assert result["ready"] is True

    def test_opencode_acp_help_fails_returns_not_ready(self) -> None:
        """When opencode acp --help fails, return unavailable."""
        from agent_crossbar.acp_lifecycle import check_opencode_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "opencode 1.0.0\n", ""),  # opencode --version OK
                _completed(1, "", "acp: unknown subcommand\n"),  # acp --help fails
            ]
        )
        result = check_opencode_acp_readiness(runner)
        assert result["ready"] is False, f"Expected not ready, got: {result}"
        assert result["error_code"] == "opencode_acp_unavailable"
        assert result["state"] == "unavailable"

    def test_opencode_readiness_probes_native_acp_help(self) -> None:
        """OpenCode readiness must probe opencode acp --help, not acpx."""
        from agent_crossbar.acp_lifecycle import check_opencode_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "opencode 1.0.0\n", ""),
                _completed(0, "Usage: opencode acp ...\n", ""),
            ]
        )
        check_opencode_acp_readiness(runner)
        # Two probes: opencode --version, then opencode acp --help
        assert len(runner.calls) == 2, f"Expected 2 calls, got {len(runner.calls)}: {runner.calls}"
        # Probe 1: opencode --version
        assert runner.calls[0]["args"] == ["opencode", "--version"]
        # Probe 2: opencode acp --help
        assert runner.calls[1]["args"] == ["opencode", "acp", "--help"], (
            f"Second probe must be opencode acp --help, got: {runner.calls[1]['args']}"
        )

    def test_opencode_acp_probe_failed_returns_probe_failed(self) -> None:
        """When opencode acp --help raises an exception, return probe_failed."""
        from agent_crossbar.acp_lifecycle import check_opencode_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(0, "opencode 1.0.0\n", ""),  # opencode --version OK
                OSError("permission denied"),  # acp --help crashes
            ]
        )
        result = check_opencode_acp_readiness(runner)
        assert result["ready"] is False
        assert result["error_code"] == "opencode_acp_probe_failed"
        assert result["state"] == "probe_failed"

    def test_opencode_version_fails_returns_broken(self) -> None:
        """When opencode --version fails, return broken before probing acp."""
        from agent_crossbar.acp_lifecycle import check_opencode_acp_readiness

        runner = FakeSubprocessRunner(
            [
                _completed(1, "", "segfault"),  # opencode --version fails
            ]
        )
        result = check_opencode_acp_readiness(runner)
        assert result["ready"] is False
        assert result["error_code"] == "opencode_broken"
        assert result["state"] == "missing_binary"


# ──────────────────────────────────────────────────────────────────────
#  ACP agent command builder tests  (build_acp_agent_command)
# ──────────────────────────────────────────────────────────────────────


class TestAcpAgentCommand:
    """build_acp_agent_command produces correct native ACP argv for each provider."""

    def test_module_exports_build_acp_agent_command(self) -> None:
        """acp_runtime must export build_acp_agent_command."""
        from agent_crossbar import acp_runtime

        assert hasattr(acp_runtime, "build_acp_agent_command")
        assert callable(acp_runtime.build_acp_agent_command)

    def test_opencode_returns_native_acp_command(self) -> None:
        """build_acp_agent_command('opencode') returns ['opencode', 'acp']."""
        from agent_crossbar.acp_runtime import build_acp_agent_command

        cmd = build_acp_agent_command("opencode")
        assert cmd == ["opencode", "acp"], f"Expected ['opencode', 'acp'], got: {cmd}"

    def test_codex_returns_pnpm_dlx_codex_acp_command(self) -> None:
        """build_acp_agent_command('codex') returns pnpm dlx for codex-acp package."""
        from agent_crossbar.acp_runtime import build_acp_agent_command

        cmd = build_acp_agent_command("codex")
        assert cmd == ["pnpm", "dlx", "@agentclientprotocol/codex-acp@1.1.7"], (
            f"Expected pnpm dlx command with pinned version, got: {cmd}"
        )

    def test_unknown_profile_raises_value_error(self) -> None:
        """build_acp_agent_command must raise ValueError for unknown profiles."""
        from agent_crossbar.acp_runtime import build_acp_agent_command

        with pytest.raises(ValueError, match="Unknown ACP provider"):
            build_acp_agent_command("claude")


# ──────────────────────────────────────────────────────────────────────
#  ACP integration — agent_start preflight & routing tests
# ──────────────────────────────────────────────────────────────────────


class TestAgentStartAcpPreflight:
    """agent_start must run ACP readiness preflight before creating jobs."""

    def test_codex_acp_preflight_failure_returns_error_no_job(self, tmp_path, monkeypatch) -> None:
        """When Codex ACP preflight fails, agent_start returns error with no job created."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        # Mock the readiness probe so we reach the ACP preflight gate
        import agent_crossbar.readiness as rmod
        from agent_crossbar import server

        def fake_probe(profile, _runner=None, use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return ReadinessResult(
                profile=profile,
                state="ready",
                support_tier="supported",
                authenticated=True,
                probe_version=1,
                timestamp=time.time(),
            )

        monkeypatch.setattr(rmod, "probe_profile", fake_probe)

        # Pass a failing preflight runner
        with mock.patch(
            "agent_crossbar.acp_lifecycle.check_codex_acp_readiness",
            return_value={
                "ready": False,
                "error_code": "pnpm_missing",
                "remediation": "Install pnpm",
            },
        ):
            result = server.agent_start(
                profile="codex",
                prompt="test preflight",
                task="dev",
                cwd=str(tmp_path),
            )
            assert result["ok"] is False
            assert result["error"] == "pnpm_missing"
            assert result["job_created"] is False
            assert "Install pnpm" in result.get("message", "")

    def test_opencode_acp_preflight_failure_returns_error_no_job(
        self, tmp_path, monkeypatch
    ) -> None:
        """When OpenCode ACP preflight fails, agent_start returns error with no job created."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        # Mock the readiness probe so we reach the ACP preflight gate
        import agent_crossbar.readiness as rmod
        from agent_crossbar import server

        def fake_probe(profile, _runner=None, use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return ReadinessResult(
                profile=profile,
                state="ready",
                support_tier="supported",
                authenticated=True,
                probe_version=1,
                timestamp=time.time(),
            )

        monkeypatch.setattr(rmod, "probe_profile", fake_probe)

        # Mock discovery so "opencode/test-model" resolves as a valid model —
        # model validation now runs before the ACP preflight gate this test targets.
        from agent_crossbar.adapters.base import ModelCatalog, ModelInfo

        mock_catalog = ModelCatalog(
            models=("opencode/test-model",),
            default_model="opencode/test-model",
            native_efforts=("high", "max"),
            source="mock",
            model_info=(
                ModelInfo(
                    id="opencode/test-model",
                    supported_efforts=("high", "max"),
                    default_effort="high",
                ),
            ),
        )
        monkeypatch.setattr(
            "agent_crossbar.discovery.discover_profile_models",
            lambda sr, profile, refresh=False: mock_catalog,
        )

        with mock.patch(
            "agent_crossbar.acp_lifecycle.check_opencode_acp_readiness",
            return_value={
                "ready": False,
                "error_code": "opencode_acp_probe_failed",
                "remediation": "opencode ACP check failed",
            },
        ):
            result = server.agent_start(
                profile="opencode",
                prompt="test preflight",
                task="dev",
                cwd=str(tmp_path),
                model="opencode/test-model",
            )
            assert result["ok"] is False
            assert result["error"] == "opencode_acp_probe_failed"
            assert result["job_created"] is False

    def test_interactive_codex_acp_returns_not_ready(self, tmp_path, monkeypatch) -> None:
        """Interactive Codex via ACP must return interactive_not_ready."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="codex",
            prompt="interactive test",
            task="dev",
            interactive=True,
            cwd=str(tmp_path),
        )
        assert result["ok"] is False
        assert result["error"] == "interactive_not_supported"
        assert result["job_created"] is False

    def test_interactive_opencode_acp_returns_not_ready(self, tmp_path, monkeypatch) -> None:
        """Interactive OpenCode via ACP must return interactive_not_ready."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="opencode",
            prompt="interactive test",
            task="dev",
            interactive=True,
            cwd=str(tmp_path),
        )
        assert result["ok"] is False
        assert result["error"] == "interactive_not_supported"
        assert result["job_created"] is False


# ──────────────────────────────────────────────────────────────────────
#  agent_start ACP routing tests — effort routing
# ──────────────────────────────────────────────────────────────────────


class TestAgentStartAcpRouting:
    """agent_start must route Codex/OpenCode: omitted effort → ACP, explicit effort → legacy print."""

    @pytest.fixture(autouse=True)
    def _mock_acp_runtime(self, monkeypatch):
        """Mock ACP runtime and legacy runners to track calls."""
        self.acp_calls: list[dict[str, Any]] = []
        self.print_job_calls: list[dict[str, Any]] = []
        self.tmux_job_calls: list[dict[str, Any]] = []

        # Mock readiness probe — our probes are read-only and require
        # locally cached ACP bridges; this test verifies routing, not probing.
        import agent_crossbar.readiness as rmod

        def fake_probe(profile, _runner=None, use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return ReadinessResult(
                profile=profile,
                state="ready",
                support_tier="supported",
                authenticated=True,
                probe_version=1,
                timestamp=time.time(),
            )

        monkeypatch.setattr(rmod, "probe_profile", fake_probe)

        async def fake_run_acp_job(
            store,
            job_id,
            *,
            provider,
            model,
            effort,
            prompt,
            cwd,
            task,
            autonomy,
            max_runtime_sec,
            **kwargs,
        ):
            self.acp_calls.append(
                {
                    "provider": provider,
                    "model": model,
                    "effort": effort,
                    "prompt": prompt,
                    "cwd": cwd,
                    "task": task,
                    "autonomy": autonomy,
                    "max_runtime_sec": max_runtime_sec,
                }
            )
            store.set_result(job_id, ok=True, summary="ACP_OK\n")

        monkeypatch.setattr(
            "agent_crossbar.server._run_acp_job",
            fake_run_acp_job,
        )

        def fake_start_print_job(store, job_id, req, **kw):
            self.print_job_calls.append({"req": dict(req), "kw": dict(kw)})
            store.set_result(job_id, True, summary="PRINT_OK\n")

        monkeypatch.setattr(
            "agent_crossbar.server.start_print_job",
            fake_start_print_job,
        )

        def fake_start_tmux_job(store, job_id, req, **kw):
            self.tmux_job_calls.append({"req": dict(req), "kw": dict(kw)})
            store.set_result(job_id, True, summary="TMUX_OK\n")

        monkeypatch.setattr(
            "agent_crossbar.server.start_tmux_job",
            fake_start_tmux_job,
        )

    def test_codex_omitted_effort_uses_acp(self, tmp_path, monkeypatch):
        """Codex without explicit effort must use ACP backend."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="codex",
            prompt="test task",
            task="dev",
            model="gpt-5.6-sol",
            # NO effort → ACP
            cwd=str(tmp_path),
        )
        assert result["ok"] is True, f"agent_start failed: {result}"
        assert len(self.acp_calls) == 1, (
            f"Expected 1 ACP call, got {len(self.acp_calls)}. "
            f"print={len(self.print_job_calls)}, tmux={len(self.tmux_job_calls)}"
        )
        assert len(self.print_job_calls) == 0, "start_print_job must NOT be called for ACP path"

    def test_codex_explicit_effort_uses_acp(self, tmp_path, monkeypatch):
        """Codex configures explicit effort through its ACP session selector."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="codex",
            prompt="test task",
            task="dev",
            model="gpt-5.6-sol",
            effort="high",
            cwd=str(tmp_path),
        )
        assert result["ok"] is True, f"agent_start failed: {result}"
        assert result["backend"] == "acp"
        assert len(self.acp_calls) == 1
        assert self.acp_calls[0]["effort"] == "high"
        assert len(self.print_job_calls) == 0

    def test_codex_explicit_effort_does_not_produce_print_fallback_warning(
        self, tmp_path, monkeypatch
    ):
        """Codex ACP effort must not be reported as a print fallback."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="codex",
            prompt="test task",
            task="dev",
            model="gpt-5.6-sol",
            effort="high",
            cwd=str(tmp_path),
        )
        assert result["ok"] is True, f"agent_start failed: {result}"
        assert result["backend"] == "acp"
        warnings = result.get("warnings", [])
        codes = {w.get("code") for w in warnings if isinstance(w, dict)}
        assert "effort_forced_print_fallback" not in codes

    def test_codex_explicit_effort_falls_back_when_acp_version_is_too_old(
        self, tmp_path, monkeypatch
    ):
        """Older codex-acp keeps the established print path for explicit effort."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        monkeypatch.setattr(
            "agent_crossbar.acp_lifecycle.check_codex_acp_readiness",
            lambda _runner: {
                "ready": False,
                "state": "version_too_low",
                "error_code": "codex_acp_version_too_low",
            },
        )

        result = server.agent_start(
            profile="codex",
            prompt="test task",
            task="dev",
            model="gpt-5.6-sol",
            effort="high",
            cwd=str(tmp_path),
        )
        assert result["ok"] is True, result
        assert result["backend"] == "print"
        assert len(self.acp_calls) == 0
        assert len(self.print_job_calls) == 1
        assert result["warnings"][0]["code"] == "effort_forced_print_fallback"

    def test_opencode_omitted_effort_uses_acp(self, tmp_path, monkeypatch):
        """OpenCode without explicit effort must use ACP backend."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        # Mock discovery for determinism
        from agent_crossbar.adapters.base import ModelCatalog, ModelInfo

        mock_catalog = ModelCatalog(
            models=("opencode/test-model",),
            default_model="opencode/test-model",
            native_efforts=("high", "max"),
            source="mock",
            model_info=(
                ModelInfo(
                    id="opencode/test-model",
                    supported_efforts=("high", "max"),
                    default_effort="high",
                ),
            ),
        )
        monkeypatch.setattr(
            "agent_crossbar.discovery.discover_profile_models",
            lambda sr, profile, refresh=False: mock_catalog,
        )

        from agent_crossbar import server

        result = server.agent_start(
            profile="opencode",
            prompt="test task",
            model="opencode/test-model",
            task="dev",
            # NO effort → ACP
            cwd=str(tmp_path),
        )
        assert result["ok"] is True, f"agent_start failed: {result}"
        assert len(self.acp_calls) == 1
        assert len(self.print_job_calls) == 0

    def test_opencode_explicit_effort_uses_acp(self, tmp_path, monkeypatch):
        """OpenCode configures explicit effort through its ACP session selector."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        # Mock discovery to return a catalog where the model supports high effort
        from agent_crossbar.adapters.base import ModelCatalog, ModelInfo

        mock_catalog = ModelCatalog(
            models=("opencode/test-model",),
            default_model="opencode/test-model",
            native_efforts=("high", "max"),
            source="mock",
            model_info=(
                ModelInfo(
                    id="opencode/test-model",
                    supported_efforts=("high", "max"),
                    default_effort="high",
                ),
            ),
        )
        monkeypatch.setattr(
            "agent_crossbar.discovery.discover_profile_models",
            lambda sr, profile, refresh=False: mock_catalog,
        )

        from agent_crossbar import server

        result = server.agent_start(
            profile="opencode",
            prompt="test task",
            task="dev",
            model="opencode/test-model",
            effort="high",
            cwd=str(tmp_path),
        )
        assert result["ok"] is True, f"agent_start failed: {result}"
        assert result["backend"] == "acp"
        assert len(self.acp_calls) == 1
        assert self.acp_calls[0]["effort"] == "high"
        assert len(self.print_job_calls) == 0

    def test_opencode_unsupported_effort_rejected(self, tmp_path, monkeypatch):
        """OpenCode with explicit effort NOT supported by model must be rejected before job creation."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        # Mock discovery with a model that has NO effort variants
        from agent_crossbar.adapters.base import ModelCatalog, ModelInfo

        mock_catalog = ModelCatalog(
            models=("opencode/no-variants-model",),
            default_model="opencode/no-variants-model",
            native_efforts=(),
            source="mock",
            model_info=(
                ModelInfo(
                    id="opencode/no-variants-model",
                    supported_efforts=(),
                    default_effort=None,
                ),
            ),
        )
        monkeypatch.setattr(
            "agent_crossbar.discovery.discover_profile_models",
            lambda sr, profile, refresh=False: mock_catalog,
        )

        from agent_crossbar import server

        result = server.agent_start(
            profile="opencode",
            prompt="test task",
            task="dev",
            model="opencode/no-variants-model",
            effort="high",  # not supported
            cwd=str(tmp_path),
        )
        assert result["ok"] is False, "Unsupported effort must be rejected"
        assert result["error"] == "unsupported_effort_for_model"
        assert result["job_created"] is False
        assert len(self.acp_calls) == 0
        assert len(self.print_job_calls) == 0

    def test_invalid_effort_rejected_before_job_creation(self, tmp_path, monkeypatch):
        """Invalid effort must be rejected by validation before any job is created."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="codex",
            prompt="test task",
            task="dev",
            effort="INSANE",  # invalid effort
            cwd=str(tmp_path),
        )
        assert result["ok"] is False, "Invalid effort must be rejected"
        assert result["job_created"] is False
        assert len(self.acp_calls) == 0
        assert len(self.print_job_calls) == 0

    def test_agent_start_reasonix_still_uses_print_legacy(self, tmp_path, monkeypatch):
        """Non-ACP profiles (Reasonix) must still use legacy start_print_job."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="reasonix",
            prompt="test task",
            task="ask",
            cwd=str(tmp_path),
        )
        assert result["ok"] is True
        assert len(self.acp_calls) == 0, "Reasonix must NOT use ACP"
        assert len(self.print_job_calls) == 1, "Reasonix must use legacy print"

    def test_agent_start_max_runtime_sec_passed_to_acp(self, tmp_path, monkeypatch):
        """max_runtime_sec must be passed through to ACP runtime."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="codex",
            prompt="test task",
            task="dev",
            cwd=str(tmp_path),
            max_runtime_sec=300,
        )
        assert result["ok"] is True
        assert len(self.acp_calls) == 1
        assert self.acp_calls[0]["max_runtime_sec"] == 300

    def test_interactive_acp_returns_not_ready_no_state_mutation(self, tmp_path, monkeypatch):
        """Interactive ACP must return interactive_not_ready and create NO job."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        from agent_crossbar import server

        result = server.agent_start(
            profile="codex",
            prompt="interactive test",
            task="dev",
            interactive=True,
            cwd=str(tmp_path),
        )
        assert result["ok"] is False
        assert result["error"] == "interactive_not_supported"
        assert result["job_created"] is False
        assert len(self.acp_calls) == 0
        assert len(self.print_job_calls) == 0
        assert len(self.tmux_job_calls) == 0


# ══════════════════════════════════════════════════════════════════════
#  JobStore: set_result guard + get_result stopped
# ══════════════════════════════════════════════════════════════════════


class TestSetResultStoppedGuard:
    """set_result must refuse to overwrite stopped (and other terminal) status."""

    def test_set_result_refuses_after_stop_job(self, tmp_path):
        """After stop_job sets status=stopped, set_result must return error."""
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="codex",
            operation="review",
            transport="print",
            sensitivity="normal",
        )

        # Mark stopped
        stop_result = store.stop_job(job.job_id, reason="user_cancelled")
        assert stop_result["ok"] is True

        meta = store._read_job_meta(job.path)
        assert meta["status"] == "stopped"

        # Background completion tries to write succeeded
        result = store.set_result(job.job_id, ok=True, summary="completed after stop")
        assert result["ok"] is False, f"Must refuse: {result}"
        assert result["error"] == "job_already_terminal"
        assert result["current_status"] == "stopped"

        # Status must still be stopped (not overwritten)
        meta = store._read_job_meta(job.path)
        assert meta["status"] == "stopped"

    def test_set_result_works_normally_for_running_jobs(self, tmp_path):
        """set_result must still work for running jobs (no regression)."""
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="codex",
            operation="review",
            transport="print",
            sensitivity="normal",
        )

        result = store.set_result(job.job_id, ok=True, summary="done")
        assert result["ok"] is True

        meta = store._read_job_meta(job.path)
        assert meta["status"] == "succeeded"

    def test_set_result_refuses_on_already_failed_status(self, tmp_path):
        """After set_result wrote failed, second set_result must be refused."""
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="codex",
            operation="review",
            transport="print",
            sensitivity="normal",
        )

        store.set_result(job.job_id, ok=False, summary="first failure")
        result = store.set_result(job.job_id, ok=True, summary="late success")
        assert result["ok"] is False
        assert result["error"] == "job_already_terminal"
        assert result["current_status"] == "failed"


class TestGetResultStopped:
    """get_result for stopped jobs must return a stable response, not result_not_ready."""

    def test_stopped_job_returns_stable_result(self, tmp_path):
        """After stop_job, get_result returns stopped response, not result_not_ready."""
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="codex",
            operation="review",
            transport="print",
            sensitivity="normal",
        )

        store.stop_job(job.job_id, reason="user_cancelled")

        result = store.get_result(job.job_id)
        assert result["ok"] is True, f"Stopped job must return ok=True: {result}"
        assert result["status"] == "stopped"
        assert result["stop_reason"] == "user_cancelled"
        assert "stopped" in result.get("summary", "").lower()

    def test_stopped_job_result_is_idempotent(self, tmp_path):
        """get_result for stopped job must return same result repeatedly."""
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="codex",
            operation="review",
            transport="print",
            sensitivity="normal",
        )

        store.stop_job(job.job_id, reason="user_cancelled")

        r1 = store.get_result(job.job_id)
        r2 = store.get_result(job.job_id)
        assert r1 == r2, f"Stopped result must be idempotent: {r1} vs {r2}"

    def test_running_job_still_returns_result_not_ready(self, tmp_path):
        """Running job without result.json still returns result_not_ready."""
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="codex",
            operation="review",
            transport="print",
            sensitivity="normal",
        )

        result = store.get_result(job.job_id)
        assert result["ok"] is False
        assert result["error"] == "result_not_ready"


class TestDeterministicRace:
    """Deterministic race regression: background set_result cannot resurrect stopped."""

    def test_background_completion_cannot_resurrect_stopped_job(self, tmp_path):
        """Simulate the exact race: stop marks stopped, then late completion calls set_result."""
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="codex",
            operation="review",
            transport="print",
            sensitivity="normal",
        )

        # Simulate: user stops the job
        store.stop_job(job.job_id, reason="user_cancelled")

        # Simulate: background ACP completion arrives late
        late = store.set_result(
            job.job_id,
            ok=True,
            summary="Task completed successfully!",
            envelope={
                "schema_version": "1",
                "status": "completed",
                "stop_reason": "end_turn",
                "output": "All done",
            },
        )

        assert late["ok"] is False
        assert late["error"] == "job_already_terminal"

        # Verify status is still stopped
        meta = store._read_job_meta(job.path)
        assert meta["status"] == "stopped"

        # Verify get_result returns stopped, not the late completion data
        result = store.get_result(job.job_id)
        assert result["status"] == "stopped"
        assert result["ok"] is True

        # Verify no result.json was written by the late set_result
        result_path = job.path / "result.json"
        assert not result_path.exists(), "Late set_result must not write result.json on stopped job"
