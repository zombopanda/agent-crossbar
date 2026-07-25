"""Integration tests for agent_start(profile="claude") using adapter registry.

These prove the public agent_start tool dispatches to the ClaudeAdapter
and native claude_bg backend, not legacy providers.py/tmux/print paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_crossbar.adapters.claude import (
    RunResult,
)
from agent_crossbar.jobs import JobStore
from agent_crossbar.server import agent_start as _agent_start
from agent_crossbar.server import job_send, job_stop


def agent_start(*args, **kwargs):
    """Claude integration tests always exercise an explicit Sonnet selection."""
    kwargs.setdefault("model", "claude-sonnet-5")
    return _agent_start(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRunner:
    """Records calls and returns pre-configured results in order."""

    def __init__(self, results: list[RunResult] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict[str, object]] = []

    def run(self, args, *, timeout=None, cwd=None, env=None):
        self.calls.append({"args": list(args), "timeout": timeout, "cwd": cwd, "env": env})
        if not self.results:
            raise AssertionError(f"Unexpected subprocess call: {args}")
        return self.results.pop(0)


def _auth_ok() -> RunResult:
    return RunResult(
        0,
        json.dumps(
            {
                "loggedIn": True,
                "authMethod": "claude.ai",
                "apiProvider": "firstParty",
                "subscriptionType": "max",
            }
        ),
        "",
    )


def _launch_ok(session_id: str = "e2accc98") -> RunResult:
    return RunResult(
        0,
        f"backgrounded · {session_id} · relay-test\n  claude attach {session_id}\n",
        "",
    )


def _agents_entry(state: str, session_id: str = "e2accc98") -> RunResult:
    return RunResult(
        0,
        json.dumps(
            [
                {
                    "id": session_id,
                    "sessionId": f"{session_id}-9fd6-4813-a6e6-3cb0d134de46",
                    "state": state,
                    "status": "idle" if state == "done" else "running",
                    "waitingFor": "permission prompt" if state == "blocked" else None,
                    "cwd": "/repo",
                    "startedAt": 1784739414928,
                }
            ]
        ),
        "",
    )


def _logs_ok(output: str = "final output") -> RunResult:
    return RunResult(0, output, "")


def _cancel_ok() -> RunResult:
    return RunResult(0, "stopped", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def claude_state_root(tmp_path: Path, monkeypatch) -> Path:
    """Redirect state and isolate the generic cached readiness preflight.

    These tests exercise the Claude adapter's own auth probe with ``FakeRunner``.
    They must not also depend on the developer machine's live Claude login.
    """
    state_root = tmp_path / "agent-harness-state"
    state_root.mkdir()
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(state_root))
    monkeypatch.setattr(
        "agent_crossbar.readiness.probe_profile",
        lambda *args, **kwargs: SimpleNamespace(
            state="ready",
            error_code=None,
            remediation=None,
        ),
    )
    return state_root


@pytest.fixture
def suppress_background_monitor(monkeypatch):
    """Prevent agent_start from launching a background monitor thread.

    Tests call monitor_agent_job manually to control timing.
    """

    def _noop(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "agent_crossbar.server.start_agent_job",
        _noop,
    )
    monkeypatch.setattr(
        "agent_crossbar.agent_runner.start_agent_job",
        _noop,
    )


# ---------------------------------------------------------------------------
# Test 1: agent_start(profile="claude") uses adapter registry, not legacy
# ---------------------------------------------------------------------------


def test_agent_start_claude_uses_adapter_not_legacy_providers(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Prove agent_start dispatches to the adapter and native claude_bg backend."""
    runner = FakeRunner([_auth_ok(), _launch_ok("abc12345")])

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    # Prevent any legacy runner from launching real processes
    monkeypatch.setattr(
        "agent_crossbar.server.start_print_job",
        lambda *a, **kw: pytest.fail("legacy start_print_job called"),
    )
    monkeypatch.setattr(
        "agent_crossbar.server.start_tmux_job",
        lambda *a, **kw: pytest.fail("legacy start_tmux_job called"),
    )
    monkeypatch.setattr(
        "agent_crossbar.server.start_gui_job",
        lambda *a, **kw: pytest.fail("legacy start_gui_job called"),
    )

    result = agent_start(
        profile="claude",
        prompt="hello world",
        task="ask",
        interactive=False,
        client_name="test",
    )

    assert result.get("ok") is True, result
    assert result.get("job_id"), "agent_start must return a durable job_id"
    assert result.get("profile") == "claude"
    # Must NOT use legacy transport annotation
    assert result.get("transport") != "tmux", "must not use legacy tmux backend"
    assert result.get("backend") == "claude_bg", "must use native claude_bg backend"

    # Verify the adapter was called for readiness and launch
    assert len(runner.calls) >= 2
    readiness_call = runner.calls[0]["args"]
    launch_call = runner.calls[1]["args"]
    assert readiness_call[0:3] == ["claude", "auth", "status"]
    assert launch_call[0:2] == ["claude", "--bg"]
    assert "-p" not in launch_call
    assert "bypassPermissions" not in launch_call
    assert launch_call[launch_call.index("--model") + 1] == "claude-sonnet-5"

    # Verify job directory was created with proper metadata
    store = JobStore(claude_state_root)
    job = store.get_job(result["job_id"])
    assert job is not None
    meta = store._read_job_meta(job.path)
    assert meta.get("backend") == "claude_bg"
    assert meta.get("native_session_id") == "abc12345"


# ---------------------------------------------------------------------------
# Test 2: readiness failure creates no job directory
# ---------------------------------------------------------------------------


def test_readiness_failure_creates_no_job_directory(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Prove that auth/readiness failure returns error without creating a job dir."""
    runner = FakeRunner(
        [
            RunResult(1, "", "not authenticated"),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    result = agent_start(
        profile="claude",
        prompt="hello",
        task="ask",
        interactive=False,
        client_name="test",
    )

    assert result.get("ok") is False, result
    assert result.get("error") is not None

    # No job directory should exist
    jobs_dir = claude_state_root / "jobs"
    if jobs_dir.exists():
        entries = list(jobs_dir.iterdir())
        assert len(entries) == 0, f"Expected no job dirs, found {entries}"


# ---------------------------------------------------------------------------
# Test 3: launch stores native session metadata
# ---------------------------------------------------------------------------


def test_launch_stores_native_session_ids_and_resolved_settings(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Prove the job meta stores all resolved adapter fields."""
    runner = FakeRunner([_auth_ok(), _launch_ok("deadbeef")])

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    result = agent_start(
        profile="claude",
        prompt="implement foo",
        task="dev",
        model="claude-sonnet-5",
        effort="high",
        cwd="/tmp/test-repo",
        interactive=False,
        client_name="test",
    )

    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    job = store.get_job(job_id)
    assert job is not None
    meta = store._read_job_meta(job.path)

    assert meta.get("native_session_id") == "deadbeef"
    assert meta.get("backend") == "claude_bg"
    assert meta.get("model") == "claude-sonnet-5"
    assert meta.get("effort") == "high"
    assert meta.get("task") == "dev"
    assert meta.get("interactive") is False
    assert meta.get("cwd") == "/tmp/test-repo"


# ---------------------------------------------------------------------------
# Test 4: monitor finalizes terminal states via adapter
# ---------------------------------------------------------------------------


def test_monitor_polls_status_until_done_and_normalizes_result(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Prove the background monitor calls adapter.status/get_logs and finalizes."""
    # Simulate: working → working → done
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b2c3d4"),
            # status poll 1: working
            _agents_entry("working", "a1b2c3d4"),
            _logs_ok(""),
            # status poll 2: working
            _agents_entry("working", "a1b2c3d4"),
            _logs_ok(""),
            # status poll 3: done
            _agents_entry("done", "a1b2c3d4"),
            # get_logs
            _logs_ok("success output"),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )

    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")

    # Run monitor manually (normally this runs in a background thread)
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    # Job should be finalized
    final = store.get_result(job_id)
    assert final.get("ok") is True, final
    assert final.get("summary") == "success output"
    meta = store._read_job_meta(store.get_job(job_id).path)
    assert meta["native_full_session_id"] == "a1b2c3d4-9fd6-4813-a6e6-3cb0d134de46"


def test_monitor_detects_blocked_as_failure_for_non_interactive_jobs(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Blocked non-interactive jobs produce structured failure, not generic timeout."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("deadb33f"),
            # status: blocked
            _agents_entry("blocked", "deadb33f"),
            # get_logs
            _logs_ok("needs approval"),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )

    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")

    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    # Non-interactive blocked → failure
    assert final.get("ok") is False, (
        f"Expected failure for blocked non-interactive job, got {final}"
    )
    assert final.get("summary") == "needs approval"


# ---------------------------------------------------------------------------
# Test 5: job_stop wires to adapter.cancel
# ---------------------------------------------------------------------------


def test_job_stop_calls_adapter_cancel_for_claude_bg_jobs(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """job_stop must call adapter.cancel for adapter-based Claude jobs."""
    # Auth, launch, then cancel
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("cafebabe"),
            # cancel call
            _cancel_ok(),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    result = agent_start(
        profile="claude",
        prompt="long task",
        task="ask",
        interactive=False,
        client_name="test",
    )

    assert result.get("ok") is True, result
    job_id = result["job_id"]

    # Call job_stop — it should call adapter.cancel
    stop_result = job_stop(
        job_id=job_id,
        reason="user_cancelled",
        client_name="test",
    )

    assert stop_result.get("ok") is True, stop_result

    # Verify cancel was called (runner.calls[2] should be claude stop)
    cancel_calls = [c for c in runner.calls if c["args"][:2] == ["claude", "stop"]]
    assert len(cancel_calls) >= 1, f"Expected claude stop call, got {runner.calls}"


# ---------------------------------------------------------------------------
# Test 6: interactive rejected before job creation
# ---------------------------------------------------------------------------


def test_agent_start_accepts_interactive_claude_and_launches_tmux_attach(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Prove interactive=true Claude launch succeeds and stores tmux metadata."""
    runner = FakeRunner([_auth_ok(), _launch_ok("cafebabe")])
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    # Mock tmux calls to avoid real tmux dependency
    mock_tmux_result = type("Result", (), {"returncode": 0, "stderr": ""})()

    def fake_tmux(*args, **kwargs):
        return mock_tmux_result

    monkeypatch.setattr("subprocess.run", fake_tmux)

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=True,
        client_name="test",
    )

    assert result.get("ok") is True, result
    assert result.get("backend") == "claude_bg_pty"

    # Verify job meta stores tmux session info
    store = JobStore(claude_state_root)
    job = store.get_job(result["job_id"])
    assert job is not None
    meta = store._read_job_meta(job.path)
    assert meta.get("interactive") is True
    assert meta.get("tmux_session") is not None
    assert "agents-" in meta["tmux_session"]
    assert meta.get("transport") == "tmux"
    assert meta.get("native_session_id") == "cafebabe"


def test_interactive_claude_job_send_routes_to_tmux(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """job_send for interactive Claude routes to tmux send-keys."""
    runner = FakeRunner([_auth_ok(), _launch_ok("feedface")])
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    mock_tmux_result = type("Result", (), {"returncode": 0, "stderr": ""})()

    def fake_tmux(*args, **kwargs):
        return mock_tmux_result

    monkeypatch.setattr("subprocess.run", fake_tmux)

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=True,
        client_name="test",
    )

    assert result.get("ok") is True
    job_id = result["job_id"]

    # Now send input — should succeed via tmux send-keys
    send_result = job_send(
        job_id=job_id,
        text="more input please",
        client_name="test",
    )

    assert send_result.get("ok") is True, (
        f"Expected job_send to succeed for interactive Claude, got {send_result}"
    )


def test_noninteractive_claude_job_send_still_rejected(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Non-interactive Claude bg jobs must still reject job_send."""
    runner = FakeRunner([_auth_ok(), _launch_ok("feedface")])
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )

    assert result.get("ok") is True
    job_id = result["job_id"]

    # job_send should reject non-interactive Claude bg
    send_result = job_send(
        job_id=job_id,
        text="more input",
        client_name="test",
    )

    assert send_result.get("ok") is False, (
        f"Expected job_send to fail for non-interactive Claude, got {send_result}"
    )


def test_interactive_claude_launch_tmux_failure_propagates(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """When tmux session creation fails, the error propagates to agent_start."""
    runner = FakeRunner([_auth_ok(), _launch_ok("deadbeef"), _cancel_ok()])
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    # Mock tmux failure
    mock_fail = type("Result", (), {"returncode": 1, "stderr": "tmux: command not found"})()
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_fail)

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=True,
        client_name="test",
    )

    assert result.get("ok") is False, result
    assert result.get("error") == "tmux_launch_failed"
    assert any(call["args"][:2] == ["claude", "stop"] for call in runner.calls)


def test_interactive_claude_job_stop_kills_tmux_and_native_session(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """job_stop for interactive Claude kills tmux session and calls claude stop."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("cafebabe"),
            _cancel_ok(),  # claude stop call
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    tmux_calls = []

    def fake_tmux(args, **kwargs):
        tmux_calls.append(args)
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("subprocess.run", fake_tmux)

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=True,
        client_name="test",
    )

    assert result.get("ok") is True
    job_id = result["job_id"]

    # Call job_stop
    stop_result = job_stop(
        job_id=job_id,
        reason="user_cancelled",
        client_name="test",
    )

    assert stop_result.get("ok") is True, stop_result

    # Verify tmux kill-session was called
    kill_calls = [c for c in tmux_calls if c[:2] == ["tmux", "kill-session"]]
    assert len(kill_calls) >= 1, f"Expected tmux kill-session, got {tmux_calls}"

    # Verify claude stop was called
    cancel_calls = [c for c in runner.calls if c["args"][:2] == ["claude", "stop"]]
    assert len(cancel_calls) >= 1, f"Expected claude stop call, got {runner.calls}"


def test_foreign_session_cannot_cancel_interactive_claude_job(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Ownership is checked before Claude or tmux receive any stop command."""
    runner = FakeRunner([_auth_ok(), _launch_ok("cafebabe")])
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    tmux_calls = []

    def fake_tmux(args, **kwargs):
        tmux_calls.append(args)
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("subprocess.run", fake_tmux)
    started = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=True,
        client_session_id="owner-session",
    )

    stopped = job_stop(
        job_id=started["job_id"],
        reason="foreign_attempt",
        client_session_id="other-session",
    )

    assert stopped["error"] == "job_not_found"
    assert not [call for call in runner.calls if call["args"][:2] == ["claude", "stop"]]
    assert not [call for call in tmux_calls if call[:2] == ["tmux", "kill-session"]]


# ---------------------------------------------------------------------------
# Test 7: monitor result envelope — state mapping
# ---------------------------------------------------------------------------


def test_monitor_maps_done_to_completed_envelope(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Monitor must produce envelope with status=completed for done sessions."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b2c3d4"),
            _agents_entry("done", "a1b2c3d4"),
            _logs_ok("success output"),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["ok"] is True
    assert final["schema_version"] == "1"
    assert final["status"] == "completed"
    assert final["stop_reason"] == "done"
    assert final["failure"] is None


def test_monitor_maps_done_without_recoverable_output_to_failed_envelope(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Native done must not become success when Claude logs contain only TUI chrome."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("bad0feed"),
            _agents_entry("done", "bad0feed"),
            _logs_ok(
                """[Screen Reader Mode: on via flag]
$claude: Summary
don't ask on (shift+tab to cycle) · esc to interrupt
0 tokens
$Worked for 2s
"""
            ),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["ok"] is False
    assert final["status"] == "failed"
    assert final["stop_reason"] == "result_output_unavailable"
    assert final["failure"]["stage"] == "finalization"
    assert final["failure"]["code"] == "result_output_unavailable"
    assert final["failure"]["next_action"] == "inspect_logs_and_retry"


def test_monitor_maps_failed_to_failed_with_failure_block(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Monitor must produce envelope with status=failed + failure block."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("deadb33f"),
            _agents_entry("failed", "deadb33f"),
            _logs_ok("execution error"),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["ok"] is False
    assert final["status"] == "failed"
    assert final["stop_reason"] == "failed"
    assert final["failure"] is not None
    assert final["failure"]["stage"] == "execution"
    assert final["failure"]["code"] == "native_failed"


def test_monitor_maps_stopped_to_cancelled(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Monitor must produce envelope with status=cancelled for stopped sessions."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("cafebabe"),
            _agents_entry("stopped", "cafebabe"),
            _logs_ok("stopped by user"),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["status"] == "cancelled"
    assert final["stop_reason"] == "stopped"


# ---------------------------------------------------------------------------
# Test 8: blocked noninteractive → failed with structured failure
# ---------------------------------------------------------------------------


def test_monitor_blocked_noninteractive_produces_execution_failure(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Blocked noninteractive must produce status=failed, not generic failure."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("deadbeef"),
            _agents_entry("blocked", "deadbeef"),
            _logs_ok("needs approval"),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["ok"] is False
    assert final["schema_version"] == "1"
    assert final["status"] == "failed"
    assert final["stop_reason"] == "blocked"
    assert final["failure"] is not None
    assert final["failure"]["stage"] == "execution"
    assert final["failure"]["code"] == "blocked_noninteractive"
    assert final["failure"]["retryable"] is False
    assert final["failure"]["next_action"] is not None
    assert final["summary"] == "needs approval"


def test_monitor_claude_limit_is_actionable_and_never_returns_raw_tui(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """A Claude usage limit is not an interactive-consent block."""
    raw_logs = (
        "\x1b[2J\x1b[HClaude Code\n"
        "\x1b[31mYou've hit your org's monthly spend limit\x1b[0m"
        " · run /usage-credits to ask your admin for a higher limit\n"
    )
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("deadbeef"),
            _agents_entry("blocked", "deadbeef"),
            _logs_ok(raw_logs),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(
        store,
        result["job_id"],
        reg.get_adapter("claude"),
        poll_interval_sec=0.01,
    )

    final = store.get_result(result["job_id"])
    assert final["ok"] is False
    assert final["failure"]["code"] == "provider_limit_exhausted"
    assert final["failure"]["retryable"] is True
    assert final["failure"]["next_action"] == "check_provider_limits_or_retry_after_reset"
    assert "usage limit" in final["summary"].lower()
    assert "retry_with_interactive_mode" not in str(final)
    assert "\x1b" not in str(final)
    assert "/Users/" not in str(final["failure"]["diagnostics"])
    assert len(final["summary"]) < 300


def test_monitor_claude_missing_login_is_auth_failure(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("deadbeef"),
            _agents_entry("blocked", "deadbeef"),
            _logs_ok("\x1b[31mNot logged in · Please run /login\x1b[0m"),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(
        store,
        result["job_id"],
        reg.get_adapter("claude"),
        poll_interval_sec=0.01,
    )
    final = store.get_result(result["job_id"])
    assert final["failure"]["stage"] == "auth"
    assert final["failure"]["code"] == "provider_needs_auth"
    assert final["failure"]["retryable"] is False
    assert final["failure"]["next_action"] == "authenticate_provider"
    assert final["summary"] == "Claude is not authenticated. Start Claude and run /login."


# ---------------------------------------------------------------------------
# Test 9: runtime deadline failure
# ---------------------------------------------------------------------------


def test_monitor_runtime_deadline_produces_execution_failure_with_timeout_layer(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Runtime deadline exceeded must produce stage=execution code=max_runtime_exceeded."""
    from agent_crossbar.agent_runner import _run_adapter_job

    # Create job manually since server validation rejects max_runtime_sec=0
    store = JobStore(claude_state_root)
    job = store.create_job(
        profile="claude",
        operation="ask",
        transport="claude_bg",
        client_name="test",
        cwd="/tmp/test",
    )
    store.update_job_meta(
        job.job_id,
        {
            "backend": "claude_bg",
            "native_session_id": "deadbeef",
            "model": "claude-sonnet-5",
            "effort": "medium",
            "task": "ask",
            "interactive": False,
            "cwd": "/tmp/test",
            "adapter_name": "claude",
            "profile": "claude",
            "created": "2025-01-01T00:00:00Z",
        },
    )
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")

    # Use max_runtime_sec=0 so deadline triggers immediately
    _run_adapter_job(
        store,
        job.job_id,
        adapter,
        session_id="deadbeef",
        poll_interval_sec=0.01,
        max_runtime_sec=0,
    )

    final = store.get_result(job.job_id)
    assert final["ok"] is False
    assert final["status"] == "failed"
    assert final["stop_reason"] == "max_runtime_exceeded"
    assert final["failure"] is not None
    assert final["failure"]["stage"] == "execution"
    assert final["failure"]["code"] == "max_runtime_exceeded"
    assert final["failure"]["retryable"] is True
    assert final["failure"]["next_action"] == "retry_with_higher_timeout"
    # Diagnostics must name the timeout layer
    diag = final["failure"]["diagnostics"]
    assert diag["layer"] == "timeout"
    assert "max_runtime_sec" in diag
    assert diag["max_runtime_sec"] == 0


# ---------------------------------------------------------------------------
# Test 10: monitor/finalization exception failure
# ---------------------------------------------------------------------------


def test_monitor_finalization_exception_produces_finalization_failure(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Unhandled exception in monitor thread must produce stage=finalization code=monitor_failure."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("deadbeef"),
            # First poll — blow up
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    # Make adapter.status raise after the first call
    original_status = (
        __import__("agent_crossbar.adapters.claude", fromlist=["ClaudeAdapter"])
        .ClaudeAdapter()
        .status
    )

    call_count = [0]

    def exploding_status(adapter_self, runner_arg, session_id):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("monitor exploded")
        return original_status(runner_arg, session_id)

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.ClaudeAdapter.status",
        exploding_status,
    )

    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["ok"] is False
    assert final["status"] == "failed"
    assert final["stop_reason"] == "monitor_failure"
    assert final["failure"] is not None
    assert final["failure"]["stage"] == "finalization"
    assert final["failure"]["code"] == "monitor_failure"
    assert final["failure"]["retryable"] is False


# ---------------------------------------------------------------------------
# Test 11: full session ID persisted on every poll
# ---------------------------------------------------------------------------


def test_monitor_persists_full_session_id_on_every_poll_not_only_terminal(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Full native session ID must be persisted on every status poll, not just terminal."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("abc12345"),
            # Poll 1: working, with full session_id
            _agents_entry("working", "abc12345"),
            _logs_ok(""),
            # Poll 2: done
            _agents_entry("done", "abc12345"),
            _logs_ok("success"),
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["schema_version"] == "1"
    # Full session ID must be in the technical block
    assert final["technical"]["native_full_session_id"] == "abc12345-9fd6-4813-a6e6-3cb0d134de46"
    assert final["technical"]["native_session_id"] == "abc12345"


# ---------------------------------------------------------------------------
# Test 12: diagnostics bounded, no raw environment/secrets
# ---------------------------------------------------------------------------


def test_monitor_failure_diagnostics_bounded_and_no_secrets(
    claude_state_root: Path, monkeypatch, suppress_background_monitor
):
    """Failure diagnostics must be bounded to 2 KiB and stripped of env/secrets."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("deadbeef"),
            _agents_entry("failed", "deadbeef"),
            _logs_ok("x" * 5000),  # large output
        ]
    )
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="do thing",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["failure"] is not None
    diag = final["failure"]["diagnostics"]
    # output in diagnostics must be bounded
    output_diag = diag.get("output", "")
    assert len(output_diag.encode("utf-8")) <= 2048
    # No raw env or secret keys
    flat = str(diag)
    assert "SECRET" not in flat
    assert "API_KEY" not in flat
    assert "env" not in diag


# ---------------------------------------------------------------------------
# Test 10: incremental log_delta events during execution
# ---------------------------------------------------------------------------


def test_monitor_emits_log_delta_events_during_working_state(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """Prove incremental get_logs deltas are emitted as log_delta events during working state."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b00001"),
            # status poll 1: working → logs have "line 1"
            _agents_entry("working", "a1b00001"),
            _logs_ok("line 1\n"),
            # status poll 2: working → logs grew to "line 1\nline 2\n"
            _agents_entry("working", "a1b00001"),
            _logs_ok("line 1\nline 2\n"),
            # status poll 3: done
            _agents_entry("done", "a1b00001"),
            _logs_ok("line 1\nline 2\nfinal output"),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="stream this",
        task="ask",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    adapter = reg.get_adapter("claude")
    monitor_agent_job(store, job_id, adapter, poll_interval_sec=0.01)

    # Check events
    tail = store.job_tail(job_id)
    delta_events = [e for e in tail["events"] if e["type"] == "log_delta"]
    assert len(delta_events) >= 1, f"Expected at least 1 log_delta event, got {tail['events']}"

    # First delta: "line 1"
    assert any("line 1" in e["data"].get("text", "") for e in delta_events), (
        f"Expected 'line 1' in delta events: {delta_events}"
    )

    # Final result still captured
    final = store.get_result(job_id)
    assert final.get("ok") is True
    assert "final output" in final.get("summary", "")


def test_monitor_log_delta_handles_empty_or_identical_logs_gracefully(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """Prove that get_logs returning empty or unchanged logs doesn't crash or emit noise."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b00002"),
            # poll 1: working, logs empty
            _agents_entry("working", "a1b00002"),
            _logs_ok(""),
            # poll 2: working, logs empty again (no change)
            _agents_entry("working", "a1b00002"),
            _logs_ok(""),
            # poll 3: done with content
            _agents_entry("done", "a1b00002"),
            _logs_ok("final result"),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="quiet task",
        task="ask",
        interactive=False,
        client_name="test",
    )
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    tail = store.job_tail(job_id)
    delta_events = [e for e in tail["events"] if e["type"] == "log_delta"]
    assert len(delta_events) == 0, (
        f"Expected zero log_delta events for empty logs, got {delta_events}"
    )

    final = store.get_result(job_id)
    assert final.get("ok") is True
    assert final.get("summary") == "final result"


def test_monitor_log_delta_deduplicates_sliding_log_tails(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """A bounded provider log tail must not re-emit its overlapping prefix."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b00025"),
            _agents_entry("working", "a1b00025"),
            _logs_ok("line 1\nline 2\n"),
            _agents_entry("working", "a1b00025"),
            _logs_ok("line 2\nline 3\n"),
            _agents_entry("done", "a1b00025"),
            _logs_ok("line 2\nline 3\nfinal result"),
        ]
    )
    monkeypatch.setattr("agent_crossbar.adapters.claude.LocalSubprocessRunner.run", runner.run)

    from agent_crossbar.agent_runner import monitor_agent_job

    job_id = agent_start(
        profile="claude", prompt="sliding logs", task="ask", interactive=False, client_name="test"
    )["job_id"]
    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)
    deltas = [
        event["data"]["text"]
        for event in store.job_tail(job_id)["events"]
        if event["type"] == "log_delta"
    ]
    assert deltas == ["line 1\nline 2", "line 3"]


def test_monitor_emits_working_heartbeat_when_provider_is_silent(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """A silent but working provider remains observable without being stopped."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b00026"),
            _agents_entry("working", "a1b00026"),
            _logs_ok(""),
            _agents_entry("done", "a1b00026"),
            _logs_ok("finished after quiet reasoning"),
        ]
    )
    monkeypatch.setattr("agent_crossbar.adapters.claude.LocalSubprocessRunner.run", runner.run)

    from agent_crossbar.agent_runner import monitor_agent_job

    job_id = agent_start(
        profile="claude",
        prompt="quiet reasoning",
        task="ask",
        interactive=False,
        client_name="test",
    )["job_id"]
    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)
    events = store.job_tail(job_id)["events"]
    heartbeats = [event for event in events if event["type"] == "provider_heartbeat"]
    assert len(heartbeats) == 1
    assert heartbeats[0]["data"]["native_state"] == "working"
    assert not any(call["args"][:2] == ["claude", "stop"] for call in runner.calls)


def test_monitor_log_delta_survives_get_logs_failure(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """Prove that a failing get_logs call during polling doesn't crash the monitor."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b00003"),
            # poll 1: working, get_logs fails (non-zero rc)
            _agents_entry("working", "a1b00003"),
            RunResult(1, "", "permission denied"),
            # poll 2: working, get_logs works
            _agents_entry("working", "a1b00003"),
            _logs_ok("recovered output"),
            # poll 3: done
            _agents_entry("done", "a1b00003"),
            _logs_ok("recovered output\ndone"),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="flaky logs",
        task="ask",
        interactive=False,
        client_name="test",
    )
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    # Must not raise
    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final.get("ok") is True
    assert "done" in final.get("summary", "")


def test_monitor_log_delta_sanitizes_ansi_control_chars(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """Prove that ANSI escape sequences and control chars are stripped from delta events."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b00004"),
            # poll 1: working, raw logs with ANSI
            _agents_entry("working", "a1b00004"),
            _logs_ok("\x1b[32mOK\x1b[0m\nprocessing...\x07"),
            # poll 2: done
            _agents_entry("done", "a1b00004"),
            _logs_ok("\x1b[32mOK\x1b[0m\nprocessing...\x07\nfinal"),
        ]
    )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.LocalSubprocessRunner.run",
        runner.run,
    )

    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="colorful",
        task="ask",
        interactive=False,
        client_name="test",
    )
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    tail = store.job_tail(job_id)
    delta_events = [e for e in tail["events"] if e["type"] == "log_delta"]
    assert len(delta_events) >= 1

    for e in delta_events:
        text = e["data"].get("text", "")
        # No raw ANSI escape sequences
        assert "\x1b" not in text, f"ANSI escape in delta: {repr(text)}"
        # No BEL control char
        assert "\x07" not in text, f"Control char in delta: {repr(text)}"


# ---------------------------------------------------------------------------
# Test: Claude idle-after-final-response lifecycle fix (regression)
# ---------------------------------------------------------------------------


def _agents_entry_working_idle(session_id="f43e1980"):
    """Claude agents entry with state=working, status=idle — the bug condition."""
    return RunResult(
        0,
        json.dumps(
            [
                {
                    "id": session_id,
                    "sessionId": f"{session_id}-9fd6-4813-a6e6-3cb0d134de46",
                    "state": "working",
                    "status": "idle",
                    "waitingFor": None,
                    "cwd": "/repo",
                    "startedAt": 1784739414928,
                }
            ]
        ),
        "",
    )


def test_monitor_finalizes_working_idle_with_final_response(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """When Claude reports state=working,status=idle with a final response in
    logs, the monitor must detect the completed session and finalize it."""
    final_output = (
        "[Screen Reader Mode: on via flag]\n"
        "$claude: Summary\n"
        "\n"
        "Here is the contract drift analysis for pandahome:\n"
        "\n"
        "The contract has three material changes:\n"
        "1. Payment terms changed from net-30 to net-15\n"
        "2. Liability cap increased from $100k to $250k\n"
        "3. Termination notice period reduced from 90 to 30 days\n"
        "\n"
        "don't ask on (shift+tab to cycle) · esc to interrupt\n"
        "1284 tokens\n"
        "$Worked for 45s\n"
    )
    # Each poll: status + idle-detection get_logs + regular get_logs.
    # Poll 3 triggers idle finalization (breaks before regular get_logs).
    # Post-break: terminal block calls get_logs once more.
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("f43e1980"),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok(final_output),
            _logs_ok(final_output),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok(final_output),
            _logs_ok(final_output),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok(final_output),
            _logs_ok(final_output),
        ]
    )
    monkeypatch.setattr("agent_crossbar.adapters.claude.LocalSubprocessRunner.run", runner.run)
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="audit pandahome contract",
        task="review",
        interactive=False,
        client_name="test",
    )
    assert result.get("ok") is True, result
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["ok"] is True, (
        f"Expected completed, got {final.get('status')}: {final.get('summary', '')[:200]}"
    )
    assert final["status"] == "completed"
    assert "contract drift" in final.get("summary", "")

    # Verify idle_finalized event was emitted
    tail = store.job_tail(job_id)
    idle_events = [e for e in tail["events"] if e["type"] == "idle_finalized"]
    assert len(idle_events) >= 1, (
        f"Expected idle_finalized event, got {[e['type'] for e in tail['events']]}"
    )
    assert idle_events[0]["data"]["consecutive_idle_polls"] >= 2


def test_monitor_does_not_false_finalize_without_final_response(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """When Claude reports state=working,status=idle but logs contain NO
    complete final response, the monitor must NOT falsely finalize."""
    incomplete = (
        "[Screen Reader Mode: on via flag]\n"
        "$claude: Summary\n"
        "esc to interrupt\n"
        "0 tokens\n"
        "$Worked for 2s\n"
    )
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("f43e1980"),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok(incomplete),
            _logs_ok(incomplete),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok(incomplete),
            _logs_ok(incomplete),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok(incomplete),
            _logs_ok(incomplete),
            _agents_entry("done", "f43e1980"),
            _logs_ok("eventual output"),
        ]
    )
    monkeypatch.setattr("agent_crossbar.adapters.claude.LocalSubprocessRunner.run", runner.run)
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="something",
        task="ask",
        interactive=False,
        client_name="test",
    )
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["status"] == "completed"
    tail = store.job_tail(job_id)
    idle_events = [e for e in tail["events"] if e["type"] == "idle_finalized"]
    assert len(idle_events) == 0, "Must not emit idle_finalized without final response"


def test_monitor_resets_idle_counter_on_changed_logs(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """When logs change between working+idle polls, the idle counter resets."""
    final_output = "[Screen Reader Mode: on via flag]\n$claude: Summary\ndone\n$Worked for 5s\n"
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("f43e1980"),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok("partial A"),
            _logs_ok("partial A"),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok("partial B"),
            _logs_ok("partial B"),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok("partial B"),
            _logs_ok("partial B"),
            _agents_entry_working_idle("f43e1980"),
            _logs_ok("partial B"),
            _logs_ok("partial B"),
            _agents_entry("done", "f43e1980"),
            _logs_ok(final_output),
        ]
    )
    monkeypatch.setattr("agent_crossbar.adapters.claude.LocalSubprocessRunner.run", runner.run)
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="something",
        task="ask",
        interactive=False,
        client_name="test",
    )
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    final = store.get_result(job_id)
    assert final["status"] == "completed"
    tail = store.job_tail(job_id)
    idle_events = [e for e in tail["events"] if e["type"] == "idle_finalized"]
    assert len(idle_events) == 0, "Must not finalize via idle when logs changed, resetting counter"


def test_job_tail_preserves_log_delta_events_for_claude_bg(
    claude_state_root, monkeypatch, suppress_background_monitor
):
    """job_tail must surface incremental log_delta events emitted during
    Claude bg polling — these were recently added and must be preserved."""
    runner = FakeRunner(
        [
            _auth_ok(),
            _launch_ok("a1b2c3d4"),
            _agents_entry("working", "a1b2c3d4"),
            _logs_ok("\x1b[32mincremental\x1b[0m\n"),
            _logs_ok("\x1b[32mincremental\x1b[0m\n"),
            _agents_entry("done", "a1b2c3d4"),
            _logs_ok("final output"),
        ]
    )
    monkeypatch.setattr("agent_crossbar.adapters.claude.LocalSubprocessRunner.run", runner.run)
    from agent_crossbar.agent_runner import monitor_agent_job

    result = agent_start(
        profile="claude",
        prompt="test",
        task="ask",
        interactive=False,
        client_name="test",
    )
    job_id = result["job_id"]

    store = JobStore(claude_state_root)
    import agent_crossbar.adapters.registry as reg

    monitor_agent_job(store, job_id, reg.get_adapter("claude"), poll_interval_sec=0.01)

    tail = store.job_tail(job_id)
    delta_events = [e for e in tail["events"] if e["type"] == "log_delta"]
    assert len(delta_events) >= 1, (
        f"Must preserve log_delta events; got {[e['type'] for e in tail['events']]}"
    )
    for e in delta_events:
        text = e["data"].get("text", "")
        assert "\x1b" not in text, f"ANSI escape in delta: {repr(text)}"
