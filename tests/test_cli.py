"""CLI help and version regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from agent_crossbar import __version__
from agent_crossbar.cli import main
from agent_crossbar.jobs import JobStore


def test_help_exits_without_starting_the_mcp_server(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-crossbar", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Run the Agent Crossbar MCP server" in output
    assert "doctor" in output


def test_version_prints_installed_package_version(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-crossbar", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"agent-crossbar {__version__}\n"


def test_wait_job_help_documents_explicit_state_dir(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-crossbar", "wait-job", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert "--state-dir" in capsys.readouterr().out


def test_writer_lease_cli_keeps_lease_commands_outside_mcp_surface():
    from agent_crossbar.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "writer-lease",
            "acquire",
            "--state-dir",
            "/tmp/state",
            "--cwd",
            "/tmp/workspace",
            "--owner-id",
            "controller",
            "--owner-kind",
            "local",
        ]
    )
    assert args.command == "writer-lease"
    assert args.writer_lease_command == "acquire"
    assert args.owner_kind == "local"

    recover = _build_parser().parse_args(
        ["writer-lease", "recover", "--cwd", "/tmp/workspace", "--acknowledgement", "ack"]
    )
    assert recover.writer_lease_command == "recover"
    assert recover.acknowledgement == "ack"


def _run_wait_job(
    state_root,
    job_id: str,
    timeout_sec: float = 1.0,
    state_dir: str | None = None,
):
    env = {**os.environ, "AGENT_CROSSBAR_STATE_DIR": str(state_root)}
    args = [
        sys.executable,
        "-m",
        "agent_crossbar.cli",
        "wait-job",
        "--job-id",
        job_id,
        "--timeout-sec",
        str(timeout_sec),
        "--poll-interval-sec",
        "0.01",
    ]
    if state_dir is not None:
        args.extend(["--state-dir", state_dir])
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _run_terminalize_job(state_root, job_id: str, reason: str = "blocking_prompt"):
    env = {**os.environ, "AGENT_CROSSBAR_STATE_DIR": str(state_root)}
    args = [
        sys.executable,
        "-m",
        "agent_crossbar.cli",
        "terminalize-job",
        "--job-id",
        job_id,
        "--timeout-sec",
        "1",
        "--poll-interval-sec",
        "0.01",
        "--reason",
        reason,
    ]
    return subprocess.run(args, check=False, capture_output=True, text=True, env=env)


def test_wait_job_deadline_does_not_stop_a_silent_running_job(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix", operation="dev", transport="print", cwd=str(tmp_path)
    )

    timed_out = _run_wait_job(tmp_path, job.job_id, timeout_sec=0.05)

    assert timed_out.returncode == 2
    assert store.job_status(job.job_id) == "running"


def test_terminalize_job_explicitly_stops_then_collects_terminal_result(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix", operation="dev", transport="print", cwd=str(tmp_path)
    )
    store.update_job_meta(job.job_id, {"status": "awaiting_input", "waiting_for": "follow_up"})

    terminal = _run_terminalize_job(tmp_path, job.job_id, reason="blocking_prompt")

    assert terminal.returncode == 3
    result = json.loads(terminal.stdout)
    assert result["status"] == "stopped"
    assert store.job_status(job.job_id) == "stopped"


def test_terminalize_blocking_prompt_refuses_running_job_without_stopping(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix", operation="dev", transport="print", cwd=str(tmp_path)
    )

    terminal = _run_terminalize_job(tmp_path, job.job_id, reason="blocking_prompt")

    assert terminal.returncode == 5
    result = json.loads(terminal.stdout)
    assert result["error"] == "terminalize_reason_not_permitted"
    assert result["status"] == "running"
    assert store.job_status(job.job_id) == "running"


def test_terminalize_runtime_deadline_refuses_before_recorded_deadline(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix", operation="dev", transport="print", cwd=str(tmp_path)
    )
    store.update_job_meta(
        job.job_id,
        {"max_runtime_sec": 3600, "started_at": "2099-01-01T00:00:00+00:00"},
    )

    terminal = _run_terminalize_job(tmp_path, job.job_id, reason="runtime_deadline")

    assert terminal.returncode == 5
    result = json.loads(terminal.stdout)
    assert result["error"] == "runtime_deadline_not_reached"
    assert result["status"] == "running"
    assert store.job_status(job.job_id) == "running"


def test_terminalize_runtime_deadline_stops_after_recorded_deadline_plus_grace(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix", operation="dev", transport="print", cwd=str(tmp_path)
    )
    store.update_job_meta(
        job.job_id,
        {"max_runtime_sec": 1, "started_at": "2000-01-01T00:00:00+00:00"},
    )

    terminal = _run_terminalize_job(tmp_path, job.job_id, reason="runtime_deadline")

    assert terminal.returncode == 3
    result = json.loads(terminal.stdout)
    assert result["status"] == "stopped"
    assert store.job_status(job.job_id) == "stopped"


def test_terminalize_job_missing_job_is_not_a_replacement_signal(tmp_path) -> None:
    terminal = _run_terminalize_job(tmp_path, "123456789-missing")

    assert terminal.returncode == 4
    assert json.loads(terminal.stdout)["error"] == "job_not_found"


def test_terminalize_job_stops_awaiting_input_state(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix", operation="dev", transport="print", cwd=str(tmp_path)
    )
    store.update_job_meta(job.job_id, {"status": "awaiting_input", "waiting_for": "follow_up"})

    terminal = _run_terminalize_job(tmp_path, job.job_id, reason="blocking_prompt")

    assert terminal.returncode == 3
    result = json.loads(terminal.stdout)
    assert result["status"] == "stopped"
    assert store.job_status(job.job_id) == "stopped"


def test_wait_job_completed_returns_zero(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode", operation="dev", transport="print", cwd=str(tmp_path)
    )
    store.set_result(job.job_id, ok=True, summary="done", envelope={"status": "completed"})

    completed = _run_wait_job(tmp_path, job.job_id)

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["summary"] == "done"


def test_wait_job_terminal_failure_returns_three(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode", operation="dev", transport="print", cwd=str(tmp_path)
    )
    store.set_result(job.job_id, ok=False, summary="provider failed", envelope={"status": "failed"})

    completed = _run_wait_job(tmp_path, job.job_id)

    assert completed.returncode == 3
    assert json.loads(completed.stdout)["summary"] == "provider failed"


def test_wait_job_not_found_returns_four(tmp_path) -> None:
    completed = _run_wait_job(tmp_path, "1787323619407-job")

    assert completed.returncode == 4
    assert json.loads(completed.stdout)["error"] == "job_not_found"


def test_wait_job_deadline_returns_two_without_stopping_running_job(tmp_path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode", operation="dev", transport="print", cwd=str(tmp_path)
    )

    completed = _run_wait_job(tmp_path, job.job_id, timeout_sec=0.01)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error"] == "terminal_wait_timeout"
    assert payload["last_result"]["error"] == "result_not_ready"
    assert store._read_job_meta(job.path).get("status", "running") == "running"


def test_wait_job_explicit_state_dir_reads_agents_mcp_state(tmp_path) -> None:
    mcp_state = tmp_path / "agents-mcp-state"
    other_state = tmp_path / "other-state"
    store = JobStore(mcp_state)
    job = store.create_job(
        profile="opencode", operation="dev", transport="print", cwd=str(tmp_path)
    )
    store.set_result(job.job_id, ok=True, summary="shared-state-done")

    completed = _run_wait_job(other_state, job.job_id, state_dir=str(mcp_state))

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["summary"] == "shared-state-done"


def test_wait_job_without_matching_state_dir_returns_four(tmp_path) -> None:
    mcp_state = tmp_path / "agents-mcp-state"
    wrong_state = tmp_path / "wrong-state"
    store = JobStore(mcp_state)
    job = store.create_job(
        profile="opencode", operation="dev", transport="print", cwd=str(tmp_path)
    )

    completed = _run_wait_job(wrong_state, job.job_id)

    assert completed.returncode == 4
    assert json.loads(completed.stdout)["error"] == "job_not_found"
