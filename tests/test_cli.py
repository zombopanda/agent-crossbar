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
