#!/usr/bin/env python3
"""Minimal regression harness for the quiet-but-live ACP controller failure.

The harness deliberately models the controller decision, not an ACP provider:
the subprocess is real, quiet for a short interval, and then exits successfully.
This keeps the regression deterministic and credential-free.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from agent_crossbar.jobs import JobStore


@dataclass(frozen=True)
class HarnessResult:
    process_alive_during_quiet_window: bool
    quiet_tail_status: str
    terminal_result: str
    unsafe_stall_inference: bool
    job_stop_events: int


def _create_quiet_job(
    quiet_sec: float,
) -> tuple[tempfile.TemporaryDirectory[str], JobStore, str, subprocess.Popen[str]]:
    state_root = tempfile.TemporaryDirectory(prefix="acp-quiet-live-")
    store = JobStore(state_root.name)
    job = store.create_job(
        profile="opencode",
        operation="dev",
        transport="print",
        cwd=state_root.name,
    )
    writer_code = (
        "import sys,time; "
        "time.sleep(float(sys.argv[3])); "
        "from agent_crossbar.jobs import JobStore; "
        "JobStore(sys.argv[1]).set_result(sys.argv[2], True, summary='ACP_QUIET_LIVE_OK')"
    )
    writer = subprocess.Popen(
        [sys.executable, "-c", writer_code, state_root.name, job.job_id, str(quiet_sec)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return state_root, store, job.job_id, writer


def run_harness(*, quiet_sec: float = 0.15) -> HarnessResult:
    """Run a quiet live subprocess and record the pre-fix unsafe decision."""
    state_root, store, job_id, process = _create_quiet_job(quiet_sec)
    time.sleep(quiet_sec / 2)
    alive = process.poll() is None
    quiet_tail = store.job_tail(job_id, max_bytes=20000, client_session_id="*")
    # This is the incident's pre-fix decision: no result and no workspace diff
    # were treated as proof of a stall, even though the process was alive.
    result_not_ready = process.poll() is None
    no_workspace_diff = True
    unsafe_stall = result_not_ready and no_workspace_diff
    process_stdout, process_stderr = process.communicate(timeout=2)
    if process.returncode != 0:
        raise RuntimeError(f"quiet subprocess failed: {process_stderr[-500:]}")
    result = store.get_result(job_id, client_session_id="*")
    final_tail = store.job_tail(job_id, max_bytes=20000, client_session_id="*")
    state_root.cleanup()
    stop_events = sum(event.get("type") == "stopped" for event in final_tail.get("events", []))
    return HarnessResult(
        alive,
        str(quiet_tail.get("status")),
        str(result.get("summary") or process_stdout.strip()),
        unsafe_stall,
        stop_events,
    )


def run_fixed_harness(*, quiet_sec: float = 0.15) -> HarnessResult:
    """Run the same durable job through the callable CLI waiter."""
    state_root, _store, job_id, process = _create_quiet_job(quiet_sec)
    alive = process.poll() is None
    waiter = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_crossbar.cli",
            "wait-job",
            "--job-id",
            job_id,
            "--timeout-sec",
            str(max(1.0, quiet_sec * 10)),
            "--poll-interval-sec",
            str(min(0.02, quiet_sec / 4)),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "AGENT_CROSSBAR_STATE_DIR": state_root.name},
    )
    time.sleep(quiet_sec / 2)
    quiet_tail = _store.job_tail(job_id, max_bytes=20000, client_session_id="*")
    waiter_stdout, waiter_stderr = waiter.communicate(timeout=5)
    process_stdout, process_stderr = process.communicate(timeout=2)
    final_tail = _store.job_tail(job_id, max_bytes=20000, client_session_id="*")
    state_root.cleanup()
    if process.returncode != 0:
        raise RuntimeError(f"quiet subprocess failed: {process_stderr[-500:]}")
    if waiter.returncode != 0:
        raise RuntimeError(
            f"wait-job failed rc={waiter.returncode}: {waiter_stderr[-500:]} {waiter_stdout[-500:]}"
        )
    result = json.loads(waiter_stdout)
    stop_events = sum(event.get("type") == "stopped" for event in final_tail.get("events", []))
    return HarnessResult(
        process_alive_during_quiet_window=alive,
        quiet_tail_status=str(quiet_tail.get("status")),
        terminal_result=str(result.get("summary") or process_stdout.strip()),
        unsafe_stall_inference=False,
        job_stop_events=stop_events,
    )


def main() -> int:
    fixed = "--fixed" in sys.argv[1:]
    result = run_fixed_harness() if fixed else run_harness()
    print(json.dumps(result.__dict__, sort_keys=True))
    if fixed:
        return 0 if result.terminal_result == "ACP_QUIET_LIVE_OK" else 1
    return 0 if result.process_alive_during_quiet_window and result.unsafe_stall_inference else 1


if __name__ == "__main__":
    raise SystemExit(main())
