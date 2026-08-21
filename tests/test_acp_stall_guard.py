"""Regression tests for ACP quiet-job supervision."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_crossbar.terminal_wait import TerminalWaitTimeout, wait_for_terminal_result


def test_pre_fix_harness_captures_quiet_live_unsafe_gap() -> None:
    """The minimal pre-fix harness proves the bad inference was possible."""
    harness = Path(__file__).parents[1] / "scripts" / "acp_quiet_live_harness.py"
    completed = subprocess.run(
        [sys.executable, str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed == {
        "process_alive_during_quiet_window": True,
        "quiet_tail_status": "running",
        "terminal_result": "ACP_QUIET_LIVE_OK",
        "unsafe_stall_inference": True,
        "job_stop_events": 0,
    }


def test_post_fix_harness_waits_for_the_same_quiet_live_process() -> None:
    harness = Path(__file__).parents[1] / "scripts" / "acp_quiet_live_harness.py"
    completed = subprocess.run(
        [sys.executable, str(harness), "--fixed"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed == {
        "process_alive_during_quiet_window": True,
        "quiet_tail_status": "running",
        "terminal_result": "ACP_QUIET_LIVE_OK",
        "unsafe_stall_inference": False,
        "job_stop_events": 0,
    }


def test_waiter_keeps_quiet_live_job_until_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        [
            {"ok": False, "error": "result_not_ready"},
            {"ok": False, "error": "result_not_ready"},
            {"ok": True, "status": "completed", "summary": "ACP_QUIET_LIVE_OK"},
        ]
    )
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def read_result() -> dict:
        return next(results)

    result = asyncio.run(wait_for_terminal_result(read_result, timeout_sec=30, poll_interval_sec=2))

    assert result["status"] == "completed"
    assert sleeps == [2, 2]


def test_waiter_does_not_cancel_on_quiet_intermediate_result() -> None:
    reads = 0

    async def read_result() -> dict:
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"ok": False, "error": "result_not_ready"}
        return {"ok": True, "status": "completed"}

    result = asyncio.run(
        wait_for_terminal_result(read_result, timeout_sec=30, poll_interval_sec=0.001)
    )
    assert result["status"] == "completed"
    assert reads == 2


def test_waiter_returns_real_process_failure_as_terminal() -> None:
    async def read_result() -> dict:
        return {
            "ok": False,
            "status": "failed",
            "failure": {"code": "acp_launch_error"},
        }

    result = asyncio.run(wait_for_terminal_result(read_result, timeout_sec=30))
    assert result["status"] == "failed"
    assert result["failure"]["code"] == "acp_launch_error"


def test_explicit_job_stop_remains_immediate_for_acp(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_crossbar.jobs import JobStore
    from agent_crossbar.server import job_stop

    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode",
        operation="dev",
        transport="print",
        cwd=str(tmp_path),
    )
    meta = store._read_job_meta(job.path)
    store.update_job_meta(job.job_id, {**meta, "backend": "acp", "status": "running"})

    stopped = job_stop(job.job_id, reason="explicit_user_stop")

    assert stopped["ok"] is True
    result = store.get_result(job.job_id)
    assert result["status"] == "cancelled"
    assert result["stop_reason"] == "explicit_user_stop"


def test_waiter_raises_terminal_timeout_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 0.0, 31.0])

    def fake_monotonic() -> float:
        return next(clock, 31.0)

    monkeypatch.setattr("agent_crossbar.terminal_wait.time.monotonic", fake_monotonic)

    async def read_result() -> dict:
        return {"ok": False, "error": "result_not_ready"}

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    with pytest.raises(TerminalWaitTimeout) as exc:
        asyncio.run(wait_for_terminal_result(read_result, timeout_sec=30, poll_interval_sec=2))
    assert exc.value.last_result["error"] == "result_not_ready"


def test_waiter_bounds_slow_async_result_read() -> None:
    async def read_result() -> dict:
        await asyncio.sleep(0.2)
        return {"ok": True, "status": "completed"}

    started = time.perf_counter()
    with pytest.raises(TerminalWaitTimeout):
        asyncio.run(wait_for_terminal_result(read_result, timeout_sec=0.01))
    assert time.perf_counter() - started < 0.1


def test_waiter_bounds_slow_async_not_ready_observer() -> None:
    async def read_result() -> dict:
        return {"ok": False, "error": "result_not_ready"}

    async def slow_observer(_result: dict) -> None:
        await asyncio.sleep(0.2)

    started = time.perf_counter()
    with pytest.raises(TerminalWaitTimeout):
        asyncio.run(
            wait_for_terminal_result(
                read_result,
                timeout_sec=0.01,
                on_not_ready=slow_observer,
            )
        )
    assert time.perf_counter() - started < 0.1


def test_waiter_rejects_sync_observer_without_leaking_a_coroutine() -> None:
    async def read_result() -> dict:
        return {"ok": False, "error": "result_not_ready"}

    def sync_observer(_result: dict) -> None:
        return None

    with pytest.raises(TypeError, match="on_not_ready must return an awaitable"):
        asyncio.run(
            wait_for_terminal_result(
                read_result,
                timeout_sec=1,
                on_not_ready=sync_observer,  # type: ignore[arg-type]
            )
        )


def test_waiter_reports_cancellation_resistant_read_after_it_settles() -> None:
    async def cancellation_resistant_read() -> dict:
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            # A provider/MCP operation may defer cancellation briefly.  The
            # waiter reports the deadline once this cooperative operation ends.
            await asyncio.sleep(0.02)
        return {"ok": False, "error": "result_not_ready"}

    started = time.perf_counter()
    with pytest.raises(TerminalWaitTimeout):
        asyncio.run(wait_for_terminal_result(cancellation_resistant_read, timeout_sec=0.01))
    elapsed = time.perf_counter() - started
    assert 0.015 <= elapsed < 0.1
