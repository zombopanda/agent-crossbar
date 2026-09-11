"""Claude interactive monitor continuation, question fidelity, and fast reply.

Uses a scripted lifecycle adapter so the monitor's polling loop can be driven
deterministically without a real Claude CLI or tmux session.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from agent_crossbar.agent_runner import (
    _run_adapter_job,
    _terminalize_runtime_deadline,
    start_agent_job,
)
from agent_crossbar.jobs import EventWriter, JobStore


@pytest.fixture(autouse=True)
def _stub_tmux(monkeypatch):
    """job_send delivers tmux keystrokes; stub the tmux binary."""
    monkeypatch.setattr(
        "agent_crossbar.jobs.subprocess.run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0})(),
    )


class _ScriptedAdapter:
    """Lifecycle adapter whose ``status`` is chosen by a callback."""

    def __init__(self, decide):
        self._decide = decide

    def status(self, runner, session_id):
        return self._decide()

    def get_logs(self, runner, session_id):
        return ""

    def normalize_result(self, entry, logs):
        return SimpleNamespace(
            status="completed",
            stop_reason="done",
            output="final answer",
            error=None,
            error_stage=None,
        )

    def cancel(self, runner, session_id):
        return True


class _RetryThenHungAdapter(_ScriptedAdapter):
    """Expose API-retry output, then wedge the status probe forever."""

    def __init__(self):
        super().__init__(lambda: {"state": "working", "session_id": "sess-1"})
        self.status_calls = 0
        self.cancelled = threading.Event()

    def status(self, runner, session_id):
        self.status_calls += 1
        if self.status_calls == 1:
            return {"state": "working", "session_id": session_id}
        # Simulate a provider CLI retry that never returns control to the
        # monitor. The lifecycle watchdog must not depend on this unblocking.
        self.cancelled.wait()
        return {"state": "stopped", "session_id": session_id}

    def get_logs(self, runner, session_id):
        return "API error · Retrying in 1s · attempt 1/10"

    def cancel(self, runner, session_id):
        self.cancelled.set()
        return True


class _CountingCleanupAdapter(_ScriptedAdapter):
    def __init__(self):
        super().__init__(lambda: {"state": "working", "session_id": "sess-1"})
        self.cancel_calls = 0
        self._lock = threading.Lock()

    def cancel(self, runner, session_id):
        with self._lock:
            self.cancel_calls += 1
        time.sleep(0.03)
        return True


def _setup_job(store, *, question=None):
    job = store.create_job(profile="claude", operation="advice", transport="tmux")
    updates = {
        "interactive": True,
        "native_session_id": "sess-1",
        "status": "running",
    }
    output_path = job.path / "tmux-output.log"
    if question is not None:
        output_path.write_text(question, encoding="utf-8")
        updates["tmux_output_path"] = str(output_path)
    store.update_job_meta(job.job_id, updates)
    return job, output_path


def _wait_for_status(store, job_id, expected, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.job_status(job_id) == expected:
            return True
        time.sleep(0.01)
    return False


def _start_monitor(store, job, adapter, *, max_runtime_sec=None):
    thread = threading.Thread(
        target=lambda: _run_adapter_job(
            store,
            job.job_id,
            adapter,
            session_id="sess-1",
            poll_interval_sec=0.01,
            max_runtime_sec=max_runtime_sec,
        ),
        daemon=True,
    )
    thread.start()
    return thread


def _blocked():
    return {
        "state": "blocked",
        "waiting_for": "permission prompt",
        "session_id": "sess-1",
    }


def _done():
    return {"state": "done", "session_id": "sess-1"}


def _working():
    return {"state": "working", "session_id": "sess-1"}


def _input_generation(store, job) -> int:
    try:
        return int(store._read_job_meta(job.path).get("input_generation", 0))
    except (TypeError, ValueError):
        return 0


def test_monitor_survives_two_awaiting_input_cycles(tmp_path):
    store = JobStore(tmp_path)
    job, _ = _setup_job(store)
    state = {"working_served": False}

    def decide():
        # Driven off the durable ``input_generation`` counter — the same
        # source of truth ``send_user_input`` itself writes — rather than a
        # counter mutated by the test thread. A real CLI's native state can
        # only change as a *result* of keystrokes actually being delivered,
        # never before; polling a test-thread counter that is incremented
        # around (rather than durably tied to) the ``send_user_input`` call
        # lets the monitor observe "working" while the job is still durably
        # ``awaiting_input``, which can spuriously re-enter awaiting_input
        # with no matching reply and hang forever.
        if _input_generation(store, job) < 2:
            return _blocked()
        if not state["working_served"]:
            state["working_served"] = True
            return _working()
        return _done()

    thread = _start_monitor(store, job, _ScriptedAdapter(decide))

    assert _wait_for_status(store, job.job_id, "awaiting_input"), "first await never surfaced"
    assert store.send_user_input(job.job_id, "answer one", _sleep=lambda _: None)["ok"]
    assert _wait_for_status(store, job.job_id, "awaiting_input"), "second await never surfaced"
    assert store.send_user_input(job.job_id, "answer two", _sleep=lambda _: None)["ok"]

    thread.join(timeout=5)
    assert not thread.is_alive()
    result = store.get_result(job.job_id)
    assert result["status"] == "completed"


def test_background_watchdog_terminalizes_when_status_probe_hangs_after_api_retry(tmp_path):
    """A wedged provider status call cannot suppress deadline cleanup."""
    store = JobStore(tmp_path)
    job, _ = _setup_job(store)
    adapter = _RetryThenHungAdapter()

    thread = start_agent_job(
        store,
        job.job_id,
        adapter,
        session_id="sess-1",
        poll_interval_sec=0.01,
        max_runtime_sec=0.12,
    )

    assert _wait_for_status(store, job.job_id, "failed", timeout=3.0)
    thread.join(timeout=3)
    assert not thread.is_alive()

    result = store.get_result(job.job_id)
    assert result["stop_reason"] == "max_runtime_exceeded"
    assert result["failure"]["code"] == "max_runtime_exceeded"
    technical = result["technical"]
    assert technical["watchdog"] is True
    assert technical["cleanup_confirmed"] is True
    assert technical["lease_disposition"] == "released"
    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    assert any(event["type"] == "provider_api_retry" for event in events)
    assert any(event["type"] == "timeout" for event in events)
    assert adapter.status_calls >= 2


def test_deadline_watchdog_and_monitor_share_one_durable_terminalization_claim(tmp_path):
    """Concurrent deadline owners perform exactly one cleanup/result/event."""
    store = JobStore(tmp_path)
    job, _ = _setup_job(store)
    adapter = _CountingCleanupAdapter()
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def terminalize(watchdog: bool) -> None:
        barrier.wait(timeout=2)
        outcomes.append(
            _terminalize_runtime_deadline(
                store,
                job.job_id,
                adapter,
                session_id="sess-1",
                effective_max_runtime=1,
                watchdog=watchdog,
            )
        )

    threads = [
        threading.Thread(target=terminalize, args=(True,)),
        threading.Thread(target=terminalize, args=(False,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert sorted(outcomes) == [False, True]
    assert adapter.cancel_calls == 1
    assert store.job_status(job.job_id) == "failed"
    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    assert len([event for event in events if event["type"] == "timeout"]) == 1
    assert len([event for event in events if event["type"] == "result"]) == 1


@pytest.mark.parametrize("failure_surface", ["send_event", "events_write"])
def test_deadline_claim_survives_timeout_telemetry_failure(tmp_path, monkeypatch, failure_surface):
    """A timeout event I/O error cannot strand the claimed job without a result."""
    store = JobStore(tmp_path)
    job, _ = _setup_job(store)
    adapter = _CountingCleanupAdapter()

    if failure_surface == "send_event":
        original_send_event = store.send_event

        def fail_timeout_event(*args, **kwargs):
            if kwargs.get("type") == "timeout":
                raise OSError("events unavailable")
            return original_send_event(*args, **kwargs)

        monkeypatch.setattr(store, "send_event", fail_timeout_event)
    else:

        def fail_event_write(*_args, **_kwargs):
            raise OSError("events unavailable")

        monkeypatch.setattr(EventWriter, "write", fail_event_write)

    assert (
        _terminalize_runtime_deadline(
            store,
            job.job_id,
            adapter,
            session_id="sess-1",
            effective_max_runtime=1,
            watchdog=True,
        )
        is True
    )

    result = store.get_result(job.job_id)
    assert result["status"] == "failed"
    assert result["stop_reason"] == "max_runtime_exceeded"
    assert result["technical"]["cleanup_confirmed"] is True
    assert result["technical"]["lease_disposition"] == "released"
    telemetry = result["failure"]["diagnostics"]["telemetry_failure"]
    assert telemetry["event"] == "timeout"
    assert telemetry["error_type"]
    assert adapter.cancel_calls == 1


def test_monitor_persists_bounded_status_stall_before_its_deadline(tmp_path, monkeypatch):
    """A status probe timeout is durable even when the watchdog is not used."""
    monkeypatch.setattr("agent_crossbar.agent_runner._STATUS_PROBE_TIMEOUT_SEC", 0.02)
    store = JobStore(tmp_path)
    job, _ = _setup_job(store)
    adapter = _RetryThenHungAdapter()

    _run_adapter_job(
        store,
        job.job_id,
        adapter,
        session_id="sess-1",
        poll_interval_sec=0.005,
        max_runtime_sec=0.08,
    )

    result = store.get_result(job.job_id)
    assert result["stop_reason"] == "max_runtime_exceeded"
    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    status_timeouts = [event for event in events if event["type"] == "provider_status_timeout"]
    assert status_timeouts
    assert status_timeouts[0]["data"]["timeout_sec"] == 0.02


def test_monitor_preserves_full_question_text_over_2kib(tmp_path):
    long_question = "Please confirm: " + ("x" * 3000) + " END-OF-QUESTION\n⏺ waiting\n"
    store = JobStore(tmp_path)
    job, _ = _setup_job(store, question=long_question)
    state = {"sends": 0, "working_served": False}

    def decide():
        if state["sends"] == 0:
            return _blocked()
        if not state["working_served"]:
            state["working_served"] = True
            return _working()
        return _done()

    thread = _start_monitor(store, job, _ScriptedAdapter(decide))
    assert _wait_for_status(store, job.job_id, "awaiting_input")

    meta = store._read_job_meta(job.path)
    question = meta.get("question")
    assert question, "awaiting_input did not carry the full question text"
    assert len(question) > 2048, "question text was truncated to the 2 KiB prefix"
    assert "END-OF-QUESTION" in question
    # The native label is still present for compatibility.
    assert meta.get("waiting_for") == "permission prompt"

    state["sends"] += 1
    store.send_user_input(job.job_id, "proceed", _sleep=lambda _: None)
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_fast_reply_uses_transcript_growth_without_working_observation(tmp_path):
    store = JobStore(tmp_path)
    job, output_path = _setup_job(store, question="initial prompt\n")
    state = {"done_readings": 0}

    def decide():
        # Use the durable generation as the provider's turn boundary.  A
        # transient test flag can expose done before send_user_input commits
        # the reply and make the stale-done observation scheduler-dependent.
        if _input_generation(store, job) == 0:
            return _blocked()
        # The first post-reply "done" is the stale pre-reply reading; the
        # second one is backed by a Claude completion marker appended after
        # the durable pre-send transcript boundary.
        state["done_readings"] += 1
        if state["done_readings"] >= 2:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write("\n⏺ answered quickly\n❯\n⏵⏵ bypass permissions on\n")
        return _done()

    thread = _start_monitor(store, job, _ScriptedAdapter(decide))
    assert _wait_for_status(store, job.job_id, "awaiting_input")
    # Give the monitor a poll while the job is durably awaiting_input, so the
    # subsequent job_send is recognized as a resume.
    time.sleep(0.05)
    assert store.send_user_input(job.job_id, "quick answer", _sleep=lambda _: None)["ok"]

    thread.join(timeout=5)
    assert not thread.is_alive()
    result = store.get_result(job.job_id)
    assert result["status"] == "completed"

    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    assert any(event.get("type") == "stale_done_ignored" for event in events), (
        "expected the stale pre-reply done reading to be ignored"
    )


def test_repeated_done_and_echo_only_never_complete_new_turn(tmp_path):
    """A stale done frame and an echoed user line are not completion proof."""
    store = JobStore(tmp_path)
    job, output_path = _setup_job(store, question="initial prompt\n")
    state = {"blocked_calls": 0, "done_readings": 0}
    stale_blocked_status_started = threading.Event()
    release_stale_blocked_status = threading.Event()

    def decide():
        # Drive the scripted provider from the same durable resume boundary
        # the monitor observes.  Hold the second stale blocked status call so
        # the owner reply is committed while status() is still returning the
        # pre-reply state; this deterministically exercises the ordering guard.
        if _input_generation(store, job) == 0:
            state["blocked_calls"] += 1
            if state["blocked_calls"] >= 2:
                stale_blocked_status_started.set()
                release_stale_blocked_status.wait(timeout=5)
            return _blocked()
        state["done_readings"] += 1
        if state["done_readings"] == 2:
            # This is only the TUI's echo/redraw of the submitted input.
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write("\n❯ quick answer\n")
        if state["done_readings"] >= 4:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write("\n⏺ fresh answer\n❯\n⏵⏵ bypass permissions on\n")
        return _done()

    thread = _start_monitor(store, job, _ScriptedAdapter(decide))
    assert _wait_for_status(store, job.job_id, "awaiting_input")
    assert stale_blocked_status_started.wait(timeout=5)
    try:
        send_result = store.send_user_input(job.job_id, "quick answer", _sleep=lambda _: None)
    finally:
        release_stale_blocked_status.set()
    assert send_result["ok"]
    thread.join(timeout=5)
    assert not thread.is_alive()
    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    assert any(event.get("type") == "stale_blocked_ignored" for event in events)
    assert sum(event.get("type") == "stale_done_ignored" for event in events) >= 3
    assert store.get_result(job.job_id)["status"] == "completed"


def test_monitor_accepts_screen_reader_final_after_resume(tmp_path):
    """A real screen-reader final must end stale-done polling after reply."""
    store = JobStore(tmp_path)
    job, output_path = _setup_job(store, question="initial prompt\n")
    state = {"done_readings": 0}

    def decide():
        if _input_generation(store, job) == 0:
            return _blocked()
        state["done_readings"] += 1
        if state["done_readings"] == 2:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(
                    "\n$claude: FINAL: Барні\n"
                    "Calculating…\n"
                    "auto mode on (shift+tab to cycle) · esc to interrupt\n"
                    "36502 tokens\n"
                    "effort: medium · /effort\n"
                    "$Cogitated for 7s\n"
                    "auto mode on (shift+tab to cycle)\n"
                )
        return _done()

    thread = _start_monitor(store, job, _ScriptedAdapter(decide))
    assert _wait_for_status(store, job.job_id, "awaiting_input")
    assert store.send_user_input(job.job_id, "quick answer", _sleep=lambda _: None)["ok"]
    thread.join(timeout=5)
    assert not thread.is_alive()

    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    assert any(event.get("type") == "stale_done_ignored" for event in events)
    assert store.get_result(job.job_id)["status"] == "completed"


def test_monitor_accepts_post_reply_screen_reader_final_while_native_state_stays_blocked(
    tmp_path,
):
    """A complete post-reply transcript wins over stale native ``blocked``."""
    store = JobStore(tmp_path)
    job, output_path = _setup_job(store, question="initial prompt\n")
    appended = {"value": False}

    def decide():
        if _input_generation(store, job) == 0:
            return _blocked()
        if not appended["value"]:
            output_path.write_text(
                "initial prompt\n$claude: FINAL: Спрінт\nBrewed for 12s\n",
                encoding="utf-8",
            )
            appended["value"] = True
        return _blocked()

    thread = _start_monitor(store, job, _ScriptedAdapter(decide))
    assert _wait_for_status(store, job.job_id, "awaiting_input")
    assert store.send_user_input(job.job_id, "continue", _sleep=lambda _: None)["ok"]
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert store.get_result(job.job_id)["status"] == "completed"
    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    assert any(event.get("type") == "blocked_output_finalized" for event in events)


@pytest.mark.parametrize(
    "suffix",
    [
        "auto mode on (shift+tab to cycle)\n",
        "Brewed for 12s\nesc to interrupt\n",
    ],
    ids=["no-footer", "busy-after-footer"],
)
def test_monitor_does_not_finalize_blocked_without_clean_post_reply_footer(tmp_path, suffix):
    """No footer or later busy evidence must remain fail-closed."""
    store = JobStore(tmp_path)
    job, output_path = _setup_job(store, question="initial prompt\n")
    appended = {"value": False}

    def decide():
        if _input_generation(store, job) == 0:
            return _blocked()
        if not appended["value"]:
            output_path.write_text(
                "initial prompt\n$claude: FINAL: Спрінт\n" + suffix,
                encoding="utf-8",
            )
            appended["value"] = True
        return _blocked()

    thread = _start_monitor(store, job, _ScriptedAdapter(decide), max_runtime_sec=0.08)
    assert _wait_for_status(store, job.job_id, "awaiting_input")
    assert store.send_user_input(job.job_id, "continue", _sleep=lambda _: None)["ok"]
    thread.join(timeout=5)
    assert not thread.is_alive()
    result = store.get_result(job.job_id)
    assert result["status"] == "failed"
    assert result["stop_reason"] == "max_runtime_exceeded"
    events = store.job_tail(job.job_id, max_bytes=200000)["events"]
    assert not any(event.get("type") == "blocked_output_finalized" for event in events)
