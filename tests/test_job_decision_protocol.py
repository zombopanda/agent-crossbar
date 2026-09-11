"""Protocol-shaped tests for the shared ``job_send`` decision contract.

Covers correlation, ownership/wildcard, staleness, duplicate delivery,
timeout/cancel settlement, restart fail-closed, and waiter exposure.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from datetime import datetime, timedelta, timezone

import pytest

from agent_crossbar.jobs import JobStore
from agent_crossbar.pending_permissions import pending_permissions


@pytest.fixture(autouse=True)
def _clean_registry():
    pending_permissions.clear()
    yield
    pending_permissions.clear()


def _make_job(store, *, transport="acp", client_session_id=None):
    return store.create_job(
        profile="opencode",
        operation="dev",
        transport=transport,
        client_session_id=client_session_id,
    )


def _set_pending(store, job, request_id="req-1", decisions=("allow", "reject"), state="pending"):
    store.update_job_meta(
        job.job_id,
        {
            "status": "awaiting_input" if state == "pending" else "running",
            "waiting_for": "permission",
            "pending_request": {
                "request_id": request_id,
                "kind": "other",
                "decisions": list(decisions),
                "state": state,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    )


def _live(job_id, request_id):
    loop = asyncio.new_event_loop()
    pending = pending_permissions.register(job_id, request_id, loop=loop)
    return loop, pending


def _decision(request_id, decision):
    return json.dumps({"request_id": request_id, "decision": decision})


def _send_from_controller_process(state_root, job_id, request_id, queue):
    """Controller-side job_send entrypoint for the IPC handoff test."""
    store = JobStore(state_root)
    queue.put(store.send_user_input(job_id, _decision(request_id, "allow")))


# ── syntax-first parsing / plain-text fallback ──────────────────────────


def _tmux_job(store, monkeypatch, *, client_session_id=None):
    monkeypatch.setattr(
        "agent_crossbar.jobs.subprocess.run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0})(),
    )
    return store.create_job(
        profile="claude",
        operation="advice",
        transport="tmux",
        client_session_id=client_session_id,
    )


def test_plain_text_is_not_a_decision_even_with_pending_request(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = _tmux_job(store, monkeypatch)
    loop, _ = _live(job.job_id, "req-1")
    try:
        _set_pending(store, job, "req-1")
        result = store.send_user_input(job.job_id, "continue please", _sleep=lambda _: None)
        # Plain text resumes the job; it does not resolve the permission.
        assert result["ok"] is True
        meta = store._read_job_meta(job.path)
        assert meta.get("pending_request", {}).get("state") == "pending"
    finally:
        loop.close()


def test_malformed_json_falls_back_to_plain_text(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = _tmux_job(store, monkeypatch)
    result = store.send_user_input(job.job_id, "{not valid json", _sleep=lambda _: None)
    assert result["ok"] is True
    assert "request_not_pending" not in json.dumps(result)


def test_json_object_without_request_fields_falls_back_to_plain_text(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = _tmux_job(store, monkeypatch)
    result = store.send_user_input(
        job.job_id, json.dumps({"hello": "world"}), _sleep=lambda _: None
    )
    assert result["ok"] is True


# ── correlation / staleness / duplicates ────────────────────────────────


def test_allow_decision_resolves_pending_request(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    loop, pending = _live(job.job_id, "req-1")
    try:
        _set_pending(store, job, "req-1")
        result = store.send_user_input(job.job_id, _decision("req-1", "allow"))
        assert result["ok"] is True
        assert result["request_id"] == "req-1"
        assert result["decision"] == "allow"
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.result() == "allow"
        # Durable pending record is cleared.
        assert "pending_request" not in store._read_job_meta(job.path)
    finally:
        loop.close()


def test_unknown_request_id_is_rejected(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1")
    result = store.send_user_input(job.job_id, _decision("nope", "allow"))
    assert result["ok"] is False
    assert result["error"] == "request_not_pending"
    # The real pending request is untouched.
    assert store._read_job_meta(job.path)["pending_request"]["state"] == "pending"


def test_invalid_decision_value_is_rejected_and_stays_pending(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    loop, _ = _live(job.job_id, "req-1")
    try:
        _set_pending(store, job, "req-1")
        result = store.send_user_input(job.job_id, _decision("req-1", "allow_always"))
        assert result["ok"] is False
        assert result["error"] == "invalid_decision"
        assert store._read_job_meta(job.path)["pending_request"]["state"] == "pending"
    finally:
        loop.close()


def test_duplicate_decision_does_not_double_apply(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    loop, pending = _live(job.job_id, "req-1")
    try:
        _set_pending(store, job, "req-1")
        first = store.send_user_input(job.job_id, _decision("req-1", "allow"))
        second = store.send_user_input(job.job_id, _decision("req-1", "reject"))
        assert first["ok"] is True
        assert second["ok"] is False
        assert second["error"] == "request_not_pending"
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.result() == "allow"
    finally:
        loop.close()


def test_terminal_job_rejects_decision_as_not_pending(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    store.update_job_meta(job.job_id, {"status": "stopped", "stop_reason": "user_cancelled"})
    result = store.send_user_input(job.job_id, _decision("req-1", "allow"))
    assert result["ok"] is False
    assert result["error"] == "request_not_pending"


# ── ownership / wildcard ────────────────────────────────────────────────


def test_foreign_session_decision_is_denied(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store, client_session_id="thread-a")
    loop, _ = _live(job.job_id, "req-1")
    try:
        _set_pending(store, job, "req-1")
        result = store.send_user_input(
            job.job_id, _decision("req-1", "allow"), client_session_id="thread-b"
        )
        assert result["error"] == "job_not_found"
    finally:
        loop.close()


def test_wildcard_allows_cross_session_decision(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store, client_session_id="thread-a")
    loop, pending = _live(job.job_id, "req-1")
    try:
        _set_pending(store, job, "req-1")
        result = store.send_user_input(
            job.job_id, _decision("req-1", "allow"), client_session_id="*"
        )
        assert result["ok"] is True
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.result() == "allow"
    finally:
        loop.close()


# ── restart fail-closed ─────────────────────────────────────────────────


def test_durable_pending_without_live_callback_fails_closed_and_cleans_up(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    # Durable pending exists, but nothing is registered in this process
    # (simulating a restart).
    _set_pending(store, job, "req-1")
    result = store.send_user_input(job.job_id, _decision("req-1", "allow"))
    assert result["ok"] is False
    assert result["error"] == "request_not_pending"
    meta = store._read_job_meta(job.path)
    assert meta["pending_request"]["state"] == "expired"


def test_durable_decision_is_accepted_when_provider_process_is_live(tmp_path):
    """A controller process may resolve a request owned by another process."""
    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1")
    # Simulate the ACP child still being alive while this controller has no
    # in-process Future for it. The provider coroutine polls this durable
    # inbox and applies the decision exactly once.
    from agent_crossbar.jobs import _process_start_identity

    store.update_job_meta(
        job.job_id,
        {
            "acp_pid": os.getpid(),
            "acp_process_start": _process_start_identity(os.getpid()),
            "acp_pgid": os.getpgid(os.getpid()),
        },
    )
    result = store.send_user_input(job.job_id, _decision("req-1", "allow"))
    assert result["ok"] is True
    pending = store._read_job_meta(job.path)["pending_request"]
    assert pending["state"] == "resolved"
    assert pending["decision"] == "allow"


def test_durable_decision_crosses_real_controller_process_once(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1")
    from agent_crossbar.jobs import _process_start_identity

    # The parent stands in for the ACP process holding the live callback;
    # only the controller-side JobStore runs in the child process.
    store.update_job_meta(
        job.job_id,
        {
            "acp_pid": os.getpid(),
            "acp_process_start": _process_start_identity(os.getpid()),
            "acp_pgid": os.getpgid(os.getpid()),
        },
    )
    loop, pending = _live(job.job_id, "req-1")
    try:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(
            target=_send_from_controller_process,
            args=(str(tmp_path), job.job_id, "req-1", queue),
        )
        child.start()
        child.join(timeout=5)
        assert child.exitcode == 0
        assert queue.get(timeout=2)["ok"] is True
        # The provider-side poll consumes the durable inbox and wakes the
        # callback exactly once, even though the decision writer was separate.
        assert store._read_job_meta(job.path)["pending_request"]["state"] == "resolved"
        assert pending_permissions.resolve(job.job_id, "req-1", "allow")["ok"] is True
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.result() == "allow"
        assert pending_permissions.resolve(job.job_id, "req-1", "reject")["error"] == (
            "request_not_pending"
        )
    finally:
        loop.close()


def test_recycled_pid_without_matching_start_identity_fails_closed(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1")
    store.update_job_meta(
        job.job_id,
        {"acp_pid": os.getpid(), "acp_process_start": "recycled-process"},
    )
    result = store.send_user_input(job.job_id, _decision("req-1", "allow"))
    assert result["ok"] is False
    assert result["error"] == "request_not_pending"
    assert store._read_job_meta(job.path)["pending_request"]["state"] == "expired"


def test_matching_start_identity_but_wrong_process_group_fails_closed(tmp_path):
    """Start-time alone has only one-second resolution on macOS; a process
    group mismatch is a second, independent identity check that must also
    fail closed rather than being ignored once the start token matches."""
    from agent_crossbar.jobs import _process_start_identity

    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1")
    store.update_job_meta(
        job.job_id,
        {
            "acp_pid": os.getpid(),
            "acp_process_start": _process_start_identity(os.getpid()),
            "acp_pgid": os.getpgid(os.getpid()) + 9999,
        },
    )
    result = store.send_user_input(job.job_id, _decision("req-1", "allow"))
    assert result["ok"] is False
    assert result["error"] == "request_not_pending"
    assert store._read_job_meta(job.path)["pending_request"]["state"] == "expired"


def test_pgid_equal_to_pid_is_accepted_as_setsid_handoff(tmp_path):
    """The repository-owned ACP wrapper calls setsid() after spawn, so a
    live process can legitimately become its own group leader (pgid == pid)
    even when a different pgid was recorded at launch time (the pre-setsid
    group observed by the SDK callback). That expected handoff must be
    accepted, not treated as process-group reuse."""
    import agent_crossbar.jobs as jobs_module
    from agent_crossbar.jobs import _process_start_identity

    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1")
    pid = os.getpid()
    store.update_job_meta(
        job.job_id,
        {
            "acp_pid": pid,
            "acp_process_start": _process_start_identity(pid),
            # A pgid deliberately different from both the live pgid and the
            # pid itself, so the match can only succeed through the
            # ``actual_pgid == pid`` handoff branch, not a coincidental
            # equality with the recorded value.
            "acp_pgid": pid + 424242,
            # Explicitly identify this as the repository-owned pre-setsid
            # group handoff; arbitrary mismatches must remain fail-closed.
            "acp_parent_pgid": pid + 424242,
        },
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jobs_module, "_process_group_id", lambda _pid: pid)
        result = store.send_user_input(job.job_id, _decision("req-1", "allow"))
    assert result["ok"] is True


# ── timeout / cancel settlement ─────────────────────────────────────────


def test_timeout_settles_pending_and_late_decision_cannot_resurrect(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    loop, pending = _live(job.job_id, "req-1")
    try:
        started = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        store.update_job_meta(
            job.job_id,
            {"started_at": started, "max_runtime_sec": 1, "backend": "acp"},
        )
        _set_pending(store, job, "req-1")
        assert store._reap_deadline_expired_job(job) is True
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.cancelled()
        assert store.job_status(job.job_id) == "failed"
        late = store.send_user_input(job.job_id, _decision("req-1", "allow"))
        assert late["ok"] is False
        assert late["error"] == "request_not_pending"
    finally:
        loop.close()


def test_stop_settles_pending_and_late_decision_cannot_resurrect(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    loop, pending = _live(job.job_id, "req-1")
    try:
        _set_pending(store, job, "req-1")
        stopped = store.stop_job(job.job_id, reason="user_cancelled")
        assert stopped["ok"] is True
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.cancelled()
        late = store.send_user_input(job.job_id, _decision("req-1", "allow"))
        assert late["ok"] is False
        assert late["error"] == "request_not_pending"
        assert store.job_status(job.job_id) == "stopped"
    finally:
        loop.close()


# ── waiter exposure / lease ─────────────────────────────────────────────


def test_job_tail_exposes_pending_request(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1")
    tail = store.job_tail(job.job_id)
    assert tail["ok"] is True
    assert tail["pending_request"]["request_id"] == "req-1"
    assert tail["pending_request"]["kind"] == "other"
    assert tail["pending_request"]["decisions"] == ["allow", "reject"]


def test_job_tail_hides_settled_request(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    _set_pending(store, job, "req-1", state="cancelled")
    tail = store.job_tail(job.job_id)
    assert tail["pending_request"] is None


def test_awaiting_input_is_reaped_by_tail_and_sweep_after_deadline(tmp_path):
    store = JobStore(tmp_path)
    job = _make_job(store)
    started = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    store.update_job_meta(
        job.job_id,
        {
            "started_at": started,
            "max_runtime_sec": 1,
            "backend": "acp",
        },
    )
    _set_pending(store, job, "req-1")
    tail = store.job_tail(job.job_id)
    assert tail["status"] == "failed"
    assert store._read_job_meta(job.path)["pending_request"]["state"] == "expired"

    # A second overdue awaiting_input job is found by the public sweep too.
    second = _make_job(store)
    store.update_job_meta(
        second.job_id,
        {"started_at": started, "max_runtime_sec": 1, "backend": "acp"},
    )
    _set_pending(store, second, "req-2")
    assert store.reap_expired_jobs() == 1
    assert store.job_status(second.job_id) == "failed"


def test_writer_lease_retained_while_pending(tmp_path):
    from agent_crossbar.writer_lease import WriterLeaseStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    leases = WriterLeaseStore(state)
    pending_lease = leases.acquire(str(workspace), owner_id="owner", owner_kind="pending_dev")
    assert pending_lease.ok and pending_lease.token

    store = JobStore(state)
    job = _make_job(store, transport="print")
    assert leases.attach(pending_lease.token, job_id=job.job_id)
    store.update_job_meta(job.job_id, {"writer_lease_token": pending_lease.token})
    _set_pending(store, job, "req-1")

    store.job_tail(job.job_id)
    # A pending request must not release the writer lease early.
    assert leases.acquire(str(workspace), owner_id="other").error == "writer_busy"
