"""Regression tests for provider-neutral dev writer serialization."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from agent_crossbar.writer_lease import RECOVERY_ACKNOWLEDGEMENT, WriterLeaseStore


def test_canonical_cwd_identity_collapses_relative_and_symlink_paths(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    store = WriterLeaseStore(tmp_path / "state")

    first = store.acquire(str(workspace), owner_id="external-job")
    assert first.ok is True
    assert first.canonical_cwd == str(workspace.resolve())

    second = store.acquire(str(alias / ".." / alias.name), owner_id="local-fallback")
    assert second.ok is False
    assert second.error == "writer_busy"
    assert second.canonical_cwd == first.canonical_cwd


def test_same_cwd_race_allows_one_owner_across_processes(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)

    def acquire(owner: str, queue) -> None:
        barrier.wait()
        result = WriterLeaseStore(state).acquire(str(workspace), owner_id=owner)
        queue.put((result.ok, result.error))

    queue = context.Queue()
    processes = [
        context.Process(target=acquire, args=(f"job-{index}", queue)) for index in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)

    assert sorted(results) == [(False, "writer_busy"), (True, None)]


def test_thread_race_does_not_depend_on_process_local_lock(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state")
    barrier = threading.Barrier(2)
    results = []

    def acquire(owner: str) -> None:
        barrier.wait()
        results.append(store.acquire(str(workspace), owner_id=owner))

    threads = [threading.Thread(target=acquire, args=(f"job-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted((result.ok, result.error) for result in results) == [
        (False, "writer_busy"),
        (True, None),
    ]


def test_terminal_job_reconciliation_releases_external_lease(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    jobs = state / "jobs" / "123456789-job"
    jobs.mkdir(parents=True)
    (jobs / "meta.json").write_text('{"status":"succeeded","cwd":"%s"}\n' % workspace)
    (jobs / "result.json").write_text('{"ok":true,"summary":"done"}\n')
    store = WriterLeaseStore(state)
    lease = store.acquire(str(workspace), owner_id="123456789-job", owner_kind="external_job")
    assert lease.ok is True
    assert store.reconcile() == 1
    assert store.acquire(str(workspace), owner_id="next-job").ok is True


@pytest.mark.parametrize("job_meta", ["running", "missing", "corrupt"])
def test_nonterminal_or_unverifiable_external_job_is_never_age_reconciled(
    tmp_path: Path, job_meta: str
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    job_id = "123456789-job"
    job_dir = state / "jobs" / job_id
    if job_meta != "missing":
        job_dir.mkdir(parents=True)
        (job_dir / "meta.json").write_text(
            '{"status":"running"}\n' if job_meta == "running" else "not-json\n"
        )
    store = WriterLeaseStore(state, stale_after_sec=1)
    lease = store.acquire(str(workspace), owner_id=job_id, owner_kind="external_job")
    assert lease.ok is True and lease.token
    lease_path = store.lease_path(lease.canonical_cwd)
    payload = json.loads(lease_path.read_text())
    payload["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    lease_path.write_text(json.dumps(payload))
    old = time.time() - 120
    os.utime(lease_path, (old, old))

    assert store.reconcile() == 0
    blocked = store.acquire(str(workspace), owner_id="next-job")
    assert blocked.ok is False
    assert blocked.error == "writer_busy"


def test_corrupt_lease_state_fails_closed(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state")
    lease_path = store.lease_path(str(workspace))
    lease_path.parent.mkdir(parents=True)
    lease_path.write_text("not-json\n")

    result = store.acquire(str(workspace), owner_id="next-job")
    assert result.ok is False
    assert result.error == "writer_lease_corrupt"


def test_explicit_recovery_can_clear_only_missing_external_job(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state")
    lease = store.acquire(str(workspace), owner_id="123456789-missing", owner_kind="external_job")
    assert lease.ok is True
    refused = store.recover(str(workspace), acknowledgement="wrong")
    assert refused.ok is False
    assert refused.error == "writer_recovery_confirmation_required"
    recovered = store.recover(str(workspace), acknowledgement=RECOVERY_ACKNOWLEDGEMENT)
    assert recovered.ok is True
    assert store.acquire(str(workspace), owner_id="next-job").ok is True


def test_explicit_recovery_never_clears_nonterminal_external_job(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    job_id = "123456789-running"
    job_dir = state / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text('{"status":"running"}\n')
    store = WriterLeaseStore(state)
    lease = store.acquire(str(workspace), owner_id=job_id, owner_kind="external_job")
    assert lease.ok is True
    refused = store.recover(str(workspace), acknowledgement=RECOVERY_ACKNOWLEDGEMENT)
    assert refused.ok is False
    assert refused.error == "writer_recovery_unsafe"
    assert store.acquire(str(workspace), owner_id="next-job").error == "writer_busy"


def test_root_integration_lease_serializes_external_dev_writer(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state")
    root = store.acquire(str(workspace), owner_id="root", owner_kind="root_integration")
    assert root.ok is True
    blocked = store.acquire(str(workspace), owner_id="123456789-job", owner_kind="external_job")
    assert blocked.ok is False
    assert blocked.error == "writer_busy"
    assert store.release(root.token) is True
    assert (
        store.acquire(str(workspace), owner_id="123456789-job", owner_kind="external_job").ok
        is True
    )


def test_agent_start_dev_rejects_root_integration_lease_before_provider_launch(
    tmp_path: Path, monkeypatch
):
    from agent_crossbar import server
    from agent_crossbar.readiness import ReadinessResult

    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = WriterLeaseStore(state).acquire(
        str(workspace), owner_id="root", owner_kind="root_integration"
    )
    assert root.ok is True
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(state))
    monkeypatch.setattr(
        "agent_crossbar.readiness.probe_profile",
        lambda *args, **kwargs: ReadinessResult(
            profile="reasonix",
            state="ready",
            support_tier="experimental",
            authenticated=True,
        ),
    )
    launched = []
    monkeypatch.setattr(server, "start_print_job", lambda *args, **kwargs: launched.append(True))

    result = server.agent_start(
        profile="reasonix",
        prompt="change the workspace",
        model="deepseek-v4-flash",
        task="dev",
        cwd=str(workspace),
    )
    assert result["ok"] is False
    assert result["error"] == "writer_busy"
    assert launched == []


def test_stale_root_integration_lease_reconciles(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state", stale_after_sec=1)
    root = store.acquire(str(workspace), owner_id="root", owner_kind="root_integration")
    assert root.ok is True
    lease_path = store.lease_path(root.canonical_cwd)
    payload = json.loads(lease_path.read_text())
    payload["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    lease_path.write_text(json.dumps(payload))
    os.utime(lease_path, (time.time() - 120, time.time() - 120))

    assert store.reconcile() == 1
    assert store.acquire(str(workspace), owner_id="next", owner_kind="local").ok is True


def test_stale_lease_is_reconciled_without_stranding_cwd(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state", stale_after_sec=1)
    lease = store.acquire(str(workspace), owner_id="dead-job", owner_kind="pending_dev")
    assert lease.ok is True
    lease_path = store.lease_path(lease.canonical_cwd)
    payload = json.loads(lease_path.read_text())
    payload["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    lease_path.write_text(json.dumps(payload))
    old = time.time() - 120
    os.utime(lease_path, (old, old))

    assert store.reconcile() == 1
    next_lease = store.acquire(str(workspace), owner_id="next-job")
    assert next_lease.ok is True


def test_stale_heartbeat_is_reconciled_even_if_file_mtime_is_fresh(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state", stale_after_sec=1)
    lease = store.acquire(str(workspace), owner_id="dead-job", owner_kind="local")
    assert lease.ok is True
    lease_path = store.lease_path(lease.canonical_cwd)
    payload = json.loads(lease_path.read_text())
    payload["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    lease_path.write_text(json.dumps(payload))

    assert store.reconcile() == 1


def test_local_fallback_and_external_agent_share_lease_state(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state")

    local = store.acquire(str(workspace), owner_id="controller-1", owner_kind="local")
    assert local.ok is True
    blocked = WriterLeaseStore(tmp_path / "state").acquire(
        str(workspace), owner_id="external-job", owner_kind="external_job"
    )
    assert blocked.ok is False
    assert blocked.error == "writer_busy"

    assert store.release(local.token) is True
    external = store.acquire(str(workspace), owner_id="external-job", owner_kind="external_job")
    assert external.ok is True
    blocked_local = store.acquire(str(workspace), owner_id="controller-2", owner_kind="local")
    assert blocked_local.ok is False
    assert store.release(external.token) is True


@pytest.mark.parametrize("terminal_action", ["result", "stop"])
def test_jobstore_releases_lease_only_when_job_becomes_terminal(
    tmp_path: Path, terminal_action: str
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    leases = WriterLeaseStore(state)
    lease = leases.acquire(str(workspace), owner_id="pending", owner_kind="pending_dev")
    assert lease.ok is True and lease.token

    from agent_crossbar.jobs import JobStore

    jobs = JobStore(state)
    job = jobs.create_job("opencode", "dev", transport="print", cwd=str(workspace))
    assert leases.attach(lease.token, job_id=job.job_id)
    jobs.update_job_meta(job.job_id, {"writer_lease_token": lease.token})
    assert leases.acquire(str(workspace), owner_id="blocked").error == "writer_busy"

    if terminal_action == "result":
        assert jobs.set_result(job.job_id, ok=True, summary="done")["ok"] is True
    else:
        assert jobs.stop_job(job.job_id, reason="test")["ok"] is True
    assert leases.acquire(str(workspace), owner_id="next").ok is True


def test_jobstore_heartbeat_keeps_long_running_lease_fresh(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    leases = WriterLeaseStore(state)
    lease = leases.acquire(str(workspace), owner_id="pending", owner_kind="pending_dev")
    assert lease.ok and lease.token

    from agent_crossbar.jobs import JobStore

    jobs = JobStore(state)
    job = jobs.create_job("opencode", "dev", transport="print", cwd=str(workspace))
    assert leases.attach(lease.token, job_id=job.job_id)
    jobs.update_job_meta(job.job_id, {"writer_lease_token": lease.token})
    assert jobs.heartbeat_writer_lease(job.job_id) is True
    assert leases.acquire(str(workspace), owner_id="next").error == "writer_busy"


def test_agent_start_rejects_busy_dev_before_provider_launch(tmp_path: Path, monkeypatch):
    from agent_crossbar import server
    from agent_crossbar.readiness import ReadinessResult

    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path / "state"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    held = WriterLeaseStore(tmp_path / "state").acquire(
        str(workspace), owner_id="controller", owner_kind="local"
    )
    assert held.ok is True
    launched = []
    monkeypatch.setattr(
        server,
        "start_print_job",
        lambda *args, **kwargs: launched.append(True),
    )
    monkeypatch.setattr(
        "agent_crossbar.readiness.probe_profile",
        lambda *args, **kwargs: ReadinessResult(
            profile="reasonix",
            state="ready",
            support_tier="experimental",
            authenticated=True,
        ),
    )

    result = server.agent_start(
        profile="reasonix",
        prompt="change the workspace",
        model="deepseek-v4-flash",
        task="dev",
        cwd=str(workspace),
    )
    assert result["ok"] is False
    assert result["error"] == "writer_busy"
    assert launched == []


def test_agent_start_dev_releases_lease_after_terminal_provider_result(tmp_path: Path, monkeypatch):
    from agent_crossbar import server
    from agent_crossbar.readiness import ReadinessResult

    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(state))
    monkeypatch.setattr(
        "agent_crossbar.readiness.probe_profile",
        lambda *args, **kwargs: ReadinessResult(
            profile="reasonix",
            state="ready",
            support_tier="experimental",
            authenticated=True,
        ),
    )

    def finish(store, job_id, req, **kwargs):
        return store.set_result(job_id, True, summary="terminal")

    monkeypatch.setattr(server, "start_print_job", finish)
    result = server.agent_start(
        profile="reasonix",
        prompt="change the workspace",
        model="deepseek-v4-flash",
        task="dev",
        cwd=str(workspace),
    )
    assert result["ok"] is True
    assert WriterLeaseStore(state).acquire(str(workspace), owner_id="next").ok is True


def test_reaper_releases_external_lease_unblocking_fallback(tmp_path: Path):
    """A deadline-expired orphaned dev job's lease must be released when the
    reaper terminalizes it, so a replacement writer is admitted only after the
    preceding job reached its terminal result."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    leases = WriterLeaseStore(state)
    lease = leases.acquire(str(workspace), owner_id="pending", owner_kind="pending_dev")
    assert lease.ok and lease.token

    from agent_crossbar.jobs import JobStore

    store = JobStore(state)
    orphan = store.create_job("opencode", "dev", transport="print", cwd=str(workspace))
    assert leases.attach(lease.token, job_id=orphan.job_id)
    store.update_job_meta(
        orphan.job_id,
        {
            "writer_lease_token": lease.token,
            "started_at": "2000-01-01T00:00:00+00:00",
            "max_runtime_sec": 1,
        },
    )

    # Nonterminal predecessor blocks the replacement writer.
    assert leases.acquire(str(workspace), owner_id="blocked").error == "writer_busy"

    # Reaping the orphaned predecessor unblocks the fallback.
    assert store.reap_expired_jobs(cwd=str(workspace)) == 1
    assert store.job_status(orphan.job_id) == "failed"
    assert leases.acquire(str(workspace), owner_id="fallback").ok is True


def test_reaper_keeps_lease_while_live_predecessor_is_nonterminal(tmp_path: Path):
    """A live dev job within its declared deadline must keep its lease — the
    fallback stays blocked until the preceding job is terminal."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    leases = WriterLeaseStore(state)
    lease = leases.acquire(str(workspace), owner_id="pending", owner_kind="pending_dev")
    assert lease.ok and lease.token

    from agent_crossbar.jobs import JobStore

    store = JobStore(state)
    live = store.create_job("opencode", "dev", transport="print", cwd=str(workspace))
    assert leases.attach(lease.token, job_id=live.job_id)
    store.update_job_meta(
        live.job_id,
        {
            "writer_lease_token": lease.token,
            "started_at": "2099-01-01T00:00:00+00:00",
            "max_runtime_sec": 3600,
        },
    )

    assert store.reap_expired_jobs(cwd=str(workspace)) == 0
    assert store.job_status(live.job_id) == "running"
    assert leases.acquire(str(workspace), owner_id="blocked").error == "writer_busy"


def test_agent_start_dev_reaps_expired_orphan_then_admits_fallback(tmp_path: Path, monkeypatch):
    """agent_start(task='dev') admission reaps a deadline-expired orphaned
    predecessor for the same workspace before acquiring the writer lease, so a
    replacement job is admitted without breaking the writer-busy gate for a
    genuinely live predecessor."""
    from agent_crossbar import server
    from agent_crossbar.jobs import JobStore
    from agent_crossbar.readiness import ReadinessResult

    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(state))
    monkeypatch.setattr(
        "agent_crossbar.readiness.probe_profile",
        lambda *args, **kwargs: ReadinessResult(
            profile="reasonix",
            state="ready",
            support_tier="experimental",
            authenticated=True,
        ),
    )

    leases = WriterLeaseStore(state)
    lease = leases.acquire(str(workspace), owner_id="pending", owner_kind="pending_dev")
    assert lease.ok and lease.token
    store = JobStore(state)
    orphan = store.create_job("reasonix", "dev", transport="print", cwd=str(workspace))
    assert leases.attach(lease.token, job_id=orphan.job_id)
    store.update_job_meta(
        orphan.job_id,
        {
            "writer_lease_token": lease.token,
            "started_at": "2000-01-01T00:00:00+00:00",
            "max_runtime_sec": 1,
        },
    )

    launched = []
    monkeypatch.setattr(
        server,
        "start_print_job",
        lambda *args, **kwargs: launched.append(True),
    )
    result = server.agent_start(
        profile="reasonix",
        prompt="fallback work",
        model="deepseek-v4-flash",
        task="dev",
        cwd=str(workspace),
    )
    assert result["ok"] is True, f"fallback admission failed: {result}"
    assert launched == [True]
    assert store.job_status(orphan.job_id) == "failed"
