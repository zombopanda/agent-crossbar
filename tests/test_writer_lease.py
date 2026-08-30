"""Regression tests for provider-neutral dev writer serialization."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from agent_crossbar.writer_lease import WriterLeaseStore


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
    store = WriterLeaseStore(state)
    lease = store.acquire(str(workspace), owner_id="123456789-job", owner_kind="external_job")
    assert lease.ok is True
    assert store.reconcile() == 1
    assert store.acquire(str(workspace), owner_id="next-job").ok is True


def test_stale_lease_is_reconciled_without_stranding_cwd(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WriterLeaseStore(tmp_path / "state", stale_after_sec=1)
    lease = store.acquire(str(workspace), owner_id="dead-job")
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
    lease = store.acquire(str(workspace), owner_id="dead-job")
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
