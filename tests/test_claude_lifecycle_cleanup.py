from __future__ import annotations

from types import SimpleNamespace

from agent_crossbar.adapters.claude import LaunchResult
from agent_crossbar.adapters.claude_lifecycle import start_claude_job
from agent_crossbar.jobs import JobStore


class _Adapter:
    name = "claude"

    def __init__(self, *, cancel_result: bool):
        self.cancel_result = cancel_result
        self.cancel_calls: list[str] = []

    def check_readiness(self, runner):
        return SimpleNamespace(authenticated=True)

    def launch(self, runner, **kwargs):
        return LaunchResult("deadbeef", "claude_bg")

    def cancel(self, runner, session_id):
        self.cancel_calls.append(session_id)
        return self.cancel_result


def _start(store, adapter, *, interactive=False):
    return start_claude_job(
        result={"profile": "claude", "operation": "review", "warnings": []},
        adapter=adapter,
        interactive=interactive,
        client=None,
        client_session_id=None,
        client_name="test",
        model="claude-sonnet-5",
        effort=None,
        task="ask",
        prompt="hello",
        effective_cwd=str(store.state_root),
        max_runtime_sec=30,
        run_req={"sensitivity": "normal"},
        store=store,
        attach_writer_lease=lambda *_args: None,
        session_id_for=lambda *_args: "client",
        metadata_for=lambda *_args: {"name": "test"},
        agent_starter=lambda *_args, **_kwargs: None,
        tool_error=lambda error, message: {"ok": False, "error": error, "message": message},
    )


def test_native_cleanup_failure_keeps_claude_lease_pending(tmp_path):
    store = JobStore(tmp_path)
    released: list[str] = []
    store._release_writer_lease = lambda job_id, meta=None: released.append(job_id) or True
    adapter = _Adapter(cancel_result=False)
    original_update = store.update_job_meta
    calls = 0

    def fail_final_metadata(job_id, updates):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("metadata unavailable")
        return original_update(job_id, updates)

    store.update_job_meta = fail_final_metadata
    result = _start(store, adapter)
    job_id = result["job_id"]

    assert result["ok"] is False
    assert adapter.cancel_calls == ["deadbeef"]
    assert released == []
    assert store.get_result(job_id)["ok"] is False
    meta = store._read_job_meta(store.get_job(job_id).path)
    assert meta["cleanup_pending"] is True
    assert meta["native_session_id"] == "deadbeef"


def test_interactive_tmux_rollback_failure_keeps_claude_lease_pending(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    released: list[str] = []
    store._release_writer_lease = lambda job_id, meta=None: released.append(job_id) or True
    adapter = _Adapter(cancel_result=False)
    monkeypatch.setattr(
        "agent_crossbar.adapters.claude_lifecycle.start_claude_interactive_tmux",
        lambda **_kwargs: SimpleNamespace(returncode=1, stderr="tmux unavailable"),
    )

    result = _start(store, adapter, interactive=True)
    job_id = result["job_id"]

    assert result["ok"] is False
    assert adapter.cancel_calls == ["deadbeef"]
    assert released == []
    assert store.get_result(job_id)["ok"] is False
    meta = store._read_job_meta(store.get_job(job_id).path)
    assert meta["cleanup_pending"] is True
    assert meta["native_session_id"] == "deadbeef"
