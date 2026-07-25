import pytest

import agent_crossbar.jobs as jobs_module
from agent_crossbar.jobs import JobStore


def test_job_dir_permissions_and_path(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    assert job.path == tmp_path / "jobs" / job.job_id
    assert oct(tmp_path.stat().st_mode & 0o777) == "0o700"
    assert oct((tmp_path / "jobs").stat().st_mode & 0o777) == "0o700"
    assert oct(job.path.stat().st_mode & 0o777) == "0o700"
    assert oct((job.path / "events.jsonl").stat().st_mode & 0o777) == "0o600"


def test_event_sequence_is_atomic(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    for idx in range(5):
        job.events.write(level="info", type="progress", message=str(idx), data={})
    events = job.events.read_since(0)
    assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]


# Fix 1: job_tail after reloading a job from disk must compute last_seq/next_seq from events.jsonl
def test_job_tail_reloads_seq_from_disk(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    # Write 3 events
    for i in range(3):
        job.events.write(level="info", type="progress", message=f"evt-{i}")

    # Simulate reloading from disk via get_job (new JobStore instance)
    store2 = JobStore(tmp_path)
    reloaded = store2.get_job(job.job_id)
    assert reloaded is not None
    assert reloaded.events.last_seq == 3
    assert reloaded.events.next_seq == 4

    tail = store2.job_tail(job.job_id)
    assert tail["ok"] is True
    assert tail["last_seq"] == 3
    assert tail["next_seq"] == 4


# Fix 2: job_tail response must include job_id and status
def test_job_tail_includes_job_id_and_status(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    tail = store.job_tail(job.job_id)
    assert tail["ok"] is True
    assert tail["job_id"] == job.job_id
    assert tail["status"] == "running"

    # Also check error response includes job_id and status
    tail_err = store.job_tail("99999999-nonexistent")
    assert tail_err["ok"] is False
    assert tail_err["job_id"] == "99999999-nonexistent"
    assert tail_err["status"] is None


def test_job_tail_includes_tmux_output_tail(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text("booting\nthinking\nediting file\n", encoding="utf-8")
    store.update_job_meta(job.job_id, {"tmux_output_path": str(output_path)})

    tail = store.job_tail(job.job_id, since_seq=job.events.last_seq)

    assert tail["ok"] is True
    assert tail["events"] == []
    assert tail["output_tail"]["path"] == str(output_path)
    assert tail["output_tail"]["text"] == "booting\nthinking\nediting file\n"
    assert tail["output_tail"]["truncated"] is False


def test_job_tail_sequence_window_is_self_consistent_when_truncated(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    for idx in range(3):
        job.events.write(level="info", type="progress", message=str(idx))

    tail = store.job_tail(job.job_id, max_events=1)

    assert tail["truncated"] is True
    assert [event["seq"] for event in tail["events"]] == [1]
    assert tail["last_seq"] == 1
    assert tail["next_seq"] == 2

    continued = store.job_tail(job.job_id, since_seq=tail["last_seq"])
    assert [event["seq"] for event in continued["events"]] == [2, 3]


def test_job_tail_does_not_advance_cursor_when_limit_returns_no_events(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    job.events.write(level="info", type="progress", message="pending")

    tail = store.job_tail(job.job_id, since_seq=0, max_events=0)

    assert tail["truncated"] is True
    assert tail["events"] == []
    assert tail["last_seq"] == 0
    assert tail["next_seq"] == 1


def test_job_tail_can_read_incremental_tmux_output(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text("first\nsecond\n", encoding="utf-8")
    store.update_job_meta(job.job_id, {"tmux_output_path": str(output_path)})

    first = store.job_tail(job.job_id, output_since_bytes=0, max_bytes=6)
    second = store.job_tail(
        job.job_id,
        output_since_bytes=first["output_next_bytes"],
        max_bytes=100,
    )

    assert first["output_tail"]["text"] == "first\n"
    assert first["output_tail"]["bytes"] == 6
    assert first["output_next_bytes"] == 6
    assert second["output_tail"]["text"] == "second\n"
    assert second["output_next_bytes"] == 13


def test_job_tail_incremental_output_never_splits_utf8(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text("ї🙂z", encoding="utf-8")
    store.update_job_meta(job.job_id, {"tmux_output_path": str(output_path)})

    chunks = []
    offset = 0
    while offset < output_path.stat().st_size:
        tail = store.job_tail(job.job_id, output_since_bytes=offset, max_bytes=1)
        chunks.append(tail["output_tail"]["text"])
        assert "\ufffd" not in chunks[-1]
        assert tail["output_next_bytes"] > offset
        offset = tail["output_next_bytes"]

    assert "".join(chunks) == "ї🙂z"


def test_job_tail_lazy_finalizes_completed_tmux_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    (job.path / "tmux-exit-status.txt").write_text("0\n", encoding="utf-8")
    (job.path / "tmux-output.log").write_text("done\n", encoding="utf-8")

    tail = store.job_tail(job.job_id)

    assert tail["status"] == "succeeded"
    assert (job.path / "result.json").exists()


def test_job_tail_includes_print_output_tail(tmp_path):
    """job_tail serves output_tail for print transport (noninteractive Reasonix)."""
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review", transport="print")
    output_path = job.path / "stdout.log"
    output_path.write_text("analysis line 1\nanalysis line 2\nfinal result\n", encoding="utf-8")
    store.update_job_meta(job.job_id, {"print_output_path": str(output_path)})

    tail = store.job_tail(job.job_id, since_seq=job.events.last_seq)

    assert tail["ok"] is True
    assert tail["events"] == []
    assert tail["output_tail"] is not None
    assert tail["output_tail"]["path"] == str(output_path)
    assert tail["output_tail"]["text"] == "analysis line 1\nanalysis line 2\nfinal result\n"
    assert tail["output_tail"]["truncated"] is False


def test_job_tail_can_read_incremental_print_output(tmp_path):
    """job_tail supports output_since_bytes for incremental print output reads."""
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="print")
    output_path = job.path / "stdout.log"
    output_path.write_text("chunk1\nchunk2\n", encoding="utf-8")
    store.update_job_meta(job.job_id, {"print_output_path": str(output_path)})

    first = store.job_tail(job.job_id, output_since_bytes=0, max_bytes=7)
    second = store.job_tail(
        job.job_id,
        output_since_bytes=first["output_next_bytes"],
        max_bytes=100,
    )

    assert first["output_tail"]["text"] == "chunk1\n"
    assert first["output_tail"]["bytes"] == 7
    assert first["output_next_bytes"] == 7
    assert second["output_tail"]["text"] == "chunk2\n"
    assert second["output_next_bytes"] == 14


def test_job_tail_print_output_bounds_and_ignores_unsafe_path(tmp_path):
    """Print output tail respects max_bytes and rejects paths outside job dir."""
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review", transport="print")
    output_path = job.path / "stdout.log"
    output_path.write_text("0123456789\nabcdefghij\n", encoding="utf-8")
    unsafe_path = tmp_path / "outside.log"
    unsafe_path.write_text("do not read me\n", encoding="utf-8")
    store.update_job_meta(job.job_id, {"print_output_path": str(unsafe_path)})

    tail = store.job_tail(job.job_id, max_bytes=8)

    # Unsafe path is rejected; falls back to default stdout.log in job dir.
    assert tail["output_tail"]["path"] == str(output_path)
    assert tail["output_tail"]["text"] == "defghij\n"
    assert tail["output_tail"]["truncated"] is True


def test_job_list_lazy_finalizes_completed_tmux_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    (job.path / "tmux-exit-status.txt").write_text("1\n", encoding="utf-8")

    listed = {item["job_id"]: item for item in store.list_jobs()}

    assert listed[job.job_id]["status"] == "failed"


def test_job_list_filters_and_correlates_jobs_by_client_session(tmp_path):
    store = JobStore(tmp_path)
    owned = store.create_job(
        profile="claude",
        operation="review",
        transport="tmux",
        client_session_id="thread-a",
        client_name="codex",
        cwd="/repo/a",
    )
    store.create_job(
        profile="claude",
        operation="review",
        transport="tmux",
        client_session_id="thread-b",
        client_name="codex",
        cwd="/repo/b",
    )

    listed = store.list_jobs(client_session_id="thread-a")

    assert listed == [
        {
            "job_id": owned.job_id,
            "profile": "claude",
            "operation": "review",
            "transport": "tmux",
            "status": "running",
            "client_session_id": "thread-a",
            "client_name": "codex",
            "cwd": "/repo/a",
        }
    ]


def test_scoped_job_list_does_not_finalize_another_sessions_job(tmp_path):
    store = JobStore(tmp_path)
    foreign = store.create_job(
        profile="claude", operation="review", transport="tmux", client_session_id="thread-b"
    )
    (foreign.path / "tmux-exit-status.txt").write_text("0\n", encoding="utf-8")

    assert store.list_jobs(client_session_id="thread-a") == []
    assert store._read_job_meta(foreign.path).get("status", "running") == "running"
    assert not (foreign.path / "result.json").exists()


def test_job_access_rejects_different_client_session(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )

    assert store.job_tail(job.job_id, client_session_id="thread-b")["error"] == "job_not_found"
    assert store.get_result(job.job_id, client_session_id="thread-b")["error"] == "job_not_found"
    assert store.stop_job(job.job_id, client_session_id="thread-b")["error"] == "job_not_found"
    assert store.job_tail(job.job_id)["error"] == "job_not_found"


def test_job_tail_bounds_tmux_output_tail_and_ignores_unsafe_path(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text("0123456789\nabcdefghij\n", encoding="utf-8")
    unsafe_path = tmp_path / "outside.log"
    unsafe_path.write_text("do not read me\n", encoding="utf-8")
    store.update_job_meta(job.job_id, {"tmux_output_path": str(unsafe_path)})

    tail = store.job_tail(job.job_id, max_bytes=8)

    assert tail["output_tail"]["path"] == str(output_path)
    assert tail["output_tail"]["text"] == "defghij\n"
    assert tail["output_tail"]["truncated"] is True


# Fix 3: EventWriter.write must include ts and redacted fields
def test_event_write_includes_ts_and_redacted(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    seq = job.events.write(level="info", type="progress", message="hello")
    events = job.events.read_since(0)
    assert len(events) == 1
    ev = events[0]
    assert ev["seq"] == seq
    assert "ts" in ev
    assert isinstance(ev["ts"], str) and "T" in ev["ts"]  # ISO format
    assert ev["redacted"] is False

    # Test redacted=True
    seq2 = job.events.write(level="warn", type="stdout", message="secret", redacted=True)
    events2 = job.events.read_since(seq)
    assert len(events2) == 1
    assert events2[0]["redacted"] is True
    assert events2[0]["seq"] == seq2


def test_event_raw_jsonl_has_required_fields(tmp_path):
    """Verify raw events.jsonl lines contain ts and redacted per spec schema."""
    import json

    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="review")
    job.events.write(level="info", type="progress", message="test")
    raw_line = (job.path / "events.jsonl").read_text().strip()
    ev = json.loads(raw_line)
    assert "seq" in ev
    assert "ts" in ev
    assert "level" in ev
    assert "type" in ev
    assert "message" in ev
    assert "redacted" in ev
    assert "data" in ev
    assert ev["redacted"] is False


def test_get_result_lazy_finalizes_completed_tmux_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    (job.path / "tmux-output.log").write_text("CURRENT_TIME_OK\n")
    (job.path / "tmux-exit-status.txt").write_text("0\n")

    result = store.get_result(job.job_id)

    assert result["ok"] is True
    assert result["summary"] == "CURRENT_TIME_OK\n"
    assert result["raw_artifacts"] == [str(job.path / "tmux-output.log")]
    tail = store.job_tail(job.job_id)
    assert tail["status"] == "succeeded"
    assert [event["type"] for event in tail["events"]][-2:] == ["tmux_exited", "result"]


def test_get_result_lazy_finalizes_interactive_tmux_output_without_exit_status(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "◇ you · just now\n"
        "↳ Reply with the single word OK and nothing else.\n"
        "\x1b[38;5;157m‹\x1b[1C\x1b[1mreply\x1b[22m v4-flash\n"
        "OK\n"
        "›▌askanything·slashforcommands\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {
            "tmux_output_path": str(output_path),
            "tmux_session": "agents-test",
            "interactive": True,
        },
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is True
    assert "OK" in result["summary"]
    assert result["raw_artifacts"] == [str(output_path)]
    tail = store.job_tail(job.job_id)
    assert tail["status"] == "succeeded"
    assert [event["type"] for event in tail["events"]][-2:] == ["tmux_output_complete", "result"]


def test_get_result_does_not_finalize_busy_reasonix_tmux_output(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "› Reply with the single word OK and nothing else.\n"
        "✦ The user asked for a single word.\n"
        "⠏ Fiddling with the character creation screen... (5s · esc to cancel)\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is False
    assert result["error"] == "result_not_ready"


def test_get_result_does_not_finalize_echoed_task_done_prompt(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "› Say task done when complete.\n*   Type your message or @path/to/file\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is False
    assert result["error"] == "result_not_ready"


def test_get_result_does_not_finalize_reasonix_reply_before_prompt_returns(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "◇ you · just now\n"
        "↳ Reply OK.\n"
        "\x1b[38;5;157m‹\x1b[1C\x1b[1mreply\x1b[22m v4-flash\n"
        "streaming partial response",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is False
    assert result["error"] == "result_not_ready"


def test_get_result_fails_reasonix_resumed_session_output(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        '✓▸resumed session "code-claude" with 10 prior messages · /new to start fresh\n'
        "◇ you · just now\n"
        "↳ Reply OK.\n"
        "‹ reply v4-flash\n"
        "OK\n"
        "ask anything · slash for commands\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is False
    assert result["summary"].startswith("Reasonix resumed an existing session")
    assert store.job_tail(job.job_id)["status"] == "failed"


def test_get_result_lazy_finalizes_claude_style_tmux_output_after_prompt_returns(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="claude", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "❯ Reply with the single word OK and nothing else.\n"
        "⏺OK\n"
        "✻Baked for 4s\n"
        "❯\n"
        "\x1b]777;notify;Claude Code;Claude is waiting for your input\x07\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is True
    assert "⏺OK" in result["summary"]


def test_get_result_does_not_finalize_busy_opencode_tmux_output(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="opencode", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "█▀▀█  OpenCode\n"
        "█  █  ~/src/example-project\n"
        "▀▀▀▀\n\n"
        "› Reply exactly OPENCODE_TMUX_DETECT_OK and nothing else.\n\n\n"
        " BUILD  ⬝⬝⬝⬝⬝⬝■■ esc interrupt                                       ctrl+p cmd\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is False
    assert result["error"] == "result_not_ready"


def test_get_result_lazy_finalizes_opencode_tmux_output_after_answer(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="opencode", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "█▀▀█  OpenCode\n"
        "█  █  ~/src/example-project\n"
        "▀▀▀▀\n\n"
        "› Reply exactly OPENCODE_TMUX_DETECT_OK and nothing else.\n\n"
        "Thinking: The user\n"
        'wants me to reply exactly "OPENCODE_TMUX_DETECT_OK" and nothing else.\n\n'
        "OPENCODE_TMUX_DETECT_OK\n\n"
        "▣ Build · DeepSeek V4 Flash · 4.3s\n\n"
        " BUILD                                          50.3K (5%) · $0.01 · ctrl+p cmd\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is True
    assert "OPENCODE_TMUX_DETECT_OK" in result["summary"]


def test_get_result_lazy_finalizes_codex_tmux_output_after_answer(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="codex", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "› Reply with the single word OK and nothing else.\n"
        "• UserPromptSubmit hook (completed)\n"
        "• OK\n"
        "• Running Stop hook:mem0 fact extraction\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-test", "interactive": True},
    )

    result = store.get_result(job.job_id)

    assert result["ok"] is True
    assert "• OK" in result["summary"]


def test_get_result_does_not_lazy_finalize_a_live_codex_tmux_session(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = store.create_job(profile="codex", operation="dev", transport="tmux")
    output_path = job.path / "tmux-output.log"
    output_path.write_text(
        "• UserPromptSubmit hook (completed)\n• Intermediate commentary\n",
        encoding="utf-8",
    )
    store.update_job_meta(
        job.job_id,
        {"tmux_output_path": str(output_path), "tmux_session": "agents-live", "interactive": True},
    )

    class Alive:
        returncode = 0

    monkeypatch.setattr(jobs_module.subprocess, "run", lambda *args, **kwargs: Alive())

    result = store.get_result(job.job_id)

    assert result["ok"] is False
    assert result["error"] == "result_not_ready"


def test_get_result_lazy_finalizes_failed_tmux_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    (job.path / "tmux-output.log").write_text("approval required\n")
    (job.path / "tmux-exit-status.txt").write_text("1\n")

    result = store.get_result(job.job_id)

    assert result["ok"] is False
    assert result["summary"] == "approval required\n"
    assert store.job_tail(job.job_id)["status"] == "failed"


# Fix 4: get_job/job_tail must reject invalid job_id / path traversal
def test_job_tail_rejects_path_traversal(tmp_path):
    store = JobStore(tmp_path)
    tail = store.job_tail("../x")
    assert tail["ok"] is False
    assert tail["error"] == "invalid_job_id"

    # get_job also rejects
    assert store.get_job("../x") is None


def test_job_tail_rejects_invalid_job_id(tmp_path):
    store = JobStore(tmp_path)
    # Too short, no hyphen suffix
    tail = store.job_tail("abc")
    assert tail["ok"] is False
    assert tail["error"] == "invalid_job_id"


# Fix 5: state root and jobs dir should be 0700
def test_state_root_and_jobs_dir_permissions(tmp_path):
    store = JobStore(tmp_path / "state")
    job = store.create_job(profile="reasonix", operation="review")
    assert oct(store.state_root.stat().st_mode & 0o777) == "0o700"
    assert oct((store.state_root / "jobs").stat().st_mode & 0o777) == "0o700"
    assert oct(job.path.stat().st_mode & 0o777) == "0o700"


def test_create_job_hardens_preexisting_dir_permissions(tmp_path):
    """If state_root and jobs dir already exist with loose permissions, create_job must chmod them to 0700."""
    state_root = tmp_path / "state"
    jobs_dir = state_root / "jobs"
    # Precreate with world-readable permissions
    state_root.mkdir(mode=0o755)
    jobs_dir.mkdir(mode=0o755)
    store = JobStore(state_root)
    job = store.create_job(profile="reasonix", operation="review")
    # Both should now be tightened to 0700
    assert oct(store.state_root.stat().st_mode & 0o777) == "0o700"
    assert oct(jobs_dir.stat().st_mode & 0o777) == "0o700"
    assert oct(job.path.stat().st_mode & 0o777) == "0o700"


def test_send_user_input_tmux_uses_enter_not_c_m(monkeypatch, tmp_path):
    """Regression: send_user_input must use Enter, not C-m, for Reasonix TUI."""
    from agent_crossbar.jobs import JobStore

    captured = []

    def fake_run(args, **kwargs):
        captured.append(args)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("agent_crossbar.jobs.subprocess.run", fake_run)

    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="advice", transport="tmux")
    store.send_user_input(job.job_id, "hello world")

    # Must have two subprocess calls: send-keys with text, then submit
    assert len(captured) == 2, f"expected 2 calls, got {len(captured)}: {captured}"
    submit_args = captured[1]
    assert "Enter" in submit_args, f"Enter not in {submit_args}"
    assert "C-m" not in submit_args, f"C-m found in {submit_args}"


def test_send_user_input_tmux_settles_between_text_and_enter(monkeypatch, tmp_path):
    """Regression: send_user_input must sleep 0.5s between -l text and Enter."""
    from agent_crossbar.jobs import JobStore

    captured = []

    def fake_run(args, **kwargs):
        captured.append(args)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("agent_crossbar.jobs.subprocess.run", fake_run)

    sleep_calls = []

    def fake_sleep(secs):
        sleep_calls.append(secs)

    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="advice", transport="tmux")
    store.send_user_input(job.job_id, "hello world", _sleep=fake_sleep)

    # Must have exactly 1 sleep call of 0.5s between the two subprocess calls
    assert len(sleep_calls) == 1, f"expected 1 sleep call, got {len(sleep_calls)}"
    assert sleep_calls[0] == 0.5, f"expected 0.5s sleep, got {sleep_calls[0]}"
    assert len(captured) == 2, f"expected 2 subprocess calls, got {len(captured)}"


def test_send_user_input_marks_awaiting_job_running(monkeypatch, tmp_path):
    """A delivered follow-up resumes an interactive job's lifecycle state."""
    from agent_crossbar.jobs import JobStore

    monkeypatch.setattr(
        "agent_crossbar.jobs.subprocess.run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0})(),
    )
    store = JobStore(tmp_path)
    job = store.create_job(profile="claude", operation="advice", transport="tmux")
    store.update_job_meta(job.job_id, {"status": "awaiting_input", "waiting_for": "user"})

    result = store.send_user_input(job.job_id, "continue", _sleep=lambda _: None)

    assert result["ok"] is True
    meta = store._read_job_meta(job.path)
    assert meta["status"] == "running"
    assert "waiting_for" not in meta


# ── Cross-session access: operator token ─────────────────────────────


@pytest.fixture
def operator_session(monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_OPERATOR_TOKEN", "operator-token")
    return "operator-token"


def test_foreign_access_denied_by_default(tmp_path):
    """Without an operator token, foreign session access is rejected for all tools."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    # tail
    assert store.job_tail(job.job_id, client_session_id="thread-b")["error"] == "job_not_found"
    # result
    assert store.get_result(job.job_id, client_session_id="thread-b")["error"] == "job_not_found"
    # stop
    assert store.stop_job(job.job_id, client_session_id="thread-b")["error"] == "job_not_found"
    # send_user_input (uses _get_owned_job internally)
    assert (
        store.send_user_input(job.job_id, "hello", client_session_id="thread-b")["error"]
        == "job_not_found"
    )
    # None session (anonymous caller) is also rejected
    assert store.job_tail(job.job_id)["error"] == "job_not_found"
    assert store.get_result(job.job_id)["error"] == "job_not_found"


def test_operator_session_allows_cross_session_tail(tmp_path, operator_session):
    """The configured operator token allows cross-session job_tail."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    result = store.job_tail(job.job_id, client_session_id=operator_session)
    assert result["ok"] is True
    assert result["job_id"] == job.job_id
    assert result["status"] == "running"


def test_operator_session_allows_cross_session_result(tmp_path, operator_session):
    """The configured operator token allows cross-session get_result."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    # Without result.json, it returns result_not_ready but NOT job_not_found
    result = store.get_result(job.job_id, client_session_id=operator_session)
    assert result["error"] == "result_not_ready"


def test_operator_session_allows_cross_session_stop(tmp_path, operator_session):
    """The configured operator token allows cross-session stop_job."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    result = store.stop_job(job.job_id, client_session_id=operator_session)
    assert result["ok"] is True
    assert result["job_id"] == job.job_id


def test_operator_session_allows_cross_session_send_input(tmp_path, operator_session):
    """The configured operator token allows cross-session send_user_input."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    # Without interactive flag, it should return job_not_interactive, NOT job_not_found
    result = store.send_user_input(job.job_id, "hello", client_session_id=operator_session)
    assert result["error"] == "job_not_interactive"


def test_job_list_stays_session_isolated_without_operator_token(tmp_path):
    """job_list filters by client_session_id unless an operator token is used."""
    store = JobStore(tmp_path)
    store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
        client_name="codex",
        cwd="/a",
    )
    store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-b",
        client_name="codex",
        cwd="/b",
    )

    listed_a = store.list_jobs(client_session_id="thread-a")
    assert len(listed_a) == 1
    assert listed_a[0]["client_session_id"] == "thread-a"

    listed_b = store.list_jobs(client_session_id="thread-b")
    assert len(listed_b) == 1
    assert listed_b[0]["client_session_id"] == "thread-b"

    # Anonymous caller (client_session_id=None) sees all jobs at store level;
    # the server layer filters out jobs with client_session_id when session is None.
    listed_none = store.list_jobs()
    assert len(listed_none) == 2  # store level returns all


def test_job_list_operator_session_shows_all_jobs(tmp_path, operator_session):
    """An operator token lists jobs from all sessions."""
    store = JobStore(tmp_path)
    store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
        client_name="codex",
        cwd="/a",
    )
    store.create_job(
        profile="reasonix",
        operation="dev",
        client_session_id="thread-b",
        client_name="claude",
        cwd="/b",
    )

    listed = store.list_jobs(client_session_id=operator_session)
    assert len(listed) == 2
    sessions = {j["client_session_id"] for j in listed}
    assert sessions == {"thread-a", "thread-b"}


def test_random_session_still_denied(tmp_path):
    """Random client_session_id string does NOT grant cross-session access."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    # Random string as client_session_id should NOT grant access
    assert (
        store.job_tail(job.job_id, client_session_id="some-random-id")["error"] == "job_not_found"
    )
    assert (
        store.get_result(job.job_id, client_session_id="some-random-id")["error"] == "job_not_found"
    )


def test_operator_session_does_not_leak_into_job_creation(tmp_path, operator_session):
    """Jobs created normally still store their real client_session_id."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    meta = store._read_job_meta(job.path)
    assert meta["client_session_id"] == "thread-a"
    assert store.job_tail(job.job_id, client_session_id=operator_session)["ok"] is True


def test_operator_session_bad_job_id_still_rejected(tmp_path, operator_session):
    """Even with an operator token, invalid job IDs are rejected before ownership check."""
    store = JobStore(tmp_path)
    assert store.job_tail("../etc", client_session_id=operator_session)["error"] == "invalid_job_id"
    assert store.job_tail("abc", client_session_id=operator_session)["error"] == "invalid_job_id"


# ── Shared default state root (regression) ──────────────────────────────────


def test_default_state_root_is_shared_not_per_session(monkeypatch):
    """default_state_root must return the stable shared path, not a per-process dir."""
    from agent_crossbar.jobs import default_state_root

    monkeypatch.delenv("AGENT_CROSSBAR_STATE_DIR", raising=False)
    monkeypatch.delenv("AGENT_HARNESS_STATE_DIR", raising=False)
    root = default_state_root()
    assert root.name == "agent-crossbar", f"Expected .../agent-crossbar, got {root.name}"
    assert root.parent.name == "state", f"Expected .../state/agent-crossbar, got {root.parent}"
    assert ".local" in str(root), f"Expected ~/.local/... path, got {root}"


def test_default_state_root_respects_override(monkeypatch):
    """AGENT_CROSSBAR_STATE_DIR must override the default path."""
    from agent_crossbar.jobs import default_state_root

    override = "/tmp/test-shared-state"
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", override)
    assert str(default_state_root()) == override


def test_two_job_stores_share_state_root(tmp_path):
    """Two independent JobStore instances with the same root see each other's jobs."""
    store_a = JobStore(tmp_path)
    job = store_a.create_job(profile="claude", operation="review")
    job.events.write(level="info", type="progress", message="hello from A")

    store_b = JobStore(tmp_path)
    assert store_b.get_job(job.job_id) is not None
    tail = store_b.job_tail(job.job_id)
    assert tail["ok"] is True
    assert tail["job_id"] == job.job_id
    assert any(e["message"] == "hello from A" for e in tail["events"])


def test_job_store_default_constructor_uses_shared_path(monkeypatch):
    """JobStore() with no arguments must use default_state_root()."""
    from agent_crossbar.jobs import default_state_root

    monkeypatch.delenv("AGENT_CROSSBAR_STATE_DIR", raising=False)
    monkeypatch.delenv("AGENT_HARNESS_STATE_DIR", raising=False)
    store = JobStore()
    assert store.state_root == default_state_root()


def test_job_store_default_and_explicit_are_consistent(monkeypatch):
    """JobStore() and JobStore(default_state_root()) must be equivalent."""
    from agent_crossbar.jobs import default_state_root

    monkeypatch.delenv("AGENT_CROSSBAR_STATE_DIR", raising=False)
    monkeypatch.delenv("AGENT_HARNESS_STATE_DIR", raising=False)
    s1 = JobStore()
    s2 = JobStore(default_state_root())
    assert s1.state_root == s2.state_root


def test_list_jobs_sees_jobs_from_another_instance(tmp_path):
    """list_jobs from one JobStore must see jobs created by another."""
    store_a = JobStore(tmp_path)
    job_a = store_a.create_job(profile="claude", operation="review", client_name="alice")

    store_b = JobStore(tmp_path)
    listed = store_b.list_jobs()
    assert any(j["job_id"] == job_a.job_id for j in listed)
    assert any(j["client_name"] == "alice" for j in listed)


# ── Cross-session note: response hint ────────────────────────────────────


def test_cross_session_note_in_denied_responses(tmp_path):
    """Denied cross-session access includes an actionable non-secret hint."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )
    expected_note = (
        "pass the configured operator token as client_session_id for local cross-session access"
    )

    # job_tail
    tail = store.job_tail(job.job_id, client_session_id="thread-b")
    assert tail["error"] == "job_not_found"
    assert tail.get("cross_session_note") == expected_note

    # get_result
    result = store.get_result(job.job_id, client_session_id="thread-b")
    assert result["error"] == "job_not_found"
    assert result.get("cross_session_note") == expected_note

    # send_user_input
    send = store.send_user_input(job.job_id, "hello", client_session_id="thread-b")
    assert send["error"] == "job_not_found"
    assert send.get("cross_session_note") == expected_note

    # stop_job
    stop = store.stop_job(job.job_id, client_session_id="thread-b")
    assert stop["error"] == "job_not_found"
    assert stop.get("cross_session_note") == expected_note


def test_cross_session_note_absent_on_own_job(tmp_path):
    """Own-session access does NOT include the cross_session_note hint."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="review",
        client_session_id="thread-a",
    )

    tail = store.job_tail(job.job_id, client_session_id="thread-a")
    assert tail["ok"] is True
    assert "cross_session_note" not in tail

    stop = store.stop_job(job.job_id, client_session_id="thread-a")
    assert stop["ok"] is True
    assert "cross_session_note" not in stop


def test_cross_session_note_absent_when_job_not_found_at_all(tmp_path):
    """Bogus job_id (valid format but doesn't exist) has no cross_session_note."""
    store = JobStore(tmp_path)
    tail = store.job_tail("99999999-nonexistent", client_session_id="any-session")
    assert tail["error"] == "job_not_found"
    assert "cross_session_note" not in tail
