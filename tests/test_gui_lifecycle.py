"""Deterministic tests for the ChatGPT GUI turn lifecycle primitives."""

from __future__ import annotations

import os

import pytest

from agent_crossbar.gui_lifecycle import (
    ChatGptSessionManager,
    CompletionTracker,
    DomHealthTracker,
    JobArtifactStore,
    ResponseObservation,
    TurnLifecycle,
    prompt_mismatch_diagnostics,
    redact_gui_evidence,
    verify_owned_window,
)


def observation(
    at: float,
    *,
    present: bool = True,
    running: bool = False,
    action: bool = True,
    text: str | None = "final answer",
) -> ResponseObservation:
    return ResponseObservation(
        at=at,
        response_present=present,
        running=running,
        completion_action=action,
        text=text,
    )


# ── Turn lifecycle ──────────────────────────────────────────────────────────


def test_lifecycle_records_ordered_stages():
    clock = {"now": 0.0}
    turn = TurnLifecycle(clock=lambda: clock["now"])
    for stage in ("bootstrap", "authenticated", "model_selected", "submitted", "streaming"):
        clock["now"] += 1
        turn.enter(stage)
    turn.enter("complete", chars=12)

    assert turn.stages() == [
        "bootstrap",
        "authenticated",
        "model_selected",
        "submitted",
        "streaming",
        "complete",
    ]
    assert turn.reached("submitted") is True
    assert turn.terminal_stage == "complete"


def test_lifecycle_terminal_stage_is_single_owner():
    turn = TurnLifecycle()
    turn.enter("submitted")
    turn.enter("cancelled")

    assert turn.enter("complete") is None
    assert turn.terminal_stage == "cancelled"
    assert turn.stages().count("cancelled") == 1


def test_lifecycle_rejects_unknown_stage():
    with pytest.raises(ValueError):
        TurnLifecycle().enter("almost_done")


def test_lifecycle_events_are_bounded():
    turn = TurnLifecycle(max_entries=3)
    for _ in range(10):
        turn.enter("streaming")

    assert len(turn.events()) == 3


# ── Prompt verification ─────────────────────────────────────────────────────


def test_prompt_mismatch_reports_structure_not_content():
    diagnostics = prompt_mismatch_diagnostics("abcdef", "abcXY")

    assert diagnostics == {
        "expected_len": 6,
        "actual_len": 5,
        "common_prefix_len": 3,
        "actual_present": True,
        "actual_empty": False,
    }
    assert "abc" not in str(diagnostics)


def test_prompt_mismatch_handles_unreadable_composer():
    diagnostics = prompt_mismatch_diagnostics("abc", None)

    assert diagnostics["actual_present"] is False
    assert diagnostics["actual_empty"] is True


# ── Completion tracker ──────────────────────────────────────────────────────


def test_completion_requires_two_stable_observations():
    tracker = CompletionTracker()

    assert tracker.observe(observation(1.0)) is None
    assert tracker.reason == "text_changed"
    assert tracker.observe(observation(3.0)) == "final answer"


def test_completion_resets_when_text_keeps_changing():
    tracker = CompletionTracker()
    tracker.observe(observation(1.0, text="part"))
    assert tracker.observe(observation(3.0, text="part two")) is None
    assert tracker.observe(observation(5.0, text="part two")) == "part two"


def test_completion_rejects_running_empty_and_actionless_states():
    tracker = CompletionTracker()
    assert tracker.observe(observation(1.0, present=False)) is None
    assert tracker.reason == "response_absent"
    assert tracker.observe(observation(2.0, running=True)) is None
    assert tracker.reason == "running"
    assert tracker.observe(observation(3.0, text="   ")) is None
    assert tracker.reason == "final_text_empty"
    assert tracker.observe(observation(4.0, action=False)) is None
    assert tracker.reason == "completion_action_absent"


def test_completion_honours_the_stability_interval():
    tracker = CompletionTracker(stability_sec=5.0)
    tracker.observe(observation(1.0))
    assert tracker.observe(observation(2.0)) is None
    assert tracker.reason == "awaiting_stability"
    assert tracker.observe(observation(6.0)) == "final answer"


def test_stale_marker_while_running_never_completes():
    tracker = CompletionTracker()
    for at in (1.0, 3.0, 5.0, 7.0):
        assert tracker.observe(observation(at, running=True, text="stale")) is None


# ── DOM health ──────────────────────────────────────────────────────────────


def test_dom_health_flags_response_that_never_appears():
    tracker = DomHealthTracker(missing_grace_sec=10.0)
    assert tracker.observe(observation(0.0, present=False, text=None)) is None
    assert tracker.observe(observation(9.0, present=False, text=None)) is None
    assert tracker.observe(observation(10.0, present=False, text=None)) == "response_never_appeared"


def test_dom_health_flags_vanished_response():
    tracker = DomHealthTracker(vanished_grace_sec=5.0)
    tracker.observe(observation(0.0, running=True, text=None))
    assert tracker.observe(observation(1.0, present=False, text=None)) is None
    assert tracker.observe(observation(6.5, present=False, text=None)) == "response_vanished"
    assert tracker.detail["absent_for_sec"] == 5.5


def test_dom_health_flags_terminal_empty_response():
    tracker = DomHealthTracker(empty_grace_sec=4.0)
    assert tracker.observe(observation(0.0, text="")) is None
    assert tracker.observe(observation(4.0, text="")) == "response_completed_empty"


def test_dom_health_stays_quiet_while_streaming():
    tracker = DomHealthTracker(empty_grace_sec=1.0)
    for at in (0.0, 5.0, 20.0):
        assert tracker.observe(observation(at, running=True, text=None)) is None


def test_dom_health_recovers_when_response_returns():
    tracker = DomHealthTracker(vanished_grace_sec=5.0)
    tracker.observe(observation(0.0, running=True, text=None))
    tracker.observe(observation(1.0, present=False, text=None))
    assert tracker.observe(observation(2.0, running=True, text=None)) is None
    assert tracker.observe(observation(5.0, present=False, text=None)) is None


# ── Session manager ─────────────────────────────────────────────────────────


def test_session_capacity_is_one_and_serializes_jobs():
    manager = ChatGptSessionManager()
    first, error = manager.acquire("job-1", "helium", "Helium")
    assert first is not None and error is None

    second, error = manager.acquire("job-2", "helium", "Helium")
    assert second is None
    assert error == "busy"

    manager.retire("job-1", "turn_complete")
    third, error = manager.acquire("job-2", "helium", "Helium")
    assert third is not None and error is None


def test_reacquiring_the_same_job_retires_the_previous_session():
    manager = ChatGptSessionManager()
    first, _ = manager.acquire("job-1", "helium", "Helium")
    second, _ = manager.acquire("job-1", "safari", "Safari")

    assert first is not None and second is not None
    assert first.retired is True
    assert first.retired_reason == "superseded_by_new_turn"
    assert manager.live_count() == 1


def test_retired_session_is_not_active():
    manager = ChatGptSessionManager()
    session, _ = manager.acquire("job-1", "helium", "Helium")
    assert session is not None
    session.bind(123, 456)
    assert manager.active("job-1") is session

    manager.retire("job-1", "contaminated")
    assert manager.active("job-1") is None
    assert session.snapshot()["retired_reason"] == "contaminated"


def test_window_verification_detects_closed_and_ambiguous_windows():
    manager = ChatGptSessionManager()
    session, _ = manager.acquire("job-1", "helium", "Helium")
    assert session is not None

    assert verify_owned_window(session, [{"pid": 123, "window_id": 456}]) == (
        False,
        "session_not_bound",
    )
    session.bind(123, 456)
    assert verify_owned_window(session, [{"pid": 123, "window_id": 456}]) == (True, None)
    assert verify_owned_window(session, [{"pid": 123, "window_id": 999}]) == (
        False,
        "window_closed",
    )
    assert verify_owned_window(
        session, [{"pid": 123, "window_id": 456}, {"pid": 123, "window_id": 456}]
    ) == (False, "window_ambiguous")
    assert verify_owned_window(session, None) == (False, "window_list_unavailable")
    assert session.matches(123, 457) is False
    assert session.matches("nope", None) is False


# ── Evidence redaction and artifacts ────────────────────────────────────────


def test_redaction_strips_context_envelopes_and_hidden_reasoning():
    text = (
        "BEGIN_AGENTS_MCP_CONTEXT\n--- secret.py ---\nvalue\nEND_AGENTS_MCP_CONTEXT\n"
        "Thought for 12 seconds\n"
        "visible answer\n"
        "api_key=abcdef123456\n"
    )

    redacted = redact_gui_evidence(text)

    assert "secret.py" not in redacted
    assert "[REDACTED:context_envelope]" in redacted
    assert "[REDACTED:hidden_reasoning]" in redacted
    assert "abcdef123456" not in redacted
    assert "visible answer" in redacted


def test_redaction_bounds_long_evidence():
    redacted = redact_gui_evidence("x" * 100, max_chars=10)

    assert "TRUNCATED" in redacted
    assert len(redacted) < 100


def test_artifact_store_writes_job_local_redacted_evidence(tmp_path):
    store = JobArtifactStore(tmp_path)
    path = store.write(
        "last-tree.txt",
        "AXWindow token=supersecretvalue",
        stage="streaming",
        kind="ax_tree",
        session={"session_id": "job-1-1", "browser": "Helium", "window_id": 456},
    )

    assert path is not None
    written = tmp_path / "chatgpt-pro" / "last-tree.txt"
    assert written.read_text(encoding="utf-8").find("supersecretvalue") == -1
    manifest = store.manifest()
    assert manifest[0]["kind"] == "ax_tree"
    assert manifest[0]["stage"] == "streaming"
    assert manifest[0]["session"] == {
        "session_id": "job-1-1",
        "browser": "Helium",
        "window_id": 456,
    }
    assert store.paths() == [str(written)]


def test_artifact_store_sanitizes_traversal_names(tmp_path):
    store = JobArtifactStore(tmp_path)
    path = store.write("../../escape.txt", "evidence")

    assert path is not None
    assert str(tmp_path.resolve()) in str(path)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_artifact_store_rejects_symlinked_target(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    store = JobArtifactStore(tmp_path)
    store.dir.mkdir(parents=True, exist_ok=True)
    (store.dir / "evidence.txt").symlink_to(outside)

    assert store.write("evidence.txt", "replacement") is None
    assert outside.read_text(encoding="utf-8") == "original"
    assert store.rejections()[0]["reason"] == "symlink_target"


def test_artifact_store_rejects_hardlinked_target(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    store = JobArtifactStore(tmp_path)
    store.dir.mkdir(parents=True, exist_ok=True)
    os.link(outside, store.dir / "evidence.txt")

    assert store.write("evidence.txt", "replacement") is None
    assert outside.read_text(encoding="utf-8") == "original"
    assert store.rejections()[0]["reason"] == "hardlinked_target"
