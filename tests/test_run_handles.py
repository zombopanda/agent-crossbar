"""Generic run-handle registry: cancellation, idempotence, bounded cleanup."""

from __future__ import annotations

import threading

from agent_crossbar.run_handles import RunHandleRegistry


def test_cancel_sets_event_and_returns_metadata():
    registry = RunHandleRegistry()
    handle = registry.register("job-1", on_cancel=lambda: {"transport": "gui"})

    data = registry.cancel("job-1")

    assert handle.cancel_event.is_set() is True
    assert data == {"cancel_requested": True, "repeated": False, "transport": "gui"}


def test_cancel_is_idempotent_and_reports_repeats():
    registry = RunHandleRegistry()
    calls: list[int] = []
    registry.register("job-1", on_cancel=lambda: calls.append(1) or {"n": len(calls)})

    first = registry.cancel("job-1")
    second = registry.cancel("job-1")

    assert calls == [1]
    assert first["repeated"] is False
    assert second["repeated"] is True


def test_cancel_unknown_job_returns_none():
    assert RunHandleRegistry().cancel("missing") is None


def test_cancel_metadata_error_never_breaks_a_stop():
    registry = RunHandleRegistry()

    def boom() -> dict:
        raise RuntimeError("snapshot failed")

    registry.register("job-1", on_cancel=boom)
    data = registry.cancel("job-1")

    assert data["cancel_requested"] is True
    assert "RuntimeError" in data["cancel_metadata_error"]
    assert registry.is_cancelled("job-1") is True


def test_stop_before_registration_is_not_lost():
    registry = RunHandleRegistry()
    registry.register("job-1")
    registry.cancel("job-1")

    # The worker registers its real handle after the stop already arrived.
    handle = registry.register("job-1", cancel_event=threading.Event())

    assert handle.cancelled is True


def test_release_is_idempotent_and_drops_the_handle():
    registry = RunHandleRegistry()
    registry.register("job-1")
    registry.release("job-1")
    registry.release("job-1")

    assert registry.get("job-1") is None
    assert registry.size() == 0


def test_registry_is_bounded():
    registry = RunHandleRegistry(max_handles=3)
    for index in range(10):
        registry.register(f"job-{index}")

    assert registry.size() == 3
    assert registry.get("job-9") is not None
