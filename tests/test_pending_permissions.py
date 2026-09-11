"""Unit tests for the provider-neutral pending request registry."""

from __future__ import annotations

import asyncio

import pytest

from agent_crossbar.pending_permissions import PendingPermissionRegistry, PendingRequestLimitError


@pytest.fixture
def registry():
    reg = PendingPermissionRegistry()
    yield reg
    reg.clear()


def _loop():
    return asyncio.new_event_loop()


def test_resolve_delivers_decision_to_live_future(registry):
    loop = _loop()
    try:
        pending = registry.register("job-1", "req-1", loop=loop)
        result = registry.resolve("job-1", "req-1", "allow")
        assert result["ok"] is True
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.done()
        assert pending.future.result() == "allow"
    finally:
        loop.close()


def test_resolve_is_one_shot_cas(registry):
    loop = _loop()
    try:
        registry.register("job-1", "req-1", loop=loop)
        first = registry.resolve("job-1", "req-1", "allow")
        second = registry.resolve("job-1", "req-1", "reject")
        assert first["ok"] is True
        assert second["ok"] is False
        assert second["error"] == "request_not_pending"
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()


def test_resolve_unknown_request_is_rejected(registry):
    result = registry.resolve("job-1", "nope", "allow")
    assert result["ok"] is False
    assert result["error"] == "request_not_pending"


def test_settle_job_cancels_outstanding_waits(registry):
    loop = _loop()
    try:
        pending = registry.register("job-1", "req-1", loop=loop)
        second = registry.register("job-1", "req-2", loop=loop)
        settled = registry.settle_job("job-1", outcome="cancelled")
        assert settled == ["req-1", "req-2"]
        loop.run_until_complete(asyncio.sleep(0))
        assert pending.future.cancelled()
        assert second.future.cancelled()
        # A later decision cannot resurrect the settled request.
        assert registry.resolve("job-1", "req-1", "allow")["error"] == "request_not_pending"
    finally:
        loop.close()


def test_discard_removes_without_delivering(registry):
    loop = _loop()
    try:
        pending = registry.register("job-1", "req-1", loop=loop)
        registry.discard("job-1", "req-1")
        assert registry.get("job-1", "req-1") is None
        assert not pending.future.done()
        assert registry.resolve("job-1", "req-1", "allow")["error"] == "request_not_pending"
    finally:
        loop.close()


def test_registry_is_bounded_per_job():
    registry = PendingPermissionRegistry(max_pending_per_job=2)
    loop = _loop()
    try:
        registry.register("job-1", "req-1", loop=loop)
        registry.register("job-1", "req-2", loop=loop)
        with pytest.raises(PendingRequestLimitError):
            registry.register("job-1", "req-3", loop=loop)
        assert registry.size() == 2
        assert registry.get("job-1", "req-1") is not None
        assert registry.get("job-1", "req-3") is None
    finally:
        loop.close()
