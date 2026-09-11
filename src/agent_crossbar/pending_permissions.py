"""Provider-neutral owner-mediated pending request registry.

A provider that pauses on an owner decision (an ACP permission request) holds
a live ``asyncio.Future`` here while the durable record lives in the job's
metadata.  Splitting the two lets ``jobs.py`` settle a pending request on any
terminal transition without branching on a provider name, and lets a decision
arriving in a *different* process/thread be delivered onto the owning event
loop with ``call_soon_threadsafe``.

Only the live callback lives here.  The durable job record is the source of
truth for request state and is also the inter-process decision inbox; a
controller in another process writes the decision there and the provider
owner polls it.  The registry is an optimisation for same-process delivery,
never a liveness test for the provider.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

MAX_PENDING_PER_JOB = 16


class PendingRequestLimitError(RuntimeError):
    """The bounded per-job live wait registry is full."""


@dataclass
class PendingRequest:
    """A live owner-decision wait for one ``(job_id, request_id)``."""

    job_id: str
    request_id: str
    loop: asyncio.AbstractEventLoop
    future: "asyncio.Future[str]"
    resolved: bool = False
    settled_outcome: str | None = None


class PendingPermissionRegistry:
    """Thread-safe registry of live owner-decision waits."""

    def __init__(self, max_pending_per_job: int = MAX_PENDING_PER_JOB) -> None:
        self._lock = threading.Lock()
        self._by_job: dict[str, dict[str, PendingRequest]] = {}
        self._max = max_pending_per_job

    def register(
        self, job_id: str, request_id: str, *, loop: asyncio.AbstractEventLoop
    ) -> PendingRequest:
        """Create and register the live wait for *request_id*. Bounded per job."""
        future: "asyncio.Future[str]" = loop.create_future()
        pending = PendingRequest(
            job_id=job_id,
            request_id=request_id,
            loop=loop,
            future=future,
        )
        with self._lock:
            bucket = self._by_job.setdefault(job_id, {})
            if len(bucket) >= self._max:
                if not bucket:
                    self._by_job.pop(job_id, None)
                raise PendingRequestLimitError(f"pending request limit reached for job {job_id!r}")
            bucket[request_id] = pending
        return pending

    def get(self, job_id: str, request_id: str) -> PendingRequest | None:
        with self._lock:
            bucket = self._by_job.get(job_id)
            return bucket.get(request_id) if bucket else None

    def resolve(self, job_id: str, request_id: str, decision: str) -> dict[str, object]:
        """One-shot CAS: resolve a pending wait with *decision*.

        Returns ``{"ok": False, "error": "request_not_pending"}`` when the
        request is unknown or was already resolved/settled — never applies a
        second side effect.
        """
        with self._lock:
            bucket = self._by_job.get(job_id)
            pending = bucket.get(request_id) if bucket else None
            if pending is None or pending.resolved:
                return {"ok": False, "error": "request_not_pending"}
            pending.resolved = True
        self._deliver(pending, lambda: _set_result(pending.future, decision))
        return {"ok": True, "job_id": job_id, "request_id": request_id, "decision": decision}

    def discard(self, job_id: str, request_id: str) -> None:
        """Drop one wait without delivering a decision. Idempotent."""
        with self._lock:
            bucket = self._by_job.get(job_id)
            if bucket is not None:
                bucket.pop(request_id, None)
                if not bucket:
                    self._by_job.pop(job_id, None)

    def settle_job(self, job_id: str, *, outcome: str = "expired") -> list[str]:
        """Cancel every outstanding wait for *job_id*; return settled request ids."""
        with self._lock:
            bucket = self._by_job.pop(job_id, {})
            pendings = list(bucket.values())
        settled: list[str] = []
        for pending in pendings:
            if not pending.resolved:
                pending.resolved = True
                pending.settled_outcome = outcome
                # Bind the current request.  A late-bound lambda would cancel
                # only the final future when multiple requests settle.
                self._deliver(pending, lambda p=pending: _cancel(p.future))
            settled.append(pending.request_id)
        return settled

    def release(self, job_id: str) -> None:
        """Drop all waits for *job_id*. Idempotent."""
        with self._lock:
            self._by_job.pop(job_id, None)

    def clear(self) -> None:
        with self._lock:
            self._by_job.clear()

    def size(self) -> int:
        with self._lock:
            return sum(len(bucket) for bucket in self._by_job.values())

    @staticmethod
    def _deliver(pending: PendingRequest, callback) -> None:
        try:
            pending.loop.call_soon_threadsafe(callback)
        except RuntimeError:
            # The owning event loop is closed; the durable record still
            # settles, so the decision is rejected rather than lost silently.
            pass


def _set_result(future: "asyncio.Future[str]", decision: str) -> None:
    if not future.done():
        future.set_result(decision)


def _cancel(future: "asyncio.Future[str]") -> None:
    if not future.done():
        future.cancel()


pending_permissions = PendingPermissionRegistry()
