"""Generic in-process run handles for durable job cancellation.

``JobStore.stop_job`` writes terminal metadata first and then asks this
registry to cancel the still-running provider work.  The registry is
transport-neutral on purpose: ``jobs.py`` must never branch on a provider
name to cancel a job.

A handle owns a ``threading.Event`` that the worker polls between provider
actions, plus an optional ``on_cancel`` callback used to attach bounded
metadata (e.g. the owned browser session identity) to the stop event.  The
callback runs on the *stopping* caller's thread, so it must not perform
foreground automation — the worker thread performs provider-side cleanup
when it observes the cancellation.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

MAX_HANDLES = 256


class RunHandle:
    """Cancellation handle for one durable job."""

    def __init__(
        self,
        job_id: str,
        cancel_event: threading.Event | None = None,
        on_cancel: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.job_id = job_id
        self.cancel_event = cancel_event or threading.Event()
        self.on_cancel = on_cancel
        self._lock = threading.Lock()
        self._cancel_data: dict[str, Any] | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> dict[str, Any]:
        """Request cancellation.  Idempotent; repeated calls report ``repeated``."""
        with self._lock:
            if self._cancel_data is not None:
                data = dict(self._cancel_data)
                data["repeated"] = True
                return data
            self.cancel_event.set()
            data = {"cancel_requested": True, "repeated": False}
            if self.on_cancel is not None:
                try:
                    extra = self.on_cancel()
                except Exception as exc:  # never let cleanup metadata break a stop
                    extra = {"cancel_metadata_error": f"{exc.__class__.__name__}: {exc}"}
                if isinstance(extra, dict):
                    data.update(extra)
            self._cancel_data = dict(data)
            return dict(data)


class RunHandleRegistry:
    """Bounded, thread-safe registry of live run handles."""

    def __init__(self, max_handles: int = MAX_HANDLES) -> None:
        self._handles: dict[str, RunHandle] = {}
        self._lock = threading.Lock()
        self._max_handles = max_handles

    def register(
        self,
        job_id: str,
        cancel_event: threading.Event | None = None,
        on_cancel: Callable[[], dict[str, Any] | None] | None = None,
    ) -> RunHandle:
        """Register (or replace) the handle for *job_id* and return it.

        A stop that arrived before registration is preserved: the freshly
        registered handle inherits the already-cancelled state so the worker
        cannot miss a stop that raced its own startup.
        """
        handle = RunHandle(job_id, cancel_event=cancel_event, on_cancel=on_cancel)
        with self._lock:
            previous = self._handles.get(job_id)
            if previous is not None and previous.cancelled:
                handle.cancel_event.set()
            self._handles[job_id] = handle
            while len(self._handles) > self._max_handles:
                oldest = next(iter(self._handles))
                if oldest == job_id:
                    break
                del self._handles[oldest]
        return handle

    def get(self, job_id: str) -> RunHandle | None:
        with self._lock:
            return self._handles.get(job_id)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        """Cancel *job_id* if a handle exists.  Returns bounded stop metadata."""
        handle = self.get(job_id)
        if handle is None:
            return None
        return handle.cancel()

    def is_cancelled(self, job_id: str) -> bool:
        handle = self.get(job_id)
        return bool(handle and handle.cancelled)

    def release(self, job_id: str) -> None:
        """Drop the handle for *job_id*.  Idempotent."""
        with self._lock:
            self._handles.pop(job_id, None)

    def clear(self) -> None:
        with self._lock:
            self._handles.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._handles)


run_handles = RunHandleRegistry()
