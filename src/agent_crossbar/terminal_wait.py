"""Provider-neutral deterministic waiting for durable job results.

Controllers must not infer a stall from a quiet event stream, an unchanged
workspace, or an intermediate ``result_not_ready`` response.  This helper
keeps that lifecycle decision in code: it returns only a terminal result or an
observed deadline timeout and never calls ``job_stop``.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any


class TerminalWaitTimeout(TimeoutError):
    """The durable job did not publish a terminal result before its deadline."""

    def __init__(self, *, timeout_sec: float, last_result: dict[str, Any]) -> None:
        super().__init__(f"terminal result not available before {timeout_sec:g}s deadline")
        self.timeout_sec = timeout_sec
        self.last_result = last_result


class _OperationDeadlineExceeded(Exception):
    """Internal marker for a single waiter operation crossing its deadline."""


async def _await_with_deadline(value: Awaitable[Any], deadline: float) -> Any:
    """Await one cooperative operation while checking *deadline*.

    A coroutine created after the deadline is closed explicitly so a rejected
    poll/observer cannot leak an un-awaited coroutine warning.  Python cannot
    forcibly stop an awaitable that ignores cancellation; callers must provide
    their own operation timeout for that case.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        if inspect.iscoroutine(value):
            value.close()
        elif isinstance(value, asyncio.Future):
            value.cancel()
        raise _OperationDeadlineExceeded
    try:
        result = await asyncio.wait_for(value, timeout=remaining)
        if time.monotonic() >= deadline:
            raise _OperationDeadlineExceeded
        return result
    except asyncio.TimeoutError as exc:
        raise _OperationDeadlineExceeded from exc


async def wait_for_terminal_result(
    read_result: Callable[[], Awaitable[dict[str, Any]]],
    *,
    timeout_sec: float,
    poll_interval_sec: float = 2.0,
    on_not_ready: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Wait until *read_result* returns a terminal job result.

    ``result_not_ready`` is an expected intermediate state.  It is never
    converted into a cancellation or fallback decision.  The deadline is
    checked between polls and cooperative async reads/observers are bounded by
    the remaining budget.  An awaitable that ignores cancellation can still
    overrun this budget until its own operation timeout settles; MCP callers
    must retain their bounded request timeout.  A caller may inspect
    :class:`TerminalWaitTimeout` at the deadline and decide how to report the
    timeout, but must not treat the exception as evidence that a live job was
    stalled before the deadline.
    """
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if poll_interval_sec <= 0:
        raise ValueError("poll_interval_sec must be positive")

    deadline = time.monotonic() + timeout_sec
    last_result: dict[str, Any] = {
        "ok": False,
        "error": "result_not_ready",
        "message": "Result not yet available",
    }
    first_poll = True
    while True:
        if not first_poll and time.monotonic() >= deadline:
            raise TerminalWaitTimeout(timeout_sec=timeout_sec, last_result=last_result)
        first_poll = False
        current = read_result()
        if not inspect.isawaitable(current):
            raise TypeError("read_result must return an awaitable")
        try:
            current = await _await_with_deadline(current, deadline)
        except _OperationDeadlineExceeded as exc:
            raise TerminalWaitTimeout(timeout_sec=timeout_sec, last_result=last_result) from exc
        if not isinstance(current, dict):
            raise TypeError("read_result must return a dict")
        last_result = current
        if current.get("error") != "result_not_ready":
            return current

        if on_not_ready is not None:
            observed = on_not_ready(current)
            if not inspect.isawaitable(observed):
                raise TypeError("on_not_ready must return an awaitable")
            try:
                await _await_with_deadline(observed, deadline)
            except _OperationDeadlineExceeded as exc:
                raise TerminalWaitTimeout(timeout_sec=timeout_sec, last_result=last_result) from exc

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TerminalWaitTimeout(timeout_sec=timeout_sec, last_result=last_result)
        try:
            await _await_with_deadline(asyncio.sleep(min(poll_interval_sec, remaining)), deadline)
        except _OperationDeadlineExceeded as exc:
            raise TerminalWaitTimeout(timeout_sec=timeout_sec, last_result=last_result) from exc
