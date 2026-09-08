"""Optional, fail-closed admission callback for configured local controllers.

The protocol is intentionally private and small. It is disabled by default;
strict mode is enabled only by the configured local Agents process. The
callback receives one canonical JSON request on stdin and must return exactly
``{"version": 1, "decision": "allow"|"deny", "reason": "..."}``.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any

_MODE_ENV = "AGENT_CROSSBAR_ADMISSION_MODE"
_COMMAND_ENV = "AGENT_CROSSBAR_ADMISSION_COMMAND"
_TIMEOUT_ENV = "AGENT_CROSSBAR_ADMISSION_TIMEOUT_SEC"
_DEFAULT_TIMEOUT = 5.0
_MAX_INPUT = 64 * 1024
_MAX_OUTPUT = 64 * 1024


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    error: str | None = None
    message: str = ""


def _configured_command() -> tuple[list[str] | None, float | None, str | None]:
    mode = os.environ.get(_MODE_ENV, "off").strip().lower()
    if mode not in {"off", "strict"}:
        return None, None, "admission mode must be 'off' or 'strict'"
    raw = os.environ.get(_COMMAND_ENV, "").strip()
    if mode == "off":
        return None, None, None
    if not raw:
        return None, None, "strict admission requires a callback command"
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, None, f"admission command must be a JSON argv array: {exc}"
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part or "\x00" in part for part in command)
    ):
        return None, None, "admission command must be a non-empty JSON argv array"
    timeout_raw = os.environ.get(_TIMEOUT_ENV, "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else _DEFAULT_TIMEOUT
    except ValueError:
        return None, None, "invalid admission timeout"
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 60:
        return None, None, "invalid admission timeout"
    return command, timeout, None


def _kill_process_group(process: subprocess.Popen[bytes], pgid: int | None) -> None:
    """Kill the owned process group, retaining the launch-time PGID if leader died."""
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _invoke(command: list[str], timeout: float, payload: bytes) -> AdmissionDecision:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        pgid = os.getpgid(process.pid)
    except OSError as exc:
        return AdmissionDecision(
            False, "admission_unavailable", f"admission callback failed: {exc}"
        )

    selector = selectors.DefaultSelector()
    streams: dict[str, bytearray] = {}
    pending = memoryview(payload)
    if process.stdin is not None:
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is not None:
            selector.register(stream, selectors.EVENT_READ, name)
            streams[name] = bytearray()
    deadline = time.monotonic() + timeout
    timed_out = False
    output_limit = False
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            ready = selector.select(remaining)
            if not ready:
                timed_out = True
                break
            for key, _mask in ready:
                if key.data == "stdin":
                    try:
                        written = os.write(key.fileobj.fileno(), pending)
                    except (BrokenPipeError, OSError):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        pending = memoryview(b"")
                    else:
                        pending = pending[written:]
                        if not pending:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    continue
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                streams[key.data].extend(chunk)
                if len(streams[key.data]) > _MAX_OUTPUT:
                    output_limit = True
                    break
            if output_limit:
                break
    finally:
        selector.close()

    def close_pipes() -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    if timed_out or output_limit:
        _kill_process_group(process, pgid)
        close_pipes()
        return AdmissionDecision(
            False,
            "admission_timeout" if timed_out else "admission_output_limit",
            "admission callback timed out"
            if timed_out
            else "admission callback output exceeded limit",
        )
    try:
        returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        _kill_process_group(process, pgid)
        close_pipes()
        return AdmissionDecision(False, "admission_timeout", "admission callback timed out")
    finally:
        close_pipes()
    stdout = bytes(streams.get("stdout", b""))
    stderr = bytes(streams.get("stderr", b""))
    if returncode != 0:
        detail = stderr.decode("utf-8", "replace")[:500]
        return AdmissionDecision(False, "admission_failed", detail or "admission callback failed")
    try:
        response = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return AdmissionDecision(False, "admission_malformed", f"invalid admission response: {exc}")
    if not isinstance(response, dict) or set(response) != {"version", "decision", "reason"}:
        return AdmissionDecision(
            False, "admission_malformed", "admission response schema is invalid"
        )
    if (
        type(response.get("version")) is not int
        or response.get("version") != 1
        or not isinstance(response.get("decision"), str)
        or response.get("decision") not in {"allow", "deny"}
    ):
        return AdmissionDecision(
            False, "admission_malformed", "admission response version or decision is invalid"
        )
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return AdmissionDecision(
            False, "admission_malformed", "admission response reason is required"
        )
    if response["decision"] == "deny":
        return AdmissionDecision(False, "admission_denied", reason[:500])
    return AdmissionDecision(True, message=reason[:500])


def admit_request(request: dict[str, Any]) -> AdmissionDecision:
    """Evaluate the canonical request through the optional local callback."""
    command, timeout, error = _configured_command()
    if command is None:
        if error:
            return AdmissionDecision(False, "admission_config", error)
        return AdmissionDecision(True, message="admission disabled")
    canonical = {
        "profile": request.get("profile"),
        "model": request.get("model"),
        "operation": request.get("operation"),
        "task": request.get("task"),
        "effort": request.get("effort"),
        "cwd": request.get("cwd"),
    }
    try:
        payload = (json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n").encode()
    except (TypeError, ValueError) as exc:
        return AdmissionDecision(
            False, "admission_malformed", f"cannot encode admission request: {exc}"
        )
    if len(payload) > _MAX_INPUT:
        return AdmissionDecision(False, "admission_input_limit", "admission request exceeded limit")
    return _invoke(command, timeout or _DEFAULT_TIMEOUT, payload)
