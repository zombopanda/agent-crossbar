"""Job directory, metadata, lifecycle, and event writer."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_crossbar.tmux_output import (
    interactive_tmux_output_complete,
    interactive_tmux_output_summary,
    interactive_tmux_session_resumed,
)

_JOB_ID_RE = re.compile(r"^[0-9]{8,}-[a-zA-Z0-9_-]+$")
_FILE_MODE = 0o600
_DIR_MODE = 0o700
_OUTPUT_TAIL_FALLBACK_BYTES = 12000




def default_state_root() -> Path:
    """Return the shared default state root directory.

    - AGENT_CROSSBAR_STATE_DIR (or deprecated AGENT_HARNESS_STATE_DIR)
      overrides the default.
    - Otherwise ~/.local/state/agent-crossbar is used — this is a
      shared, durable location, NOT per-session or per-process.
    """
    from agent_crossbar.env_compat import getenv

    env_dir = getenv("AGENT_CROSSBAR_STATE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".local" / "state" / "agent-crossbar"


def _generate_job_id(existing_ids: set[str]) -> str:
    """Generate a unique job ID matching the required regex."""
    import time

    base = int(time.time() * 1000)  # epoch millis → >=8 digits
    suffix = 0
    while True:
        if suffix == 0:
            candidate = f"{base}-job"
        else:
            candidate = f"{base}-job-{suffix}"
        if candidate not in existing_ids:
            return candidate
        suffix += 1


@dataclass
class EventWriter:
    """Monotonic per-job event writer backed by JSONL with a per-job lock."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: int = 0

    def __post_init__(self) -> None:
        """Reload sequence counter from existing events on disk."""
        self._reload_seq()

    def _reload_seq(self) -> None:
        """Scan events.jsonl and set _seq to the highest seq found."""
        if not self.path.exists():
            self._seq = 0
            return
        max_seq = 0
        for raw in self.path.read_text().splitlines():
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
                s = event.get("seq", 0)
                if s > max_seq:
                    max_seq = s
            except (json.JSONDecodeError, KeyError):
                continue
        self._seq = max_seq

    def write(
        self,
        level: str,
        type: str,  # noqa: A002 – spec field name
        message: str,
        data: dict[str, Any] | None = None,
        redacted: bool = False,
    ) -> int:
        """Append one event line; returns the assigned sequence number."""
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "type": type,
                "message": message,
                "redacted": redacted,
                "data": data or {},
            }
            line = json.dumps(event, separators=(",", ":")) + "\n"
            with open(self.path, "a") as f:
                f.write(line)
            return self._seq

    def read_since(self, after_seq: int) -> list[dict[str, Any]]:
        """Read events with seq > after_seq, in order."""
        results: list[dict[str, Any]] = []
        if not self.path.exists():
            return results
        with self._lock:
            for raw in self.path.read_text().splitlines():
                if not raw.strip():
                    continue
                event = json.loads(raw)
                if event["seq"] > after_seq:
                    results.append(event)
        return results

    @property
    def last_seq(self) -> int:
        """Current highest sequence number."""
        return self._seq

    @property
    def next_seq(self) -> int:
        """Sequence number that will be assigned to the next event."""
        return self._seq + 1


@dataclass
class Job:
    """A running job with its directory and event writer."""

    job_id: str
    path: Path
    profile: str
    operation: str
    events: EventWriter
    transport: str = "auto"
    interactive: bool = False
    sensitivity: str = "normal"


class JobStore:
    """Persistent job store under a state root directory."""

    def __init__(self, state_root: str | Path | None = None) -> None:
        if state_root is None:
            state_root = default_state_root()
        self.state_root = Path(state_root)
        self._known_ids: set[str] = set()
        self._lock = threading.Lock()

    # ── helpers ───────────────────────────────────────────────────────────

    def _read_job_meta(self, job_dir: Path) -> dict[str, Any]:
        """Read meta.json from a job directory."""
        meta_path = job_dir / "meta.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_job_meta(self, job_dir: Path, meta: dict[str, Any]) -> None:
        """Atomically write meta.json with restricted permissions."""
        meta_path = job_dir / "meta.json"
        tmp_path = job_dir / f".meta-{os.getpid()}-{threading.get_ident()}.tmp"
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(meta, separators=(",", ":")) + "\n")
        os.replace(tmp_path, meta_path)
        meta_path.chmod(_FILE_MODE)

    @contextmanager
    def _job_meta_lock(self, job_dir: Path):
        """Serialize metadata state transitions across store instances."""
        lock_path = job_dir / ".meta.lock"
        with self._lock:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, _FILE_MODE)
            try:
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - Windows fallback
                    fcntl = None
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _safe_job_artifact_path(self, job: Job, value: Any, fallback_name: str) -> Path:
        """Resolve a job-local artifact path, ignoring unsafe metadata paths."""
        candidate = Path(value) if isinstance(value, str) and value else job.path / fallback_name
        try:
            candidate.resolve().relative_to(job.path.resolve())
        except (OSError, ValueError):
            return job.path / fallback_name
        return candidate

    def _read_output_tail(self, path: Path, max_bytes: int) -> dict[str, Any] | None:
        """Return a bounded UTF-8 tail for a provider output artifact."""
        if max_bytes <= 0 or not path.exists() or not path.is_file():
            return None

        try:
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > max_bytes:
                    f.seek(-max_bytes, os.SEEK_END)
                    raw = f.read(max_bytes)
                    truncated = True
                else:
                    raw = f.read()
                    truncated = False
        except OSError:
            return None

        if not raw:
            return None

        return {
            "path": str(path),
            "bytes": len(raw),
            "truncated": truncated,
            "text": raw.decode("utf-8", errors="replace"),
        }

    def _read_output_since(
        self, path: Path, offset: int, max_bytes: int
    ) -> tuple[dict[str, Any] | None, int]:
        """Read a bounded forward slice and return its next byte offset."""
        if offset < 0 or max_bytes <= 0 or not path.exists() or not path.is_file():
            return None, max(0, offset)
        try:
            size = path.stat().st_size
            start = min(offset, size)
            with open(path, "rb") as f:
                while start < size:
                    f.seek(start)
                    first = f.read(1)
                    if not first or first[0] & 0xC0 != 0x80:
                        break
                    start += 1
                f.seek(start)
                raw = f.read(max_bytes + 3)
        except OSError:
            return None, max(0, offset)
        if not raw:
            return None, start
        preferred_end = min(max_bytes, len(raw))
        decoded: str | None = None
        end = preferred_end
        while end <= len(raw):
            try:
                decoded = raw[:end].decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                if exc.reason != "unexpected end of data":
                    break
                end += 1
        if decoded is None:
            end = preferred_end
            decoded = raw[:end].decode("utf-8", errors="replace")
        next_offset = start + end
        return {
            "path": str(path),
            "bytes": end,
            "truncated": next_offset < size,
            "text": decoded,
        }, next_offset

    def update_job_meta(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge updates into a job's meta.json."""
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            meta.update(updates)
            self._write_job_meta(job.path, meta)
        return {"ok": True, "job_id": job_id}

    def _refresh_known_ids_locked(self) -> None:
        """Load existing job IDs so new JobStore instances avoid collisions."""
        jobs_dir = self.state_root / "jobs"
        if not jobs_dir.is_dir():
            return
        for entry in jobs_dir.iterdir():
            if entry.is_dir() and _JOB_ID_RE.match(entry.name):
                self._known_ids.add(entry.name)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def create_job(
        self,
        profile: str,
        operation: str,
        transport: str | None = None,
        sensitivity: str | None = None,
        client_session_id: str | None = None,
        client_name: str | None = None,
        cwd: str | None = None,
    ) -> Job:
        """Create a new job directory and event file with restricted permissions."""
        with self._lock:
            self._refresh_known_ids_locked()
            job_id = _generate_job_id(self._known_ids)
            self._known_ids.add(job_id)

        jobs_dir = self.state_root / "jobs"
        self.state_root.mkdir(parents=True, mode=_DIR_MODE, exist_ok=True)
        self.state_root.chmod(_DIR_MODE)
        jobs_dir.mkdir(mode=_DIR_MODE, exist_ok=True)
        jobs_dir.chmod(_DIR_MODE)

        job_dir = jobs_dir / job_id
        try:
            job_dir.mkdir(parents=True, mode=_DIR_MODE, exist_ok=False)
        except FileExistsError:
            raise RuntimeError(f"Job directory already exists: {job_dir}")

        events_path = job_dir / "events.jsonl"
        fd = os.open(str(events_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        os.close(fd)

        transport = transport or "auto"
        interactive = transport in ("tmux", "gui")
        sensitivity = sensitivity or "normal"

        meta = {
            "profile": profile,
            "operation": operation,
            "created": datetime.now(timezone.utc).isoformat(),
            "transport": transport,
            "interactive": interactive,
            "sensitivity": sensitivity,
            "client_session_id": client_session_id,
            "client_name": client_name,
            "cwd": cwd,
        }
        meta_path = job_dir / "meta.json"
        fd = os.open(str(meta_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(meta, separators=(",", ":")) + "\n")

        event_writer = EventWriter(path=events_path)
        return Job(
            job_id=job_id,
            path=job_dir,
            profile=profile,
            operation=operation,
            events=event_writer,
            transport=transport,
            interactive=interactive,
            sensitivity=sensitivity,
        )

    def get_job(self, job_id: str) -> Job | None:
        """Look up an existing job by ID, or None if not found."""
        if not _JOB_ID_RE.match(job_id):
            return None
        job_dir = self.state_root / "jobs" / job_id
        try:
            job_dir.resolve().relative_to(self.state_root.resolve())
        except ValueError:
            return None
        if not job_dir.is_dir():
            return None
        events_path = job_dir / "events.jsonl"
        event_writer = EventWriter(path=events_path)
        meta = self._read_job_meta(job_dir)
        return Job(
            job_id=job_id,
            path=job_dir,
            profile=meta.get("profile", ""),
            operation=meta.get("operation", ""),
            events=event_writer,
            transport=meta.get("transport", "auto"),
            interactive=meta.get("interactive", False),
            sensitivity=meta.get("sensitivity", "normal"),
        )

    def _get_owned_job(
        self,
        job_id: str,
        client_session_id: str | None = None,
    ) -> tuple[Job | None, str | None]:
        """Return (job, cross_session_note).

        *cross_session_note* is a non-secret hint string when the job
        exists on disk but belongs to a different session.  Callers
        SHOULD include it in error responses so the client sees an
        actionable path to authenticated local cross-session access.
        """
        job = self.get_job(job_id)
        if job is None:
            return None, None
        owner = self._read_job_meta(job.path).get("client_session_id")
        if owner is None or owner == client_session_id:
            return job, None
        if client_session_id == "*":
            return job, None
        return None, (
            "pass client_session_id=\"*\" for explicit local cross-session access"
        )

    @staticmethod
    def _inject_cross_session_note(
        result: dict[str, Any],
        note: str | None,
    ) -> None:
        """Add *note* to *result* when non-None."""
        if note is not None:
            result["cross_session_note"] = note

    # ── tail ──────────────────────────────────────────────────────────────

    def job_tail(
        self,
        job_id: str,
        since_seq: int = 0,
        max_events: int | None = None,
        max_bytes: int = 12000,
        output_since_bytes: int | None = None,
        client_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return tail events for a job in the spec shape."""
        if not _JOB_ID_RE.match(job_id):
            return {
                "ok": False,
                "error": "invalid_job_id",
                "job_id": job_id,
                "status": None,
                "last_seq": 0,
                "next_seq": 1,
                "truncated": False,
                "events": [],
            }

        job, cross_session_note = self._get_owned_job(job_id, client_session_id)
        if job is None:
            result = {
                "ok": False,
                "error": "job_not_found",
                "job_id": job_id,
                "status": None,
                "last_seq": 0,
                "next_seq": 1,
                "truncated": False,
                "events": [],
            }
            self._inject_cross_session_note(result, cross_session_note)
            return result

        meta = self._read_job_meta(job.path)
        if meta.get("status", "running") == "running":
            self._finalize_completed_tmux_job(job)

        events = job.events.read_since(since_seq)
        last_seq = job.events.last_seq
        next_seq = job.events.next_seq

        original_event_count = len(events)

        # Apply max_events limit first (if specified)
        if max_events is not None and len(events) > max_events:
            events = events[:max_events]

        # Apply max_bytes limit — complete events only, no partial JSON
        truncated = False
        total_bytes = 0
        clipped: list[dict[str, Any]] = []
        for event in events:
            event_json = json.dumps(event, separators=(",", ":"))
            event_bytes = len(event_json.encode("utf-8"))
            if clipped and total_bytes + event_bytes > max_bytes:
                truncated = True
                break
            clipped.append(event)
            total_bytes += event_bytes

        # If max_events caused truncation but max_bytes didn't, still mark truncated
        if not truncated and max_events is not None and original_event_count > max_events:
            truncated = True

        if truncated:
            # ``last_seq`` describes the last event in this response, not an
            # unseen global high-water mark.  Clients can safely continue with
            # ``since_seq=last_seq`` when ``truncated`` is true.
            last_seq = int(clipped[-1]["seq"]) if clipped else since_seq
            next_seq = last_seq + 1
        meta = self._read_job_meta(job.path)
        transport = meta.get("transport", job.transport)
        output_tail = None
        output_next_bytes = output_since_bytes
        if transport in ("tmux", "print"):
            fallback_name = "tmux-output.log" if transport == "tmux" else "stdout.log"
            meta_key = "tmux_output_path" if transport == "tmux" else "print_output_path"
            output_path = self._safe_job_artifact_path(job, meta.get(meta_key), fallback_name)
            if output_since_bytes is None:
                output_tail = self._read_output_tail(
                    output_path, max_bytes or _OUTPUT_TAIL_FALLBACK_BYTES
                )
                if output_tail is not None:
                    # When reading a full tail (no offset), the next byte to
                    # read from is the current file size — this lets callers
                    # switch to incremental mode after the first poll.
                    try:
                        output_next_bytes = output_path.stat().st_size
                    except OSError:
                        output_next_bytes = output_tail["bytes"]
            else:
                output_tail, output_next_bytes = self._read_output_since(
                    output_path, output_since_bytes, max_bytes or _OUTPUT_TAIL_FALLBACK_BYTES
                )
        return {
            "ok": True,
            "job_id": job_id,
            "status": meta.get("status", "running"),
            "last_seq": last_seq,
            "next_seq": next_seq,
            "truncated": truncated,
            "events": clipped,
            "output_tail": output_tail,
            "output_next_bytes": output_next_bytes,
        }

    # ── result ────────────────────────────────────────────────────────────

    def set_result(
        self,
        job_id: str,
        ok: bool,
        summary: str = "",
        artifacts: list[str] | None = None,
        envelope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write result.json for a job and record a result event (internal/provider use).

        When *envelope* is provided (adapter-based jobs), its fields are
        stored in result.json and surfaced by get_result as top-level keys.
        """
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        # Guard: never overwrite a terminal status — a stopped job must not be
        # resurrected by a late-arriving background completion.  Check BEFORE
        # writing result.json so the file system is never touched on a terminal job.
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            current_status = meta.get("status", "running")
            if current_status not in ("running", None, ""):
                return {
                    "ok": False,
                    "error": "job_already_terminal",
                    "job_id": job_id,
                    "current_status": current_status,
                }
            result_data: dict[str, Any] = {
                "ok": ok,
                "summary": summary,
                "artifacts": artifacts or [],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if envelope is not None:
                result_data["envelope"] = envelope
            result_path = job.path / "result.json"
            fd = os.open(str(result_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(result_data, separators=(",", ":")) + "\n")
            result_path.chmod(_FILE_MODE)
            meta["status"] = "succeeded" if ok else "failed"
            self._write_job_meta(job.path, meta)
        job.events.write(level="info", type="result", message=summary, data=result_data)
        return {"ok": True, "job_id": job_id}

    def set_stopped_result(
        self,
        job_id: str,
        *,
        summary: str,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist the terminal envelope for a job already marked stopped."""
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        meta = self._read_job_meta(job.path)
        if meta.get("status") != "stopped":
            return {
                "ok": False,
                "error": "job_not_stopped",
                "job_id": job_id,
            }
        result_path = job.path / "result.json"
        if result_path.exists():
            return {"ok": True, "job_id": job_id, "already_persisted": True}
        result_data = {
            "ok": False,
            "summary": summary,
            "artifacts": [],
            "envelope": envelope,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        fd = os.open(str(result_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(result_data, separators=(",", ":")) + "\n")
        result_path.chmod(_FILE_MODE)
        job.events.write(level="info", type="result", message=summary, data=result_data)
        return {"ok": True, "job_id": job_id}

    def _finalize_completed_tmux_job(self, job: Job) -> None:
        """Persist tmux results left behind by a short-lived MCP process."""
        if job.transport != "tmux":
            return
        meta = self._read_job_meta(job.path)
        exit_status_path = Path(
            meta.get("tmux_exit_status_path") or job.path / "tmux-exit-status.txt"
        )
        output_path = Path(meta.get("tmux_output_path") or job.path / "tmux-output.log")
        transcript_path = Path(meta.get("tmux_transcript_path") or job.path / "transcript.jsonl")
        output = ""
        try:
            if output_path.exists():
                output = output_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            output = ""

        artifacts = [str(path) for path in (transcript_path, output_path) if path.exists()]
        if not exit_status_path.exists():
            tmux_session = meta.get("tmux_session")
            if meta.get("interactive") is True and tmux_session:
                try:
                    alive = subprocess.run(
                        ["tmux", "has-session", "-t", str(tmux_session)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                except (OSError, subprocess.SubprocessError):
                    alive = None
                if alive is not None and alive.returncode == 0:
                    return
            if meta.get("interactive") is True and interactive_tmux_session_resumed(
                output,
                profile=str(meta.get("profile") or job.profile or ""),
            ):
                message = "Reasonix resumed an existing session; tmux dev jobs must start isolated sessions"
                job.events.write(
                    level="error",
                    type="tmux_session_resumed",
                    message=message,
                    data={
                        "tmux_session": meta.get("tmux_session"),
                        "tmux_output_path": str(output_path),
                        "lazy_finalized": True,
                    },
                )
                self.set_result(job.job_id, ok=False, summary=message, artifacts=artifacts)
                return
            if meta.get("interactive") is True and interactive_tmux_output_complete(
                output,
                profile=str(meta.get("profile") or job.profile or ""),
            ):
                summary = interactive_tmux_output_summary(
                    output,
                    profile=str(meta.get("profile") or job.profile or ""),
                )
                job.events.write(
                    level="info",
                    type="tmux_output_complete",
                    message="Interactive tmux output completed",
                    data={
                        "tmux_session": meta.get("tmux_session"),
                        "tmux_output_path": str(output_path),
                        "lazy_finalized": True,
                    },
                )
                self.set_result(job.job_id, ok=True, summary=summary, artifacts=artifacts)
            return

        try:
            exit_code = int(exit_status_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            exit_code = 1

        ok = exit_code == 0
        summary = output[-4000:] if output else f"Tmux session exited with code {exit_code}"
        job.events.write(
            level="info" if ok else "error",
            type="tmux_exited",
            message=f"Tmux session exited with code {exit_code}",
            data={
                "tmux_session": meta.get("tmux_session"),
                "exit_code": exit_code,
                "lazy_finalized": True,
            },
        )
        self.set_result(job.job_id, ok=ok, summary=summary, artifacts=artifacts)

    def get_result(self, job_id: str, client_session_id: str | None = None) -> dict[str, Any]:
        """Read the final result for a job (public API)."""
        job, cross_session_note = self._get_owned_job(job_id, client_session_id)
        if job is None:
            result = {
                "ok": False,
                "error": "job_not_found",
                "job_id": job_id,
                "warnings": [],
                "job_created": False,
            }
            self._inject_cross_session_note(result, cross_session_note)
            return result
        meta = self._read_job_meta(job.path)
        if meta.get("status") == "stopped":
            return {
                "ok": True,
                "job_id": job_id,
                "status": "stopped",
                "stop_reason": meta.get("stop_reason", "user_cancelled"),
                "summary": f"Job stopped: {meta.get('stop_reason', 'user_cancelled')}",
                "artifacts": [],
                "warnings": [],
            }
        result_path = job.path / "result.json"
        if not result_path.exists():
            self._finalize_completed_tmux_job(job)
            meta = self._read_job_meta(job.path)  # re-read after lazy finalize
        if not result_path.exists():
            # Stopped jobs return a stable "stopped" response even without result.json
            if meta.get("status") == "stopped":
                return {
                    "ok": True,
                    "job_id": job_id,
                    "status": "stopped",
                    "stop_reason": meta.get("stop_reason", "user_cancelled"),
                    "summary": f"Job stopped: {meta.get('stop_reason', 'user_cancelled')}",
                    "artifacts": [],
                    "warnings": [],
                }
            return {
                "ok": False,
                "error": "result_not_ready",
                "message": "Result not yet available",
                "warnings": [],
                "job_created": False,
            }
        result_data = json.loads(result_path.read_text())
        meta = self._read_job_meta(job.path)

        response: dict[str, Any] = {"ok": True, "job_id": job_id}
        response.update(result_data)

        # Surface envelope fields at top level for adapter-based jobs
        if result_data.get("envelope"):
            response.update(result_data["envelope"])

        sensitivity = meta.get("sensitivity", "normal")
        if sensitivity in ("private", "secret"):
            response["sensitivity_warning"] = f"Job has {sensitivity} sensitivity"

        if result_data.get("artifacts"):
            response["raw_artifacts"] = result_data["artifacts"]

        return response

    # ── events ────────────────────────────────────────────────────────────

    def send_event(
        self,
        job_id: str,
        level: str,
        type: str,  # noqa: A002
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an event to a job's event log (internal/provider use)."""
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        seq = job.events.write(level=level, type=type, message=message, data=data)
        return {"ok": True, "job_id": job_id, "seq": seq}

    def send_user_input(
        self, job_id: str, text: str, client_session_id: str | None = None, _sleep: Any = None
    ) -> dict[str, Any]:
        """Send user input to an interactive job. Redacts the raw text.

        The *_sleep* callable (default ``time.sleep``) is injected for
        deterministic test control of inter-keystroke settle delays.
        """
        job, cross_session_note = self._get_owned_job(job_id, client_session_id)
        if job is None:
            result = {
                "ok": False,
                "error": "job_not_found",
                "job_id": job_id,
                "warnings": [],
                "job_created": False,
            }
            self._inject_cross_session_note(result, cross_session_note)
            return result
        meta = self._read_job_meta(job.path)
        interactive = bool(meta.get("interactive", job.interactive))
        if not interactive:
            return {
                "ok": False,
                "error": "job_not_interactive",
                "message": f"Job transport '{job.transport}' is not interactive",
                "warnings": [],
                "job_created": False,
            }
        n_bytes = len(text.encode("utf-8"))
        seq = job.events.write(
            level="info",
            type="user_input",
            message=f"[redacted user input, {n_bytes} bytes]",
            data={"bytes": n_bytes},
            redacted=True,
        )

        # Deliver keystrokes to the tmux session if this is a tmux job.
        transport = job.transport
        if transport == "tmux":
            import re as _re

            safe = _re.sub(r"[^A-Za-z0-9_-]+", "-", job_id).strip("-")
            session = f"agents-{safe}"
            try:
                sleep = _sleep if _sleep is not None else time.sleep
                subprocess.run(
                    ["tmux", "send-keys", "-t", session, "-l", text],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                sleep(0.5)  # bounded settle between text and submit
                subprocess.run(
                    ["tmux", "send-keys", "-t", session, "Enter"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return {
                    "ok": False,
                    "error": "send_user_input_failed",
                    "message": str(exc),
                    "job_id": job_id,
                    "warnings": [],
                    "job_created": False,
                }

        # A successful reply starts another provider turn.  This matters for
        # adapters that previously surfaced a native `awaiting_input` state.
        if meta.get("status") == "awaiting_input":
            meta["status"] = "running"
            meta.pop("waiting_for", None)
            self._write_job_meta(job.path, meta)

        return {"ok": True, "job_id": job_id, "seq": seq}

    # ── stop / list ───────────────────────────────────────────────────────

    def stop_job(
        self,
        job_id: str,
        reason: str = "user_cancelled",
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        client_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Write a stopped event for a job."""
        job, cross_session_note = self._get_owned_job(job_id, client_session_id)
        if job is None:
            result = {"ok": False, "error": "job_not_found", "job_id": job_id}
            self._inject_cross_session_note(result, cross_session_note)
            return result
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            # Make a stop terminal before attempting provider cleanup. A runner
            # that observes its process exit while cleanup is in flight must not
            # publish a late success over the caller's explicit stop request.
            meta["status"] = "stopped"
            meta["stop_reason"] = reason
            self._write_job_meta(job.path, meta)
        data: dict[str, Any] = {"reason": reason}
        tmux_session = meta.get("tmux_session")
        if job.transport == "tmux" and tmux_session:
            runner = run or subprocess.run
            data["tmux_session"] = tmux_session
            try:
                exists = runner(
                    ["tmux", "has-session", "-t", tmux_session],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if exists.returncode == 0:
                    killed = runner(
                        ["tmux", "kill-session", "-t", tmux_session],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    data["tmux_stop"] = "killed" if killed.returncode == 0 else "kill_failed"
                    data["tmux_stop_returncode"] = killed.returncode
                    if killed.stderr:
                        data["tmux_stop_stderr"] = killed.stderr[-1000:]
                else:
                    data["tmux_stop"] = "missing"
                    data["tmux_has_session_returncode"] = exists.returncode
                    if exists.stderr:
                        data["tmux_has_session_stderr"] = exists.stderr[-1000:]
            except (OSError, subprocess.SubprocessError) as exc:
                data["tmux_stop"] = "error"
                data["tmux_stop_error"] = str(exc)

        print_pid = meta.get("print_pid")
        print_pgid = meta.get("print_pgid")
        if job.transport == "print" and isinstance(print_pid, int) and isinstance(print_pgid, int):
            try:
                if os.name == "posix":
                    if os.getpgid(print_pid) != print_pgid:
                        data["print_stop"] = "stale_process"
                    else:
                        os.killpg(print_pgid, signal.SIGTERM)
                        data["print_stop"] = "terminated"
                        data["print_pid"] = print_pid
                else:
                    os.kill(print_pid, signal.SIGTERM)
                    data["print_stop"] = "terminated"
                    data["print_pid"] = print_pid
            except ProcessLookupError:
                data["print_stop"] = "missing"
            except OSError as exc:
                data["print_stop"] = "error"
                data["print_stop_error"] = str(exc)

        job.events.write(
            level="info",
            type="stopped",
            message=f"Job stopped: {reason}",
            data=data,
        )
        return {"ok": True, "job_id": job_id}

    def list_jobs(self, client_session_id: str | None = None) -> list[dict[str, Any]]:
        """List all existing jobs under the state root."""
        jobs_dir = self.state_root / "jobs"
        if not jobs_dir.is_dir():
            return []
        result: list[dict[str, Any]] = []
        for entry in sorted(jobs_dir.iterdir()):
            if not entry.is_dir():
                continue
            job_id = entry.name
            if not _JOB_ID_RE.match(job_id):
                continue
            meta: dict[str, Any] = {}
            meta_path = entry / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            if (
                client_session_id is not None
                and client_session_id != "*"
                and meta.get("client_session_id") != client_session_id
            ):
                continue
            if meta.get("status", "running") == "running":
                job = self.get_job(job_id)
                if job is not None:
                    self._finalize_completed_tmux_job(job)
                    meta = self._read_job_meta(entry)
            item = {
                "job_id": job_id,
                "profile": meta.get("profile", ""),
                "operation": meta.get("operation", ""),
                "transport": meta.get("transport", "auto"),
                "status": meta.get("status", "running"),
            }
            for key in ("client_session_id", "client_name", "cwd"):
                if meta.get(key) is not None:
                    item[key] = meta[key]
            result.append(item)
        return result
