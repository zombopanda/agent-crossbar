"""Job directory, metadata, lifecycle, and event writer."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_crossbar.envelope import build_result_envelope
from agent_crossbar.pending_permissions import pending_permissions
from agent_crossbar.run_handles import run_handles
from agent_crossbar.tmux_output import (
    interactive_tmux_output_complete,
    interactive_tmux_output_summary,
    interactive_tmux_session_resumed,
)
from agent_crossbar.usage import resolve_usage_for_meta

_JOB_ID_RE = re.compile(r"^[0-9]{8,}-[a-zA-Z0-9_-]+$")
_FILE_MODE = 0o600
_DIR_MODE = 0o700
_OUTPUT_TAIL_FALLBACK_BYTES = 12000
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "stopped", "cancelled"})
_EVENT_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_EVENT_PROCESS_LOCKS_GUARD = threading.Lock()

# A job is only reaped once its declared runtime deadline plus this bounded
# grace window has elapsed. A live worker publishes its own terminal result at
# the declared deadline, so a job still nonterminal past the grace window
# proves its worker is gone — reaping it never races a healthy execution.
DEADLINE_REAP_GRACE_SEC = 15.0


def _parse_decision_attempt(text: str) -> tuple[str, str] | None:
    """Return ``(request_id, decision)`` when *text* is a structured decision.

    Syntax-first by contract: any JSON object carrying string ``request_id``
    and ``decision`` fields is a decision *attempt*.  A syntactically valid
    attempt that does not match a live pending request is reported as
    ``request_not_pending`` by the caller — it is never reinterpreted as plain
    interactive text.  Anything that is not such an object (including
    malformed JSON or a JSON array) returns ``None`` and keeps today's
    plain-text meaning.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    request_id = parsed.get("request_id")
    decision = parsed.get("decision")
    if not isinstance(request_id, str) or not isinstance(decision, str):
        return None
    return request_id, decision


def _provider_process_is_alive(meta: dict[str, Any]) -> bool:
    """Return whether the recorded ACP provider process is still alive.

    A missing in-process callback does not prove that the provider is gone:
    ``job_send`` commonly runs in a separate CLI/controller process.  The
    durable decision inbox may therefore be used when the provider PID is
    live.  An unidentifiable or dead PID fails closed.
    """
    raw_pid = meta.get("acp_pid")
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return False
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return False

    # A live PID is not an authorization fence: it may have been recycled
    # after the ACP child exited.  Match the immutable process-start token
    # captured alongside the PID at launch.  Without it, a restart fails
    # closed and cannot persist an allow decision into an unrelated process.
    expected_start = meta.get("acp_process_start")
    if expected_start is None or not str(expected_start):
        return False
    actual_start = _process_start_identity(pid)
    if actual_start is None or actual_start != str(expected_start):
        return False
    # ``lstart`` has only one-second wall-clock resolution on macOS (no
    # ``/proc`` ticks-based start time is available there), so start-time
    # alone can collide for a short-lived process reusing the same PID
    # within the same second.  Cross-checking the recorded process group is
    # a second, independent identity signal that must also match; a
    # mismatch fails closed exactly like a start-time mismatch.
    #
    # The repository-owned ACP wrapper calls setsid() immediately after
    # spawn, so the group recorded at launch can be the pre-exec group while
    # the live process has since become its own group leader (pgid == pid).
    # Accept that expected handoff instead of treating it as reuse — mirrors
    # the same tolerance ``acp_runtime.safe_acp_termination`` already applies.
    expected_pgid = meta.get("acp_pgid")
    try:
        expected_pgid_int = int(expected_pgid)
    except (TypeError, ValueError):
        return False
    actual_pgid = _process_group_id(pid)
    if actual_pgid is None:
        return False
    if actual_pgid == expected_pgid_int:
        return True
    # The repository-owned ACP wrapper calls setsid() before exec.  A
    # callback may durably capture the controller's pre-handoff group while a
    # later probe sees the child as its own group leader.  Accept that case
    # only when the expected group is the launch-time parent group recorded by
    # run_acp_job; never accept an arbitrary mismatched group.
    if actual_pgid != pid:
        return False
    try:
        parent_pgid = int(meta.get("acp_parent_pgid"))
    except (TypeError, ValueError):
        return False
    return expected_pgid_int == parent_pgid


def _process_group_id(pid: int) -> int | None:
    """Return the process group id of *pid*, or None if unidentifiable."""
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _process_start_identity(pid: int) -> str | None:
    """Read a stable start token for *pid* on Linux or macOS."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(fields) > 21:
            return fields[21]
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a JSON document atomically for lock-free readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, _FILE_MODE)
        os.replace(temp_path, path)
        path.chmod(_FILE_MODE)
    finally:
        temp_path.unlink(missing_ok=True)


def _cleanup_confirmed(data: dict[str, Any] | None) -> bool:
    """Return whether provider cleanup was positively confirmed."""
    if not data:
        return True
    if data.get("terminated") is False:
        return False
    handle = data.get("run_handle_stop")
    if isinstance(handle, dict) and handle.get("adapter_cancel_confirmed") is False:
        return False
    if (
        isinstance(handle, dict)
        and handle.get("provider_stop") == "requested"
        and not handle.get("provider_stop_confirmed", False)
    ):
        return False
    for key in ("tmux_stop", "print_stop"):
        if key in data and data[key] not in {"terminated", "killed", "missing"}:
            return False
    acp = data.get("acp_stop")
    if isinstance(acp, dict) and acp.get("terminated") is False:
        return False
    return True


def _event_process_lock(path: Path) -> threading.RLock:
    """Return the process-local lock shared by all writers for *path*."""
    key = str(path.resolve())
    with _EVENT_PROCESS_LOCKS_GUARD:
        lock = _EVENT_PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _EVENT_PROCESS_LOCKS[key] = lock
        return lock


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
    """Monotonic per-job event writer with inter-instance serialization."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: int = 0

    def __post_init__(self) -> None:
        """Reload sequence counter from existing events on disk."""
        with self._lock:
            with self._shared_lock():
                self._reload_seq()

    @contextmanager
    def _shared_lock(self):
        """Serialize event reads/writes across threads and processes."""
        process_lock = _event_process_lock(self.path.with_name(".events.lock"))
        with process_lock:
            lock_path = self.path.with_name(".events.lock")
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, _FILE_MODE)
            try:
                os.fchmod(fd, _FILE_MODE)
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - Windows fallback
                    fcntl = None
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

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
            with self._shared_lock():
                # Every writer instance reloads the high-water mark while the
                # inter-process lock is held.  Its private _lock alone cannot
                # protect a heartbeat thread from another JobStore instance.
                self._reload_seq()
                next_seq = self._seq + 1
                event = {
                    "seq": next_seq,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "level": level,
                    "type": type,
                    "message": message,
                    "redacted": redacted,
                    "data": data or {},
                }
                line = (json.dumps(event, separators=(",", ":")) + "\n").encode()
                fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, _FILE_MODE)
                try:
                    os.fchmod(fd, _FILE_MODE)
                    offset = 0
                    while offset < len(line):
                        offset += os.write(fd, line[offset:])
                finally:
                    os.close(fd)
                self._seq = next_seq
                return next_seq

    def _read_since_locked(self, after_seq: int) -> tuple[list[dict[str, Any]], int]:
        """Read events and capture one high-water mark under the shared lock."""
        self._reload_seq()
        results: list[dict[str, Any]] = []
        if not self.path.exists():
            return results, self._seq
        for raw in self.path.read_text().splitlines():
            if not raw.strip():
                continue
            event = json.loads(raw)
            if event["seq"] > after_seq:
                results.append(event)
        return results, self._seq

    def read_since_with_cursor(self, after_seq: int) -> tuple[list[dict[str, Any]], int]:
        """Read events and return the matching high-water mark atomically."""
        with self._lock:
            with self._shared_lock():
                return self._read_since_locked(after_seq)

    def read_since(self, after_seq: int) -> list[dict[str, Any]]:
        """Read events with seq > after_seq, in order."""
        events, _last_seq = self.read_since_with_cursor(after_seq)
        return events

    @property
    def last_seq(self) -> int:
        """Current highest sequence number."""
        with self._lock:
            with self._shared_lock():
                self._reload_seq()
                return self._seq

    @property
    def next_seq(self) -> int:
        """Sequence number that will be assigned to the next event."""
        with self._lock:
            with self._shared_lock():
                self._reload_seq()
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

    def job_status(self, job_id: str) -> str | None:
        """Return the durable status for *job_id*, or None when it is unknown.

        Workers use this to observe a stop that raced their own startup before
        they touch any provider state.
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        return str(self._read_job_meta(job.path).get("status") or "running")

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

    def transition_job_status(
        self,
        job_id: str,
        status: str,
        *,
        allowed_from: set[str] | frozenset[str],
        updates: dict[str, Any] | None = None,
        remove: tuple[str, ...] = (),
        require_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically apply a nonterminal lifecycle transition.

        Terminal statuses are never overwritten.  Workers use this helper for
        ``running``/``awaiting_input`` transitions so a stop racing a native
        completion cannot resurrect a job by writing metadata directly.

        *require_meta*, when given, is a compare-and-swap guard checked
        against the durable meta under the same lock as the write: every key
        must equal the recorded value, or the transition is rejected as
        ``stale_transition`` rather than applied.  A caller whose decision to
        transition was based on native provider evidence read separately
        from (and therefore possibly staler than) this durable meta — e.g.
        the Claude monitor's per-poll ``adapter.status()`` call versus the
        job's ``input_generation`` — uses this to guarantee the transition
        cannot silently overwrite a state change (such as an owner reply)
        that has already landed since that evidence was gathered.
        """
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            current_status = meta.get("status", "running")
            if current_status not in allowed_from:
                return {
                    "ok": False,
                    "error": "job_already_terminal",
                    "job_id": job_id,
                    "current_status": current_status,
                }
            if require_meta is not None:
                for key, expected in require_meta.items():
                    if meta.get(key) != expected:
                        return {
                            "ok": False,
                            "error": "stale_transition",
                            "job_id": job_id,
                            "current_status": current_status,
                        }
            meta.update(updates or {})
            for key in remove:
                meta.pop(key, None)
            meta["status"] = status
            self._write_job_meta(job.path, meta)
        return {"ok": True, "job_id": job_id, "status": status}

    def claim_terminalization(
        self,
        job_id: str,
        *,
        reason: str,
        owner: str,
    ) -> dict[str, Any]:
        """Durably claim the one terminalization operation for a job.

        Deadline watchdogs, monitors, and orphan reapers can all observe the
        same expired job.  This compare-and-set happens under the metadata
        lock *before* provider cleanup, so only the winner may cancel, emit a
        timeout marker, or publish the terminal result.
        """
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            status = meta.get("status", "running")
            if status not in {"running", "awaiting_input", None, ""}:
                return {
                    "ok": False,
                    "error": "job_already_terminal",
                    "job_id": job_id,
                    "current_status": status,
                }
            if meta.get("terminalization_state") == "claimed":
                return {
                    "ok": False,
                    "error": "terminalization_claimed",
                    "job_id": job_id,
                    "current_status": status,
                    "terminalization_owner": meta.get("terminalization_owner"),
                }
            now = datetime.now(timezone.utc).isoformat()
            meta.update(
                {
                    "terminalization_state": "claimed",
                    "terminalization_reason": str(reason)[:128],
                    "terminalization_owner": str(owner)[:128],
                    "terminalization_claimed_at": now,
                }
            )
            self._write_job_meta(job.path, meta)
        return {"ok": True, "job_id": job_id, "status": status, "meta": meta}

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
        return None, ('pass client_session_id="*" for explicit local cross-session access')

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
        if meta.get("status", "running") in {"running", "awaiting_input"}:
            self._reap_deadline_expired_job(job)
            self._finalize_completed_tmux_job(job)

        events, last_seq = job.events.read_since_with_cursor(since_seq)
        next_seq = last_seq + 1

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
        pending_request = meta.get("pending_request")
        if not (isinstance(pending_request, dict) and pending_request.get("state") == "pending"):
            pending_request = None
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
            "pending_request": pending_request,
        }

    # ── result ────────────────────────────────────────────────────────────

    def set_result(
        self,
        job_id: str,
        ok: bool,
        summary: str = "",
        artifacts: list[str] | None = None,
        envelope: dict[str, Any] | None = None,
        release_writer_lease: bool = True,
        terminalization_owner: str | None = None,
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
            if current_status not in ("running", "awaiting_input", None, ""):
                return {
                    "ok": False,
                    "error": "job_already_terminal",
                    "job_id": job_id,
                    "current_status": current_status,
                }
            claimed_owner = meta.get("terminalization_owner")
            if (
                meta.get("terminalization_state") == "claimed"
                and claimed_owner != terminalization_owner
            ):
                return {
                    "ok": False,
                    "error": "terminalization_claimed",
                    "job_id": job_id,
                    "terminalization_owner": claimed_owner,
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
            _atomic_write_json(result_path, result_data)
            meta["status"] = "succeeded" if ok else "failed"
            self._write_job_meta(job.path, meta)
        event_error: str | None = None
        try:
            job.events.write(level="info", type="result", message=summary, data=result_data)
        except Exception as exc:  # durable result remains authoritative
            event_error = type(exc).__name__
        finally:
            if release_writer_lease:
                self._release_writer_lease(job_id, meta)
            else:
                self.update_job_meta(job_id, {"cleanup_pending": True})
        result = {"ok": True, "job_id": job_id}
        if event_error is not None:
            result["warnings"] = [{"code": "result_event_write_failed", "error": event_error}]
        return result

    def _release_writer_lease(self, job_id: str, meta: dict[str, Any] | None = None) -> bool:
        """Release a dev writer lease after publishing a terminal job state."""
        metadata = meta
        if metadata is None:
            job = self.get_job(job_id)
            metadata = self._read_job_meta(job.path) if job is not None else {}
        token = metadata.get("writer_lease_token")
        if not isinstance(token, str) or not token:
            return False
        from agent_crossbar.writer_lease import WriterLeaseStore

        try:
            return WriterLeaseStore(self.state_root).release(token)
        except OSError:
            # The terminal result is authoritative even if cleanup is
            # temporarily unavailable; the next acquire reconciles it.
            return False

    def _writer_lease_token_state(self, token: str) -> str:
        """Return whether a token is present, absent, or unreadable.

        Cleanup retry needs to distinguish an already-reconciled lease from a
        real release failure.  An unreadable lease remains conservative and is
        reported as a failure; only a clean scan with no matching token is
        classified as ``already_absent``.
        """
        if not token:
            return "absent"
        from agent_crossbar.writer_lease import WriterLeaseStore

        lease_store = WriterLeaseStore(self.state_root)
        try:
            paths = list(lease_store.leases_root.glob("*.json"))
        except OSError:
            return "unknown"
        unreadable = False
        for path in paths:
            payload = WriterLeaseStore._read(path)
            if payload is None:
                unreadable = True
                continue
            if payload.get("token") == token:
                return "present"
        return "unknown" if unreadable else "absent"

    def heartbeat_writer_lease(self, job_id: str) -> bool:
        """Refresh a running dev lease so long jobs are not mistaken for stale ones."""
        job = self.get_job(job_id)
        if job is None:
            return False
        token = self._read_job_meta(job.path).get("writer_lease_token")
        if not isinstance(token, str) or not token:
            return False
        from agent_crossbar.writer_lease import WriterLeaseStore

        try:
            return WriterLeaseStore(self.state_root).heartbeat(token)
        except OSError:
            return False

    @staticmethod
    def _mark_pending_settled(meta: dict[str, Any], outcome: str) -> str | None:
        """Mark a still-pending request in *meta* settled. Returns its id.

        Callers hold the job-meta lock; this mutates the in-memory dict only
        so the durable write happens in the same critical section as the
        terminal transition.
        """
        pending = meta.get("pending_request")
        if not (isinstance(pending, dict) and pending.get("state") == "pending"):
            return None
        request_id = pending.get("request_id")
        meta["pending_request"] = {
            **pending,
            "state": outcome,
            "settled_at": datetime.now(timezone.utc).isoformat(),
        }
        return str(request_id) if isinstance(request_id, str) else None

    def settle_pending_request(self, job_id: str, outcome: str = "expired") -> dict[str, Any]:
        """Settle any open pending request for *job_id* (durable + live).

        Idempotent.  Marks the durable record expired/cancelled and cancels
        the live wait, so a later structured decision is rejected instead of
        resurrecting a terminal job.
        """
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            settled_id = self._mark_pending_settled(meta, outcome)
            if settled_id is not None:
                self._write_job_meta(job.path, meta)
        settled_ids = pending_permissions.settle_job(job_id, outcome=outcome)
        return {"ok": True, "job_id": job_id, "settled": settled_ids}

    def set_stopped_result(
        self,
        job_id: str,
        *,
        summary: str,
        envelope: dict[str, Any],
        release_writer_lease: bool = True,
    ) -> dict[str, Any]:
        """Persist the terminal envelope for a job already marked stopped."""
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}
        with self._job_meta_lock(job.path):
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
            _atomic_write_json(result_path, result_data)
        event_error: str | None = None
        try:
            job.events.write(level="info", type="result", message=summary, data=result_data)
        except Exception as exc:  # durable result remains authoritative
            event_error = type(exc).__name__
        finally:
            if release_writer_lease:
                self._release_writer_lease(job_id, meta)
            else:
                self.update_job_meta(job_id, {"cleanup_pending": True})
        result = {"ok": True, "job_id": job_id}
        if event_error is not None:
            result["warnings"] = [{"code": "result_event_write_failed", "error": event_error}]
        return result

    def record_terminal_cleanup_retry(
        self,
        job_id: str,
        *,
        reason: str,
        prior_terminal_status: str,
        cleanup_attempt: dict[str, Any],
        cleanup_result: dict[str, Any],
        cleanup_confirmed: bool,
    ) -> dict[str, Any]:
        """Persist a provider cleanup retry for an already-terminal job.

        A terminal result is normally immutable, but a provider process can
        outlive the worker that published it.  This narrow recovery path
        updates only cleanup evidence and keeps the writer lease until the
        provider reports confirmed cleanup.  Lease release happens after the
        evidence is durable and is never attempted for an unconfirmed result.
        """
        job = self.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job_not_found", "job_id": job_id}

        now = datetime.now(timezone.utc).isoformat()
        attempt = {
            **cleanup_attempt,
            "reason": reason,
            "prior_terminal_status": prior_terminal_status,
            "recorded_at": now,
        }
        safe_result = dict(cleanup_result)
        diagnostic: dict[str, Any] = {
            "job_id": job_id,
            "reason": reason,
            "prior_terminal_status": prior_terminal_status,
            "cleanup_attempt": attempt,
            "cleanup_result": safe_result,
            "cleanup_confirmed": bool(cleanup_confirmed),
        }

        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            current_status = str(meta.get("status") or "")
            if current_status not in _TERMINAL_STATUSES:
                return {
                    "ok": False,
                    "error": "job_not_terminal",
                    "job_id": job_id,
                    "status": current_status,
                }
            result_path = job.path / "result.json"
            try:
                result_data = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                return {
                    "ok": False,
                    "error": "result_not_ready",
                    "job_id": job_id,
                    "status": current_status,
                }
            if not isinstance(result_data, dict):
                return {
                    "ok": False,
                    "error": "result_not_ready",
                    "job_id": job_id,
                    "status": current_status,
                }

            envelope = result_data.get("envelope")
            if not isinstance(envelope, dict):
                envelope = {}
            technical = envelope.get("technical")
            if not isinstance(technical, dict):
                technical = {}
            previous_attempts = technical.get("cleanup_attempts")
            attempts = list(previous_attempts) if isinstance(previous_attempts, list) else []
            attempts.append(attempt)
            # Keep repeated explicit recovery bounded while preserving the
            # latest evidence and enough history to diagnose repeated leaks.
            technical["cleanup_attempts"] = attempts[-8:]
            technical["cleanup_attempt"] = attempt
            technical["cleanup_result"] = safe_result
            # Keep the established provider_cleanup field in sync with the
            # retry receipt; leaving an earlier false receipt beside a later
            # confirmed result would make the terminal envelope contradictory.
            technical["provider_cleanup"] = safe_result
            technical["cleanup_confirmed"] = bool(cleanup_confirmed)
            technical["cleanup_pending"] = not bool(cleanup_confirmed)
            technical["lease_disposition"] = (
                "pending_release" if cleanup_confirmed else "retained_cleanup_unconfirmed"
            )
            envelope["technical"] = technical
            failure = envelope.get("failure")
            if isinstance(failure, dict):
                failure = dict(failure)
                failure_diagnostics = failure.get("diagnostics")
                if not isinstance(failure_diagnostics, dict):
                    failure_diagnostics = {}
                failure["diagnostics"] = {**failure_diagnostics, **diagnostic}
                envelope["failure"] = failure
            result_data["envelope"] = envelope
            result_data["cleanup_retry"] = diagnostic
            _atomic_write_json(result_path, result_data)

            meta["cleanup_pending"] = not bool(cleanup_confirmed)
            meta["cleanup_confirmed"] = bool(cleanup_confirmed)
            meta["cleanup_last_attempt"] = attempt
            meta["cleanup_last_result"] = safe_result
            self._write_job_meta(job.path, meta)

        token = meta.get("writer_lease_token")
        if cleanup_confirmed:
            if isinstance(token, str) and token:
                released = self._release_writer_lease(job_id, meta)
                if released:
                    lease_disposition = "released"
                elif self._writer_lease_token_state(token) == "absent":
                    # The provider cleanup is confirmed and the exact lease
                    # token is already gone (for example, reconciliation won
                    # the race). This is successful cleanup, not a release
                    # failure and must not leave a false retained disposition.
                    lease_disposition = "already_absent"
                else:
                    lease_disposition = "retained_release_failed"
            else:
                released = False
                lease_disposition = "not_present"
        else:
            released = False
            lease_disposition = "retained_cleanup_unconfirmed"

        diagnostic["lease_disposition"] = lease_disposition
        diagnostic["lease_released"] = bool(released)

        # Publish the final lease disposition after the guarded release.  If
        # the release itself failed, the durable evidence explicitly says the
        # lease remains retained and a later retry can safely recover it.
        with self._job_meta_lock(job.path):
            result_path = job.path / "result.json"
            try:
                result_data = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                result_data = None
            if isinstance(result_data, dict):
                envelope = result_data.get("envelope")
                if isinstance(envelope, dict):
                    technical = envelope.get("technical")
                    if isinstance(technical, dict):
                        technical["lease_disposition"] = lease_disposition
                        envelope["technical"] = technical
                    failure = envelope.get("failure")
                    if isinstance(failure, dict):
                        failure = dict(failure)
                        failure_diagnostics = failure.get("diagnostics")
                        if not isinstance(failure_diagnostics, dict):
                            failure_diagnostics = {}
                        failure["diagnostics"] = {**failure_diagnostics, **diagnostic}
                        envelope["failure"] = failure
                    result_data["envelope"] = envelope
                result_data["cleanup_retry"] = diagnostic
                _atomic_write_json(result_path, result_data)
            meta = self._read_job_meta(job.path)
            meta["lease_disposition"] = lease_disposition
            self._write_job_meta(job.path, meta)

        try:
            job.events.write(
                level="info" if cleanup_confirmed else "warn",
                type="provider_cleanup_retry",
                message=(
                    "Provider cleanup retry confirmed"
                    if cleanup_confirmed
                    else "Provider cleanup retry remains unconfirmed"
                ),
                data=diagnostic,
            )
        except Exception as exc:  # durable result remains authoritative
            diagnostic["event_warning"] = {
                "code": "cleanup_retry_event_write_failed",
                "error": type(exc).__name__,
            }

        return {"ok": True, **diagnostic}

    @staticmethod
    def _build_stopped_envelope(
        meta: dict[str, Any],
        *,
        reason: str,
        summary: str,
        technical: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the durable terminal envelope for a provider-neutral stop."""
        requested = {
            "profile": meta.get("profile"),
            "model": meta.get("model"),
            "effort": meta.get("effort"),
            "task": meta.get("task"),
            "interactive": meta.get("interactive", False),
            "cwd": meta.get("cwd"),
        }
        resolved = {
            **requested,
            "backend": meta.get("backend"),
        }
        return build_result_envelope(
            status="cancelled",
            stop_reason=reason,
            output=summary,
            summary=summary,
            created_at=meta.get("created") or meta.get("started_at") or "",
            started_at=meta.get("started_at"),
            requested=requested,
            resolved=resolved,
            usage=resolve_usage_for_meta(meta),
            technical=technical,
        )

    @staticmethod
    def _build_tmux_result_envelope(
        meta: dict[str, Any],
        *,
        summary: str,
        ok: bool,
        exit_code: int | None,
        artifacts: list[str],
        technical: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the same durable envelope for lazy tmux finalization.

        A short-lived MCP process may exit before the main runner publishes an
        envelope.  Lazy finalization must still expose the Claude transcript
        usage and preserve the provider-neutral result shape.
        """
        requested = {
            "profile": meta.get("profile"),
            "model": meta.get("model"),
            "effort": meta.get("effort"),
            "task": meta.get("task"),
            "interactive": meta.get("interactive", False),
            "cwd": meta.get("cwd"),
        }
        resolved = {**requested, "backend": meta.get("backend")}
        failure = None
        if not ok:
            failure = {
                "code": "tmux_provider_exit",
                "stage": "execution",
                "retryable": True,
                "next_action": "inspect_provider_output",
                "diagnostics": {"exit_code": exit_code},
            }
        return build_result_envelope(
            status="completed" if ok else "failed",
            stop_reason="completed" if ok else "provider_exit",
            output=summary,
            summary=summary,
            created_at=meta.get("created") or meta.get("started_at") or "",
            started_at=meta.get("started_at"),
            requested=requested,
            resolved=resolved,
            exit_code=exit_code,
            failure=failure,
            usage=resolve_usage_for_meta(meta),
            artifacts=artifacts,
            technical=technical,
        )

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
                self.set_result(
                    job.job_id,
                    ok=False,
                    summary=message,
                    artifacts=artifacts,
                    envelope=self._build_tmux_result_envelope(
                        meta,
                        summary=message,
                        ok=False,
                        exit_code=None,
                        artifacts=artifacts,
                        technical={
                            "lifecycle_events": self._lifecycle_event_count(job),
                            "native_session_id": meta.get("native_session_id"),
                            "native_full_session_id": meta.get("native_full_session_id"),
                            "lazy_finalized": True,
                        },
                    ),
                )
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
                self.set_result(
                    job.job_id,
                    ok=True,
                    summary=summary,
                    artifacts=artifacts,
                    envelope=self._build_tmux_result_envelope(
                        meta,
                        summary=summary,
                        ok=True,
                        exit_code=0,
                        artifacts=artifacts,
                        technical={
                            "lifecycle_events": self._lifecycle_event_count(job),
                            "native_session_id": meta.get("native_session_id"),
                            "native_full_session_id": meta.get("native_full_session_id"),
                            "lazy_finalized": True,
                        },
                    ),
                )
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
        envelope = self._build_tmux_result_envelope(
            meta,
            summary=summary,
            ok=ok,
            exit_code=exit_code,
            artifacts=artifacts,
            technical={
                "lifecycle_events": self._lifecycle_event_count(job),
                "native_session_id": meta.get("native_session_id"),
                "native_full_session_id": meta.get("native_full_session_id"),
                "lazy_finalized": True,
            },
        )
        self.set_result(
            job.job_id,
            ok=ok,
            summary=summary,
            artifacts=artifacts,
            envelope=envelope,
        )

    def _reap_deadline_expired_job(self, job: Job) -> bool:
        """Terminalize an orphaned nonterminal job whose runtime deadline elapsed.

        A job is an orphan when it is still ``running`` after
        ``max_runtime_sec`` plus :data:`DEADLINE_REAP_GRACE_SEC` has elapsed
        since its recorded start.  Live workers publish their own terminal
        result at the declared deadline (``agent_runner``/``acp_runtime``
        timeout paths), so a job that is still nonterminal past the grace
        window proves its worker is gone (interrupted, crashed, or the owning
        process exited).  Reaping is guarded by ``set_result`` so exactly one
        durable terminal result is ever persisted, even when a late worker or
        a second store instance races the same job.

        ``awaiting_input`` jobs still have a declared runtime bound. Once that
        deadline plus grace elapses they follow the same durable terminal
        lifecycle; otherwise ``set_result`` rejects the timeout and the
        writer lease remains stuck forever.

        Returns True only when this call persisted the terminal result.
        """
        meta = self._read_job_meta(job.path)
        if meta.get("status", "running") not in ("running", "awaiting_input", None, ""):
            return False
        if not self._deadline_expired(meta):
            return False
        try:
            max_runtime = float(meta["max_runtime_sec"])
        except (TypeError, ValueError):
            return False

        claim = self.claim_terminalization(
            job.job_id,
            reason="max_runtime_exceeded",
            owner="deadline_reaper",
        )
        if not claim.get("ok"):
            return False
        # Use the compare-and-set snapshot for all evidence assembled below;
        # another worker cannot mutate the lifecycle while this reaper owns
        # the terminalization operation.
        meta = claim.get("meta", meta)

        summary = f"max_runtime_sec ({max_runtime:g}s) exceeded and job was orphaned"
        finished_at = datetime.now(timezone.utc).isoformat()
        started_at = meta.get("started_at") or meta.get("created")
        created_at = meta.get("created") or started_at or finished_at

        envelope = build_result_envelope(
            status="failed",
            stop_reason="max_runtime_exceeded",
            output=summary,
            summary=summary,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            requested={
                "profile": meta.get("profile"),
                "model": meta.get("model"),
                "effort": meta.get("effort"),
                "task": meta.get("task"),
                "interactive": meta.get("interactive", False),
                "cwd": meta.get("cwd"),
            },
            resolved={
                "profile": meta.get("profile"),
                "model": meta.get("model"),
                "effort": meta.get("effort"),
                "task": meta.get("task"),
                "interactive": meta.get("interactive", False),
                "backend": meta.get("backend"),
                "cwd": meta.get("cwd"),
            },
            failure={
                "stage": "execution",
                "code": "max_runtime_exceeded",
                "retryable": True,
                "next_action": "retry_with_higher_timeout",
                "diagnostics": {
                    "layer": "timeout",
                    "max_runtime_sec": max_runtime,
                    "grace_sec": DEADLINE_REAP_GRACE_SEC,
                },
            },
            usage=resolve_usage_for_meta(meta),
            technical={
                "lifecycle_events": self._lifecycle_event_count(job),
                "native_session_id": meta.get("native_session_id"),
                "native_full_session_id": meta.get("native_full_session_id"),
            },
        )

        # Request provider cleanup before publishing the terminal result. The
        # transport-neutral run handle owns the provider callback; this core
        # module never branches on provider names. The result write below remains
        # the single race-safe terminal claim: if a late worker or another
        # reaper won first, set_result returns job_already_terminal and no
        # second reap event is emitted.
        termination: dict[str, Any] | None = None
        run_handle_stop = run_handles.cancel(
            job.job_id,
            preserve=meta.get("backend") == "acp",
        )
        if run_handle_stop is not None:
            termination = {"run_handle_stop": run_handle_stop}
        if meta.get("backend") == "acp":
            from agent_crossbar.acp_runtime import safe_acp_termination

            try:
                acp_termination = safe_acp_termination(meta)
                termination = {**(termination or {}), "acp_stop": acp_termination}
            except Exception as exc:  # pragma: no cover - defensive cleanup
                # Provider cleanup is best effort; it must never prevent the
                # durable timeout result from being published.
                termination = {
                    **(termination or {}),
                    "terminated": False,
                    "reason": "termination_error",
                    "error": type(exc).__name__,
                    "pid": meta.get("acp_pid"),
                }
            envelope["technical"]["acp_stop"] = termination
        elif termination is not None:
            envelope["technical"]["provider_cleanup"] = termination

        cleanup_ok = _cleanup_confirmed(termination)
        if meta.get("task") == "dev" and termination is None:
            cleanup_ok = False
        if (
            meta.get("backend") in {"acp", "tmux", "print", "claude_bg", "claude_bg_pty"}
            and termination is None
        ):
            # A crashed controller leaves no in-memory callback. Unknown
            # process state is not proof of death, so retain the writer lease.
            cleanup_ok = False
        envelope["technical"]["cleanup_confirmed"] = cleanup_ok
        # Settle any pending owner-mediated request in the same terminalization
        # operation, so a late structured decision cannot resurrect the job.
        self.settle_pending_request(job.job_id, outcome="expired")
        result = self.set_result(
            job.job_id,
            ok=False,
            summary=summary,
            envelope=envelope,
            release_writer_lease=cleanup_ok,
            terminalization_owner="deadline_reaper",
        )
        if not result.get("ok", False):
            return False

        # Publish the reaping marker only after the terminal claim succeeds,
        # so concurrent sweepers cannot leave duplicate reap evidence behind.
        self.send_event(
            job.job_id,
            level="error",
            type="deadline_reaped",
            message=summary,
            data={
                "max_runtime_sec": max_runtime,
                "grace_sec": DEADLINE_REAP_GRACE_SEC,
                **({"acp_stop": termination} if termination is not None else {}),
            },
        )
        return True

    @staticmethod
    def _deadline_expired(meta: dict[str, Any]) -> bool:
        """True when *meta* records a positive runtime bound past its deadline + grace."""
        max_runtime_raw = meta.get("max_runtime_sec")
        started_raw = meta.get("started_at") or meta.get("created")
        try:
            max_runtime = float(max_runtime_raw)
            if max_runtime <= 0:
                return False
            started_ts = datetime.fromisoformat(str(started_raw)).timestamp()
        except (TypeError, ValueError):
            return False
        deadline = started_ts + max_runtime + DEADLINE_REAP_GRACE_SEC
        return datetime.now(timezone.utc).timestamp() >= deadline

    @staticmethod
    def _lifecycle_event_count(job: Job) -> int:
        """Return the job's highest event sequence number; 0 on any error."""
        try:
            return int(job.events.last_seq)
        except Exception:
            return 0

    def reap_expired_jobs(self, cwd: str | Path | None = None) -> int:
        """Reap deadline-expired orphaned jobs and return how many were terminalized.

        When *cwd* is given, only jobs created for that canonical workspace are
        considered.  Controllers use this before acquiring a dev writer lease so
        a dead predecessor never blocks a replacement writer past its declared
        deadline.
        """
        target: str | None = None
        if cwd is not None:
            from agent_crossbar.writer_lease import canonical_cwd as _canonical_cwd

            target = _canonical_cwd(cwd)
        jobs_dir = self.state_root / "jobs"
        if not jobs_dir.is_dir():
            return 0
        reaped = 0
        for entry in sorted(jobs_dir.iterdir()):
            if not entry.is_dir() or not _JOB_ID_RE.match(entry.name):
                continue
            meta = self._read_job_meta(entry)
            if meta.get("status", "running") not in (
                "running",
                "awaiting_input",
                None,
                "",
            ):
                continue
            if target is not None:
                raw_cwd = meta.get("cwd")
                if raw_cwd is None or _canonical_cwd(str(raw_cwd)) != target:
                    continue
            job = self.get_job(entry.name)
            if job is not None and self._reap_deadline_expired_job(job):
                reaped += 1
        return reaped

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
        result_path = job.path / "result.json"
        if meta.get("status") == "stopped":
            # If a result envelope was persisted (e.g. ACP stop), read it.
            if result_path.exists():
                try:
                    result_data = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError):
                    # A crash can leave a temp file but never a partial
                    # replacement. Treat an unreadable result as intermediate
                    # state so callers can retry instead of seeing a decode
                    # exception or a false terminal result.
                    return {
                        "ok": False,
                        "job_id": job_id,
                        "status": "stopped",
                        "error": "result_not_ready",
                        "stop_reason": meta.get("stop_reason", "user_cancelled"),
                        "message": "Stopped job terminal result is not readable yet",
                        "warnings": [],
                        "job_created": True,
                    }
                envelope = result_data.get("envelope", {})
                envelope_technical = envelope.get("technical") or {}
                result = {
                    "ok": True,
                    "job_id": job_id,
                    # Provider-neutral stores expose ``stopped`` while the
                    # ACP server exposes its adapter-level ``cancelled``
                    # status.  Preserve both existing API shapes.
                    "status": (
                        "stopped"
                        if "stop" in envelope_technical
                        else envelope.get("status", "stopped")
                    ),
                    "stop_reason": envelope.get(
                        "stop_reason", meta.get("stop_reason", "user_cancelled")
                    ),
                    "summary": envelope.get(
                        "summary", f"Job stopped: {meta.get('stop_reason', 'user_cancelled')}"
                    ),
                    "artifacts": [],
                    "warnings": [],
                }
                # Pass through envelope technical / failure / resolved fields
                for key in ("technical", "failure", "resolved", "usage"):
                    if key in envelope:
                        result[key] = envelope[key]
                return result
            return {
                "ok": False,
                "job_id": job_id,
                "status": "stopped",
                "error": "result_not_ready",
                "stop_reason": meta.get("stop_reason", "user_cancelled"),
                "message": "Stopped job has no durable terminal result envelope",
                "warnings": [],
                "job_created": True,
            }
        if not result_path.exists():
            self._reap_deadline_expired_job(job)
            self._finalize_completed_tmux_job(job)
            meta = self._read_job_meta(job.path)  # re-read after lazy finalize
        if not result_path.exists():
            # Stopped jobs return a stable "stopped" response even without result.json
            if meta.get("status") == "stopped":
                return {
                    "ok": False,
                    "job_id": job_id,
                    "status": "stopped",
                    "error": "result_not_ready",
                    "stop_reason": meta.get("stop_reason", "user_cancelled"),
                    "message": "Stopped job has no durable terminal result envelope",
                    "warnings": [],
                    "job_created": True,
                }
            return {
                "ok": False,
                "error": "result_not_ready",
                "message": "Result not yet available",
                "warnings": [],
                "job_created": False,
            }
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return {
                "ok": False,
                "job_id": job_id,
                "error": "result_not_ready",
                "message": "Result is being published; retry shortly",
                "warnings": [],
                "job_created": True,
            }
        meta = self._read_job_meta(job.path)

        response: dict[str, Any] = {"ok": True, "job_id": job_id}
        response.update(result_data)
        # Legacy/provider results may not carry an envelope.  Surface the
        # durable metadata status so callers never infer completion from a
        # bare ``ok``/summary response.
        response.setdefault("status", meta.get("status"))

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
        # Ownership is checked above and again here; a structured decision
        # attempt is resolved (or rejected) before any plain-text handling so
        # a stale/unknown/terminal request_id is never reinterpreted as
        # ordinary interactive input.
        attempt = _parse_decision_attempt(text)
        if attempt is not None:
            request_id, decision = attempt
            return self._resolve_pending_decision(job, request_id, decision)
        meta = self._read_job_meta(job.path)
        initial_status = meta.get("status", "running")
        if initial_status not in {"running", "awaiting_input"}:
            return {
                "ok": False,
                "error": "job_already_terminal",
                "job_id": job_id,
                "status": initial_status,
                "warnings": [],
                "job_created": True,
            }
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
        input_transcript_baseline: int | None = None
        raw_output_path = meta.get("tmux_output_path")
        if transport == "tmux" and raw_output_path:
            try:
                input_transcript_baseline = Path(str(raw_output_path)).stat().st_size
            except OSError:
                input_transcript_baseline = None
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

        # Persist a monotonic reply generation after the provider accepted
        # the input.  The monitor can observe this even when a fast provider
        # changes awaiting_input -> done between two polls; relying on an
        # intermediate working state loses that valid completion.
        with self._job_meta_lock(job.path):
            current = self._read_job_meta(job.path)
            if current.get("status", "running") not in {"running", "awaiting_input"}:
                return {
                    "ok": False,
                    "error": "job_already_terminal",
                    "job_id": job_id,
                    "status": current.get("status"),
                    "warnings": [],
                    "job_created": True,
                }
            try:
                generation = int(current.get("input_generation", 0)) + 1
            except (TypeError, ValueError):
                generation = 1
            current["input_generation"] = generation
            current["last_input_at"] = datetime.now(timezone.utc).isoformat()
            if input_transcript_baseline is not None:
                current["input_transcript_baseline"] = input_transcript_baseline
            self._write_job_meta(job.path, current)

        # A successful reply starts another provider turn.  This matters for
        # adapters that previously surfaced a native `awaiting_input` state.
        if initial_status == "awaiting_input":
            resumed = self.transition_job_status(
                job_id,
                "running",
                allowed_from={"awaiting_input"},
                remove=("waiting_for", "question"),
            )
            if not resumed.get("ok"):
                return resumed

        return {"ok": True, "job_id": job_id, "seq": seq}

    def _resolve_pending_decision(self, job: Job, request_id: str, decision: str) -> dict[str, Any]:
        """Resolve one structured decision against the job's pending request.

        The live-registry CAS is authoritative for exactly-once delivery; the
        durable record is checked first so unknown/stale/terminal requests are
        rejected with ``request_not_pending`` without touching the registry.
        """
        live = pending_permissions.get(job.job_id, request_id)
        durable_resolved = False
        decisions: list[str] = []
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            status = meta.get("status", "running")
            pending = meta.get("pending_request")
            if (
                status not in {"running", "awaiting_input"}
                or not isinstance(pending, dict)
                or pending.get("request_id") != request_id
                or pending.get("state") != "pending"
            ):
                return {
                    "ok": False,
                    "error": "request_not_pending",
                    "job_id": job.job_id,
                    "request_id": request_id,
                    "warnings": [],
                    "job_created": True,
                }
            raw_decisions = pending.get("decisions")
            decisions = (
                [item for item in raw_decisions if isinstance(item, str)]
                if isinstance(raw_decisions, list)
                else []
            )
            # The public wire representation is deliberately provider-neutral.
            # Never accept an ACP option id or an escalating allow_always.
            if decision not in {"allow", "reject"} or decision not in decisions:
                return {
                    "ok": False,
                    "error": "invalid_decision",
                    "job_id": job.job_id,
                    "request_id": request_id,
                    "decisions": decisions,
                    "warnings": [],
                    "job_created": True,
                }
            if live is None and not _provider_process_is_alive(meta):
                # No callback and no live provider: this is a restart/death,
                # so settle the durable inbox before returning.  A missing
                # local registry alone is not sufficient evidence to do this.
                meta["pending_request"] = {
                    **pending,
                    "state": "expired",
                    "settled_at": datetime.now(timezone.utc).isoformat(),
                }
                self._write_job_meta(job.path, meta)
                return {
                    "ok": False,
                    "error": "request_not_pending",
                    "job_id": job.job_id,
                    "request_id": request_id,
                    "message": "Pending request has no live provider callback",
                    "warnings": [],
                    "job_created": True,
                }
            # This compare-and-set is the inter-process decision handoff.  It
            # must happen before any live callback delivery, while the same
            # metadata lock that stop/deadline settlement uses is held.
            meta["pending_request"] = {
                **pending,
                "state": "resolved",
                "decision": decision,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_job_meta(job.path, meta)
            durable_resolved = True

        if durable_resolved and live is not None:
            # The callback is only an acceleration; the owner process can be
            # separate and the ACP coroutine also polls the durable inbox.
            pending_permissions.resolve(job.job_id, request_id, decision)
            # A live same-process callback has already received the decision
            # through the registry. Remove the inbox record so ordinary
            # callers retain the historical post-resolution metadata shape.
            # Cross-process owners leave the resolved record in place for the
            # provider coroutine's durable poll to consume.
            with self._job_meta_lock(job.path):
                current = self._read_job_meta(job.path)
                resolved_pending = current.get("pending_request")
                if (
                    isinstance(resolved_pending, dict)
                    and resolved_pending.get("request_id") == request_id
                    and resolved_pending.get("state") == "resolved"
                ):
                    current.pop("pending_request", None)
                    self._write_job_meta(job.path, current)
        return {
            "ok": True,
            "job_id": job.job_id,
            "request_id": request_id,
            "decision": decision,
        }

    # ── stop / list ───────────────────────────────────────────────────────

    def stop_job(
        self,
        job_id: str,
        reason: str = "user_cancelled",
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        client_session_id: str | None = None,
        *,
        persist_result: bool = True,
        release_writer_lease: bool = True,
    ) -> dict[str, Any]:
        """Stop a job and publish its durable terminal result envelope.

        ``persist_result=False`` and ``release_writer_lease=False`` are used
        only by the ACP server path, which must add provider termination
        metadata before claiming the stopped result.  Every public stop still
        defaults to the fail-closed durable result path.
        """
        job, cross_session_note = self._get_owned_job(job_id, client_session_id)
        if job is None:
            result = {"ok": False, "error": "job_not_found", "job_id": job_id}
            self._inject_cross_session_note(result, cross_session_note)
            return result
        # Set the cancellation event before publishing the terminal status for
        # ACP workers.  Other transports retain their historical no-handle
        # stop behavior; their durable stop code owns cleanup directly.
        # Workers use the same startup lock to fence provider spawn, so a
        # pre-start stop cannot be overtaken by a late registration.
        pre_meta = self._read_job_meta(job.path)
        pre_handle_data = run_handles.cancel(
            job_id,
            preserve=pre_meta.get("backend") == "acp",
        )
        with self._job_meta_lock(job.path):
            meta = self._read_job_meta(job.path)
            current_status = meta.get("status", "running")
            if current_status == "stopped":
                # Stopping an already stopped job is idempotent.  The first
                # stop owns the durable result; callers may safely repeat the
                # request without changing its terminal envelope.
                if (job.path / "result.json").exists():
                    repeated_data: dict[str, Any] = {"reason": meta.get("stop_reason", reason)}
                    repeated_handle = run_handles.cancel(job_id)
                    if repeated_handle is not None:
                        repeated_data["run_handle_stop"] = repeated_handle
                    repeated_warning: dict[str, Any] | None = None
                    try:
                        job.events.write(
                            level="info",
                            type="stopped",
                            message=f"Job stopped: {meta.get('stop_reason', reason)}",
                            data=repeated_data,
                        )
                    except Exception as exc:  # result already exists
                        repeated_warning = {
                            "code": "stopped_event_write_failed",
                            "error": type(exc).__name__,
                        }
                    response = {
                        "ok": True,
                        "job_id": job_id,
                        "already_terminal": True,
                        "status": current_status,
                    }
                    if repeated_warning is not None:
                        response["warnings"] = [repeated_warning]
                    return response
                return {
                    "ok": False,
                    "error": "result_not_ready",
                    "job_id": job_id,
                    "status": current_status,
                }
            if current_status not in ("running", "awaiting_input", None, ""):
                return {
                    "ok": False,
                    "error": "job_already_terminal",
                    "job_id": job_id,
                    "status": current_status,
                }
            # Make a stop terminal before attempting provider cleanup. A runner
            # that observes its process exit while cleanup is in flight must not
            # publish a late success over the caller's explicit stop request.
            meta["status"] = "stopped"
            meta["stop_reason"] = reason
            # Settle any pending owner-mediated request in the same critical
            # section that terminalizes the job, so a late structured decision
            # can never resurrect it.
            settled_request_id = self._mark_pending_settled(meta, "cancelled")
            self._write_job_meta(job.path, meta)
        if settled_request_id is not None:
            pending_permissions.settle_job(job_id, outcome="cancelled")
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

        # Generic, transport-neutral provider cancellation. A worker that
        # registered a run handle observes the cancellation on its next check
        # and performs its own provider-side cleanup; this call only requests
        # it and collects bounded metadata. No provider name is branched on
        # here — every transport uses the same interface. When the handle's
        # callback returned an ``acp_stop`` termination receipt, it is lifted
        # to the top level of ``data`` so ``_cleanup_confirmed`` and the
        # durable envelope can honor it like any other cleanup evidence.
        handle_data = pre_handle_data
        if handle_data is not None:
            data["run_handle_stop"] = handle_data
            acp_stop = handle_data.get("acp_stop")
            if isinstance(acp_stop, dict):
                data["acp_stop"] = acp_stop

        if persist_result:
            summary = f"Job stopped: {reason}"
            cleanup_confirmed = _cleanup_confirmed(data)
            technical: dict[str, Any] = {"stop": data}
            if isinstance(data.get("acp_stop"), dict):
                technical["acp_stop"] = data["acp_stop"]
            envelope = self._build_stopped_envelope(
                meta,
                reason=reason,
                summary=summary,
                technical=technical,
            )
            envelope["technical"]["cleanup_confirmed"] = cleanup_confirmed
            try:
                persisted = self.set_stopped_result(
                    job_id,
                    summary=summary,
                    envelope=envelope,
                    release_writer_lease=cleanup_confirmed,
                )
            except OSError as exc:
                persisted = {
                    "ok": False,
                    "error": "terminal_result_persist_failed",
                    "job_id": job_id,
                    "message": type(exc).__name__,
                }
            if not persisted.get("ok"):
                return persisted
        event_error: str | None = None
        try:
            job.events.write(
                level="info",
                type="stopped",
                message=f"Job stopped: {reason}",
                data=data,
            )
        except Exception as exc:  # durable result remains authoritative
            event_error = type(exc).__name__
        finally:
            if not persist_result and release_writer_lease:
                self._release_writer_lease(job_id, meta)
        result = {"ok": True, "job_id": job_id}
        if not persist_result:
            # The non-persist path is used by the ACP server stop flow, which
            # needs the collected cleanup evidence (including any ``acp_stop``
            # receipt) to finish building its durable envelope.
            result["stop_data"] = data
        warnings = list(persisted.get("warnings", [])) if persist_result else []
        if event_error is not None:
            warnings.append({"code": "stopped_event_write_failed", "error": event_error})
        if warnings:
            result["warnings"] = warnings
        return result

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
            if meta.get("status", "running") in {"running", "awaiting_input"}:
                job = self.get_job(job_id)
                if job is not None:
                    self._reap_deadline_expired_job(job)
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
