"""Durable, provider-neutral serialization for development writers.

The MCP surface intentionally stays unchanged: ``agent_start(task=\"dev\")``
uses this store internally, while controller-local fallbacks use the matching
``agent-crossbar writer-lease`` CLI.  The lease state lives below the same
durable state root as jobs, so independent MCP processes and local controllers
observe one atomic lock per canonical workspace path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_FILE_MODE = 0o600
_DIR_MODE = 0o700
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "stopped", "cancelled"})
_AGE_RECONCILABLE_OWNERS = frozenset({"pending_dev", "local", "root_integration"})
_ROOT_INTEGRATION_STALE_AFTER_SEC = 900.0
_JOB_ID_RE = re.compile(r"^[0-9]{8,}-[a-zA-Z0-9_-]+$")
RECOVERY_ACKNOWLEDGEMENT = "recover-missing-or-corrupt-job"
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def canonical_cwd(cwd: str | Path) -> str:
    """Return one stable identity for relative, absolute, and symlink paths."""
    raw = Path(cwd).expanduser()
    # strict=False preserves existing agent_start validation semantics for a
    # missing path, while resolve() still collapses symlink and ``..`` aliases.
    return str(raw.resolve(strict=False))


@dataclass(frozen=True)
class WriterLeaseResult:
    """Provider-neutral result returned by lease operations."""

    ok: bool
    canonical_cwd: str
    token: str | None = None
    owner_id: str | None = None
    owner_kind: str | None = None
    error: str | None = None
    message: str | None = None
    age_sec: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": self.ok,
            "canonical_cwd": self.canonical_cwd,
        }
        for key in ("token", "owner_id", "owner_kind", "error", "message", "age_sec"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


class WriterLeaseStore:
    """Cross-process lease store rooted at Agent Crossbar's state directory."""

    def __init__(self, state_root: str | Path, *, stale_after_sec: float = 7200.0) -> None:
        if stale_after_sec <= 0:
            raise ValueError("stale_after_sec must be positive")
        self.state_root = Path(state_root).expanduser()
        self.leases_root = self.state_root / "writer-leases"
        self.stale_after_sec = float(stale_after_sec)
        self._lock_path = self.state_root / ".writer-leases.lock"

    def lease_path(self, cwd: str | Path) -> Path:
        """Return the hashed on-disk lease path for a cwd identity."""
        identity = canonical_cwd(cwd)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.leases_root / f"{digest}.json"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize all lease transitions across threads and processes."""
        self.state_root.mkdir(parents=True, mode=_DIR_MODE, exist_ok=True)
        self.state_root.chmod(_DIR_MODE)
        key = str(self._lock_path.resolve())
        with _PROCESS_LOCKS_GUARD:
            process_lock = _PROCESS_LOCKS.setdefault(key, threading.RLock())
        with process_lock:
            fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, _FILE_MODE)
            try:
                os.fchmod(fd, _FILE_MODE)
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - Windows fallback
                    fcntl = None
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                self.leases_root.mkdir(mode=_DIR_MODE, exist_ok=True)
                self.leases_root.chmod(_DIR_MODE)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_name(
            f".{path.name}.{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}.tmp"
        )
        fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            os.replace(temp, path)
            path.chmod(_FILE_MODE)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _external_job_state(state_root: Path, payload: dict[str, Any]) -> str:
        """Classify external ownership without ever treating uncertainty as safe."""
        job_id = payload.get("job_id") or payload.get("owner_id")
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return "corrupt"
        job_dir = state_root / "jobs" / job_id
        meta_path = job_dir / "meta.json"
        if not meta_path.is_file():
            return "missing"
        meta = WriterLeaseStore._read(meta_path)
        if meta is None:
            return "corrupt"
        if meta.get("status") in _TERMINAL_STATUSES:
            return "terminal"
        return "nonterminal"

    @staticmethod
    def _job_is_terminal(state_root: Path, payload: dict[str, Any]) -> bool:
        if payload.get("owner_kind") != "external_job":
            return False
        return WriterLeaseStore._external_job_state(state_root, payload) == "terminal"

    def _is_stale(self, path: Path, payload: dict[str, Any], now: float) -> bool:
        try:
            observed_times = [path.stat().st_mtime]
        except OSError:
            return True
        # Keep the filesystem timestamp as a crash-safe fallback, but also
        # honor the durable heartbeat timestamp.  Taking the oldest valid
        # observation makes reconciliation conservative after partial writes:
        # one stale signal must never allow a second writer to overlap.
        heartbeat = payload.get("heartbeat_at")
        if isinstance(heartbeat, str):
            try:
                observed_times.append(datetime.fromisoformat(heartbeat).timestamp())
            except ValueError:
                pass
        else:
            acquired = payload.get("acquired_at")
            if isinstance(acquired, str):
                try:
                    observed_times.append(datetime.fromisoformat(acquired).timestamp())
                except ValueError:
                    pass
        stale_after = self.stale_after_sec
        if payload.get("owner_kind") == "root_integration":
            stale_after = min(stale_after, _ROOT_INTEGRATION_STALE_AFTER_SEC)
        return now - min(observed_times) >= stale_after

    def _reconcile_locked(self) -> int:
        removed = 0
        now = time.time()
        if not self.leases_root.is_dir():
            return removed
        for path in self.leases_root.glob("*.json"):
            payload = self._read(path)
            if payload is None:
                # Corrupt lease state is fail-closed.  Removing it here would
                # allow an unrelated writer to overlap an unknown owner.
                continue
            if payload.get("owner_kind") == "external_job":
                # An external job's age is never evidence of safe release:
                # only its durable terminal result or explicit stop is.
                if self._job_is_terminal(self.state_root, payload):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        continue
                    removed += 1
                continue
            if payload.get("owner_kind") not in _AGE_RECONCILABLE_OWNERS:
                continue
            if self._is_stale(path, payload, now):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed += 1
        return removed

    def reconcile(self) -> int:
        """Release terminal or stale leases and return the number removed."""
        with self._locked():
            return self._reconcile_locked()

    def acquire(
        self,
        cwd: str | Path,
        *,
        owner_id: str,
        owner_kind: str = "external_job",
    ) -> WriterLeaseResult:
        """Atomically acquire the lease or return a stable ``writer_busy`` error."""
        identity = canonical_cwd(cwd)
        if not owner_id.strip():
            raise ValueError("owner_id must not be empty")
        if not owner_kind.strip():
            raise ValueError("owner_kind must not be empty")
        path = self.lease_path(identity)
        with self._locked():
            self._reconcile_locked()
            current = self._read(path)
            if current is None and path.exists():
                return WriterLeaseResult(
                    ok=False,
                    canonical_cwd=identity,
                    error="writer_lease_corrupt",
                    message=(
                        f"writer lease state is unreadable for canonical cwd {identity}; "
                        "inspect or restore the lease file before retrying"
                    ),
                )
            if current is not None:
                try:
                    age_sec = max(0, int(time.time() - path.stat().st_mtime))
                except OSError:
                    age_sec = None
                current_owner = str(current.get("owner_id") or "unknown")
                current_kind = str(current.get("owner_kind") or "unknown")
                return WriterLeaseResult(
                    ok=False,
                    canonical_cwd=identity,
                    owner_id=current_owner,
                    owner_kind=current_kind,
                    error="writer_busy",
                    age_sec=age_sec,
                    message=(
                        f"development writer lease is held for canonical cwd {identity} by "
                        f"{current_kind} {current_owner}; wait for terminal release or stale "
                        "reconciliation"
                    ),
                )
            token = uuid.uuid4().hex
            payload = {
                "version": 1,
                "canonical_cwd": identity,
                "owner_id": owner_id,
                "owner_kind": owner_kind,
                "token": token,
                "pid": os.getpid(),
                "acquired_at": self._now_iso(),
                "heartbeat_at": self._now_iso(),
            }
            self._write(path, payload)
            return WriterLeaseResult(
                ok=True,
                canonical_cwd=identity,
                token=token,
                owner_id=owner_id,
                owner_kind=owner_kind,
            )

    def attach(self, token: str, *, job_id: str) -> bool:
        """Transfer a pending acquisition to its durable external job owner."""
        if not token or not job_id:
            return False
        with self._locked():
            for path in self.leases_root.glob("*.json"):
                payload = self._read(path)
                if payload is None or payload.get("token") != token:
                    continue
                payload["owner_id"] = job_id
                payload["owner_kind"] = "external_job"
                payload["job_id"] = job_id
                payload["heartbeat_at"] = self._now_iso()
                self._write(path, payload)
                return True
        return False

    def recover(
        self,
        cwd: str | Path,
        *,
        acknowledgement: str,
    ) -> WriterLeaseResult:
        """Explicitly recover only a valid external lease with no readable job.

        This is intentionally not part of the MCP surface.  It requires an
        exact operator acknowledgement and never recovers a nonterminal job.
        """
        identity = canonical_cwd(cwd)
        path = self.lease_path(identity)
        if acknowledgement != RECOVERY_ACKNOWLEDGEMENT:
            return WriterLeaseResult(
                ok=False,
                canonical_cwd=identity,
                error="writer_recovery_confirmation_required",
                message=f"pass acknowledgement={RECOVERY_ACKNOWLEDGEMENT!r} explicitly",
            )
        with self._locked():
            payload = self._read(path)
            if payload is None:
                return WriterLeaseResult(
                    ok=False,
                    canonical_cwd=identity,
                    error="writer_lease_corrupt",
                    message="corrupt lease state cannot be recovered automatically",
                )
            if payload.get("owner_kind") != "external_job":
                return WriterLeaseResult(
                    ok=False,
                    canonical_cwd=identity,
                    error="writer_recovery_unsafe",
                    message="only an external lease with a missing or corrupt job may be recovered",
                )
            state = self._external_job_state(self.state_root, payload)
            if state not in {"missing", "corrupt"}:
                return WriterLeaseResult(
                    ok=False,
                    canonical_cwd=identity,
                    error="writer_recovery_unsafe",
                    message=f"external job state is {state}; wait for terminal result or explicit stop",
                )
            try:
                path.unlink()
            except FileNotFoundError:
                return WriterLeaseResult(
                    ok=False,
                    canonical_cwd=identity,
                    error="writer_lease_not_found",
                    message="writer lease disappeared before recovery",
                )
            return WriterLeaseResult(
                ok=True,
                canonical_cwd=identity,
                owner_id=str(payload.get("owner_id") or "unknown"),
                owner_kind="external_job",
                message=f"recovered external lease with {state} job state",
            )

    def heartbeat(self, token: str) -> bool:
        """Refresh a lease timestamp while its owner is still running."""
        if not token:
            return False
        with self._locked():
            for path in self.leases_root.glob("*.json"):
                payload = self._read(path)
                if payload is None or payload.get("token") != token:
                    continue
                payload["heartbeat_at"] = self._now_iso()
                self._write(path, payload)
                return True
        return False

    def release(self, token: str) -> bool:
        """Release exactly the lease identified by *token*; never another owner."""
        if not token:
            return False
        with self._locked():
            for path in self.leases_root.glob("*.json"):
                payload = self._read(path)
                if payload is None or payload.get("token") != token:
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    return False
                return True
        return False
