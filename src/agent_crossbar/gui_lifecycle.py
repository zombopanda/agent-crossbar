"""ChatGPT Pro GUI turn lifecycle: stages, sessions, completion, artifacts.

These are provider-internal building blocks for the browser transport.  They
are deliberately pure state machines fed by observations so the runner's
polling behaviour can be tested without a live browser, and so every claimed
lifecycle guarantee (stable completion, DOM health, session ownership, secure
artifacts) has a deterministic unit test.

Nothing here is part of the public MCP contract.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_crossbar.envelope import sanitize_diagnostic_text

# ── Turn stages ─────────────────────────────────────────────────────────────

TURN_STAGES: tuple[str, ...] = (
    "bootstrap",
    "authenticated",
    "model_selected",
    "composer_ready",
    "prompt_verified",
    "submitted",
    "streaming",
    "complete",
    "cancelled",
    "failed",
)
TERMINAL_TURN_STAGES: frozenset[str] = frozenset({"complete", "cancelled", "failed"})

_MAX_STAGE_EVENTS = 64
_MAX_EVIDENCE_CHARS = 20_000


class TurnLifecycle:
    """Ordered, bounded record of the internal stages of one GUI turn."""

    def __init__(
        self,
        max_entries: int = _MAX_STAGE_EVENTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._entries: list[dict[str, Any]] = []
        self._max_entries = max_entries
        self._clock = clock
        self._terminal: str | None = None

    def enter(self, stage: str, **data: Any) -> dict[str, Any] | None:
        """Record *stage*.  Returns the event, or None if already terminal.

        Terminal stages are single-owner: once ``complete``/``cancelled``/
        ``failed`` is recorded no further stage may overwrite it, so a late
        provider observation cannot publish a second terminal event.
        """
        if stage not in TURN_STAGES:
            raise ValueError(f"Unknown turn stage {stage!r}")
        if self._terminal is not None:
            return None
        if stage in TERMINAL_TURN_STAGES:
            self._terminal = stage
        event = {"stage": stage, "at": self._clock()}
        if data:
            event["data"] = data
        self._entries.append(event)
        if len(self._entries) > self._max_entries:
            del self._entries[0 : len(self._entries) - self._max_entries]
        return event

    @property
    def stage(self) -> str | None:
        return self._entries[-1]["stage"] if self._entries else None

    @property
    def terminal_stage(self) -> str | None:
        return self._terminal

    def reached(self, stage: str) -> bool:
        return any(entry["stage"] == stage for entry in self._entries)

    def stages(self) -> list[str]:
        return [str(entry["stage"]) for entry in self._entries]

    def events(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._entries]


# ── Prompt verification diagnostics ─────────────────────────────────────────


def prompt_mismatch_diagnostics(expected: str, actual: str | None) -> dict[str, Any]:
    """Structural diagnostics for a composer that does not equal the prompt.

    Only lengths and structural facts are reported — never the prompt or the
    observed composer text, which may contain user content.
    """
    observed = actual or ""
    common = 0
    for expected_char, actual_char in zip(expected, observed):
        if expected_char != actual_char:
            break
        common += 1
    return {
        "expected_len": len(expected),
        "actual_len": len(observed),
        "common_prefix_len": common,
        "actual_present": actual is not None,
        "actual_empty": not observed.strip(),
    }


# ── Completion and DOM health ───────────────────────────────────────────────


@dataclass(frozen=True)
class ResponseObservation:
    """One bounded observation of the visible ChatGPT response state."""

    at: float
    response_present: bool
    running: bool
    completion_action: bool
    text: str | None = None


class CompletionTracker:
    """Require stable, non-running, non-empty, actionable response evidence.

    Completion needs every predicate at once and the correlated answer text
    unchanged across ``stability_observations`` consecutive observations that
    span at least ``stability_sec``.  In production the runner polls a couple
    of seconds apart, so two identical observations already prove the visible
    answer stopped changing; ``stability_sec`` remains available for a
    stricter wall-clock window.
    """

    def __init__(self, stability_sec: float = 0.0, stability_observations: int = 2) -> None:
        self.stability_sec = max(0.0, float(stability_sec))
        self.stability_observations = max(1, int(stability_observations))
        self.reason: str | None = None
        self._text: str | None = None
        self._first_at: float = 0.0
        self._count = 0

    def _reset(self, reason: str) -> None:
        self.reason = reason
        self._text = None
        self._count = 0

    def observe(self, obs: ResponseObservation) -> str | None:
        """Return the final answer text once completion is proven, else None."""
        if not obs.response_present:
            self._reset("response_absent")
            return None
        if obs.running:
            self._reset("running")
            return None
        if not obs.text or not obs.text.strip():
            self._reset("final_text_empty")
            return None
        if not obs.completion_action:
            self._reset("completion_action_absent")
            return None
        if obs.text != self._text:
            self._text = obs.text
            self._first_at = obs.at
            self._count = 1
            self.reason = "text_changed"
            if self.stability_observations <= 1 and self.stability_sec <= 0.0:
                self.reason = None
                return obs.text
            return None
        self._count += 1
        if self._count >= self.stability_observations and (
            obs.at - self._first_at >= self.stability_sec
        ):
            self.reason = None
            return obs.text
        self.reason = "awaiting_stability"
        return None


class DomHealthTracker:
    """Fail closed when the response DOM never appears, vanishes, or ends empty."""

    def __init__(
        self,
        missing_grace_sec: float = 180.0,
        vanished_grace_sec: float = 20.0,
        empty_grace_sec: float = 30.0,
    ) -> None:
        self.missing_grace_sec = float(missing_grace_sec)
        self.vanished_grace_sec = float(vanished_grace_sec)
        self.empty_grace_sec = float(empty_grace_sec)
        self._started_at: float | None = None
        self._seen_present = False
        self._absent_since: float | None = None
        self._empty_since: float | None = None
        self.detail: dict[str, Any] = {}

    def observe(self, obs: ResponseObservation) -> str | None:
        """Return a stable DOM-health failure category, or None while healthy."""
        if self._started_at is None:
            self._started_at = obs.at
        if not obs.response_present:
            self._empty_since = None
            if self._seen_present:
                if self._absent_since is None:
                    self._absent_since = obs.at
                elapsed = obs.at - self._absent_since
                if elapsed >= self.vanished_grace_sec:
                    self.detail = {"absent_for_sec": round(elapsed, 3)}
                    return "response_vanished"
                return None
            elapsed = obs.at - self._started_at
            if elapsed >= self.missing_grace_sec:
                self.detail = {"missing_for_sec": round(elapsed, 3)}
                return "response_never_appeared"
            return None

        self._seen_present = True
        self._absent_since = None
        idle_without_text = (
            not obs.running and obs.completion_action and not (obs.text or "").strip()
        )
        if idle_without_text:
            if self._empty_since is None:
                self._empty_since = obs.at
            elapsed = obs.at - self._empty_since
            if elapsed >= self.empty_grace_sec:
                self.detail = {"empty_for_sec": round(elapsed, 3)}
                return "response_completed_empty"
            return None
        self._empty_since = None
        return None


# ── Job-owned browser sessions ──────────────────────────────────────────────


@dataclass
class ChatGptSession:
    """A job's owned ChatGPT browser window."""

    job_id: str
    session_id: str
    candidate_key: str
    browser_name: str
    created_at: float
    pid: int | None = None
    window_id: int | None = None
    state: str = "acquired"
    retired_reason: str | None = None
    last_used_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bind(self, pid: int, window_id: int, now: float | None = None) -> None:
        self.pid = int(pid)
        self.window_id = int(window_id)
        self.state = "active"
        self.last_used_at = time.monotonic() if now is None else now

    def matches(self, pid: Any, window_id: Any) -> bool:
        if self.pid is None or self.window_id is None:
            return False
        try:
            return int(pid) == self.pid and int(window_id) == self.window_id
        except (TypeError, ValueError):
            return False

    def retire(self, reason: str) -> None:
        self.state = "retired"
        self.retired_reason = reason

    @property
    def retired(self) -> bool:
        return self.state == "retired"

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "browser": self.browser_name,
            "candidate": self.candidate_key,
            "pid": self.pid,
            "window_id": self.window_id,
            "state": self.state,
            "retired_reason": self.retired_reason,
        }


class ChatGptSessionManager:
    """Serialize foreground ChatGPT access to one job-owned session at a time.

    Capacity stays at one: foreground CUA input and the system clipboard are
    process-global, so a second concurrent turn could contaminate the first.
    """

    def __init__(self, capacity: int = 1) -> None:
        self.capacity = max(1, int(capacity))
        self._sessions: dict[str, ChatGptSession] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def acquire(
        self,
        job_id: str,
        candidate_key: str,
        browser_name: str,
    ) -> tuple[ChatGptSession | None, str | None]:
        """Acquire the session slot for *job_id*.

        Returns ``(session, None)`` or ``(None, reason)`` when the single slot
        is already owned by a different live job.
        """
        with self._lock:
            for owner, session in list(self._sessions.items()):
                if session.retired:
                    del self._sessions[owner]
            live = [s for s in self._sessions.values() if s.job_id != job_id]
            if len(live) >= self.capacity:
                return None, "busy"
            previous = self._sessions.get(job_id)
            if previous is not None and not previous.retired:
                previous.retire("superseded_by_new_turn")
            self._counter += 1
            session = ChatGptSession(
                job_id=job_id,
                session_id=f"{job_id}-{self._counter}",
                candidate_key=candidate_key,
                browser_name=browser_name,
                created_at=time.monotonic(),
            )
            self._sessions[job_id] = session
            return session, None

    def retire(self, job_id: str, reason: str) -> ChatGptSession | None:
        with self._lock:
            session = self._sessions.pop(job_id, None)
        if session is not None and not session.retired:
            session.retire(reason)
        return session

    def active(self, job_id: str) -> ChatGptSession | None:
        with self._lock:
            session = self._sessions.get(job_id)
        if session is None or session.retired:
            return None
        return session

    def live_count(self) -> int:
        with self._lock:
            return len([s for s in self._sessions.values() if not s.retired])

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


sessions = ChatGptSessionManager()


def verify_owned_window(
    session: ChatGptSession,
    windows: list[dict[str, Any]] | None,
) -> tuple[bool, str | None]:
    """Confirm the owned window is still present and unambiguous."""
    if session.pid is None or session.window_id is None:
        return False, "session_not_bound"
    if windows is None:
        return False, "window_list_unavailable"
    matched = [
        window
        for window in windows
        if session.matches(window.get("pid", session.pid), window.get("window_id"))
    ]
    if not matched:
        return False, "window_closed"
    if len(matched) > 1:
        return False, "window_ambiguous"
    return True, None


# ── Evidence redaction and job-local artifacts ──────────────────────────────

CONTEXT_ENVELOPE_BEGIN = "BEGIN_AGENTS_MCP_CONTEXT"
CONTEXT_ENVELOPE_END = "END_AGENTS_MCP_CONTEXT"

_HIDDEN_REASONING_LINE = re.compile(
    r"^\s*(?:thought(?:s)? for\b|thinking\b|reasoned\b|reasoning\b|chain[- ]of[- ]thought\b)",
    re.IGNORECASE,
)
_CONTEXT_ENVELOPE_BLOCK = re.compile(
    rf"{re.escape(CONTEXT_ENVELOPE_BEGIN)}.*?{re.escape(CONTEXT_ENVELOPE_END)}",
    re.DOTALL,
)


def redact_gui_evidence(text: str, max_chars: int = _MAX_EVIDENCE_CHARS) -> str:
    """Bound and redact browser evidence before it becomes a durable artifact.

    Removes packed context envelopes and visible hidden-reasoning markers,
    then applies the shared secret redaction/length bound.
    """
    if not text:
        return ""
    cleaned = _CONTEXT_ENVELOPE_BLOCK.sub("[REDACTED:context_envelope]", text)
    kept: list[str] = []
    for line in cleaned.splitlines():
        if _HIDDEN_REASONING_LINE.match(line):
            kept.append("[REDACTED:hidden_reasoning]")
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + f"\n[TRUNCATED after {max_chars} chars]"
    return sanitize_diagnostic_text(cleaned)


_SAFE_ARTIFACT_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class JobArtifactStore:
    """Write bounded, redacted GUI diagnostics under one job's directory."""

    def __init__(self, base_dir: str | Path, subdir: str = "chatgpt-pro") -> None:
        self.base_dir = Path(base_dir)
        self.subdir = subdir
        self._entries: list[dict[str, Any]] = []
        self._rejections: list[dict[str, Any]] = []

    @property
    def dir(self) -> Path:
        return self.base_dir / self.subdir

    def _safe_name(self, name: str) -> str:
        candidate = _SAFE_ARTIFACT_NAME.sub("-", Path(name).name).strip("-.") or "artifact.txt"
        return candidate[:120]

    def _resolve(self, name: str) -> Path | None:
        """Return a contained target path, or None when it is unsafe."""
        base = self.base_dir.resolve()
        target = self.dir / self._safe_name(name)
        parent = target.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        try:
            if parent.resolve().relative_to(base) is None:  # pragma: no cover - defensive
                return None
        except (OSError, ValueError):
            return None
        if target.is_symlink():
            self._rejections.append({"name": name, "reason": "symlink_target"})
            return None
        try:
            if target.exists() and target.stat().st_nlink > 1:
                self._rejections.append({"name": name, "reason": "hardlinked_target"})
                return None
        except OSError:
            return None
        try:
            resolved_parent = target.resolve().parent
            resolved_parent.relative_to(base)
        except (OSError, ValueError):
            self._rejections.append({"name": name, "reason": "outside_job_dir"})
            return None
        return target

    def write(
        self,
        name: str,
        content: str,
        *,
        stage: str | None = None,
        kind: str = "diagnostic",
        session: dict[str, Any] | None = None,
    ) -> str | None:
        """Write redacted *content* as a job-local artifact and register it."""
        target = self._resolve(name)
        if target is None:
            return None
        redacted = redact_gui_evidence(content)
        try:
            with open(
                os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW, 0o600),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(redacted)
        except OSError:
            self._rejections.append({"name": name, "reason": "write_failed"})
            return None
        entry: dict[str, Any] = {
            "path": str(target),
            "kind": kind,
            "chars": len(redacted),
        }
        if stage:
            entry["stage"] = stage
        if session:
            entry["session"] = {
                key: session.get(key) for key in ("session_id", "browser", "window_id")
            }
        self._entries.append(entry)
        return str(target)

    def paths(self) -> list[str]:
        return [str(entry["path"]) for entry in self._entries]

    def manifest(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._entries]

    def rejections(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._rejections]
