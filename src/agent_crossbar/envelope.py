"""Provider-neutral result envelope builder.

Produces the terminal technical-result shape consumed by job_result/get_result
for adapter-based jobs.  Every field is explicit — no guessing, no estimation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent_crossbar.redaction import redact_secrets
from agent_crossbar.usage import unavailable_usage

DIAGNOSTICS_MAX_BYTES = 2048  # 2 KiB

# The only stages a failure envelope may report — where in the job
# lifecycle the failure occurred. Any other value is a programmer error.
FAILURE_STAGES: frozenset[str] = frozenset(
    {"preflight", "launch", "auth", "prompt_delivery", "execution", "finalization"}
)
_SECRET_KEY_PATTERNS = frozenset(
    {
        "token",
        "api_key",
        "apikey",
        "secret",
        "secret_key",
        "password",
        "passwd",
        "credential",
        "private_key",
        # Compound / prefixed keys
        "openai_api_key",
        "anthropic_api_key",
        "agent_token",
        "access_key",
        "access_key_id",
        "access_token",
        "auth_token",
        "api_secret",
        "api_token",
        "client_secret",
        "refresh_token",
        "session_token",
    }
)
_ENV_KEY = "env"


# ── status mapping ────────────────────────────────────────────────────────

_ADAPTER_STATUSES = frozenset({"completed", "failed", "cancelled", "waiting"})


def map_adapter_status(status: str) -> str:
    """Map an adapter result status to a terminal envelope status.

    Only accepts the four terminal states; raises ValueError otherwise.
    """
    if status not in _ADAPTER_STATUSES:
        raise ValueError(f"Unknown adapter status: {status!r}")
    return status


# ── diagnostics ───────────────────────────────────────────────────────────


# Regex for compound secret keys that embed secret substrings
# e.g. "openai_api_key", "anthropic_api_key", "my_agent_token", "AZURE_OPENAI_API_KEY"
_SECRET_KEY_RE = __import__("re").compile(
    r"(?:^|_)(?:api[_-]?key|secret[_-]?key|access[_-]?key|"
    r"access[_-]?token|auth[_-]?token|api[_-]?secret|"
    r"api[_-]?token|client[_-]?secret|refresh[_-]?token|"
    r"session[_-]?token|private[_-]?key|credential|password|passwd|token)"
    r"(?:_|$)",
    __import__("re").IGNORECASE,
)


def _is_secret_key(key: str) -> bool:
    """Return True if *key* looks like a secret/credential field name.

    Matches both simple keys (``token``, ``api_key``) and compound keys
    (``openai_api_key``, ``anthropic_api_key``, ``agent_token``).
    """
    lower = key.lower().replace("-", "_")
    if lower in _SECRET_KEY_PATTERNS:
        return True
    return bool(_SECRET_KEY_RE.search(lower))


def truncate_diagnostics(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *data* safe for the diagnostics payload.

    - Strips ``env`` entirely.
    - Strips secret-key-looking top-level fields.
    - Recursively strips secret-key-looking fields from nested dicts.
    - Truncates any single string value to 2 KiB (UTF-8 bytes).
    """
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if key == _ENV_KEY:
            continue
        if _is_secret_key(key):
            continue
        cleaned[key] = _clean_diagnostics_value(value)
    return cleaned


def _clean_diagnostics_value(value: Any) -> Any:
    if isinstance(value, str):
        value, _ = redact_secrets(value)
        raw = value.encode("utf-8")
        ellipsis_bytes = len("\u2026".encode("utf-8"))  # 3
        effective_max = DIAGNOSTICS_MAX_BYTES - ellipsis_bytes
        if len(raw) > DIAGNOSTICS_MAX_BYTES:
            truncated = raw[:effective_max]
            # Walk back to the last valid UTF-8 boundary
            for cut in range(len(truncated), 0, -1):
                try:
                    return truncated[:cut].decode("utf-8") + "\u2026"
                except UnicodeDecodeError:
                    pass
            return truncated.decode("utf-8", errors="replace") + "\u2026"
        return value
    if isinstance(value, dict):
        return {k: _clean_diagnostics_value(v) for k, v in value.items() if not _is_secret_key(k)}
    if isinstance(value, list):
        return [_clean_diagnostics_value(v) for v in value]
    return value


def sanitize_diagnostic_text(text: str) -> str:
    """Redact secret-looking values and bound *text* to ``DIAGNOSTICS_MAX_BYTES``.

    Shared entry point for any raw provider output (e.g. an event log
    message derived from provider stdout/stderr) that must never carry
    live secrets or unbounded length, without duplicating the redaction
    or truncation rules.
    """
    return _clean_diagnostics_value(text)


def _validate_failure_schema(failure: dict[str, Any]) -> None:
    """Fail fast on a malformed failure block — never publish a bad envelope.

    A caller-constructed failure dict is a programmer input, not untrusted
    user input; a schema violation here means a bug in the caller, so we
    raise immediately instead of persisting a silently-invalid envelope.
    """
    code = failure.get("code")
    if not isinstance(code, str) or not code:
        raise ValueError(f"failure.code must be a non-empty string, got {code!r}")
    stage = failure.get("stage")
    if stage not in FAILURE_STAGES:
        raise ValueError(f"failure.stage must be one of {sorted(FAILURE_STAGES)}, got {stage!r}")
    retryable = failure.get("retryable")
    if not isinstance(retryable, bool):
        raise ValueError(f"failure.retryable must be a bool, got {retryable!r}")
    next_action = failure.get("next_action")
    if not isinstance(next_action, str) or not next_action:
        raise ValueError(f"failure.next_action must be a non-empty string, got {next_action!r}")


# ── envelope ───────────────────────────────────────────────────────────────


def build_result_envelope(
    *,
    status: str,
    stop_reason: str,
    output: str,
    created_at: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    queue_ms: int = 0,
    run_ms: int | None = None,
    total_ms: int | None = None,
    summary: str | None = None,
    requested: dict[str, Any] | None = None,
    resolved: dict[str, Any] | None = None,
    exit_code: int | None = None,
    signal: str | None = None,
    failure: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    changes: list[dict[str, Any]] | None = None,
    artifacts: list[str] | None = None,
    technical: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the spec-compliant terminal result envelope.

    Every value is explicit — callers must provide real data; the builder
    never invents or estimates unavailable values.
    """
    # Map and validate the adapter status
    envelope_status = map_adapter_status(status)

    now = datetime.now(timezone.utc).isoformat()
    effective_finished_at = finished_at or now
    effective_summary = summary if summary is not None else output

    # Compute ms values when timing data is available
    effective_run_ms: int | None = run_ms
    effective_total_ms: int | None = total_ms
    if started_at and effective_run_ms is None:
        try:
            started_dt = datetime.fromisoformat(started_at)
            finished_dt = datetime.fromisoformat(effective_finished_at)
            effective_run_ms = int((finished_dt - started_dt).total_seconds() * 1000)
        except (ValueError, TypeError):
            effective_run_ms = None
    if effective_total_ms is None:
        try:
            created_dt = datetime.fromisoformat(created_at)
            finished_dt = datetime.fromisoformat(effective_finished_at)
            effective_total_ms = int((finished_dt - created_dt).total_seconds() * 1000)
        except (ValueError, TypeError):
            if effective_run_ms is not None:
                effective_total_ms = queue_ms + effective_run_ms

    # Requested fields from the original launch request
    req_defaults: dict[str, Any] = {
        "profile": None,
        "model": None,
        "effort": None,
        "task": None,
        "interactive": None,
        "cwd": None,
    }
    if requested:
        req_defaults.update(requested)
    envelope_requested = {k: req_defaults.get(k) for k in req_defaults}

    # Resolved fields — what was *actually* used
    res_defaults: dict[str, Any] = {
        "profile": None,
        "model": None,
        "effort": None,
        "task": None,
        "interactive": None,
        "backend": None,
        "cwd": None,
    }
    if resolved:
        res_defaults.update(resolved)
    envelope_resolved = {k: res_defaults.get(k) for k in res_defaults}

    # Failure diagnostics — validate schema, then truncate when present
    safe_failure: dict[str, Any] | None = None
    if failure is not None:
        _validate_failure_schema(failure)
        safe_failure = dict(failure)
        if "diagnostics" in safe_failure and isinstance(safe_failure["diagnostics"], dict):
            safe_failure["diagnostics"] = truncate_diagnostics(safe_failure["diagnostics"])

    # Technical block
    tech_defaults: dict[str, Any] = {
        "lifecycle_events": None,
        "turns": None,
        "tool_calls": None,
        "native_session_id": None,
        "native_full_session_id": None,
    }
    if technical:
        tech_defaults.update(technical)
    envelope_technical = {k: tech_defaults.get(k) for k in tech_defaults}

    return {
        "schema_version": "1",
        "status": envelope_status,
        "stop_reason": stop_reason,
        "output": output,
        "summary": effective_summary,
        "timing": {
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": effective_finished_at,
            "queue_ms": queue_ms,
            "run_ms": effective_run_ms,
            "total_ms": effective_total_ms,
        },
        "requested": envelope_requested,
        "resolved": envelope_resolved,
        "process": {
            "exit_code": exit_code,
            "signal": signal,
        },
        "failure": safe_failure,
        "usage": usage if usage is not None else unavailable_usage("not_captured"),
        "changes": changes if changes is not None else [],
        "artifacts": artifacts if artifacts is not None else [],
        "technical": envelope_technical,
    }
