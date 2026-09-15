"""Provider/model-level token usage extraction for job result envelopes.

Builds the ``usage`` block consumed by :func:`agent_crossbar.envelope.build_result_envelope`.
Every value is either real native evidence or an explicit unavailable/partial
status — this module never invents or estimates a number, and it never reports
``available: False`` bare: a reason always accompanies it.

Two native evidence sources are supported today:

* Claude Code tmux sessions — read the on-disk transcript Claude Code itself
  writes under ``~/.claude/projects/<slug>/<session>.jsonl`` (plus nested
  ``subagents/agent-*.jsonl`` files for Task-tool subagents), keyed by the
  ``native_full_session_id`` recorded on the job. Streamed duplicate JSONL
  rows that share the same ``requestId`` are deduplicated before summing.
* OpenCode (and other) ACP jobs — the ACP ``PromptResponse.usage`` field
  (an unstable but structured part of the protocol) reported once per prompt;
  no summation/dedup is needed since it is already a session-cumulative total.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"

_USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)

_SLUG_RE = re.compile(r"[^A-Za-z0-9]")


def unavailable_usage(reason: str) -> dict[str, Any]:
    """Return the explicit structured 'no native evidence' usage block.

    Always carries a machine-readable *reason* — never a bare
    ``{"available": False}`` that could be misread as zero usage.
    """
    usage: dict[str, Any] = {
        "available": False,
        "status": STATUS_UNAVAILABLE,
        "source": None,
        "provenance": None,
        "reason": reason,
        "subagents": [],
    }
    usage.update({field: None for field in _USAGE_FIELDS})
    usage["cache_write_tokens"] = None
    return usage


def _available_usage(
    *,
    source: str,
    input_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    total_tokens: int | None,
    subagents: list[dict[str, Any]] | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    status = STATUS_PARTIAL if incomplete_reason else STATUS_COMPLETE
    return {
        "available": True,
        "status": status,
        "source": source,
        "provenance": source,
        "reason": incomplete_reason,
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_write_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "subagents": subagents or [],
    }


def _read_transcript_model(path: Path) -> str | None:
    """Return the first provider model recorded in a Claude transcript."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message")
        model = message.get("model") if isinstance(message, dict) else None
        if isinstance(model, str) and model:
            return model
    return None


# ── Claude Code tmux transcript extraction ─────────────────────────────────


def default_claude_projects_root() -> Path:
    """Return the root directory Claude Code stores project transcripts under.

    Honors ``CLAUDE_CONFIG_DIR`` (Claude Code's own override for ``~/.claude``)
    so a customized install is still discoverable.
    """
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(config_dir) if config_dir else Path.home() / ".claude"
    return base / "projects"


def claude_project_slug(cwd: str) -> str:
    """Return the project directory name Claude Code derives from *cwd*.

    Claude Code replaces every non-alphanumeric character in the absolute
    working directory with ``-`` (one-for-one, no collapsing) to name the
    project directory under ``~/.claude/projects``.
    """
    return _SLUG_RE.sub("-", cwd)


def _dedup_usage_by_request_id(path: Path) -> dict[str, dict[str, Any]]:
    """Return one raw ``usage`` dict per distinct ``requestId`` in *path*.

    Claude Code writes one JSONL row per streamed content block, and every
    row belonging to the same API turn repeats that turn's identical
    cumulative ``usage`` object under a shared ``requestId``. Keeping only
    one sample per ``requestId`` is what prevents summing the same turn's
    usage multiple times.
    """
    by_request: dict[str, dict[str, Any]] = {}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return by_request
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        request_id = entry.get("requestId") or entry.get("uuid")
        if not request_id:
            continue
        by_request[request_id] = usage
    return by_request


def _coerce_token(value: Any) -> int | None:
    """Return a valid non-negative integer token count, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _sum_usage_samples_with_status(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    if not samples:
        return None, None

    fields = (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    totals = {field: 0 for field in fields}
    seen = {field: False for field in fields}
    invalid: set[str] = set()
    for usage in samples:
        for field in fields:
            if field not in usage:
                continue
            seen[field] = True
            value = _coerce_token(usage[field])
            if value is None:
                invalid.add(field)
                continue
            totals[field] += value

    # Input/cache/output are required for an exact reconstructed total.
    required = fields[:4]
    incomplete = sorted(field for field in required if not seen[field] or field in invalid)
    if "reasoning_tokens" in invalid:
        incomplete.append("reasoning_tokens")

    aggregate: dict[str, Any] = {
        field: totals[field] if seen[field] and field not in invalid else None for field in fields
    }
    if all(seen[field] and field not in invalid for field in required):
        total_tokens = sum(totals[field] for field in required)
        if seen["reasoning_tokens"] and "reasoning_tokens" not in invalid:
            total_tokens += totals["reasoning_tokens"]
        elif "reasoning_tokens" in invalid:
            total_tokens = None
    else:
        total_tokens = None
    aggregate["total_tokens"] = total_tokens
    reason = (
        "claude_transcript_missing_or_invalid:" + ",".join(sorted(set(incomplete)))
        if incomplete
        else None
    )
    return aggregate, reason


def _sum_usage_samples(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Sum deduplicated samples while preserving the legacy private helper."""
    aggregate, _reason = _sum_usage_samples_with_status(samples)
    return aggregate


def _aggregate_transcript(path: Path) -> dict[str, Any] | None:
    by_request = _dedup_usage_by_request_id(path)
    return _sum_usage_samples(list(by_request.values()))


def _aggregate_transcript_with_status(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    by_request = _dedup_usage_by_request_id(path)
    return _sum_usage_samples_with_status(list(by_request.values()))


def _read_subagent_meta(meta_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def extract_claude_usage(
    *,
    cwd: str | None,
    full_session_id: str | None,
    projects_root: Path | None = None,
) -> dict[str, Any]:
    """Extract native Claude Code token usage for a tmux job.

    Reads the transcript Claude Code itself persisted for this session,
    deduplicates streamed duplicate rows, and nests any Task-tool subagent
    transcripts under ``subagents``. Returns an explicit unavailable status
    (never zeros) when the session or transcript cannot be located.

    ``projects_root`` overrides the ``~/.claude/projects`` discovery (used by
    tests/replay fixtures); production callers omit it.
    """
    if not cwd or not full_session_id:
        return unavailable_usage("native_session_id_missing")

    root = projects_root if projects_root is not None else default_claude_projects_root()
    project_dir = root / claude_project_slug(cwd)
    transcript_path = project_dir / f"{full_session_id}.jsonl"
    if not transcript_path.is_file():
        return unavailable_usage("claude_transcript_not_found")

    main_usage, main_reason = _aggregate_transcript_with_status(transcript_path)
    if main_usage is None:
        return unavailable_usage("claude_transcript_no_usage_entries")

    subagents: list[dict[str, Any]] = []
    subagents_dir = project_dir / full_session_id / "subagents"
    if subagents_dir.is_dir():
        for agent_file in sorted(subagents_dir.glob("agent-*.jsonl")):
            agent_usage, agent_reason = _aggregate_transcript_with_status(agent_file)
            if agent_usage is None:
                continue
            meta_path = agent_file.with_suffix("").with_suffix(".meta.json")
            agent_meta = _read_subagent_meta(meta_path)
            subagents.append(
                {
                    "agent_id": agent_file.stem.removeprefix("agent-"),
                    "agent_type": agent_meta.get("agentType"),
                    "description": agent_meta.get("description"),
                    "model": _read_transcript_model(agent_file),
                    "status": STATUS_PARTIAL if agent_reason else STATUS_COMPLETE,
                    "reason": agent_reason,
                    **agent_usage,
                }
            )

    return _available_usage(
        source="claude_code_transcript",
        subagents=subagents,
        incomplete_reason=main_reason,
        **main_usage,
    )


# ── ACP native usage extraction ────────────────────────────────────────────

_ACP_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "inputTokens"),
    "cache_creation_input_tokens": ("cached_write_tokens", "cachedWriteTokens"),
    "cache_read_input_tokens": ("cached_read_tokens", "cachedReadTokens"),
    "output_tokens": ("output_tokens", "outputTokens"),
    "reasoning_tokens": ("thought_tokens", "thoughtTokens"),
    "total_tokens": ("total_tokens", "totalTokens"),
}


def _read_acp_field(usage_obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(usage_obj, dict):
            if name in usage_obj and usage_obj[name] is not None:
                return usage_obj[name]
        else:
            value = getattr(usage_obj, name, None)
            if value is not None:
                return value
    return None


def extract_acp_usage(usage_obj: Any) -> dict[str, Any]:
    """Extract native token usage from an ACP ``PromptResponse.usage`` value.

    ``usage_obj`` may be the ``acp.schema.Usage`` model, an equivalent plain
    dict (for tests/fixtures), or ``None`` when the provider never reported
    it (an ACP-spec-optional, unstable field). Fields the provider omitted
    stay ``None`` and mark the result ``partial`` rather than being coerced
    to zero.
    """
    if usage_obj is None:
        return unavailable_usage("acp_provider_omitted_usage")

    raw_values = {
        field: _read_acp_field(usage_obj, aliases) for field, aliases in _ACP_FIELD_ALIASES.items()
    }
    invalid = [
        field
        for field, value in raw_values.items()
        if value is not None and _coerce_token(value) is None
    ]
    values = {
        field: _coerce_token(value) if value is not None else None
        for field, value in raw_values.items()
    }

    if (
        values["input_tokens"] is None
        and values["output_tokens"] is None
        and values["total_tokens"] is None
    ):
        reason = (
            "acp_usage_fields_invalid:" + ",".join(invalid)
            if invalid
            else "acp_provider_omitted_usage"
        )
        return unavailable_usage(reason)

    missing = [
        field
        for field in ("cache_read_input_tokens", "cache_creation_input_tokens", "reasoning_tokens")
        if values[field] is None
    ]
    incomplete = missing + invalid
    incomplete_reason = (
        "acp_provider_did_not_report:" + ",".join(incomplete) if incomplete else None
    )

    return _available_usage(
        source="acp_native",
        incomplete_reason=incomplete_reason,
        **values,
    )


# ── job-meta dispatch ───────────────────────────────────────────────────────


def resolve_usage_for_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Resolve a usage block from durable job metadata alone.

    Used by finalization paths (deadline reaper, tmux completion) that only
    have the job's persisted ``meta.json`` to work from — no live provider
    handle. Dispatches on ``profile``/``backend`` to the matching native
    evidence source.
    """
    profile = meta.get("profile")
    backend = meta.get("backend")

    if profile == "claude":
        return extract_claude_usage(
            cwd=meta.get("cwd"),
            full_session_id=meta.get("native_full_session_id"),
        )

    if backend == "acp":
        # ACP usage is only ever available from the live PromptResponse
        # captured at prompt-completion time (see extract_acp_usage); a job
        # finalized from durable meta alone (deadline reaper, orphan cleanup)
        # never received one.
        return unavailable_usage("acp_session_terminated_before_response")

    return unavailable_usage("usage_capture_not_supported_for_profile")
