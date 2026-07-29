"""Input artifact validation and bounded context packing.

The packer is provider-neutral: it consumes only the generic public ``cwd``
and ``scope`` inputs, walks them deterministically, and produces a bounded,
redacted text envelope plus a summary of what was included and omitted.  No
provider-specific field is introduced anywhere in the public contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_crossbar.gui_lifecycle import CONTEXT_ENVELOPE_BEGIN, CONTEXT_ENVELOPE_END
from agent_crossbar.redaction import redact_secrets

DEFAULT_TOTAL_CHARS = 60_000
DEFAULT_FILE_CHARS = 8_000
DEFAULT_CHUNK_CHARS = 2_000
MAX_FILES = 200
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024

SKIP_DIR_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".hg",
        ".idea",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "site-packages",
        "target",
        "venv",
    }
)

PRIORITY_NAMES = (
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)

TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cfg",
        ".conf",
        ".css",
        ".go",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".mjs",
        ".py",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def validate_input_artifact(
    artifact: dict[str, Any],
    sensitivity: str = "normal",
) -> dict[str, Any]:
    """Validate a single input artifact against sensitivity rules.

    Returns a result dict with ``ok: bool`` and optional ``error``/``message``.

    In ``sensitivity="secret"`` mode, artifacts with ``sanitized=False`` are
    rejected.
    """
    sensitivity_val = sensitivity.lower() if sensitivity else "normal"

    if sensitivity_val == "secret" and not artifact.get("sanitized", False):
        return {
            "ok": False,
            "error": "unsanitized_artifact_in_secret_mode",
            "message": "Unsanitized input artifact rejected in secret mode. "
            "Set sanitized=true or use a sanitized artifact.",
        }

    return {
        "ok": True,
        "error": None,
        "message": "Artifact accepted",
    }


def _error(error: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": error, "message": message}


def _priority(path: Path, root: Path) -> tuple[int, str]:
    """Deterministic ordering: repo metadata first, then sorted relative path."""
    relative = path.relative_to(root).as_posix()
    if path.name in PRIORITY_NAMES:
        return (PRIORITY_NAMES.index(path.name), relative)
    return (len(PRIORITY_NAMES), relative)


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def _collect_files(root: Path, requested: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    """Walk *requested* paths deterministically, skipping symlinks and noise."""
    files: list[Path] = []
    omissions: list[dict[str, str]] = []
    seen: set[Path] = set()

    def add_dir(directory: Path) -> None:
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.is_symlink():
                omissions.append({"path": entry.as_posix(), "reason": "symlink"})
                continue
            if entry.is_dir():
                if entry.name in SKIP_DIR_NAMES:
                    omissions.append({"path": entry.as_posix(), "reason": "generated_directory"})
                    continue
                add_dir(entry)
                continue
            if not entry.is_file():
                continue
            if not _is_text_candidate(entry):
                omissions.append({"path": entry.as_posix(), "reason": "not_text"})
                continue
            if entry not in seen:
                seen.add(entry)
                files.append(entry)

    for path in requested:
        if path.is_dir():
            add_dir(path)
        elif path.is_file():
            if path not in seen:
                seen.add(path)
                files.append(path)

    files.sort(key=lambda item: _priority(item, root))
    return files[:MAX_FILES], omissions


def _read_bounded(path: Path, file_chars: int, chunk_chars: int) -> tuple[str, str | None]:
    """Return bounded, redacted text for *path* plus an omission reason."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", "unreadable"
    text, _ = redact_secrets(raw)
    if len(text) <= file_chars:
        return text, None
    head = text[: max(chunk_chars, 0)]
    tail_budget = max(file_chars - len(head), 0)
    tail = text[-tail_budget:] if tail_budget else ""
    return f"{head}\n[... truncated ...]\n{tail}", "chunked_file_budget"


def resolve_context_paths(cwd: str | Path, scope: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the requested scope paths against *cwd* before any provider work."""
    root_raw = Path(cwd) if cwd else None
    if root_raw is None:
        return _error("cwd_required", "context packing requires cwd")
    if not root_raw.is_dir():
        return _error("context_cwd_missing", f"cwd does not exist: {root_raw}")
    root = root_raw.resolve()

    raw_paths = (scope or {}).get("paths") or []
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, list):
        return _error("invalid_scope", "scope.paths must be a list of relative paths")

    resolved: list[Path] = []
    for entry in raw_paths:
        if not isinstance(entry, str) or not entry.strip():
            return _error("invalid_scope", "scope.paths entries must be non-empty strings")
        candidate = (root / entry).expanduser()
        if candidate.is_symlink():
            return _error("context_path_symlink", f"refusing to follow symlinked context: {entry}")
        if not candidate.exists():
            return _error("context_path_missing", f"context path does not exist: {entry}")
        try:
            candidate.resolve().relative_to(root)
        except ValueError:
            return _error("context_path_outside_cwd", f"context path escapes cwd: {entry}")
        resolved.append(candidate.resolve())

    return {"ok": True, "root": root, "paths": resolved}


def validate_attachments(cwd: str | Path, scope: dict[str, Any] | None) -> dict[str, Any]:
    """Validate explicit binary attachments and their size budget."""
    raw = (scope or {}).get("attachments") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return _error("invalid_scope", "scope.attachments must be a list of relative paths")
    root = Path(cwd).resolve() if cwd else None
    if raw and root is None:
        return _error("cwd_required", "scope.attachments requires cwd")
    attachments: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            return _error("invalid_scope", "scope.attachments entries must be non-empty strings")
        assert root is not None
        candidate = (root / entry).expanduser()
        if candidate.is_symlink():
            return _error("attachment_symlink", f"refusing to attach a symlinked path: {entry}")
        if not candidate.is_file():
            return _error("attachment_missing", f"attachment does not exist: {entry}")
        try:
            candidate.resolve().relative_to(root)
        except ValueError:
            return _error("attachment_outside_cwd", f"attachment escapes cwd: {entry}")
        size = candidate.stat().st_size
        if size > MAX_ATTACHMENT_BYTES:
            return _error(
                "attachment_too_large",
                f"attachment {entry} is {size} bytes; limit is {MAX_ATTACHMENT_BYTES}",
            )
        attachments.append({"path": str(candidate.resolve()), "bytes": size})
    return {"ok": True, "attachments": attachments}


def pack_context(cwd: str | Path | None, scope: dict[str, Any] | None) -> dict[str, Any]:
    """Pack a bounded, deterministic context envelope for the requested scope.

    Returns ``{"ok": True, "text": ..., "summary": {...}, "attachments": [...]}``
    with an empty ``text`` when the scope requests no paths, or a stable
    validation error when an explicitly requested path is missing or unsafe.
    """
    scope = scope or {}
    if not scope.get("paths") and not scope.get("attachments"):
        return {
            "ok": True,
            "text": "",
            "attachments": [],
            "summary": {
                "requested": False,
                "files_scanned": 0,
                "files_included": 0,
                "files_omitted": 0,
                "chars_used": 0,
                "omissions": [],
            },
        }

    attachment_result = validate_attachments(cwd or "", scope)
    if not attachment_result["ok"]:
        return attachment_result

    if not scope.get("paths"):
        return {
            "ok": True,
            "text": "",
            "attachments": attachment_result["attachments"],
            "summary": {
                "requested": True,
                "files_scanned": 0,
                "files_included": 0,
                "files_omitted": 0,
                "chars_used": 0,
                "omissions": [],
            },
        }

    resolution = resolve_context_paths(cwd or "", scope)
    if not resolution["ok"]:
        return resolution

    root: Path = resolution["root"]
    total_chars = int(scope.get("max_chars") or DEFAULT_TOTAL_CHARS)
    file_chars = int(scope.get("max_file_chars") or DEFAULT_FILE_CHARS)
    chunk_chars = int(scope.get("max_chunk_chars") or DEFAULT_CHUNK_CHARS)

    files, omissions = _collect_files(root, resolution["paths"])
    included: list[str] = []
    chars_used = 0
    files_included = 0

    for path in files:
        relative = path.relative_to(root).as_posix()
        remaining = total_chars - chars_used
        if remaining <= 0:
            omissions.append({"path": relative, "reason": "total_budget_exhausted"})
            continue
        text, reason = _read_bounded(path, min(file_chars, remaining), chunk_chars)
        if reason == "unreadable":
            omissions.append({"path": relative, "reason": reason})
            continue
        if reason:
            omissions.append({"path": relative, "reason": reason})
        block = f"--- {relative} ---\n{text}\n"
        included.append(block)
        chars_used += len(block)
        files_included += 1

    body = "".join(included)
    envelope = f"{CONTEXT_ENVELOPE_BEGIN}\n{body}{CONTEXT_ENVELOPE_END}\n" if files_included else ""
    return {
        "ok": True,
        "text": envelope,
        "attachments": attachment_result["attachments"],
        "summary": {
            "requested": True,
            "files_scanned": len(files),
            "files_included": files_included,
            "files_omitted": len(omissions),
            "chars_used": chars_used,
            "omissions": omissions[:50],
        },
    }
