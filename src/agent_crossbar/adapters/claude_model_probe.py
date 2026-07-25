"""Claude interactive /model picker probe via bounded PTY.

Launches Claude in safe interactive mode, sends ``/model``, captures the
model-picker TUI output, and parses it into a ``ModelCatalog``.

Lifecycle status/completion continues to use ``claude agents --json --all``;
this module is discovery-only.
"""

from __future__ import annotations

import os
import re
import select
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

# ── OS guard ────────────────────────────────────────────────────────────────


def _is_posix_pty_supported() -> bool:
    """PTY probe requires macOS or Linux."""
    return sys.platform in ("darwin", "linux")


# ── Safe launch command ─────────────────────────────────────────────────────

# Minimal safe interactive command that preserves OAuth subscription.
# --disable-slash-commands is NOT included because /model is required.
# --bare is NOT used (it disables OAuth).
# -p / --print is NOT used (we need the interactive TUI).
# --ax-screen-reader flattens TUI output for deterministic parsing.
_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'

_SAFE_CLAUDE_BASE_ARGS: tuple[str, ...] = (
    "claude",
    "--safe-mode",
    "--permission-mode",
    "plan",
    "--strict-mcp-config",
    "--mcp-config",
    _EMPTY_MCP_CONFIG,
    "--ax-screen-reader",
)


def _build_probe_argv() -> list[str]:
    """Build the full argv including unique session-id and name.

    Returns a new list each call — the session-id UUID and derived name
    are unique per invocation for traceability.
    """
    session_id = str(uuid.uuid4())
    short_id = session_id[:8]
    name = f"agent-crossbar-model-probe-{short_id}"
    return list(_SAFE_CLAUDE_BASE_ARGS) + [
        "--session-id",
        session_id,
        "--name",
        name,
    ]


# ── PTY constants ────────────────────────────────────────────────────────────

_MAX_CAPTURED_BYTES = 32 * 1024  # 32 KiB
_HARD_TIMEOUT_SEC = 15.0
# Claude Code prompt: heavy right-pointing angle quotation mark ❯ (U+276F)
# or legacy ">" prompt, at end of sanitized output.
_PROMPT_PATTERN = re.compile(rb"(?:\xe2\x9d\xaf|>|\$)\s*$")
_ANSI_ESCAPE_RE = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")
# OSC sequences (e.g. title changes): ESC ] ... BEL or ESC ] ... ESC \
_OSC_RE = re.compile(rb"\x1b\].*?(\x07|\x1b\\)")
# ESC 7 / ESC 8 (save/restore cursor) — not covered by CSI regex
_ESC_7_8_RE = re.compile(rb"\x1b ?[78]")
# Charset shifts: SI (\x0f), SO (\x0e), ESC ( B (select ASCII)
_CHARSET_SHIFT_RE = re.compile(rb"[\x0e\x0f]|\x1b\(B")


# ── Public types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeResult:
    """Raw result from a model-picker probe."""

    output: str | None  # Raw captured PTY output (ANSI stripped for parsing)
    error: str | None  # Error message when probe failed


class ClaudeModelProbe(Protocol):
    """Injectable boundary for the Claude interactive /model PTY probe."""

    def probe(self) -> ProbeResult: ...


# ── POSIX implementation ────────────────────────────────────────────────────


class PosixClaudeModelProbe:
    """Production probe using ``os.forkpty()`` + ``os.execvp()``.

    Always terminates the child and closes the PTY master fd in ``finally``.

    macOS + Linux only.  Other platforms return ``unsupported`` error via
    the ``probe()`` method.
    """

    def probe(self) -> ProbeResult:
        if not _is_posix_pty_supported():
            return ProbeResult(
                output=None,
                error="unsupported: Claude model probe requires macOS or Linux PTY",
            )

        try:
            pid, master_fd = os.forkpty()
        except OSError as exc:
            return ProbeResult(
                output=None,
                error=f"PTY fork failed: {exc}",
            )

        if pid == 0:
            # ── child ────────────────────────────────────────────────
            try:
                # Ensure clean signal handling
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGHUP, signal.SIG_DFL)

                argv = _build_probe_argv()
                os.execvp("claude", argv)
            except Exception:
                os._exit(127)
            # os.execvp never returns on success
            os._exit(127)

        # ── parent ───────────────────────────────────────────────────
        result = _run_probe_session(master_fd, pid)
        return result


def _run_probe_session(master_fd: int, child_pid: int) -> ProbeResult:
    """Run the PTY probe session: wait for prompt, send /model, capture, exit.

    Sends commands with CR (\r) for real terminal execution.  Only sends
    /model after a recognized prompt character (❯ or >), never on arbitrary
    startup output.
    """
    captured = bytearray()
    deadline = time.monotonic() + _HARD_TIMEOUT_SEC
    state = "wait_prompt"  # wait_prompt → sent_model → capturing → done

    # We set a short initial wait for the prompt to appear
    model_sent_at: float | None = None

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ProbeResult(
                    output=_sanitize_output(captured),
                    error="timeout: Claude did not produce recognizable output within 15s",
                )

            # Determine read timeout for this iteration
            if state == "wait_prompt":
                read_timeout = min(remaining, 5.0)  # Wait up to 5s for prompt
            elif state == "sent_model":
                read_timeout = min(remaining, 2.0)  # Wait up to 2s after /model
            else:
                read_timeout = min(remaining, 1.0)  # Poll phase

            ready, _, _ = select.select([master_fd], [], [], read_timeout)

            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    # EOF — child exited
                    break
                captured.extend(chunk)
                if len(captured) > _MAX_CAPTURED_BYTES:
                    captured = captured[-_MAX_CAPTURED_BYTES:]
                    # Ring-buffer: keep latest bytes; still try to parse

            # State transitions
            if state == "wait_prompt":
                if _recognize_prompt(captured):
                    state = "sent_model"
                    _write_all(master_fd, b"/model\r")
                    model_sent_at = time.monotonic()
                elif time.monotonic() - (deadline - _HARD_TIMEOUT_SEC) > 5.0:
                    # 5s elapsed without prompt — check for auth/error/EOF first
                    text = _sanitize_output(captured)
                    if _looks_like_auth_prompt(text):
                        return ProbeResult(
                            output=text,
                            error="auth prompt detected before prompt — run `claude auth login` first",
                        )
                    if _looks_like_error(text):
                        return ProbeResult(
                            output=text,
                            error="claude startup error detected before prompt",
                        )
                    if not captured:
                        return ProbeResult(
                            output=None,
                            error="no output from Claude before prompt timeout (EOF or hang)",
                        )
                    # Non-prompt startup output (banner, version, etc.)
                    # Do NOT send /model — return error instead of guessing.
                    return ProbeResult(
                        output=text,
                        error="no recognized prompt after startup output — TUI may not be interactive",
                    )

            elif state == "sent_model":
                if model_sent_at is not None and time.monotonic() - model_sent_at > 2.0:
                    # Give it a bit more time for picker to fully render
                    state = "capturing"
                    # Read one more time then exit

            elif state == "capturing":
                # We've waited enough — exit
                break

        # Exit Claude cleanly: dismiss picker (Esc), then /exit
        _write_all(master_fd, b"\x1b")  # Escape to dismiss picker
        time.sleep(0.1)
        _write_all(master_fd, b"/exit\r")
        time.sleep(0.2)

        return ProbeResult(
            output=_sanitize_output(captured),
            error=None,
        )

    finally:
        # ── guaranteed cleanup ───────────────────────────────────────
        # Close master fd
        try:
            os.close(master_fd)
        except OSError:
            pass

        # Terminate child if still alive
        try:
            os.kill(child_pid, signal.SIGTERM)
        except OSError:
            pass

        # Brief wait for SIGTERM to take effect
        time.sleep(0.1)

        try:
            os.kill(child_pid, signal.SIGKILL)
        except OSError:
            pass

        # Reap the child
        # Synchronous reap — block until child is collected
        _reap_child(child_pid)


def _reap_child(child_pid: int) -> None:
    """Guaranteed synchronous child reaping after SIGKILL.

    Polls with WNOHANG for up to 200ms, then falls back to a blocking
    wait to guarantee the zombie is collected — no zombie left behind.
    """
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        try:
            wpid, _status = os.waitpid(child_pid, os.WNOHANG)
            if wpid == child_pid:
                return
        except OSError:
            return
        time.sleep(0.02)
    # Blocking wait — guaranteed collection, no zombie left behind
    try:
        os.waitpid(child_pid, 0)
    except OSError:
        pass


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes to *fd*, ignoring errors."""
    try:
        os.write(fd, data)
    except OSError:
        pass


# ── Output sanitization ──────────────────────────────────────────────────────


def _sanitize_output(raw: bytearray) -> str:
    """Strip ANSI escapes, cursor saves, charset shifts — decode to string.

    Handles CSI sequences, OSC sequences, ESC 7/8 (save/restore cursor),
    SI/SO (shift-in/shift-out charset), ESC ( B (select ASCII),
    and normalizes NBSP (U+00A0) to regular space.
    """
    text = bytes(raw)
    # Strip OSC sequences first (they can be long)
    text = _OSC_RE.sub(b"", text)
    # Strip ANSI CSI sequences
    text = _ANSI_ESCAPE_RE.sub(b"", text)
    # Strip ESC 7 / ESC 8 (save/restore cursor)
    text = _ESC_7_8_RE.sub(b"", text)
    # Strip charset shifts
    text = _CHARSET_SHIFT_RE.sub(b"", text)
    # Normalize NBSP (C2 A0 in UTF-8) to regular space
    text = text.replace(b"\xc2\xa0", b" ")
    # Decode, replacing non-UTF-8 bytes
    return text.decode("utf-8", errors="replace")


def strip_ansi(raw: str) -> str:
    """Strip ANSI escape sequences from a string (public for testing).

    Also handles ESC 7/8 (save/restore cursor), SI/SO charset shifts,
    ESC ( B (select ASCII), and normalizes NBSP to space.
    """
    import re as _re

    # OSC sequences
    raw = _re.sub(r"\x1b\].*?(\x07|\x1b\\)", "", raw)
    # CSI sequences
    raw = _re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
    # ESC 7 / ESC 8 (save/restore cursor)
    raw = _re.sub(r"\x1b ?[78]", "", raw)
    # Charset shifts: SI, SO, ESC ( B
    raw = _re.sub(r"[\x0e\x0f]|\x1b\(B", "", raw)
    # NBSP → regular space
    raw = raw.replace("\xa0", " ")
    return raw


def _recognize_prompt(raw: bytearray) -> bool:
    """Check whether sanitized PTY output ends with a recognized prompt.

    Strips ANSI/control sequences via ``_sanitize_output``, then checks
    whether the sanitized text ends with a standalone prompt character
    (``$``, ``❯``, or ``>``) followed only by optional whitespace.

    This handles the real-world case where raw PTY bytes contain
    trailing ANSI cursor/redraw sequences after the visible prompt
    character — the sanitization step strips those before recognition.
    """
    text = _sanitize_output(raw)
    # Match a prompt character at end of string (whitespace-tolerant).
    # After sanitization all ANSI/control bytes are gone, so we just
    # check whether the sanitized text ends with $, ❯, or > (with
    # optional trailing whitespace).
    import re as _re

    return bool(_re.search(r"(?:❯|>|\$)\s*$", text))


# ── Picker parser ────────────────────────────────────────────────────────────


# Known stable aliases for Claude model families.
# These are the values the Claude CLI accepts for --model and the
# aliases shown in the /model picker (opus, sonnet, fable, haiku).
# Do NOT invent versioned IDs like claude-opus-4-8 — the CLI help
# explicitly lists these aliases; versioned full IDs are only used
# when the picker itself surfaces them.
_MODEL_FAMILIES: tuple[str, ...] = ("opus", "sonnet", "fable", "haiku")

# Regex to extract an explicit full model ID from a picker line.
# Matches patterns like: claude-opus-4-8, claude-sonnet-5, claude-haiku-4-5.
# Only captures IDs for known families; unknown prefixes are ignored.
_EXPLICIT_ID_RE = re.compile(
    r"\b(claude-(?:opus|sonnet|fable|haiku)-\d+(?:[.-]\d+)*)\b",
    re.IGNORECASE,
)

# The picker exposes human-readable names such as "Opus 4.8" and "Sonnet 5".
# Claude CLI's documented full-ID convention is ``claude-<family>-<version>``;
# normalize the live picker version rather than exposing mutable family aliases.
_DISPLAY_VERSION_RE = re.compile(
    r"\b(?P<family>[a-z][a-z0-9-]*)\s+(?P<version>\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)

# Display-name patterns for matching picker lines → alias.
# Order matters — more specific patterns first.
_MODEL_DISPLAY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("opus", re.compile(r"\bopus\b", re.IGNORECASE)),
    ("sonnet", re.compile(r"\bsonnet\b", re.IGNORECASE)),
    ("fable", re.compile(r"\bfable\b", re.IGNORECASE)),
    ("haiku", re.compile(r"\bhaiku\b", re.IGNORECASE)),
]

# Effort patterns — look for effort labels near model entries
_EFFORT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("low", re.compile(r"\blow\b", re.IGNORECASE)),
    ("medium", re.compile(r"\bmedium\b", re.IGNORECASE)),
    ("high", re.compile(r"\bhigh\b", re.IGNORECASE)),
    ("max", re.compile(r"\bmax\b", re.IGNORECASE)),
]


# ── Numbered-picker section parser (Claude Code ≥2.1.211) ────────────────────

# Regex to match the start of a numbered picker entry: "N. rest"
_NUMBERED_ENTRY_RE: re.Pattern = re.compile(r"^\s*(\d+)\.\s+(.*)")

# Section boundary markers
_SELECT_MODEL_HEADER_RE: re.Pattern = re.compile(r"^\s*Select model\s*$", re.IGNORECASE)
_ENTER_SELECTION_RE: re.Pattern = re.compile(r"^\s*Enter selection\b", re.IGNORECASE)


def _find_section_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Locate the ``Select model`` → ``Enter selection`` section.

    Returns ``(header_idx, terminator_idx)`` or ``None`` if the section
    is not found.  *terminator_idx* points to the line AFTER the last
    numbered entry (exclusive bound for slicing).
    """
    header_idx: int | None = None
    terminator_idx: int | None = None

    for i, line in enumerate(lines):
        if header_idx is None and _SELECT_MODEL_HEADER_RE.match(line):
            header_idx = i
        elif header_idx is not None and _ENTER_SELECTION_RE.match(line):
            terminator_idx = i
            break

    if header_idx is not None and terminator_idx is not None and terminator_idx > header_idx:
        return header_idx, terminator_idx
    return None


def _resolve_numbered_label(label_text: str) -> tuple[str | None, bool]:
    """Map a cleaned picker label to ``(model_id, is_selected)``.

    ``label_text`` is the content after the number and dot, with the
    description after the em-dash already stripped.  Examples:

    * ``"Default (recommended)"`` → ``("default", False)``
    * ``"(selected) Sonnet"`` → ``("sonnet", True)``
    * ``"Fable"`` → ``("fable", False)``
    """
    is_selected = False

    # Detect and strip "(selected)" prefix
    selected_m = re.match(r"^\s*\(selected\)\s+(.*)", label_text, re.IGNORECASE)
    if selected_m:
        is_selected = True
        label_text = selected_m.group(1)

    # Strip parenthetical suffixes like (recommended), (1M context), etc.
    label_text = re.sub(r"\s*\([^)]*\)", "", label_text).strip()

    if not label_text:
        return None, is_selected

    label_lower = label_text.lower()

    # "Default" → standalone default alias
    if label_lower == "default":
        return "default", is_selected

    # Numbered picker labels use family aliases. They are only an internal
    # classification aid: _parse_numbered_section resolves the version from
    # the corresponding description before emitting a public model ID.
    if label_lower in _MODEL_FAMILIES:
        return label_lower, is_selected

    # Fall back to _match_model_line for the label text
    canonical, _ = _match_model_line(label_text)
    if canonical:
        return canonical, is_selected

    return None, is_selected


def _parse_numbered_section(
    lines: list[str], start: int, end: int
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse the ``Select model`` … ``Enter selection`` numbered-picker section.

    Returns ``(models, default_id, error)`` — same shape as
    ``parse_model_picker_output``.
    """
    models: list[dict[str, Any]] = []
    default_id: str | None = None
    seen_ids: set[str] = set()

    current_num: int | None = None
    current_text: str | None = None
    current_line_idx: int | None = None

    def _flush() -> None:
        nonlocal default_id
        if current_text is None or current_num is None:
            return
        # Extract label portion (before em-dash / en-dash / hyphen separator)
        label_only = re.split(r"\s*[-–—]\s*", current_text, maxsplit=1)[0]
        model_id, is_selected = _resolve_numbered_label(label_only)
        # Numbered picker labels are aliases (for example, "Opus"); the
        # description after the dash contains the live version to publish.
        full_model_id, _ = _match_model_line(current_text)
        if full_model_id is not None:
            model_id = full_model_id
        if model_id is None:
            return
        if model_id in seen_ids:
            # "Default — Sonnet 5" and "(selected) Sonnet — Sonnet 5"
            # describe one model. Preserve one entry but let the selected
            # marker set its public default_model.
            if is_selected:
                default_id = model_id
                for entry in models:
                    if entry["id"] == model_id:
                        entry["is_default"] = True
                        break
            return
        seen_ids.add(model_id)
        if is_selected:
            default_id = model_id
        # display_name: original text without the number prefix, cleaned up
        display_name = re.sub(r"\s*[-–—]\s*.*", "", current_text).strip()
        display_name = re.sub(r"^\s*\(selected\)\s+", "", display_name, flags=re.IGNORECASE)
        display_name = display_name.strip()
        efforts = (
            _extract_nearby_efforts(lines, current_line_idx) if current_line_idx is not None else []
        )
        models.append(
            {
                "id": model_id,
                "display_name": display_name,
                "is_default": is_selected,
                "efforts": efforts,
            }
        )

    for i in range(start + 1, end):
        line = lines[i].strip()
        if not line:
            continue

        m = _NUMBERED_ENTRY_RE.match(line)
        if m:
            _flush()
            current_num = int(m.group(1))
            current_text = m.group(2).strip()
            current_line_idx = i
        elif current_text is not None:
            # Continuation line — append
            current_text += " " + line

    _flush()

    if not models:
        return [], None, "unrecognized picker format — no model entries in numbered section"

    return models, default_id, None


# ── Main parser ──────────────────────────────────────────────────────────────


def parse_model_picker_output(
    raw_output: str,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Parse sanitized /model picker output into structured model entries.

    Returns ``(models, default_id, error)`` where *models* is a list of dicts
    with keys: ``id``, ``display_name``, ``is_default``, ``efforts`` (list of
    effort labels found near the entry, if any).

    Returns an empty list with *error* set when the output is unrecognized,
    appears to be an auth prompt, or contains no identifiable model entries.

    Two formats are supported:

    * **Numbered picker** (Claude Code ≥2.1.211): ``Select model`` header
      followed by ``1. Label — Description`` entries, terminated by
      ``Enter selection``.  The ``(selected)`` marker on a numbered entry
      identifies the currently-active model (becomes ``default_model``).
      ``1. Default`` is treated as its own model entry with id ``"default"``.

    * **Legacy picker**: free-form model-name matching across all lines
      (with deduplication — first occurrence of each model id wins).
    """
    if not raw_output or not raw_output.strip():
        return [], None, "empty picker output"

    # Quick auth-prompt detection
    if _looks_like_auth_prompt(raw_output):
        return [], None, "auth prompt detected — run `claude auth login` first"

    # Quick error detection
    if _looks_like_error(raw_output):
        return [], None, "claude startup error in probe output"

    lines = raw_output.splitlines()

    # ── Numbered-picker gate (Claude Code ≥2.1.211) ─────────────────────
    bounds = _find_section_bounds(lines)
    if bounds is not None:
        return _parse_numbered_section(lines, *bounds)

    # ── Legacy free-form parser (with deduplication) ────────────────────
    models: list[dict[str, Any]] = []
    default_id: str | None = None
    seen_ids: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        canonical, is_default = _match_model_line(stripped)
        if canonical and canonical not in seen_ids:
            seen_ids.add(canonical)
            efforts = _extract_nearby_efforts(lines, i)
            models.append(
                {
                    "id": canonical,
                    "display_name": _extract_display_name(stripped),
                    "is_default": is_default,
                    "efforts": efforts,
                }
            )
            if is_default and default_id is None:
                default_id = canonical

    if not models:
        return (
            [],
            None,
            f"unrecognized picker format — no model entries found in {len(lines)} lines",
        )

    return models, default_id, None


def _match_model_line(line: str) -> tuple[str | None, bool]:
    """Try to match a picker line against known model patterns.

    Returns ``(model_id, is_default)`` or ``(None, False)``.

    Strategy:
    1. Standalone "Default" as primary label → ``("default", True)``.
       Must be checked BEFORE family-name matching, otherwise lines like
       ``● Default (currently Opus 4.8)`` would be misclassified as opus.
    2. Extract an explicit full ID like ``claude-sonnet-5`` if present.
    3. Otherwise match display-name patterns → stable alias (opus/sonnet/…).
    """
    # Default marker anywhere on the line (e.g. "Opus 4.8 (Default)").
    # A ``Default — Sonnet 5`` entry is resolved from its description too.
    has_default_marker = bool(re.search(r"\bDefault\b", line, re.IGNORECASE))

    # 2. Explicit full ID (e.g. "claude-sonnet-5" in the picker text)
    explicit = _EXPLICIT_ID_RE.search(line)
    if explicit:
        return explicit.group(1).lower(), has_default_marker

    # 3. Display-name version → documented full CLI ID.
    display = _DISPLAY_VERSION_RE.search(line)
    if display:
        family = display.group("family").lower()
        version = display.group("version").replace(".", "-")
        return f"claude-{family}-{version}", has_default_marker

    return None, False


def _extract_display_name(line: str) -> str:
    """Extract a human-readable display name from a picker line."""
    # Remove common markers
    cleaned = re.sub(r"[>●✓*]\s*", "", line)
    cleaned = re.sub(r"\bDefault\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(.*?\)", "", cleaned)  # Remove parentheticals like (1M context)
    cleaned = cleaned.strip().strip("·●✓*> ")
    return cleaned or line.strip()


def _extract_nearby_efforts(lines: list[str], model_idx: int) -> list[str]:
    """Extract effort labels from lines near *model_idx*."""
    efforts: list[str] = []
    # Check one line before and after the model line
    for offset in (-1, 1):
        check_idx = model_idx + offset
        if 0 <= check_idx < len(lines):
            line = lines[check_idx]
            for effort_label, pattern in _EFFORT_PATTERNS:
                if pattern.search(line) and effort_label not in efforts:
                    efforts.append(effort_label)
    return efforts


def _looks_like_auth_prompt(text: str) -> bool:
    """Heuristic: does the output look like an auth/login prompt?"""
    auth_indicators = [
        "claude auth login",
        "not logged in",
        "loggedIn",
        "authentication required",
        "please sign in",
        "launching claude code",
    ]
    text_lower = text.lower()
    return any(indicator.lower() in text_lower for indicator in auth_indicators)


def _looks_like_error(text: str) -> bool:
    """Heuristic: does the output look like a startup error?"""
    error_indicators = [
        "command not found",
        "no such file",
        "cannot execute",
        "fatal error",
        "uncaught exception",
    ]
    text_lower = text.lower()
    return any(indicator.lower() in text_lower for indicator in error_indicators)


# ── Probe → ModelCatalog ────────────────────────────────────────────────────


def probe_to_catalog(probe_result: ProbeResult) -> dict[str, Any]:
    """Convert a ``ProbeResult`` into a cacheable dict for ``ModelCatalog``.

    Returns a dict with keys matching ``ModelCatalog`` fields:
    ``models``, ``default_model``, ``native_efforts``, ``model_info``,
    ``source``, ``error``.
    """
    if probe_result.error:
        return {
            "models": [],
            "default_model": None,
            "native_efforts": [],
            "model_info": [],
            "source": "claude interactive /model picker",
            "error": probe_result.error,
        }

    raw_output = probe_result.output or ""
    entries, default_id, parse_error = parse_model_picker_output(raw_output)

    if parse_error:
        return {
            "models": [],
            "default_model": None,
            "native_efforts": [],
            "model_info": [],
            "source": "claude interactive /model picker",
            "error": parse_error,
        }

    model_ids = [e["id"] for e in entries]
    # Never silently fall back to first model when no default is advertised.
    # The real picker has "Default" as an explicit choice or marked entry;
    # if neither is present, default_model stays None.
    default_model = default_id

    # Collect all unique efforts across all models
    all_efforts: set[str] = set()
    for e in entries:
        all_efforts.update(e["efforts"])

    model_info = []
    for e in entries:
        effs = tuple(e["efforts"])
        model_info.append(
            {
                "id": e["id"],
                "supported_efforts": list(effs),
                "default_effort": effs[0] if effs else None,
            }
        )

    return {
        "models": model_ids,
        "default_model": default_model,
        "native_efforts": sorted(all_efforts),
        "model_info": model_info,
        "source": "claude interactive /model picker",
        "error": None,
    }


# ── Session cleanup ──────────────────────────────────────────────────────────
# The probe manages its own child process: os.forkpty + os.execvp.
# The child is always terminated (SIGTERM → SIGKILL → synchronous reap)
# within _run_probe_session's finally block.  There is no persisted
# Claude session to delete — the interactive process is ephemeral.
# Do NOT invoke `claude session delete` (unsupported command) or touch
# any user-owned sessions.
