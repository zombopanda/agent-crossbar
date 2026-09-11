"""Helpers for interpreting provider TUI output captured from tmux panes."""

from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b(?:\][^\a]*(?:\a|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REASONIX_REPLY_RE = re.compile(r"‹\s*reply", re.IGNORECASE)
_REASONIX_READY_RE = re.compile(
    r"(?:ask\s*anything|type\s*a\s*message\s*to\s*start\s*your\s*session)",
    re.IGNORECASE,
)
_SESSION_RESUMED_RE = re.compile(r"resumed\s*session", re.IGNORECASE)
_CLAUDE_STYLE_ANSWER_RE = re.compile(
    r"(?m)^\s*⏺\s*(?!(?:Bash|Read|Write|Edit|Glob|Grep|Task|WebFetch|WebSearch|Tool|Skill)\b)\S"
)
_CLAUDE_SCREEN_READER_ANSWER_RE = re.compile(r"(?m)^\s*\$?claude:\s+\S")
_CLAUDE_IDLE_NOTIFICATION_RE = re.compile(
    r"\x1b\]777;notify;Claude Code;Claude is waiting for your input(?:\x07|\x1b\\)",
    re.IGNORECASE,
)
_CLAUDE_IDLE_FOOTER_RE = re.compile(
    r"(?m)^\s*⏵⏵\s+(?:bypass permissions|plan mode)\s+on\b",
    re.IGNORECASE,
)
_CLAUDE_BUSY_RE = re.compile(
    r"(?:esc\s*to\s*interrupt|^\s*[✶✽✻]\s+[^\n]*…)",
    re.IGNORECASE | re.MULTILINE,
)
_CLAUDE_FINISHED_RE = re.compile(
    r"(?m)^\s*✻\s*[^\n…]{1,48}?\s+for\s*\d+\s*[smh]?\b",
    re.IGNORECASE,
)
_CLAUDE_SCREEN_READER_FINISHED_RE = re.compile(
    r"(?m)^\s*\$?(?-i:[A-Z])[^\W\d_]{2,15}ed\s+for\s+"
    r"\d+(?:\.\d+)?\s*[smh]\b",
    re.IGNORECASE,
)
_CLAUDE_PLAN_READY_RE = re.compile(
    r"(?ms)^Claude\s*has\s*written\s*up\s*a\s*plan\s*and\s*is\s*ready\s*to\s*"
    r"execute\.?\s*Would\s*you\s*like\s*to\s*proceed\?\s*❯?\s*1\.\s*Yes,",
    re.IGNORECASE,
)
_CODEX_ANSWER_RE = re.compile(
    r"(?m)^\s*•\s+(?!(?:SessionStart|UserPromptSubmit|PreToolUse|PostToolUse|PermissionRequest|Stop|Working|Running|Ran|Called|Read|Edited|Updated|Explored|Searched|Listed|Wrote|Patched)\b)\S"
)
_CODEX_BUSY_RE = re.compile(r"\besc\s+to\s+interrupt\b", re.IGNORECASE)
_CODEX_STOP_RE = re.compile(r"\bStop\s+hook\b", re.IGNORECASE)
_CODEX_WORKED_FOR_RE = re.compile(r"\bWorked\s+for\s+\d", re.IGNORECASE)
_OPENCODE_PROMPT_RE = re.compile(r"(?m)^\s*›\s+.+$")
_OPENCODE_BUSY_RE = re.compile(r"\besc\s+interrupt\b", re.IGNORECASE)
_CURSOR_MOTION_RE = re.compile(r"\x1b\[[0-9;?]*[ABCDEFGHJKSTdfmus]", re.IGNORECASE)
_MAX_RENDER_INPUT_CHARS = 2_000_000
_MAX_RENDER_ROWS = 2_048
_MAX_RENDER_COLUMNS = 4_096
_MAX_RENDER_CELLS = 200_000
_MAX_RENDER_HISTORY_LINES = 16_384
_MAX_CSI_PARAMETER = 100_000


class _CursorRenderFailure:
    """Private sentinel for malformed/unbounded terminal control data."""


_CURSOR_RENDER_FAILED = _CursorRenderFailure()


def _render_cursor_rewrites(output: str) -> str | None | _CursorRenderFailure:
    """Render cursor-motion redraws into the visible terminal text.

    Screen-reader Claude output can split one assistant line across a redraw:
    it writes ``claude: L``, moves the cursor up/right, then writes the rest
    of the answer. Stripping ANSI alone leaves a false partial answer. This
    small screen buffer handles the cursor controls used by tmux captures and
    returns ``None`` when no cursor motion is present so ordinary transcript
    normalization remains unchanged.
    """
    if not _CURSOR_MOTION_RE.search(output):
        return None
    if len(output) > _MAX_RENDER_INPUT_CHARS:
        return _CURSOR_RENDER_FAILED

    rows: list[list[str]] = [[]]
    history: list[str] = []
    row = 0
    column = 0
    saved: tuple[int, int] | None = None
    cells_used = 0
    failed = False

    def ensure_row(index: int) -> bool:
        nonlocal failed
        if index < 0 or index >= _MAX_RENDER_ROWS:
            failed = True
            return False
        while len(rows) <= index:
            rows.append([])
        return True

    def write_char(char: str) -> None:
        nonlocal cells_used, column, row, failed
        if failed:
            return
        if char == "\n":
            row += 1
            column = 0
            ensure_row(row)
            return
        if char == "\r":
            column = 0
            return
        if char == "\b":
            column = max(0, column - 1)
            return
        if char == "\t":
            column = ((column // 8) + 1) * 8
            if column >= _MAX_RENDER_COLUMNS:
                failed = True
            return
        if ord(char) < 0x20 or ord(char) == 0x7F:
            return
        if not ensure_row(row) or column >= _MAX_RENDER_COLUMNS:
            failed = True
            return
        current = rows[row]
        while len(current) <= column:
            current.append(" ")
            cells_used += 1
            if cells_used > _MAX_RENDER_CELLS:
                failed = True
                return
        current[column] = char
        column += 1

    def param_values(raw: str) -> list[int] | None:
        nonlocal failed
        raw = raw.lstrip("?")
        if not raw:
            return []
        values: list[int] = []
        for part in raw.split(";"):
            if len(part) > len(str(_MAX_CSI_PARAMETER)):
                failed = True
                return None
            try:
                value = int(part or "1")
            except ValueError:
                value = 1
            if value < 0 or value > _MAX_CSI_PARAMETER:
                failed = True
                return None
            values.append(value)
        return values

    index = 0
    while index < len(output):
        if output[index] != "\x1b":
            write_char(output[index])
            index += 1
            continue
        if index + 1 >= len(output):
            break
        next_char = output[index + 1]
        if next_char == "]":
            # OSC title/notification sequence: it is not screen text.
            end = index + 2
            while end < len(output) and output[end] != "\x07":
                if output[end] == "\x1b" and end + 1 < len(output) and output[end + 1] == "\\":
                    end += 1
                    break
                end += 1
            index = min(end + 1, len(output))
            continue
        if next_char in "P_^":
            # DCS/APC/PM payloads (including tmux passthrough) are terminal
            # control data terminated by ST; never render their payload.
            end = output.find("\x1b\\", index + 2)
            index = len(output) if end < 0 else end + 2
            continue
        if next_char in "78":
            if next_char == "7":
                saved = (row, column)
            elif saved is not None:
                row, column = saved
            index += 2
            continue
        if next_char != "[":
            index += 2
            continue

        end = index + 2
        while end < len(output) and not ("@" <= output[end] <= "~"):
            end += 1
        if end >= len(output):
            break
        params = output[index + 2 : end]
        final = output[end]
        values = param_values(params)
        if values is None:
            break
        amount = values[0] if values and values[0] > 0 else 1
        if final in "ABCDEFGHdf" and amount > _MAX_RENDER_ROWS:
            failed = True
            break
        if final in {"C", "G"} and amount > _MAX_RENDER_COLUMNS:
            failed = True
            break
        if final == "A":
            row = max(0, row - amount)
        elif final == "B":
            row += amount
            ensure_row(row)
        elif final == "C":
            column += amount
            if column >= _MAX_RENDER_COLUMNS:
                failed = True
        elif final == "D":
            column = max(0, column - amount)
        elif final == "E":
            row += amount
            column = 0
            ensure_row(row)
        elif final == "F":
            row = max(0, row - amount)
            column = 0
        elif final == "G":
            column = max(0, amount - 1)
        elif final in {"H", "f"}:
            row = max(0, (values[0] if values else 1) - 1)
            column = max(0, (values[1] if len(values) > 1 else 1) - 1)
            if column >= _MAX_RENDER_COLUMNS or not ensure_row(row):
                failed = True
        elif final == "d":
            row = max(0, amount - 1)
            ensure_row(row)
        elif final == "s":
            saved = (row, column)
        elif final == "u" and saved is not None:
            row, column = saved
        elif final == "K":
            if not ensure_row(row):
                break
            current = "".join(rows[row]).rstrip()
            if current:
                if len(history) >= _MAX_RENDER_HISTORY_LINES:
                    failed = True
                    break
                history.append(current)
            erase_mode = values[0] if values else 0
            if erase_mode == 2:
                rows[row] = []
            elif erase_mode == 1:
                for position in range(min(column + 1, len(rows[row]))):
                    rows[row][position] = " "
            else:
                del rows[row][column:]
        elif final == "J":
            for line in rows:
                current = "".join(line).rstrip()
                if current:
                    if len(history) >= _MAX_RENDER_HISTORY_LINES:
                        failed = True
                        break
                    history.append(current)
            if failed:
                break
            rows = [[] for _ in range(row + 1)]
        index = end + 1

    if failed:
        return _CURSOR_RENDER_FAILED
    current_rows = ["".join(line).rstrip() for line in rows]
    rendered = "\n".join(history + current_rows).rstrip()
    if len(rendered) > _MAX_RENDER_INPUT_CHARS:
        return _CURSOR_RENDER_FAILED
    return rendered


def _extract_cursor_rewritten_claude_line(output: str) -> tuple[str, str] | None:
    """Recover a Claude screen-reader line split by cursor-up/right output."""
    csi = r"\x1b\[[0-9;?]*[ -/]*[@-~]"
    control = rf"(?:{csi}|\x1b\(B|\x1b[78]|[\x0e\x0f])*"
    marker_pattern = re.compile(rf"(?m)(?:^|[\r\n])\$?{control}(?P<line>\$?claude:\s+[^\r\n]*)")
    rewrite_pattern = re.compile(
        r"\x1b\[(?P<up>\d+)A\x1b\[(?P<column>\d+)(?:G|C)"
        r"(?P<continuation>[^\r\n\x1b]+)"
    )
    footer_pattern = re.compile(
        r"(?P<footer>(?:Cogitated|Worked|Brewed|Churned)\s+for\s+[^\r\n]+)",
        re.IGNORECASE,
    )
    for marker in marker_pattern.finditer(output):
        line = marker.group("line")
        after = output[marker.end() :]
        rewrite = rewrite_pattern.search(after)
        if rewrite is None:
            continue
        try:
            up = int(rewrite.group("up"))
            target = int(rewrite.group("column")) - 1
        except ValueError:
            continue
        if up <= 0 or up > _MAX_RENDER_ROWS or target < 0 or target >= _MAX_RENDER_COLUMNS:
            continue
        continuation = rewrite.group("continuation")
        if not continuation.strip():
            continue
        full_line = line[:target] + continuation
        if "claude:" not in full_line.casefold():
            full_line = line + continuation
        footer = footer_pattern.search(after, rewrite.end())
        if footer is None:
            continue
        return full_line.rstrip(), footer.group("footer").strip()
    return None


def reconstruct_tmux_output(output: str) -> str:
    """Return visible text with cursor redraws reconstructed and ANSI removed."""
    if len(output) > _MAX_RENDER_INPUT_CHARS:
        return "" if _CURSOR_MOTION_RE.search(output) else normalize_tmux_output(output)
    clean = normalize_tmux_output(output)
    rewritten = _extract_cursor_rewritten_claude_line(output)
    if rewritten is not None:
        line, footer = rewritten
        # Append the synthesized latest screen-reader frame after teardown
        # noise so Claude's chronology-aware extractor selects the full line.
        return f"{clean}\n{line}\n${footer}".rstrip()
    rendered = _render_cursor_rewrites(output)
    if rendered is _CURSOR_RENDER_FAILED:
        return ""
    return rendered if rendered is not None else clean


def normalize_tmux_output(output: str) -> str:
    """Strip terminal control sequences while keeping user-visible text."""
    text = output.replace("\r", "\n")
    previous = None
    while previous != text:
        previous = text
        text = _ANSI_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


def interactive_tmux_output_complete(
    output: str,
    *,
    baseline_bytes: int = 0,
    profile: str | None = None,
) -> bool:
    if len(output.encode("utf-8", errors="replace")) <= baseline_bytes:
        return False

    text = normalize_tmux_output(output)
    if interactive_tmux_session_resumed(output, profile=profile):
        return False

    profile = (profile or "").casefold()
    if profile in {"reasonix", "deepseek"}:
        return _reasonix_output_complete(text)
    if profile in {"claude", "opus"}:
        # Validate cursor data before ordinary marker matching.  Otherwise a
        # malformed huge movement could be treated as a clean final transcript
        # and bypass the bounded renderer entirely.
        cursor_render = _render_cursor_rewrites(output)
        if cursor_render is _CURSOR_RENDER_FAILED:
            return False
        complete = _claude_style_output_complete(
            text,
            idle_notified=_claude_latest_turn_idle_notified(output),
        )
        if complete:
            return True
        reconstructed = reconstruct_tmux_output(output)
        return reconstructed != text and _claude_style_output_complete(
            reconstructed,
            idle_notified=_claude_latest_turn_idle_notified(output),
        )
    if profile == "codex":
        return _codex_output_complete(text)
    if profile == "opencode":
        return _opencode_output_complete(text)
    if profile:
        return False

    return (
        _reasonix_output_complete(text)
        or _claude_style_output_complete(text)
        or _codex_output_complete(text)
        or _opencode_output_complete(text)
    )


def interactive_tmux_output_complete_since(
    output: str,
    *,
    baseline_bytes: int,
    profile: str | None = None,
) -> bool:
    """Recognize completion in output appended after a recorded turn boundary.

    A growing pane capture is not itself evidence of a new answer: tmux may
    append an echoed user line or a redraw of the previous screen.  The
    monitor therefore records the byte boundary before ``job_send`` and runs
    the provider parser against only the appended bytes.  A missing boundary
    is deliberately inconclusive and leaves the job running until its normal
    deadline.
    """
    try:
        raw = output.encode("utf-8", errors="replace")
        if baseline_bytes < 0 or len(raw) <= baseline_bytes:
            return False
        appended = raw[baseline_bytes:].decode("utf-8", errors="replace")
    except (TypeError, ValueError):
        return False
    return interactive_tmux_output_complete(appended, profile=profile)


def interactive_tmux_output_summary(
    output: str,
    *,
    profile: str | None = None,
    max_chars: int = 4000,
) -> str:
    text = normalize_tmux_output(output)
    if not text:
        return "Interactive tmux output completed"
    index = _completion_marker_index(text, profile=profile)
    if index >= 0:
        start = max(0, index - max_chars // 3)
        end = min(len(text), index + max_chars)
        return text[start:end][-max_chars:]
    return text[-max_chars:]


def interactive_tmux_session_resumed(output: str, *, profile: str | None = None) -> bool:
    profile = (profile or "").casefold()
    if profile and profile not in {"reasonix", "deepseek"}:
        return False
    return _SESSION_RESUMED_RE.search(normalize_tmux_output(output)) is not None


def _reasonix_output_complete(text: str) -> bool:
    reply = list(_REASONIX_REPLY_RE.finditer(text))
    if not reply:
        return False
    last_reply = reply[-1].start()
    return any(match.start() > last_reply for match in _REASONIX_READY_RE.finditer(text))


def _claude_style_output_complete(text: str, *, idle_notified: bool = False) -> bool:
    if _CLAUDE_PLAN_READY_RE.search(text):
        return True
    answer = list(_CLAUDE_STYLE_ANSWER_RE.finditer(text))
    if answer:
        last_answer = answer[-1].start()
        answer_tail = text[last_answer:]
        finished = list(_CLAUDE_FINISHED_RE.finditer(answer_tail))
        busy = list(_CLAUDE_BUSY_RE.finditer(answer_tail))
        if finished and (not busy or finished[-1].start() > busy[-1].start()):
            return True
        if text.rfind("❯") > last_answer:
            if idle_notified:
                return True
            if bool(_CLAUDE_IDLE_FOOTER_RE.search(answer_tail)) and not busy:
                return True

    # Screen-reader mode renders assistant redraws as ``$claude: ...``
    # instead of the normal ``⏺`` marker.  A marker alone is not completion
    # evidence: stale redraws can repeat after a reply.  Require the matching
    # lifecycle footer, and reject any later busy indicator.
    screen_reader_answers = list(_CLAUDE_SCREEN_READER_ANSWER_RE.finditer(text))
    if not screen_reader_answers:
        return False
    screen_reader_tail = text[screen_reader_answers[-1].start() :]
    screen_reader_finished = list(_CLAUDE_SCREEN_READER_FINISHED_RE.finditer(screen_reader_tail))
    screen_reader_busy = list(_CLAUDE_BUSY_RE.finditer(screen_reader_tail))
    return bool(
        screen_reader_finished
        and (
            not screen_reader_busy
            or screen_reader_finished[-1].start() > screen_reader_busy[-1].start()
        )
    )


def _claude_latest_turn_idle_notified(output: str) -> bool:
    notifications = list(_CLAUDE_IDLE_NOTIFICATION_RE.finditer(output))
    if not notifications:
        return False
    notification = notifications[-1].start()
    return notification > output.rfind("⏺")


def _codex_output_complete(text: str) -> bool:
    submit = text.rfind("UserPromptSubmit hook")
    search_from = submit if submit >= 0 else text.rfind("\n› ")
    haystack = text[search_from:] if search_from >= 0 else text
    answers = list(_CODEX_ANSWER_RE.finditer(haystack))
    if not answers:
        return False
    answer_tail = haystack[answers[-1].start() :]
    boundary = _CODEX_STOP_RE.search(answer_tail) or _CODEX_WORKED_FOR_RE.search(answer_tail)
    if boundary is None:
        return False
    return _CODEX_BUSY_RE.search(answer_tail[boundary.end() :]) is None


def _opencode_output_complete(text: str) -> bool:
    return _opencode_answer_index(text) >= 0


def _opencode_answer_index(text: str) -> int:
    prompts = list(_OPENCODE_PROMPT_RE.finditer(text))
    if not prompts:
        return -1
    tail_start = prompts[-1].end()
    tail = text[tail_start:]
    if _OPENCODE_BUSY_RE.search(tail):
        return -1

    offset = tail_start
    in_thinking = False
    for line in tail.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            in_thinking = False
            offset += len(line)
            continue
        if stripped.lower().startswith("thinking:"):
            in_thinking = True
            offset += len(line)
            continue
        if in_thinking:
            offset += len(line)
            continue
        if _opencode_status_line(stripped):
            offset += len(line)
            continue
        return offset + len(line) - len(line.lstrip())
    return -1


def _opencode_status_line(line: str) -> bool:
    return (
        line.startswith("▣ Build")
        or line.startswith("BUILD")
        or line.startswith("█")
        or line.startswith("▀")
    )


def _completion_marker_index(text: str, *, profile: str | None = None) -> int:
    profile = (profile or "").casefold()
    if profile in {"reasonix", "deepseek"}:
        matches = list(_REASONIX_REPLY_RE.finditer(text))
        return matches[-1].start() if matches else -1
    if profile in {"claude", "opus"}:
        plan_ready = list(_CLAUDE_PLAN_READY_RE.finditer(text))
        if plan_ready:
            return plan_ready[-1].start()
        matches = list(_CLAUDE_STYLE_ANSWER_RE.finditer(text))
        return matches[-1].start() if matches else -1
    if profile == "codex":
        matches = list(_CODEX_ANSWER_RE.finditer(text))
        return matches[-1].start() if matches else -1
    if profile == "opencode":
        return _opencode_answer_index(text)
    if profile:
        return -1

    matches: list[int] = []
    if match := _REASONIX_REPLY_RE.search(text):
        matches.append(match.start())
    for regex in (_CLAUDE_STYLE_ANSWER_RE, _CODEX_ANSWER_RE):
        found = list(regex.finditer(text))
        if found:
            matches.append(found[-1].start())
    opencode_index = _opencode_answer_index(text)
    if opencode_index >= 0:
        matches.append(opencode_index)
    return max(matches) if matches else -1
