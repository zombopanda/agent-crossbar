"""Provider execution for print/sync/GUI agent jobs."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from agent_crossbar.context import pack_context
from agent_crossbar.env_compat import getenv
from agent_crossbar.gui_lifecycle import (
    ChatGptSession,
    CompletionTracker,
    DomHealthTracker,
    JobArtifactStore,
    ResponseObservation,
    TurnLifecycle,
    prompt_mismatch_diagnostics,
    sessions,
    verify_owned_window,
)
from agent_crossbar.jobs import JobStore, default_state_root
from agent_crossbar.providers import (
    _opencode_model_id,
    build_launch_plan,
    reasonix_dev_prompt,
    reasonix_shell_env,
    reasonix_shell_mcp_spec,
)
from agent_crossbar.run_handles import run_handles
from agent_crossbar.tmux_output import (
    interactive_tmux_output_complete,
    interactive_tmux_output_summary,
    interactive_tmux_session_resumed,
)

_DEFAULT_TIMEOUT_SEC = 1800
CHATGPT_PRO_DEFAULT_TIMEOUT_SEC = 1800
_DEFAULT_CUA_CALL_TIMEOUT_SEC = 60.0
_CHATGPT_BUNDLE_ID = "com.openai.chat"
_CHATGPT_SELECTED_CANDIDATE = "ChatGPT native app via cua-driver"
_CUA_ACTION_TOOLS = {
    "click",
    "double_click",
    "hotkey",
    "launch_app",
    "page",
    "press_key",
    "right_click",
    "set_value",
    "type_text",
}
_TMUX_INHERITED_ENV_KEYS = frozenset(
    {
        "ALL_PROXY",
        "DASHSCOPE_API_KEY",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NODE_EXTRA_CA_CERTS",
        "NO_PROXY",
        "OPENAI_API_KEY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_FILE",
        "TMPDIR",
        "USER",
    }
)
_TMUX_CLEAN_ENV_KEYS = _TMUX_INHERITED_ENV_KEYS | {"TERM"}
_TMUX_LITERAL_PROMPT_MAX_BYTES = 8192
_TMUX_PASTE_SETTLE_SEC = 0.5
_chatgpt_turn_generating = threading.Event()


def _candidate_home_dirs() -> list[Path]:
    """Return likely user homes for wrapper/config lookup."""
    homes: list[Path] = []
    for raw in (
        getenv("AGENT_CROSSBAR_PROVIDER_HOME"),
        getenv("AGENT_CROSSBAR_USER_HOME"),
        os.environ.get("REAL_HOME"),
        f"/Users/{os.environ.get('USER')}"
        if sys.platform == "darwin" and os.environ.get("USER")
        else None,
        str(Path.home()),
        str(Path.home() / "bo")
        if sys.platform == "darwin" and os.environ.get("USER") == "bo"
        else None,
    ):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists() and path not in homes:
            homes.append(path)
    return homes


def _provider_home() -> str:
    """Return the real provider home used for CLI config lookup."""
    homes = _candidate_home_dirs()
    return str(homes[0]) if homes else str(Path.home())


@dataclass
class CommandCandidate:
    """Executable candidate for a provider fallback chain."""

    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    send_prompt: bool = True
    redirect_output: bool = True
    prompt_delay_sec: float = 0.0
    prompt_ready_patterns: tuple[str, ...] = ()
    prompt_submit_delay_sec: float = 0.0
    prompt_text: str | None = None
    stdin_text: str | None = None
    clean_env: bool = False


@dataclass(frozen=True)
class ChatGptBrowserCandidate:
    key: str
    name: str
    bundle_id: str


_CHATGPT_BROWSER_CANDIDATES = (
    ChatGptBrowserCandidate("helium", "Helium", "net.imput.helium"),
    ChatGptBrowserCandidate("chrome", "Chrome", "com.google.Chrome"),
    ChatGptBrowserCandidate("safari", "Safari", "com.apple.Safari"),
)


class CuaDriverClient:
    """Small JSON wrapper around the cua-driver CLI."""

    def __init__(self, bin_path: str | None = None, call_timeout_sec: float | None = None):
        self.bin_path = bin_path or _wrapper_bin(
            "AGENT_CROSSBAR_CUA_DRIVER_BIN",
            ".local/bin/cua-driver",
            "cua-driver",
        )
        self.call_timeout_sec = call_timeout_sec or _DEFAULT_CUA_CALL_TIMEOUT_SEC

    def call(
        self,
        tool: str,
        payload: dict[str, Any],
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        result = subprocess.run(
            [self.bin_path, "call", tool, json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=timeout_sec if timeout_sec is not None else self.call_timeout_sec,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or f"cua-driver {tool} failed"
            raise RuntimeError(message)
        if not result.stdout.strip():
            return {}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            if tool in _CUA_ACTION_TOOLS:
                return {"raw_output": result.stdout.strip()}
            raise

    def call_with_timeout(
        self, tool: str, payload: dict[str, Any], timeout_sec: float
    ) -> dict[str, Any]:
        return self.call(tool, payload, timeout_sec=timeout_sec)


def _wrapper_bin(env_name: str, relative_path: str, fallback: str) -> str:
    """Resolve a provider wrapper from env, likely homes, then PATH."""
    configured = os.environ.get(env_name)
    if configured:
        return configured
    for home in _candidate_home_dirs():
        candidate = home / relative_path
        if candidate.exists():
            return str(candidate)
    return fallback


def _path_bin(env_name: str, fallback: str) -> str:
    """Resolve a PATH binary before tmux startup files can alter PATH."""
    configured = os.environ.get(env_name)
    if configured:
        return configured
    return shutil.which(fallback) or fallback


def _state_root() -> Path:
    return default_state_root()


def _mise_trust_env() -> dict[str, str]:
    """Trust Bo's mise config when present without hardcoding it as required."""
    if os.environ.get("MISE_TRUSTED_CONFIG_PATHS"):
        return {}
    configured = getenv("AGENT_CROSSBAR_MISE_CONFIG")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(home / ".config" / "mise" / "config.toml" for home in _candidate_home_dirs())
    for candidate in candidates:
        if candidate.exists():
            return {"MISE_TRUSTED_CONFIG_PATHS": str(candidate)}
    return {}


def _execution_cwd(req: dict[str, Any]) -> str | None:
    """Run providers from the requested cwd or the MCP server cwd."""
    cwd = req.get("cwd")
    return str(cwd) if cwd else os.getcwd()


def _provider_env(extra: dict[str, str]) -> dict[str, str]:
    """Return subprocess env with provider config rooted at the real home."""
    home = _provider_home()
    return {**os.environ, "HOME": home, "REAL_HOME": home, **extra}


def _prepare_prompt(req: dict[str, Any]) -> str:
    """Return the prepared prompt, falling back to the raw prompt."""
    if req.get("_prepared_prompt") is not None:
        return str(req["_prepared_prompt"])
    return str(req.get("prompt") or "")


def _chatgpt_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / "agent-crossbar-chatgpt-pro.lock"


def _acquire_chatgpt_lock(stale_after_sec: int = 900) -> tuple[bool, Path, str | None]:
    path = _chatgpt_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()

    # Check legacy lock path for backward compatibility with processes that
    # were started before the lock moved to /tmp.
    legacy_path = _state_root() / "locks" / "chatgpt_pro.lock"
    if legacy_path.exists():
        try:
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            legacy_age = now - float(legacy.get("created_at", 0))
            if legacy_age <= stale_after_sec:
                return False, legacy_path, "Another chatgpt_pro GUI job is already running"
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        created_at = float(current.get("created_at", 0))
        if now - created_at > stale_after_sec:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        path.unlink(missing_ok=True)

    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False, path, "Another chatgpt_pro GUI job is already running"

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"created_at": now, "pid": os.getpid()}, handle)
    return True, path, None


def _release_chatgpt_lock(path: Path) -> None:
    path.unlink(missing_ok=True)


def _read_text_clipboard() -> str:
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _write_text_clipboard(text: str) -> None:
    try:
        subprocess.run(["pbcopy"], input=text, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _chatgpt_prompt(
    req: dict[str, Any], nonce: str, *, context_text: str = ""
) -> tuple[str, str, str]:
    begin = f"BEGIN_AGENTS_MCP_RESPONSE_{nonce}"
    end = f"END_AGENTS_MCP_RESPONSE_{nonce}"
    prompt = _prepare_prompt(req).rstrip()
    if context_text:
        prompt = f"{context_text.rstrip()}\n\n{prompt}"
    wrapped = (
        "You are the ChatGPT Pro advisor called by agents MCP. "
        "Answer the user's request directly and do not mention these transport instructions.\n\n"
        "For response correlation, wrap your complete final answer exactly like this:\n"
        f"{begin}\n"
        "<your answer>\n"
        f"{end}\n\n"
        f"User request:\n{prompt}\n"
    )
    return wrapped, begin, end


class ChatGptGuiError(RuntimeError):
    """Recoverable ChatGPT GUI failure with an artifactable AX tree."""

    def __init__(self, error: str, message: str, tree: str = ""):
        super().__init__(message)
        self.error = error
        self.tree = tree


def _find_element(tree: str, label: str, role: str = "AXButton") -> int | None:
    pattern = (
        rf"\[(\d+)\]\s+{re.escape(role)}[^\n]*(?:\({re.escape(label)}\)|\"{re.escape(label)}\")"
    )
    match = re.search(pattern, tree)
    if match:
        return int(match.group(1))
    return None


def _element_index(line: str) -> int | None:
    match = re.search(r"\[(\d+)\]", line)
    if match:
        return int(match.group(1))
    return None


def _chatgpt_model_label(line: str) -> str | None:
    match = re.search(r'AXButton\s*=\s*"([^"]+)"', line)
    if match:
        return match.group(1)
    match = re.search(r"AXButton\s+\(([^)]*)\)", line)
    if match:
        return match.group(1)
    return None


def _active_chatgpt_model_is_pro(tree: str) -> bool:
    for line in tree.splitlines():
        if "AXButton" not in line:
            continue
        if "Pick a model or GPT" not in line and "(Options)" not in line:
            continue
        label = _chatgpt_model_label(line)
        if label and re.search(r"\bPro\b", label):
            return True
    return False


def _find_chatgpt_model_picker(tree: str) -> int | None:
    for line in tree.splitlines():
        if "AXButton" not in line or "DISABLED" in line:
            continue
        if "Pick a model or GPT" in line or "(Options)" in line:
            return _element_index(line)
    return None


def _find_chatgpt_pro_option(tree: str) -> int | None:
    for line in tree.splitlines():
        if "AXButton" not in line:
            continue
        if "Research-grade intelligence" in line or "(Pro," in line or "(Pro)" in line:
            return _element_index(line)
    return None


def _find_first_text_area(tree: str) -> int | None:
    matches = list(re.finditer(r"\[(\d+)\]\s+AXTextArea\b", tree))
    for position, match in enumerate(matches):
        chunk_end = matches[position + 1].start() if position + 1 < len(matches) else len(tree)
        textarea_chunk = tree[match.start() : chunk_end]
        if not re.search(r"Ask ChatGPT|Chat with ChatGPT", textarea_chunk, re.IGNORECASE):
            continue
        return int(match.group(1))
    return None


def _extract_marked_response(tree: str, begin: str, end: str) -> str | None:
    outputs: list[str] = []
    search_from = 0
    while True:
        begin_idx = tree.find(begin, search_from)
        if begin_idx < 0:
            break
        content_start = begin_idx + len(begin)
        end_idx = tree.find(end, content_start)
        if end_idx < 0:
            break
        output = tree[content_start:end_idx]
        output = output.replace("\\n", "\n").strip(" \n")
        if not _chatgpt_placeholder_response(output):
            outputs.append(output)
        search_from = end_idx + len(end)
    if not outputs:
        return None
    return outputs[-1]


def _chatgpt_placeholder_response(output: str) -> bool:
    normalized = output.strip().strip("`").strip().casefold()
    return normalized in {"", "<your answer>", "your answer"}


def _chatgpt_time_limit_reached(tree: str) -> bool:
    return (
        "You've reached your limit on ChatGPT." in tree
        or "You’ve reached your limit on ChatGPT." in tree
    )


def _find_latest_chatgpt_copy_action(tree: str) -> int | None:
    found: int | None = None
    for line in tree.splitlines():
        if "AXButton" not in line or "actions=[" not in line or "Copy" not in line:
            continue
        index = _element_index(line)
        if index is not None:
            found = index
    return found


def _find_copy_menu_item(tree: str) -> int | None:
    menu_item = _find_element(tree, "Copy", role="AXMenuItem")
    if menu_item is not None:
        return menu_item
    return _find_element(tree, "Copy", role="AXButton")


def _copy_marked_chatgpt_response(
    cua: Any,
    pid: int,
    window_id: int,
    tree: str,
    begin: str,
    end: str,
    sleep: Callable[[float], None],
) -> str | None:
    source = _find_latest_chatgpt_copy_action(tree)
    if source is None:
        return None

    previous_clipboard = _read_text_clipboard()
    try:
        cua.call(
            "click",
            {
                "pid": pid,
                "window_id": window_id,
                "element_index": source,
                "action": "show_menu",
            },
        )
        sleep(0.2)
        menu_tree = _chatgpt_snapshot(cua, pid, window_id)
        copy_item = _find_copy_menu_item(menu_tree)
        if copy_item is None:
            return None
        cua.call("click", {"pid": pid, "window_id": window_id, "element_index": copy_item})
        sleep(0.2)
        return _extract_marked_response(_read_text_clipboard(), begin, end)
    finally:
        _write_text_clipboard(previous_clipboard)


def _chatgpt_artifact_dir(nonce: str) -> Path:
    path = _state_root() / "artifacts" / "chatgpt_pro" / nonce
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_chatgpt_artifact(nonce: str, name: str, content: str) -> str:
    path = _chatgpt_artifact_dir(nonce) / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _chatgpt_failure(error: str, message: str, *, nonce: str, tree: str = "") -> dict[str, Any]:
    artifacts: list[str] = []
    if tree:
        artifacts.append(_write_chatgpt_artifact(nonce, "last-tree.txt", tree))
    return {
        "ok": False,
        "error": error,
        "message": message,
        "attempts": [],
        "artifacts": artifacts,
    }


def _chatgpt_app(cua: Any) -> dict[str, Any]:
    apps = cua.call("list_apps", {}).get("apps", [])
    for app in apps:
        if app.get("bundle_id") == _CHATGPT_BUNDLE_ID and app.get("running") and app.get("pid"):
            return app
    launched = cua.call("launch_app", {"bundle_id": _CHATGPT_BUNDLE_ID})
    bundle_id = launched.get("bundle_id")
    pid = launched.get("pid")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or bundle_id not in (_CHATGPT_BUNDLE_ID, None, "", "?")
    ):
        raise RuntimeError("Native ChatGPT app did not launch")
    return launched


def _list_browser_windows(cua: Any, pid: int | None) -> list[dict[str, Any]] | None:
    """Return the raw window list for *pid*, or None when it cannot be read."""
    if pid is None:
        return None
    try:
        windows = cua.call("list_windows", {"pid": pid}).get("windows", [])
    except Exception:
        return None
    return [window for window in windows if isinstance(window, dict)]


def _find_windows_on_current_space(cua: Any, pid: int) -> list[dict[str, Any]]:
    """Return ChatGPT windows on the current Space, sorted by z_index descending.

    We accept windows that are on the current Space regardless of is_on_screen
    status — a minimized window can be brought back with activate + raise.
    Some cua-driver builds do not populate `on_current_space`; when that
    metadata is missing, fall back to visible on-screen windows.
    """
    windows = cua.call("list_windows", {"pid": pid}).get("windows", [])
    return sorted(
        [
            window
            for window in windows
            if window.get("pid") == pid
            and window.get("layer") == 0
            and (
                window.get("on_current_space") is True
                or (window.get("on_current_space") is None and window.get("is_on_screen") is True)
            )
        ],
        key=lambda item: item.get("z_index", 0),
        reverse=True,
    )


def _activate_chatgpt_app(cua: Any | None = None, pid: int | None = None) -> None:
    """Activate the native ChatGPT app via System Events to bring window to foreground.

    Tolerates missing osascript (e.g. on Linux CI runners). This is a best-effort
    macOS window activation — the caller should continue without it.
    """
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                """
tell application "System Events"
    tell process "ChatGPT"
        set frontmost to true
        tell window 1
            perform action "AXRaise"
        end tell
    end tell
end tell
""",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    time.sleep(0.5)


def _chatgpt_window(cua: Any, pid: int) -> dict[str, Any]:
    """Return a usable ChatGPT window, activating if needed."""
    valid = _find_windows_on_current_space(cua, pid)
    if valid:
        return valid[0]

    # App is running but no window visible on current Space
    _activate_chatgpt_app()
    valid = _find_windows_on_current_space(cua, pid)
    if valid:
        return valid[0]

    raise RuntimeError("No native ChatGPT window is available on the current Space")


def _refresh_chatgpt_window(pid: int, window_id: int) -> None:
    """Force the ChatGPT window to become visible via System Events.

    Tolerates missing osascript (e.g. on Linux CI runners). The caller handles
    the case where the refresh didn't actually happen by falling back to other
    recovery strategies.
    """
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                """
tell application "System Events"
    tell process "ChatGPT"
        set frontmost to true
        tell window 1
            perform action "AXRaise"
        end tell
    end tell
end tell
""",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    time.sleep(1.0)


def _chatgpt_snapshot(cua: Any, pid: int, window_id: int) -> str:
    state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
    tree = str(state.get("tree_markdown") or "")
    bundle_id = state.get("bundle_id")
    if bundle_id not in (_CHATGPT_BUNDLE_ID, None):
        raise RuntimeError("CUA target is not the native ChatGPT app")
    if (
        bundle_id is None
        and 'AXWindow "ChatGPT"' not in tree
        and 'AXApplication "ChatGPT"' not in tree
    ):
        raise RuntimeError("CUA target is not the native ChatGPT app")

    # If the tree looks like a sidebar-only view (no textarea for chat input,
    # no chat response content), the window may be off-screen and the AX cache
    # is stale. A valid new-chat view always has an AXTextArea.
    is_sidebar = (
        "AXTextArea" not in tree and "Thought for" not in tree and "Scroll to bottom" not in tree
    )
    if is_sidebar:
        _refresh_chatgpt_window(pid, window_id)
        time.sleep(1.0)
        state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
        tree = str(state.get("tree_markdown") or "")

    return tree


def _ensure_chatgpt_pro_model(
    cua: Any,
    pid: int,
    window_id: int,
    tree: str,
    sleep: Callable[[float], None],
) -> str:
    if _active_chatgpt_model_is_pro(tree):
        return tree

    picker = _find_chatgpt_model_picker(tree)
    if picker is None:
        raise ChatGptGuiError(
            "model_picker_not_found",
            "Could not find the native ChatGPT model picker",
            tree,
        )

    cua.call("click", {"pid": pid, "window_id": window_id, "element_index": picker})
    sleep(0.3)
    picker_tree = _chatgpt_snapshot(cua, pid, window_id)
    option = _find_chatgpt_pro_option(picker_tree)
    if option is None:
        raise ChatGptGuiError(
            "model_pro_option_not_found",
            "Could not find the native ChatGPT Pro model option",
            picker_tree,
        )

    cua.call("click", {"pid": pid, "window_id": window_id, "element_index": option})
    sleep(0.5)
    selected_tree = _chatgpt_snapshot(cua, pid, window_id)
    if not _active_chatgpt_model_is_pro(selected_tree):
        raise ChatGptGuiError(
            "model_not_pro",
            "Native ChatGPT did not switch to a Pro model",
            selected_tree,
        )
    return selected_tree


def _chatgpt_page_text(
    cua: Any,
    pid: int,
    window_id: int,
    *,
    timeout_sec: float | None = None,
) -> str:
    payload = {"action": "get_text", "pid": pid, "window_id": window_id}
    if timeout_sec is not None and hasattr(cua, "call_with_timeout"):
        result = cua.call_with_timeout("page", payload, timeout_sec)
    else:
        result = cua.call("page", payload)
    if isinstance(result, str):
        return result
    return str(result.get("text") or result.get("raw_output") or "")


def _chatgpt_web_control_label(line: str) -> str | None:
    match = re.search(
        r"AX(?:PopUpButton|Button|MenuItem|RadioButton)\b(?:\s*=\s*)?\s*"
        r'(?P<label>"[^"]*"|\([^)]*\))',
        line,
    )
    if match is None:
        return None
    return match.group("label")[1:-1].strip()


def _chatgpt_web_active_model_is_pro(tree: str) -> bool:
    picker_line = _chatgpt_web_model_picker_line(tree)
    return bool(picker_line and (_chatgpt_web_control_label(picker_line) or "").casefold() == "pro")


def _chatgpt_screen_time_limit(tree: str) -> str | None:
    """Return the macOS Screen Time limit message exposed over AX, if present."""
    if not re.search(r'AXStaticText\s*=\s*"Time Limit"', tree, re.IGNORECASE):
        return None
    match = re.search(
        r'AXStaticText\s*=\s*"(You[’\']ve reached your limit on [^"]+)"',
        tree,
        re.IGNORECASE,
    )
    return match.group(1) if match else "macOS Screen Time limit reached"


def _chatgpt_web_model_picker_line(tree: str) -> str | None:
    lines = tree.splitlines()
    for composer_position, line in enumerate(lines):
        if "AXTextArea" not in line or not re.search(
            r"Ask ChatGPT|Chat with ChatGPT", line, re.IGNORECASE
        ):
            continue
        for candidate_line in lines[composer_position + 1 : composer_position + 7]:
            if re.search(r"AX(?:PopUpButton|Button)\b", candidate_line) and (
                _chatgpt_web_control_label(candidate_line) or ""
            ).casefold() in {"auto", "pro", "low", "medium", "high"}:
                return candidate_line
    return None


def _find_chatgpt_web_model_picker(tree: str) -> int | None:
    picker_line = _chatgpt_web_model_picker_line(tree)
    if picker_line is not None:
        return _element_index(picker_line)
    return None


def _find_chatgpt_web_pro_option(tree: str) -> int | None:
    for line in tree.splitlines():
        if not re.search(r"AX(?:Button|MenuItem|RadioButton)\b", line):
            continue
        if (_chatgpt_web_control_label(line) or "").casefold() != "pro":
            continue
        index = _element_index(line)
        if index is not None:
            return index
    return None


def _find_chatgpt_web_send_button(tree: str) -> int | None:
    for line in tree.splitlines():
        if not re.search(
            r'AXButton(?:\s*=\s*)?\s*(?:\(|")Send(?: prompt| message)?(?:\)|")',
            line,
            re.IGNORECASE,
        ):
            continue
        index = _element_index(line)
        if index is not None:
            return index
    return None


def _find_chatgpt_web_stop_button(tree: str) -> int | None:
    """Return the visible Stop control while ChatGPT is generating.

    The live label is ``Stop answering``; earlier builds used ``Stop`` and
    ``Stop generating``. Match the whole family rather than a fixed list, so a
    UI wording change cannot silently make a running turn look idle.
    """
    for line in tree.splitlines():
        if not re.search(r"AXButton\b", line):
            continue
        label = (_chatgpt_web_control_label(line) or "").casefold().strip()
        if label == "stop" or label.startswith("stop "):
            index = _element_index(line)
            if index is not None:
                return index
    return None


def _chatgpt_response_is_running(tree: str) -> bool:
    """True while ChatGPT still exposes a Stop control for the active turn."""
    return _find_chatgpt_web_stop_button(tree) is not None


def _chatgpt_completion_action_visible(tree: str) -> bool:
    """True when ChatGPT exposes an idle-turn action (copy, or a fresh Send).

    ChatGPT swaps the composer's Send control for Stop while a turn streams,
    so a visible Send (or a response Copy action) is the visible evidence that
    the turn is no longer running.
    """
    if _find_latest_chatgpt_copy_action(tree) is not None:
        return True
    return _find_chatgpt_web_send_button(tree) is not None


def _chatgpt_model_choices(tree: str) -> list[str]:
    """Return the visible model/effort choice labels, deduplicated in order."""
    choices: list[str] = []
    for line in tree.splitlines():
        if not re.search(r"AX(?:PopUpButton|Button|MenuItem|RadioButton)\b", line):
            continue
        label = (_chatgpt_web_control_label(line) or "").strip()
        if not label:
            continue
        normalized = label.casefold()
        if normalized in {"auto", "pro", "low", "medium", "high", "instant", "thinking"}:
            if label not in choices:
                choices.append(label)
    return choices


_EFFORT_CHOICES = frozenset({"low", "medium", "high"})


def _chatgpt_requested_label(requested: str | None, available: list[str]) -> str | None:
    """Resolve a requested public model/effort onto a live visible choice label.

    Matching is exact on the visible label or on a token of the requested
    value (``gpt-5-pro`` → ``Pro``).  No default is invented: an unmatched
    request returns None so the caller can fail closed.
    """
    if not requested:
        return None
    normalized = requested.strip().casefold()
    if not normalized:
        return None
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    for label in available:
        candidate = label.casefold()
        if candidate == normalized or candidate in tokens:
            return label
    return None


def _chatgpt_web_composer_value(tree: str, text_area: int) -> str | None:
    marker = re.search(
        rf'(?m)^\s*(?:-\s*)?\[{text_area}\]\s+AXTextArea(?:\s+"[^"]*")?\s*=\s*"',
        tree,
    )
    if marker is None:
        # Safari exposes an empty ChatGPT composer as a role/title-only AX node,
        # rather than as an AXTextArea with an explicit empty value.
        role_only = re.search(
            rf'(?m)^\s*(?:-\s*)?\[{text_area}\]\s+AXTextArea\s+"[^\"]*"\s*\(',
            tree,
        )
        if role_only is not None:
            return ""
        return None

    start = marker.end()
    escaped = False
    for index, char in enumerate(tree[start:], start):
        if char == '"' and not escaped:
            raw_value = tree[start:index]
            try:
                return json.loads(f'"{raw_value}"')
            except json.JSONDecodeError:
                return raw_value
        escaped = char == "\\" and not escaped
    return None


def _chatgpt_web_composer_is_empty(tree: str, text_area: int) -> bool:
    placeholders = {
        "ask chatgpt",
        "message chatgpt",
        "chat with chatgpt",
        "ask anything",
    }
    value = _chatgpt_web_composer_value(tree, text_area)
    if value is None:
        return False
    normalized_value = value.strip().casefold()
    return not normalized_value or normalized_value in placeholders


def _chatgpt_web_composer_matches_prompt(actual: str | None, prompt: str) -> bool:
    """Compare ChatGPT's contenteditable canonical form without losing text."""
    if actual is None:
        return False

    def canonicalize(value: str) -> str:
        return "\n".join(line for line in value.replace("\r\n", "\n").split("\n") if line.strip())

    return canonicalize(actual) == canonicalize(prompt)


def _chatgpt_web_signed_out(text: str) -> bool:
    lowered = text.casefold()
    return "ask chatgpt" not in lowered and ("log in" in lowered or "sign up" in lowered)


def _process_is_headless(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    command = result.stdout
    return result.returncode != 0 or "--headless" in command or "--no-startup-window" in command


def _find_gui_browser_pid(candidate: ChatGptBrowserCandidate) -> int | None:
    if candidate.key != "chrome":
        return None
    try:
        result = subprocess.run(
            ["pgrep", "-x", "Google Chrome"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for value in result.stdout.splitlines():
        try:
            pid = int(value.strip())
        except ValueError:
            continue
        if pid > 0 and not _process_is_headless(pid):
            return pid
    return None


def _chatgpt_browser_app(
    cua: Any, candidate: ChatGptBrowserCandidate, sleep: Callable[[float], None]
) -> dict[str, Any]:
    def running_app() -> dict[str, Any] | None:
        for app in cua.call("list_apps", {}).get("apps", []):
            pid = app.get("pid")
            if (
                app.get("bundle_id") == candidate.bundle_id
                and app.get("running")
                and isinstance(pid, int)
                and not isinstance(pid, bool)
                and pid > 0
                and not _process_is_headless(pid)
            ):
                return app
        discovered_pid = _find_gui_browser_pid(candidate)
        if discovered_pid is not None:
            return {
                "bundle_id": candidate.bundle_id,
                "pid": discovered_pid,
                "running": True,
            }
        return None

    app = running_app()
    if app is not None:
        return app
    launch_args = (
        [
            "open",
            "-na",
            "Google Chrome",
            "--args",
            "--new-window",
            "https://chatgpt.com/",
        ]
        if candidate.key == "chrome"
        else ["open", "-b", candidate.bundle_id, "https://chatgpt.com/"]
    )
    subprocess.run(
        launch_args,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    for _ in range(10):
        sleep(0.5)
        app = running_app()
        if app is not None:
            return app
    raise RuntimeError(f"{candidate.name} is not available")


def _chatgpt_browser_window(
    cua: Any,
    candidate: ChatGptBrowserCandidate,
    pid: int,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    def browser_windows(target_pid: int) -> list[dict[str, Any]]:
        windows = cua.call("list_windows", {"pid": target_pid}).get("windows", [])
        return [
            window
            for window in windows
            if window.get("pid") == target_pid and window.get("layer") == 0
        ]

    def chatgpt_window(windows: list[dict[str, Any]], target_pid: int) -> dict[str, Any] | None:
        ordered_windows = sorted(
            windows,
            key=lambda window: (
                window.get("is_on_screen") is not True,
                0 if str(window.get("title") or "").strip().casefold() == "chatgpt" else 1,
                -int(window.get("z_index") or 0),
            ),
        )
        for window in ordered_windows:
            title = str(window.get("title") or "")
            if "just a moment" in title.casefold():
                return {**window, "pid": target_pid}
            try:
                state = cua.call(
                    "get_window_state",
                    {"pid": target_pid, "window_id": int(window["window_id"])},
                )
            except Exception:
                continue
            tree = str(state.get("tree_markdown") or "")
            if _chatgpt_screen_time_limit(tree) is not None:
                return {**window, "pid": target_pid}
            has_chatgpt_address = re.search(
                r'AXTextField\s*=\s*"[^"]*chatgpt\.com[^"]*"\s*\(Address and search bar\)',
                tree,
                re.IGNORECASE,
            )
            # Only raise the browser when the page is not already exposed over
            # accessibility: a rendered window can be driven entirely in the
            # background, without taking the user's foreground or key focus.
            needs_activation = _find_first_text_area(tree) is None
            if re.search(r'AXWebArea\s+(?:"ChatGPT"|\(ChatGPT\))', tree):
                return {**window, "pid": target_pid, "needs_activation": needs_activation}
            if has_chatgpt_address:
                return {**window, "pid": target_pid, "needs_activation": needs_activation}
        return None

    def maybe_activate(window: dict[str, Any], target_pid: int) -> dict[str, Any]:
        if candidate.key != "safari" and window.get("needs_activation", True):
            _activate_chatgpt_browser(candidate, target_pid, int(window["window_id"]))
        return window

    initial_windows = browser_windows(pid)
    window = chatgpt_window(initial_windows, pid)
    if window is not None:
        return maybe_activate(window, pid)
    for _ in range(5):
        sleep(0.2)
        window = chatgpt_window(browser_windows(pid), pid)
        if window is not None:
            return maybe_activate(window, pid)
    launch_args = (
        [
            "open",
            "-na",
            "Google Chrome",
            "--args",
            "--new-window",
            "https://chatgpt.com/",
        ]
        if candidate.key == "chrome"
        else ["open", "-b", candidate.bundle_id, "https://chatgpt.com/"]
    )
    prelaunch_pids = {pid}
    if candidate.key == "chrome":
        try:
            prelaunch_pids.update(
                app_pid
                for app in cua.call("list_apps", {}).get("apps", [])
                if app.get("bundle_id") == candidate.bundle_id
                and isinstance((app_pid := app.get("pid")), int)
                and not isinstance(app_pid, bool)
                and app_pid > 0
            )
        except Exception:
            pass
    subprocess.run(
        launch_args,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    candidate_pids = [pid]

    def refresh_chrome_pids() -> None:
        if candidate.key != "chrome":
            return
        try:
            discovered_new_pids: set[int] = set()
            for app in cua.call("list_apps", {}).get("apps", []):
                discovered_pid = app.get("pid")
                if (
                    app.get("bundle_id") == candidate.bundle_id
                    and isinstance(discovered_pid, int)
                    and discovered_pid not in prelaunch_pids
                    and not _process_is_headless(discovered_pid)
                ):
                    discovered_new_pids.add(discovered_pid)
            candidate_pids[:] = sorted(discovered_new_pids) + [pid]
        except Exception:
            pass

    for _ in range(10):
        sleep(0.5)
        refresh_chrome_pids()
        for candidate_pid in candidate_pids:
            windows = browser_windows(candidate_pid)
            window = chatgpt_window(windows, candidate_pid)
            if window is not None:
                return maybe_activate(window, candidate_pid)
    raise RuntimeError(f"{candidate.name} did not expose a ChatGPT window")


def _activate_chatgpt_browser(
    candidate: ChatGptBrowserCandidate, pid: int, window_id: int | None = None
) -> None:
    """Raise the selected browser before taking or acting on an AX snapshot."""
    script = """
on run argv
    set targetPid to (item 1 of argv) as integer
    set targetWindowNumber to (item 2 of argv) as integer
    tell application "System Events"
        set targetProcess to first application process whose unix id is targetPid
        tell targetProcess
            set frontmost to true
            if targetWindowNumber > 0 then
                repeat with targetWindow in windows
                    try
                        if value of attribute "AXWindowNumber" of targetWindow is targetWindowNumber then
                            perform action "AXRaise" of targetWindow
                            exit repeat
                        end if
                    end try
                end repeat
            else if (count of windows) > 0 then
                perform action "AXRaise" of window 1
            end if
        end tell
    end tell
end run
"""
    try:
        subprocess.run(
            ["osascript", "-e", script, str(pid), str(window_id or 0)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _verify_temporary_chat_indicator(tree: str) -> bool:
    """Return True when the AX tree contains the temporary-chat indicator.

    Checks for an AXHeading containing "Temporary Chat" or an AXButton
    containing "Turn off temporary chat" (case-insensitive). Either match
    suffices.
    """
    lowered = tree.casefold()
    for line in lowered.splitlines():
        if "axheading" in line and "temporary chat" in line:
            return True
        if "axbutton" in line and "turn off temporary chat" in line:
            return True
    return False


def _close_turn_window(
    cua: Any,
    candidate: ChatGptBrowserCandidate,
    pid: int,
    window_id: int,
    sleep: Callable[[float], None],
) -> None:
    """Close a single browser window without terminating the app.

    Never raises — cleanup code must be silent.
    """
    try:
        state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
        tree = str(state.get("tree_markdown") or "")
        close_idx: int | None = None
        for label in ("Close", "Close tab"):
            close_idx = _find_element(tree, label, role="AXButton")
            if close_idx is not None:
                break
        if close_idx is not None:
            cua.call(
                "click",
                {"pid": pid, "window_id": window_id, "element_index": close_idx},
            )
        else:
            cua.call("hotkey", {"pid": pid, "keys": ["cmd", "w"]})
    except Exception:
        pass


def _chatgpt_open_fresh_window(
    cua: Any,
    candidate: ChatGptBrowserCandidate,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Open a new ChatGPT window whose composer is empty.

    Used when the discovered window holds an unrelated draft: the user's text is
    never cleared or overwritten, the turn simply takes a fresh window. Returns
    ``{"pid", "window_id"}`` or None when no clean window could be obtained.

    When a window is found with an empty composer, the function additionally
    verifies that the visible ``Temporary Chat`` indicator is present so a
    pre-existing non-temporary window is never returned by accident.
    """
    if candidate.key == "safari":
        # Safari exposes no supported way to open a controlled extra window
        # without disturbing the user's session; retire instead.
        return None
    try:
        subprocess.run(
            [
                "open",
                "-na",
                candidate.name,
                "--args",
                "--new-window",
                "https://chatgpt.com/?temporary-chat=true",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    for _attempt in range(20):
        if _attempt > 0:
            sleep(0.5)
        try:
            apps = cua.call("list_apps", {}).get("apps", [])
        except Exception:
            # If the CUA cannot list apps on the first attempt, bail
            # immediately — polling will not help.
            if _attempt == 0:
                return None
            continue
        pids = [
            app.get("pid")
            for app in apps
            if isinstance(app, dict)
            and app.get("bundle_id") == candidate.bundle_id
            and isinstance(app.get("pid"), int)
        ]
        for pid in pids:
            for window in _list_browser_windows(cua, pid) or []:
                window_id = window.get("window_id")
                if window_id is None:
                    continue
                try:
                    state = cua.call("get_window_state", {"pid": pid, "window_id": int(window_id)})
                except Exception:
                    continue
                tree = str(state.get("tree_markdown") or "")
                text_area = _find_first_text_area(tree)
                if text_area is None:
                    continue
                if _chatgpt_web_composer_is_empty(tree, text_area):
                    if _verify_temporary_chat_indicator(tree):
                        return {"pid": int(pid), "window_id": int(window_id)}
    return None


def _chatgpt_browser_snapshot(
    cua: Any,
    candidate: ChatGptBrowserCandidate,
    pid: int,
    window_id: int,
    sleep: Callable[[float], None],
) -> str:
    tree = ""
    for attempt in range(10):
        state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
        tree = str(state.get("tree_markdown") or "")
        if _find_first_text_area(tree) is not None:
            return tree
        if attempt == 0:
            _activate_chatgpt_browser(candidate, pid, window_id)
        if attempt < 9:
            sleep(0.5)
    return tree


def _chatgpt_clear_composer(
    cua: Any, pid: int, window_id: int, text_area: int
) -> dict[str, Any] | None:
    """Clear text this turn wrote into a composer it had proven empty.

    Delivery is only attempted after the composer was verified empty, so any
    remaining content belongs to this turn. Leaving a corrupted partial prompt
    behind would pollute the user's draft and block every later turn with
    ``composer_not_empty``.
    """
    try:
        state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
        current = _chatgpt_web_composer_value(str(state.get("tree_markdown") or ""), text_area)
        if current is None or not current.strip():
            return None
        cua.call(
            "set_value",
            {"pid": pid, "window_id": window_id, "element_index": text_area, "value": ""},
        )
        return {"cleared": True, "chars": len(current)}
    except Exception as exc:
        return {"cleared": False, "error": exc.__class__.__name__}


def _chatgpt_deliver_prompt(
    cua: Any,
    pid: int,
    window_id: int,
    text_area: int,
    prompt: str,
    sleep: Callable[[float], None] = time.sleep,
    preserve_existing_focus: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> bool:
    """Paste a complete prompt and prove the selected composer received it."""

    trace = diagnostics if diagnostics is not None else {}

    def succeeded(result: Any) -> bool:
        return not isinstance(result, dict) or (
            result.get("ok") is not False and not result.get("error")
        )

    def composer_center(state: dict[str, Any]) -> tuple[float, float] | None:
        window_frames: list[tuple[float, float, float, float]] = []
        for element in state.get("elements", []):
            if element.get("role") != "AXWindow":
                continue
            frame = element.get("frame")
            if not isinstance(frame, dict):
                continue
            try:
                window_frames.append(
                    (
                        float(frame["x"]),
                        float(frame["y"]),
                        float(frame["w"]),
                        float(frame["h"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        for element in state.get("elements", []):
            if element.get("element_index") != text_area:
                continue
            frame = element.get("frame")
            if not isinstance(frame, dict):
                continue
            try:
                x = float(frame["x"])
                y = float(frame["y"])
                width = float(frame["w"])
                height = float(frame["h"])
            except (KeyError, TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                center_x = x + width / 2
                center_y = y + height / 2
                # CUA's AX frames are screen-global while pixel actions on a
                # target window consume screenshot-local coordinates. Convert
                # through the containing AXWindow; without a window frame keep
                # the legacy coordinates for older/minimal driver snapshots.
                for window_x, window_y, window_width, window_height in window_frames:
                    if (
                        window_x <= center_x <= window_x + window_width
                        and window_y <= center_y <= window_y + window_height
                    ):
                        return center_x - window_x, center_y - window_y
                return center_x, center_y
        return None

    def verify_paste() -> tuple[bool, dict[str, Any] | None]:
        last_state: dict[str, Any] | None = None
        for attempt in range(5):
            state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
            last_state = state
            tree = str(state.get("tree_markdown") or "")
            value = _chatgpt_web_composer_value(tree, text_area)
            if _chatgpt_web_composer_matches_prompt(value, prompt):
                return True, state
            # A non-empty mismatch might be user/browser content. Never retry
            # over it: leave the composer untouched and refuse to submit.
            if not _chatgpt_web_composer_is_empty(tree, text_area):
                return False, state
            if attempt < 4:
                sleep(0.2)
        return False, last_state

    # Preferred path: a background accessibility write. It never steals the
    # user's keyboard focus, never moves the cursor, and never touches the
    # clipboard. ChatGPT's composer accepts it and re-renders its Send control,
    # which the exact verification below confirms.
    trace["stage"] = "ax_set_value"
    try:
        ax_write = cua.call(
            "set_value",
            {"pid": pid, "window_id": window_id, "element_index": text_area, "value": prompt},
        )
    except Exception as exc:
        ax_write = {"ok": False, "error": exc.__class__.__name__}
    if succeeded(ax_write):
        delivered, _state = verify_paste()
        if delivered:
            trace["stage"] = "verified"
            trace["focus_mode"] = "background_ax"
            return True
        trace["ax_residue"] = _chatgpt_clear_composer(cua, pid, window_id, text_area)

    previous_clipboard = _read_text_clipboard()
    injected_clipboard = False
    try:
        trace["stage"] = "clipboard_prepare"
        _write_text_clipboard(prompt)
        injected_clipboard = True
        # Safari's AX press clears the renderer focus even when its text area
        # is already focused. Keep that native focus and deliver the same
        # foreground Cmd+V instead.
        click_result: Any = None
        if not preserve_existing_focus:
            click_result = cua.call(
                "click",
                {"pid": pid, "window_id": window_id, "element_index": text_area},
            )
        click_succeeded = preserve_existing_focus or succeeded(click_result)
        trace["ax_focus"] = {
            "attempted": not preserve_existing_focus,
            "reported_success": click_succeeded,
        }
        # Chromium accepts the clipboard only when the key event is delivered
        # as foreground input.  This is still one ordinary Cmd+V transaction;
        # cua-driver restores the previous frontmost window immediately after.
        paste_payload: dict[str, Any] = {
            "pid": pid,
            "window_id": window_id,
            "keys": ["command", "v"],
            "delivery_mode": "foreground",
        }
        if not click_succeeded:
            # Some native browser AX bridges reject AXPress on a live textarea.
            # Re-snapshot and let cua-driver do a foreground pixel focus plus the
            # same ordinary Cmd+V clipboard transaction.
            focus_state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
            center = composer_center(focus_state)
            if center is None:
                return False
            paste_payload.update({"x": center[0], "y": center[1], "delivery_mode": "foreground"})
        trace["stage"] = "ax_focused_paste"
        if not succeeded(cua.call("hotkey", paste_payload)):
            trace["failure"] = "hotkey_rejected"
            return False
        # AX snapshots can lag just behind a successful paste. Poll a bounded
        # set of fresh snapshots, but never submit unless one equals exactly.
        delivered, last_state = verify_paste()
        if delivered:
            trace["stage"] = "verified"
            trace["focus_mode"] = "existing" if preserve_existing_focus else "ax"
            return True

        last_tree = str((last_state or {}).get("tree_markdown") or "")
        if not _chatgpt_web_composer_is_empty(last_tree, text_area):
            trace["failure"] = "composer_non_empty_mismatch"
            return False
        if preserve_existing_focus:
            trace["failure"] = "composer_stayed_empty"
            return False

        center = composer_center(last_state or {})
        if center is None:
            trace["failure"] = "composer_frame_missing"
            return False
        # Re-select the complete composer contents before retrying. If the AX
        # snapshots were stale after a successful first paste, this replaces
        # that text instead of duplicating it; if focus was false-positive, it
        # safely selects the still-empty composer before the retry.
        select_all_payload = {
            "pid": pid,
            "window_id": window_id,
            "keys": ["command", "a"],
            "x": center[0],
            "y": center[1],
            "delivery_mode": "foreground",
        }
        trace["stage"] = "pixel_focused_type"
        if not succeeded(cua.call("hotkey", select_all_payload)):
            trace["failure"] = "pixel_select_all_rejected"
            return False
        clear_payload = {
            "pid": pid,
            "window_id": window_id,
            "keys": ["backspace"],
            "delivery_mode": "foreground",
        }
        if not succeeded(cua.call("hotkey", clear_payload)):
            trace["failure"] = "pixel_clear_rejected"
            return False
        # Chromium drops foreground Cmd+V even after a proven pixel focus on
        # some profiles. cua-driver's pixel-addressed type_text uses direct
        # foreground key events and is independently verified by the exact AX
        # composer read below. Cmd+A + Backspace above makes this a replace,
        # not an append, if the first paste landed but AX snapshots were stale.
        typed_segment = False
        prompt_lines = prompt.replace("\r\n", "\n").split("\n")
        for line_index, line in enumerate(prompt_lines):
            if line:
                type_payload = {
                    "pid": pid,
                    "window_id": window_id,
                    "text": line,
                    "delay_ms": 0,
                    "delivery_mode": "foreground",
                }
                if not typed_segment:
                    type_payload.update({"x": center[0], "y": center[1]})
                if not succeeded(cua.call("type_text", type_payload)):
                    trace["failure"] = "pixel_type_rejected"
                    return False
                typed_segment = True
            if line_index < len(prompt_lines) - 1:
                line_break_payload = {
                    "pid": pid,
                    "window_id": window_id,
                    "keys": ["shift", "enter"],
                    "delivery_mode": "foreground",
                }
                if not succeeded(cua.call("hotkey", line_break_payload)):
                    trace["failure"] = "line_break_rejected"
                    return False
        delivered, last_state = verify_paste()
        if delivered:
            trace["stage"] = "verified"
            trace["focus_mode"] = "pixel"
            return True
        last_tree = str((last_state or {}).get("tree_markdown") or "")
        trace["failure"] = (
            "composer_stayed_empty"
            if _chatgpt_web_composer_is_empty(last_tree, text_area)
            else "composer_non_empty_mismatch"
        )
        return False
    except Exception as exc:
        trace["failure"] = "exception"
        trace["exception_type"] = exc.__class__.__name__
        trace["exception_message"] = str(exc)
        return False
    finally:
        if injected_clipboard and _read_text_clipboard() == prompt:
            _write_text_clipboard(previous_clipboard)


def _safari_execute_javascript(javascript: str) -> str | None:
    """Run JavaScript in Safari's front ChatGPT tab through Apple Events."""

    script = (
        'const safari=Application("Safari");'
        "const doc=safari.documents[0];"
        'if(!doc || !String(doc.url()).includes("chatgpt.com")) '
        'throw new Error("front Safari tab is not ChatGPT");'
        f"safari.doJavaScript({json.dumps(javascript)},{{in:doc}});"
    )
    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\n")


def _chatgpt_page_insert_prompt(
    cua: Any,
    pid: int,
    window_id: int,
    text_area: int,
    prompt: str,
    sleep: Callable[[float], None] = time.sleep,
    diagnostics: dict[str, Any] | None = None,
) -> bool:
    """Safari-only Apple Events fallback after verified native clipboard delivery fails."""

    trace = diagnostics if diagnostics is not None else {}
    try:
        trace["stage"] = "dom_probe"
        selector = '[contenteditable="true"][aria-label="Chat with ChatGPT"]'
        dom_value = _safari_execute_javascript(
            f"document.querySelector({json.dumps(selector)}).innerText"
        )
        if dom_value is None:
            trace["failure"] = "dom_probe_unavailable"
            return False
        if _chatgpt_web_composer_matches_prompt(dom_value, prompt):
            trace["stage"] = "verified"
            return True
        if dom_value:
            trace["failure"] = "composer_non_empty_mismatch"
            return False
        trace["stage"] = "page_focus"
        click_result = cua.call(
            "page",
            {
                "action": "click_element",
                "pid": pid,
                "window_id": window_id,
                "selector": selector,
            },
        )
        if isinstance(click_result, dict) and (
            click_result.get("error") or click_result.get("ok") is False
        ):
            trace["failure"] = "page_focus_rejected"
            return False
        trace["stage"] = "dom_insert"
        inserted = _safari_execute_javascript(
            f'document.execCommand("insertText", false, {json.dumps(prompt)})'
        )
        if inserted != "true":
            trace["failure"] = "dom_insert_rejected"
            return False
        for attempt in range(5):
            state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
            tree = str(state.get("tree_markdown") or "")
            value = _chatgpt_web_composer_value(tree, text_area)
            if _chatgpt_web_composer_matches_prompt(value, prompt):
                trace["stage"] = "verified"
                return True
            if not _chatgpt_web_composer_is_empty(tree, text_area):
                trace["failure"] = "composer_non_empty_mismatch"
                return False
            if attempt < 4:
                sleep(0.2)
        trace["failure"] = "composer_stayed_empty"
    except Exception as exc:
        trace["failure"] = "exception"
        trace["exception_type"] = exc.__class__.__name__
        trace["exception_message"] = str(exc)
        return False
    return False


def _find_chatgpt_web_option(tree: str, label: str) -> int | None:
    """Return the picker option whose visible label equals *label*."""
    wanted = label.casefold()
    for line in tree.splitlines():
        if not re.search(r"AX(?:Button|MenuItem|RadioButton)\b", line):
            continue
        if (_chatgpt_web_control_label(line) or "").casefold() != wanted:
            continue
        index = _element_index(line)
        if index is not None:
            return index
    return None


def _chatgpt_pick_choice(
    cua: Any,
    candidate: ChatGptBrowserCandidate,
    pid: int,
    window_id: int,
    tree: str,
    target: str,
    sleep: Callable[[float], None],
) -> str:
    """Open the composer picker and select *target*, returning a fresh tree.

    Accessibility presses land on a backgrounded window, so the browser is only
    raised when a press did not take effect. That keeps the user's foreground
    window and keyboard focus untouched in the normal case.
    """

    def press(element_index: int, settle: float) -> str:
        cua.call(
            "click",
            {"pid": pid, "window_id": window_id, "element_index": element_index, "action": "press"},
        )
        sleep(settle)
        state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
        return str(state.get("tree_markdown") or "")

    picker = _find_chatgpt_web_model_picker(tree)
    if picker is None:
        return tree
    open_tree = press(picker, 0.3)
    option = _find_chatgpt_web_option(open_tree, target)
    if option is None:
        # The menu did not open in the background — raise the window once and
        # retry before giving up, so selection still fails closed, not silently.
        _activate_chatgpt_browser(candidate, pid, window_id)
        picker = _find_chatgpt_web_model_picker(open_tree) or picker
        open_tree = press(picker, 0.3)
        option = _find_chatgpt_web_option(open_tree, target)
        if option is None:
            return open_tree
    return press(option, 0.5)


def _chatgpt_active_label(tree: str) -> str | None:
    picker_line = _chatgpt_web_model_picker_line(tree)
    if picker_line is None:
        return None
    return _chatgpt_web_control_label(picker_line)


def _chatgpt_verify_selection(
    cua: Any,
    candidate: ChatGptBrowserCandidate,
    pid: int,
    window_id: int,
    tree: str,
    req: dict[str, Any],
    sleep: Callable[[float], None],
    diagnostics: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """Select and confirm the requested model (and effort) before any typing.

    Returns ``(tree, failure)``.  No default model is invented: an explicitly
    requested model that the live picker cannot confirm fails closed with the
    redacted list of visible choices.  When no model is supplied by an
    internal caller the historical Pro requirement is preserved.
    """
    available = _chatgpt_model_choices(tree)
    model_choices = [label for label in available if label.casefold() not in _EFFORT_CHOICES]
    requested_model = str(req.get("model") or "").strip()
    target = _chatgpt_requested_label(requested_model, model_choices)
    explicit = bool(requested_model)
    if target is None and not explicit:
        target = "Pro"

    selection: dict[str, Any] = {
        "requested_model": requested_model or None,
        "available_choices": available,
        "target": target,
    }
    diagnostics["selection"] = selection

    if target is None:
        diagnostics.update({"stage": "model_selection"})
        return tree, {
            "ok": False,
            "error": "model_not_available",
            "message": (
                f"ChatGPT model {requested_model!r} is not offered by the visible "
                f"{candidate.name} picker"
            ),
            "diagnostics": diagnostics,
        }

    if (_chatgpt_active_label(tree) or "").casefold() != target.casefold():
        tree = _chatgpt_pick_choice(cua, candidate, pid, window_id, tree, target, sleep)
    confirmed = (_chatgpt_active_label(tree) or "").casefold() == target.casefold()
    selection["confirmed_model"] = _chatgpt_active_label(tree)

    diagnostics.update(
        {
            "stage": "model_detection",
            "model_detection": {
                "composer_found": _find_first_text_area(tree) is not None,
                "picker_found": _find_chatgpt_web_model_picker(tree) is not None,
                "pro_detected": _chatgpt_web_active_model_is_pro(tree),
            },
        }
    )
    if not confirmed:
        if explicit:
            return tree, {
                "ok": False,
                "error": "model_not_available",
                "message": (
                    f"Could not confirm ChatGPT model {requested_model!r} in {candidate.name}"
                ),
                "diagnostics": diagnostics,
            }
        return tree, {
            "ok": False,
            "error": "model_not_pro",
            "message": f"Could not confirm ChatGPT Pro in {candidate.name}",
            "diagnostics": diagnostics,
        }

    requested_effort = str(req.get("effort") or "").strip()
    if requested_effort:
        effort_choices = [label for label in available if label.casefold() in _EFFORT_CHOICES]
        selection["available_efforts"] = effort_choices
        if not effort_choices:
            # The published support matrix declares effort_support=False for
            # this provider: record that the control was absent instead of
            # failing a request the contract never promised to honour.
            selection["effort"] = "control_absent"
        else:
            effort_target = _chatgpt_requested_label(requested_effort, effort_choices)
            if effort_target is None:
                return tree, {
                    "ok": False,
                    "error": "effort_not_available",
                    "message": (
                        f"ChatGPT effort {requested_effort!r} is not offered by the visible "
                        f"{candidate.name} picker"
                    ),
                    "diagnostics": diagnostics,
                }
            tree = _chatgpt_pick_choice(cua, candidate, pid, window_id, tree, effort_target, sleep)
            active = _chatgpt_active_label(tree)
            selection["confirmed_effort"] = active
            if (active or "").casefold() != effort_target.casefold():
                return tree, {
                    "ok": False,
                    "error": "effort_not_available",
                    "message": (
                        f"Could not confirm ChatGPT effort {requested_effort!r} in {candidate.name}"
                    ),
                    "diagnostics": diagnostics,
                }
    return tree, None


def _chatgpt_upload_capability(
    cua: Any,
    candidate: ChatGptBrowserCandidate,
    pid: int,
    window_id: int,
    session_id: str,
) -> tuple[bool, str]:
    """Detect, read-only, whether this browser exposes a file-attachment route.

    A ChatGPT attachment can only be delivered through CDP
    (``browser_set_input_files``); there is no safe native file-dialog path.
    Safari exposes no attachable CDP endpoint at all, and the Chromium route
    requires an *owned* DevTools endpoint on an isolated profile, which by
    construction is not the user's signed-in ChatGPT profile. This probe never
    prepares or launches an endpoint — it only asks whether one already binds.
    """
    if candidate.key == "safari":
        return False, "webkit_no_cdp_endpoint"
    try:
        state = cua.call(
            "get_browser_state",
            {"pid": pid, "window_id": window_id, "session": session_id},
        )
    except Exception as exc:
        return False, f"browser_state_unavailable:{exc.__class__.__name__}"
    if isinstance(state, dict) and state.get("status") == "refused":
        refusal = state.get("refusal") if isinstance(state.get("refusal"), dict) else {}
        return False, str(refusal.get("code") or "browser_route_unavailable")
    return True, "cdp_bound"


class ChatGptCancelled(RuntimeError):
    """Raised inside a GUI turn once the durable job has been stopped."""


def _check_cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ChatGptCancelled("job_stopped")


def _chatgpt_request_stop(
    cua: Any,
    pid: int | None,
    window_id: int | None,
    sleep: Callable[[float], None],
) -> bool:
    """Click ChatGPT's visible Stop control.  True only when it was clicked."""
    if pid is None or window_id is None:
        return False
    try:
        state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
        tree = str(state.get("tree_markdown") or "")
        stop = _find_chatgpt_web_stop_button(tree)
        if stop is None:
            return False
        result = cua.call("click", {"pid": pid, "window_id": window_id, "element_index": stop})
        sleep(0.2)
        if isinstance(result, dict) and (result.get("ok") is False or result.get("error")):
            return False
        return True
    except Exception:
        return False


def _run_chatgpt_browser_candidate(
    candidate: ChatGptBrowserCandidate,
    req: dict[str, Any],
    cua: Any,
    sleep: Callable[[float], None],
    deadline: float,
    nonce: str,
    progress: Callable[[str, str, dict[str, Any]], None] | None = None,
    *,
    cancel: threading.Event | None = None,
    session: ChatGptSession | None = None,
    lifecycle: TurnLifecycle | None = None,
    artifacts: JobArtifactStore | None = None,
    context_text: str = "",
    attachments: list[dict[str, Any]] | None = None,
    stability_sec: float = 0.0,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"stage": "browser_app"}
    turn = lifecycle if lifecycle is not None else TurnLifecycle()

    def stage(name: str, **data: Any) -> None:
        event = turn.enter(name, **data)
        if event is None or progress is None:
            return
        try:
            progress("turn_stage", f"ChatGPT turn stage: {name}", {"stage": name, **data})
        except Exception as exc:
            diagnostics.setdefault("progress_reporting", []).append(
                {
                    "event_type": "turn_stage",
                    "exception_type": exc.__class__.__name__,
                    "exception_message": str(exc),
                }
            )

    def owned(pid: int, window_id: int) -> dict[str, Any] | None:
        """Fail closed when the live window is not the job's owned session."""
        if session is None:
            return None
        if not session.matches(pid, window_id):
            diagnostics.update(
                {
                    "stage": "session_mismatch",
                    "session": session.snapshot(),
                    "observed": {"pid": pid, "window_id": window_id},
                }
            )
            session.retire("window_mismatch")
            stage("failed", reason="session_mismatch")
            return {
                "ok": False,
                "error": "session_mismatch",
                "message": (
                    "The discovered ChatGPT window is not the window this job owns; "
                    "refusing to type or submit into it"
                ),
                "diagnostics": diagnostics,
            }
        ok, reason = verify_owned_window(session, _list_browser_windows(cua, pid))
        if not ok and reason == "window_list_unavailable":
            # The bound identity is still the only window this turn addresses;
            # a transient driver read failure is recorded, not fatal.
            diagnostics["session_verification"] = {"skipped": reason}
            return None
        if not ok:
            diagnostics.update(
                {"stage": "session_unverified", "session": session.snapshot(), "reason": reason}
            )
            session.retire(reason or "window_unverified")
            stage("failed", reason=reason or "window_unverified")
            return {
                "ok": False,
                "error": "session_window_unavailable",
                "message": f"The owned ChatGPT window could not be verified ({reason})",
                "diagnostics": diagnostics,
            }
        return None

    _opened_fresh = False
    _opened_pid: int | None = None
    _opened_window_id: int | None = None

    try:
        _check_cancelled(cancel)
        stage("bootstrap", browser=candidate.name)
        app = _chatgpt_browser_app(cua, candidate, sleep)
        pid = int(app["pid"])
        diagnostics.update({"stage": "browser_window", "pid": pid})
        # Always try to open a fresh temporary-chat window first. If that
        # fails (Safari, or the browser refused), fall back to discovering an
        # existing window.
        fresh = _chatgpt_open_fresh_window(cua, candidate, sleep)
        diagnostics["fresh_window"] = bool(fresh)
        window: dict[str, Any] = {}
        if fresh is not None:
            _opened_fresh = True
            _opened_pid = int(fresh["pid"])
            _opened_window_id = int(fresh["window_id"])
            pid = _opened_pid
            window_id = _opened_window_id
        else:
            window = _chatgpt_browser_window(cua, candidate, pid, sleep)
            pid = int(window.get("pid", pid))
            window_id = int(window["window_id"])
            _opened_fresh = False
        if session is not None:
            session.bind(pid, window_id)
            diagnostics["session"] = session.snapshot()
        diagnostics.update({"stage": "page_probe", "window_id": window_id})
        _check_cancelled(cancel)
        # When we opened a fresh window, it must expose the Temporary Chat
        # indicator. A pre-existing non-temporary window is never accepted.
        if _opened_fresh:
            probe_tree = _chatgpt_browser_snapshot(cua, candidate, pid, window_id, sleep)
            if not _verify_temporary_chat_indicator(probe_tree):
                stage("failed", reason="temporary_chat_not_confirmed")
                return {
                    "ok": False,
                    "error": "temporary_chat_not_confirmed",
                    "message": (
                        f"Opened a new ChatGPT window in {candidate.name}, but the "
                        "Temporary Chat indicator was not found"
                    ),
                    "diagnostics": diagnostics,
                }
        if "just a moment" in str(window.get("title") or "").casefold():
            return {
                "ok": False,
                "error": "browser_challenge_required",
                "message": f"Manual human verification is required in {candidate.name}",
            }
        tree = _chatgpt_browser_snapshot(cua, candidate, pid, window_id, sleep)
        screen_time_message = _chatgpt_screen_time_limit(tree)
        if screen_time_message is not None:
            diagnostics.update(
                {
                    "stage": "browser_time_limit",
                    "screen_time_message": screen_time_message,
                }
            )
            return {
                "ok": False,
                "error": "browser_time_limit",
                "message": f"macOS Screen Time blocked {candidate.name}",
                "diagnostics": diagnostics,
            }
        mismatch = owned(pid, window_id)
        if mismatch is not None:
            return mismatch
        # Safari's Apple Events page probe clears the web composer focus. Its
        # fresh AX tree already exposes the signed-in composer, so use that
        # instead until after delivery has completed.
        if candidate.key == "safari":
            if _chatgpt_web_signed_out(tree):
                return {
                    "ok": False,
                    "error": "authentication_required",
                    "message": f"ChatGPT sign-in required in {candidate.name}",
                }
        else:
            page_text = _chatgpt_page_text(cua, pid, window_id)
            if _chatgpt_web_signed_out(page_text):
                return {
                    "ok": False,
                    "error": "authentication_required",
                    "message": f"ChatGPT sign-in required in {candidate.name}",
                }
            tree = _chatgpt_browser_snapshot(cua, candidate, pid, window_id, sleep)
        screen_time_message = _chatgpt_screen_time_limit(tree)
        if screen_time_message is not None:
            diagnostics.update(
                {
                    "stage": "browser_time_limit",
                    "screen_time_message": screen_time_message,
                }
            )
            return {
                "ok": False,
                "error": "browser_time_limit",
                "message": f"macOS Screen Time blocked {candidate.name}",
                "diagnostics": diagnostics,
            }
        stage("authenticated", browser=candidate.name)
        _check_cancelled(cancel)
        tree, selection_failure = _chatgpt_verify_selection(
            cua, candidate, pid, window_id, tree, req, sleep, diagnostics
        )
        if selection_failure is not None:
            stage("failed", reason=str(selection_failure.get("error")))
            return selection_failure
        stage("model_selected", model=diagnostics.get("selection", {}).get("confirmed_model"))

        text_area = _find_first_text_area(tree)
        if text_area is None:
            stage("failed", reason="message_input_not_found")
            return {
                "ok": False,
                "error": "message_input_not_found",
                "message": f"Could not find ChatGPT input in {candidate.name}",
            }
        if not _chatgpt_web_composer_is_empty(tree, text_area):
            # A non-empty composer after the fresh-window-first flow means
            # we fell back to an existing window with a stray draft. Never
            # overwrite it.
            stage("failed", reason="composer_not_empty")
            return {
                "ok": False,
                "error": "composer_not_empty",
                "message": (f"Refusing to overwrite an existing ChatGPT draft in {candidate.name}"),
            }
        stage("composer_ready")
        if attachments:
            # Explicit attachments were requested. Fail closed before the prompt
            # is delivered rather than silently dropping the caller's files.
            supported, reason = _chatgpt_upload_capability(
                cua, candidate, pid, window_id, session.session_id if session else nonce
            )
            diagnostics["attachments"] = {
                "requested": len(attachments),
                "upload_supported": supported,
                "reason": reason,
            }
            stage("failed", reason="attachment_upload_unsupported")
            return {
                "ok": False,
                "error": "attachment_upload_unsupported",
                "message": (
                    f"{candidate.name} exposes no usable ChatGPT file-attachment route "
                    f"({reason}); the prompt was not submitted. Inline the file contents "
                    "through scope.paths instead."
                ),
                "diagnostics": diagnostics,
            }
        _check_cancelled(cancel)

        prompt, begin, end = _chatgpt_prompt(req, nonce, context_text=context_text)
        prompt_diagnostics: dict[str, Any] = {}
        diagnostics.update({"stage": "prompt_delivery", "prompt_delivery": prompt_diagnostics})
        delivered = _chatgpt_deliver_prompt(
            cua,
            pid,
            window_id,
            text_area,
            prompt,
            sleep,
            preserve_existing_focus=candidate.key == "safari",
            diagnostics=prompt_diagnostics,
        )
        if not delivered and candidate.key == "safari":
            safari_diagnostics: dict[str, Any] = {}
            diagnostics["safari_page_insert"] = safari_diagnostics
            delivered = _chatgpt_page_insert_prompt(
                cua,
                pid,
                window_id,
                text_area,
                prompt,
                sleep,
                diagnostics=safari_diagnostics,
            )
        if not delivered:
            # Never leave our own partial prompt in the user's composer: it
            # would block every later turn with composer_not_empty.
            diagnostics["residue_cleanup"] = _chatgpt_clear_composer(cua, pid, window_id, text_area)
            stage("failed", reason="prompt_insertion_failed")
            return {
                "ok": False,
                "error": "prompt_insertion_failed",
                "message": f"Could not safely deliver the ChatGPT prompt in {candidate.name}",
                "diagnostics": diagnostics,
            }
        submit_tree = _chatgpt_browser_snapshot(cua, candidate, pid, window_id, sleep)
        # Verify the exact expected prompt from a fresh snapshot immediately
        # before Send. A truncated or altered composer must never be submitted.
        observed = _chatgpt_web_composer_value(submit_tree, text_area)
        if not _chatgpt_web_composer_matches_prompt(observed, prompt):
            diagnostics["residue_cleanup"] = _chatgpt_clear_composer(cua, pid, window_id, text_area)
            diagnostics["prompt_verification"] = {
                "browser": candidate.name,
                "stage": "pre_submit_verification",
                **prompt_mismatch_diagnostics(prompt, observed),
            }
            stage("failed", reason="prompt_verification_failed")
            if artifacts is not None:
                artifacts.write(
                    "prompt-verification-tree.txt",
                    submit_tree,
                    stage="prompt_verified",
                    kind="ax_tree",
                    session=session.snapshot() if session else None,
                )
            return {
                "ok": False,
                "error": "prompt_verification_failed",
                "message": (
                    f"The {candidate.name} composer did not contain the exact prompt; "
                    "refusing to submit"
                ),
                "diagnostics": diagnostics,
            }
        stage("prompt_verified", expected_len=len(prompt))
        mismatch = owned(pid, window_id)
        if mismatch is not None:
            return mismatch
        _check_cancelled(cancel)
        send_button = _find_chatgpt_web_send_button(submit_tree)
        if send_button is None:
            stage("failed", reason="prompt_submit_failed")
            return {
                "ok": False,
                "error": "prompt_submit_failed",
                "message": f"Could not find a fresh ChatGPT Send button in {candidate.name}",
            }
        submit_result = cua.call(
            "click",
            {"pid": pid, "window_id": window_id, "element_index": send_button},
        )
        if isinstance(submit_result, dict) and (
            submit_result.get("ok") is False or submit_result.get("error")
        ):
            stage("failed", reason="prompt_submit_failed")
            return {
                "ok": False,
                "error": "prompt_submit_failed",
                "message": f"Could not submit the ChatGPT prompt in {candidate.name}",
            }
        submitted_at = time.monotonic()
        last_progress_at = submitted_at
        diagnostics.update(
            {
                "stage": "generation_in_progress",
                "prompt_submitted": True,
            }
        )

        def report_progress(event_type: str, message: str, data: dict[str, Any]) -> None:
            if progress is None:
                return
            try:
                progress(event_type, message, data)
            except Exception as exc:
                diagnostics.setdefault("progress_reporting", []).append(
                    {
                        "event_type": event_type,
                        "exception_type": exc.__class__.__name__,
                        "exception_message": str(exc),
                    }
                )

        stage("submitted", browser=candidate.name)
        _chatgpt_turn_generating.set()
        report_progress(
            "prompt_submitted",
            "ChatGPT prompt submitted",
            {"browser": candidate.name},
        )

        completion = CompletionTracker(stability_sec=stability_sec)
        health = DomHealthTracker()
        streaming_reported = False
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                stopped = _chatgpt_request_stop(cua, pid, window_id, sleep)
                diagnostics.update(
                    {
                        "stage": "cancelled",
                        "provider_stop_confirmed": stopped,
                        "elapsed_sec": int(time.monotonic() - submitted_at),
                    }
                )
                stage("cancelled", provider_stop_confirmed=stopped)
                if session is not None:
                    session.retire("cancelled_after_submit")
                return {
                    "ok": False,
                    "error": "cancelled",
                    "message": (
                        "ChatGPT generation was cancelled after submission"
                        + ("; the visible Stop action was clicked" if stopped else "")
                    ),
                    "provider_stop_confirmed": stopped,
                    "diagnostics": diagnostics,
                }
            sleep(min(2.0, max(0.0, deadline - time.monotonic())))
            now = time.monotonic()
            remaining_sec = deadline - now
            if remaining_sec <= 0:
                break
            heartbeat_remaining_sec = max(0.1, 30.0 - (now - last_progress_at))
            page_read_timeout_sec = min(25.0, remaining_sec, heartbeat_remaining_sec)
            try:
                page_text = _chatgpt_page_text(
                    cua,
                    pid,
                    window_id,
                    timeout_sec=page_read_timeout_sec,
                )
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if progress is not None and now - last_progress_at >= 30:
                    report_progress(
                        "generation_in_progress",
                        "ChatGPT is still generating",
                        {
                            "browser": candidate.name,
                            "elapsed_sec": int(now - submitted_at),
                            "stage": turn.stage,
                        },
                    )
                    last_progress_at = now
                continue
            try:
                poll_state = cua.call("get_window_state", {"pid": pid, "window_id": window_id})
                poll_tree = str(poll_state.get("tree_markdown") or "")
            except Exception:
                poll_tree = ""
            running = _chatgpt_response_is_running(poll_tree)
            if poll_tree:
                completion_action = _chatgpt_completion_action_visible(poll_tree)
            else:
                # The AX tree is momentarily unreadable. Rather than fail every
                # such turn, fall back to the correlated closed nonce marker
                # plus the stability window, and record the degradation.
                completion_action = True
                diagnostics["ax_unreadable_polls"] = (
                    int(diagnostics.get("ax_unreadable_polls", 0)) + 1
                )
            output = _extract_marked_response(page_text, begin, end)
            observation = ResponseObservation(
                at=time.monotonic(),
                response_present=bool(output) or running or begin in page_text,
                running=running,
                completion_action=completion_action,
                text=output,
            )
            if running and not streaming_reported:
                streaming_reported = True
                stage("streaming")
            final = completion.observe(observation)
            if final:
                stage("complete", chars=len(final))
                return {
                    "ok": True,
                    "output": final,
                    "browser": candidate.name,
                    "diagnostics": diagnostics,
                }
            dom_failure = health.observe(observation)
            if dom_failure is not None:
                diagnostics.update(
                    {
                        "stage": "dom_health_failed",
                        "dom_health": {"failure": dom_failure, **health.detail},
                        "completion_blocked_by": completion.reason,
                    }
                )
                if artifacts is not None:
                    artifacts.write(
                        f"dom-health-{dom_failure}.txt",
                        poll_tree,
                        stage="streaming",
                        kind="ax_tree",
                        session=session.snapshot() if session else None,
                    )
                stage("failed", reason=dom_failure)
                if session is not None:
                    session.retire(dom_failure)
                return {
                    "ok": False,
                    "error": "generation_status_unavailable",
                    "message": (
                        f"ChatGPT generation could not be read safely in {candidate.name} "
                        f"({dom_failure})"
                    ),
                    "diagnostics": diagnostics,
                }
            now = time.monotonic()
            if progress is not None and now - last_progress_at >= 30:
                report_progress(
                    "generation_in_progress",
                    "ChatGPT is still generating",
                    {
                        "browser": candidate.name,
                        "elapsed_sec": int(now - submitted_at),
                        "stage": turn.stage,
                    },
                )
                last_progress_at = now
        diagnostics["completion_blocked_by"] = completion.reason
        stage("failed", reason="generation_timed_out")
        return {
            "ok": False,
            "error": "generation_timed_out",
            "message": (
                f"ChatGPT prompt was submitted in {candidate.name}, but the "
                "response did not finish before the configured timeout"
            ),
            "diagnostics": diagnostics,
        }
    except ChatGptCancelled:
        diagnostics.update({"stage": "cancelled", "prompt_submitted": False})
        stage("cancelled", provider_stop_confirmed=False)
        if session is not None:
            session.retire("cancelled_before_submit")
        return {
            "ok": False,
            "error": "cancelled",
            "message": "ChatGPT turn was cancelled before the prompt was submitted",
            "provider_stop_confirmed": False,
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        diagnostics.update(
            {
                "failure": "exception",
                "exception_type": exc.__class__.__name__,
                "exception_message": str(exc),
            }
        )
        stage("failed", reason=exc.__class__.__name__)
        if session is not None:
            session.retire("exception")
        return {
            "ok": False,
            "error": exc.__class__.__name__,
            "message": str(exc),
            "diagnostics": diagnostics,
        }
    finally:
        _chatgpt_turn_generating.clear()
        if _opened_fresh and _opened_pid is not None and _opened_window_id is not None:
            try:
                _close_turn_window(cua, candidate, _opened_pid, _opened_window_id, sleep)
            except Exception:
                pass


def _chatgpt_stability_sec() -> float:
    """Configurable completion stability window (seconds)."""
    raw = getenv("AGENT_CROSSBAR_CHATGPT_STABILITY_SEC")
    try:
        return max(0.0, float(raw)) if raw else 0.0
    except (TypeError, ValueError):
        return 0.0


def run_gui_request(
    req: dict[str, Any],
    *,
    cua: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout_sec: int | None = None,
    progress: Callable[[str, str, dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
    job_id: str | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a ChatGPT Pro request through the browser fallback chain."""
    if req.get("profile") != "chatgpt_pro":
        return {
            "ok": False,
            "error": "provider_not_implemented",
            "message": f"No GUI runner for profile={req.get('profile')}",
            "attempts": [],
        }

    nonce = uuid.uuid4().hex
    owner = job_id or nonce

    # Bounded context preparation happens before any browser work so an
    # unsafe or missing explicit path can never reach a submitted prompt.
    packed = pack_context(req.get("cwd"), req.get("scope"))
    if not packed.get("ok"):
        return {
            "ok": False,
            "error": str(packed.get("error") or "invalid_scope"),
            "message": str(packed.get("message") or "context preparation failed"),
            "attempts": [],
            "artifacts": [],
            "nonce": nonce,
        }
    context_text = str(packed.get("text") or "")
    context_summary = packed.get("summary") or {}

    if cancel is not None and cancel.is_set():
        return {
            "ok": False,
            "error": "cancelled",
            "message": "ChatGPT Pro job was stopped before the browser was touched",
            "attempts": [],
            "artifacts": [],
            "nonce": nonce,
        }

    acquired, lock_path, lock_message = _acquire_chatgpt_lock()
    if not acquired:
        return _chatgpt_failure(
            "busy", lock_message or "chatgpt_pro GUI runner is busy", nonce=nonce
        )
    owns_lock = True

    timeout = timeout_sec or int(req.get("timeout_sec") or CHATGPT_PRO_DEFAULT_TIMEOUT_SEC)
    client = cua or CuaDriverClient(
        call_timeout_sec=max(_DEFAULT_CUA_CALL_TIMEOUT_SEC, min(float(timeout), 180.0))
    )
    deadline = time.monotonic() + timeout
    artifacts = (
        JobArtifactStore(artifact_dir)
        if artifact_dir is not None
        else JobArtifactStore(_state_root() / "artifacts" / "chatgpt_pro", subdir=nonce)
    )
    lifecycle = TurnLifecycle()
    stability_sec = _chatgpt_stability_sec()

    def finish(payload: dict[str, Any]) -> dict[str, Any]:
        existing = list(payload.get("artifacts") or [])
        for path in artifacts.paths():
            if path not in existing:
                existing.append(path)
        payload["artifacts"] = existing
        payload["lifecycle"] = lifecycle.stages()
        if context_summary.get("requested"):
            payload["context_summary"] = context_summary
        manifest = artifacts.manifest()
        if manifest:
            payload["artifact_manifest"] = manifest
        return payload

    try:
        attempts: list[dict[str, Any]] = []
        for candidate in _CHATGPT_BROWSER_CANDIDATES:
            if time.monotonic() >= deadline:
                break
            if cancel is not None and cancel.is_set():
                return finish(
                    {
                        "ok": False,
                        "error": "cancelled",
                        "message": "ChatGPT Pro job was stopped before submission",
                        "attempts": attempts,
                        "nonce": nonce,
                    }
                )
            if _chatgpt_turn_generating.is_set():
                return finish(
                    {
                        "ok": False,
                        "error": "turn_generating",
                        "message": (
                            "Another ChatGPT turn is still generating; "
                            "refusing to launch a fallback browser"
                        ),
                        "attempts": attempts,
                        "nonce": nonce,
                    }
                )
            session, busy_reason = sessions.acquire(owner, candidate.key, candidate.name)
            if session is None:
                return finish(
                    {
                        "ok": False,
                        "error": "busy",
                        "message": (
                            "Another chatgpt_pro GUI turn already owns the browser session"
                        ),
                        "attempts": attempts,
                        "busy_reason": busy_reason,
                        "nonce": nonce,
                    }
                )
            result = _run_chatgpt_browser_candidate(
                candidate,
                req,
                client,
                sleep,
                deadline,
                nonce,
                progress,
                cancel=cancel,
                session=session,
                lifecycle=lifecycle,
                artifacts=artifacts,
                context_text=context_text,
                attachments=packed.get("attachments") or [],
                stability_sec=stability_sec,
            )
            attempt = {
                "browser": candidate.name,
                "bundle_id": candidate.bundle_id,
                "ok": bool(result.get("ok")),
            }
            if result.get("ok"):
                attempt["candidate"] = f"ChatGPT Pro web via {candidate.name}"
                attempts.append(attempt)
                sessions.retire(owner, "turn_complete")
                return finish(
                    {
                        "ok": True,
                        "output": result["output"],
                        "selected_candidate": attempt["candidate"],
                        "provider_exit_code": 0,
                        "attempts": attempts,
                        "nonce": nonce,
                    }
                )
            sessions.retire(owner, str(result.get("error") or "turn_failed"))
            if result.get("error") == "cancelled":
                attempt["error"] = "cancelled"
                attempt["message"] = str(result.get("message") or "cancelled")
                attempts.append(attempt)
                return finish(
                    {
                        "ok": False,
                        "error": "cancelled",
                        "message": attempt["message"],
                        "provider_stop_confirmed": bool(result.get("provider_stop_confirmed")),
                        "attempts": attempts,
                        "nonce": nonce,
                    }
                )
            attempt["error"] = str(result.get("error") or "browser_failed")
            attempt["message"] = str(result.get("message") or f"{candidate.name} failed")
            if isinstance(result.get("diagnostics"), dict):
                attempt["diagnostics"] = result["diagnostics"]
            attempts.append(attempt)
            prompt_was_submitted = bool(
                isinstance(result.get("diagnostics"), dict)
                and result["diagnostics"].get("prompt_submitted") is True
            )
            if result.get("error") == "generation_timed_out":
                return finish(
                    {
                        "ok": False,
                        "error": "generation_timed_out",
                        "message": attempt["message"],
                        "attempts": attempts,
                        "artifacts": result.get("artifacts", []),
                        "nonce": nonce,
                    }
                )
            if prompt_was_submitted:
                # Fallback is forbidden once a prompt is durably submitted:
                # another candidate would duplicate the same request.
                return finish(
                    {
                        "ok": False,
                        "error": "generation_status_unavailable",
                        "message": (
                            "ChatGPT prompt was submitted, but its generation status "
                            "could not be read safely"
                        ),
                        "cause_error": attempt["error"],
                        "attempts": attempts,
                        "artifacts": result.get("artifacts", []),
                        "nonce": nonce,
                    }
                )
        if len(attempts) == len(_CHATGPT_BROWSER_CANDIDATES) and all(
            attempt.get("error") == "browser_time_limit" for attempt in attempts
        ):
            return finish(
                {
                    "ok": False,
                    "error": "browser_time_limit",
                    "message": "macOS Screen Time blocked every ChatGPT browser",
                    "attempts": attempts,
                    "artifacts": [],
                    "nonce": nonce,
                }
            )
        return finish(
            {
                "ok": False,
                "error": "browser_fallback_exhausted",
                "message": "ChatGPT Pro failed in Helium, Chrome, and Safari",
                "attempts": attempts,
                "artifacts": [],
                "nonce": nonce,
            }
        )
    except Exception as exc:
        return _chatgpt_failure(
            exc.__class__.__name__,
            str(exc),
            nonce=nonce,
        )
    finally:
        sessions.retire(owner, "runner_exit")
        if owns_lock:
            _release_chatgpt_lock(lock_path)


def _candidates(req: dict[str, Any], prompt: str) -> list[CommandCandidate]:
    """Return executable candidates for a validated print request."""
    profile = req["profile"]
    operation = req["operation"]
    transport = req.get("transport", "print")

    if profile == "reasonix":
        model = str(req["model"])
        if operation == "dev":
            effort = "high"
            reasonix_prompt = reasonix_dev_prompt(prompt)
            reasonix_shell_args = ["--mcp", reasonix_shell_mcp_spec()]
            reasonix_env = reasonix_shell_env(req.get("cwd"))
        else:
            effort = req.get("effort") or ("medium" if operation == "text" else "high")
            reasonix_prompt = prompt
            reasonix_shell_args = []
            reasonix_env = {}
        return [
            CommandCandidate(
                name=f"reasonix run {model} --mcp shell"
                if operation == "dev"
                else f"reasonix {model}",
                argv=[
                    "reasonix",
                    "run",
                    "-m",
                    model,
                    "--effort",
                    effort,
                    *reasonix_shell_args,
                    reasonix_prompt,
                ],
                env=reasonix_env,
            )
        ]

    if profile == "codex":
        model = str(req["model"])
        effort = str(req.get("effort") or "medium")
        argv = [
            "codex",
            "exec",
            "--ephemeral",
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{effort}"',
        ]
        if operation == "review" and transport == "print":
            argv += ["--sandbox", "read-only"]
        cwd = req.get("cwd")
        if cwd:
            argv += ["-C", str(cwd)]
        argv.append(prompt)
        return [CommandCandidate(name="codex exec", argv=argv)]

    if profile == "opencode":
        model_id = _opencode_model_id(str(req.get("model") or ""))
        argv = ["opencode", "run", "-m", model_id]
        if operation == "dev":
            argv.append("--dangerously-skip-permissions")
            if req.get("cwd"):
                argv += ["--dir", str(req["cwd"])]
            argv.append(prompt)
            return [CommandCandidate(name=f"opencode run {model_id}", argv=argv)]
        return [CommandCandidate(name=f"opencode run {model_id}", argv=argv, stdin_text=prompt)]

    return []


def run_print_request(
    req: dict[str, Any],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Run a print-mode request synchronously and return normalized output."""
    prompt = _prepare_prompt(req)
    candidates = _candidates(req, prompt)
    if not candidates:
        return {
            "ok": False,
            "error": "provider_not_implemented",
            "message": f"No print runner for profile={req.get('profile')}",
            "attempts": [],
        }

    cwd = _execution_cwd(req)
    timeout = timeout_sec or int(req.get("timeout_sec") or _DEFAULT_TIMEOUT_SEC)
    attempts: list[dict[str, Any]] = []

    for candidate in candidates:
        env = _provider_env(candidate.env)
        try:
            run_kwargs: dict[str, Any] = {
                "cwd": cwd,
                "env": env,
                "capture_output": True,
                "text": True,
                "timeout": timeout,
            }
            if candidate.stdin_text is None:
                run_kwargs["stdin"] = subprocess.DEVNULL
            else:
                run_kwargs["input"] = candidate.stdin_text
            result = run(
                candidate.argv,
                **run_kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            attempts.append(
                {
                    "candidate": candidate.name,
                    "ok": False,
                    "error": "timed_out",
                    "message": str(exc),
                }
            )
            continue
        except OSError as exc:
            attempts.append(
                {
                    "candidate": candidate.name,
                    "ok": False,
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            continue

        output = result.stdout if result.stdout else result.stderr
        attempts.append(
            {
                "candidate": candidate.name,
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "stderr": result.stderr[-4000:] if result.stderr else "",
            }
        )
        if result.returncode == 0:
            return {
                "ok": True,
                "output": output,
                "selected_candidate": candidate.name,
                "provider_exit_code": result.returncode,
                "attempts": attempts,
            }

    return {
        "ok": False,
        "error": "all_candidates_failed",
        "message": "All provider candidates failed",
        "attempts": attempts,
    }


def run_print_job(
    store: JobStore,
    job_id: str,
    req: dict[str, Any],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Run a print job and persist events/result.

    Stdout is written incrementally to stdout.log inside the job
    directory so that JobStore.job_tail can serve live
    output_tail / output_since_bytes while the provider is
    still running.
    """
    job = store.get_job(job_id)
    output_path = (job.path / "stdout.log") if job is not None else None

    # Expose the output path in metadata before starting so job_tail
    # can find it immediately (provider-neutral — no adapter fields).
    if output_path is not None:
        store.update_job_meta(job_id, {"print_output_path": str(output_path)})

    store.send_event(
        job_id,
        level="info",
        type="provider_started",
        message="Provider execution started",
        data={"profile": req.get("profile"), "operation": req.get("operation")},
    )

    # Build a run wrapper that streams stdout to the file while the
    # process runs so job_tail sees incremental output.  We still need
    # the final stdout text for result persistence, so read the file
    # back after the subprocess exits.
    if output_path is not None and run is subprocess.run:

        def _file_streaming_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            # Drop capture_output / text — we handle those ourselves.
            kwargs.pop("capture_output", None)
            kwargs.pop("text", None)
            kwargs.setdefault("stdin", subprocess.DEVNULL)
            with open(output_path, "wb") as out_f:
                timeout = kwargs.pop("timeout", None)
                process = subprocess.Popen(
                    argv,
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    **kwargs,
                )
                store.update_job_meta(
                    job_id,
                    {
                        "print_pid": process.pid,
                        "print_pgid": process.pid,
                    },
                )
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise
                finally:
                    store.update_job_meta(job_id, {"print_pid": None, "print_pgid": None})
            # Read back the captured text for the caller.
            try:
                stdout_text = output_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                stdout_text = ""
            return SimpleNamespace(
                returncode=returncode,
                stdout=stdout_text,
                stderr="",
            )

        effective_run = _file_streaming_run
    else:
        effective_run = run

    result = run_print_request(req, run=effective_run, timeout_sec=timeout_sec)
    if result.get("output"):
        store.send_event(
            job_id,
            level="info",
            type="stdout",
            message=result["output"],
            data={"selected_candidate": result.get("selected_candidate")},
        )

    if result["ok"]:
        store.set_result(
            job_id,
            ok=True,
            summary=result.get("output", ""),
            artifacts=[],
        )
        return result

    message = result.get("message") or result.get("error") or "Provider failed"
    store.send_event(
        job_id,
        level="error",
        type="error",
        message=message,
        data={"attempts": result.get("attempts", [])},
    )
    store.set_result(job_id, ok=False, summary=message, artifacts=[])
    return result


def _tmux_session_name(job_id: str) -> str:
    """Return a deterministic harness-owned tmux session name."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", job_id).strip("-")
    return f"agents-{safe}"


def _tmux_prompt_buffer_name(session_name: str) -> str:
    """Return a job-scoped tmux buffer name safe for concurrent prompt delivery."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", session_name).strip("-")
    return f"{safe}-prompt"


def _tmux_candidates(
    req: dict[str, Any], job_dir: Path
) -> tuple[list[CommandCandidate], str | None]:
    """Build executable tmux candidates from the provider launch plan."""
    plan = build_launch_plan(
        profile=str(req.get("profile") or ""),
        operation=str(req.get("operation") or ""),
        transport=str(req.get("transport") or "tmux"),
        prompt=_prepare_prompt(req),
        model=req.get("model"),
        effort=req.get("effort"),
        cwd=req.get("cwd"),
        job_dir=str(job_dir),
    )
    if plan.error:
        return [], plan.message or plan.error

    candidates: list[CommandCandidate] = []
    for candidate in plan.candidates:
        argv = list(candidate.args)
        if not argv:
            continue
        clean_env = False
        if argv[0] == "claude":
            argv[0] = _wrapper_bin("CLAUDE_BIN", ".bin/claude", "claude")
        if argv[0] == "opencode":
            argv[0] = _path_bin("OPENCODE_BIN", "opencode")
        candidates.append(
            CommandCandidate(
                name=candidate.name,
                argv=argv,
                env=dict(candidate.env),
                send_prompt=candidate.send_prompt,
                redirect_output=candidate.redirect_output,
                prompt_delay_sec=candidate.prompt_delay_sec,
                prompt_ready_patterns=tuple(candidate.prompt_ready_patterns),
                prompt_submit_delay_sec=candidate.prompt_submit_delay_sec,
                prompt_text=candidate.prompt_text,
                clean_env=clean_env,
            )
        )
    return candidates, None


def _tmux_shell_command(
    candidate: CommandCandidate,
    exit_status_path: Path,
    output_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Wrap provider command so tmux writes an exit status when it ends."""
    status_path = shlex.quote(str(exit_status_path))
    if candidate.clean_env:
        source_env = {**os.environ, **(env or {})}
        clean_env = {
            key: ("dumb" if key == "TERM" else value)
            for key in sorted(_TMUX_CLEAN_ENV_KEYS)
            if (value := source_env.get(key))
        }
        clean_env.setdefault("TERM", "dumb")
        command = shlex.join(
            [
                "env",
                "-i",
                *(f"{key}={value}" for key, value in clean_env.items()),
                *candidate.argv,
            ]
        )
    else:
        command = shlex.join(candidate.argv)
    if candidate.redirect_output and not candidate.send_prompt and output_path is not None:
        command = f"{command} > {shlex.quote(str(output_path))} 2>&1 < /dev/null"
    script = (
        f"{command}; "
        "__agents_status=$?; "
        f"printf '%s\\n' \"$__agents_status\" > {status_path}; "
        "exit $__agents_status"
    )
    # tmux evaluates its shell-command through the user's default shell.  Make
    # that shell replace itself so a completed provider cannot fall back to an
    # idle interactive fish pane and leave the job running without a status.
    return shlex.join(["exec", "/bin/sh", "-lc", script])


def _tmux_inherited_env() -> dict[str, str]:
    """Return provider/proxy env that tmux panes need from this MCP process."""
    source = _provider_env({})
    return {
        key: value for key, value in source.items() if key in _TMUX_INHERITED_ENV_KEYS and value
    }


def _completed_returncode(result: Any) -> int:
    return int(getattr(result, "returncode", 1))


def _completed_stderr(result: Any) -> str:
    return str(getattr(result, "stderr", "") or "")


def _completed_stdout(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "")


def _wait_for_tmux_target(
    target: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    sleep: Callable[[float], None],
    attempts: int = 20,
    interval_sec: float = 0.05,
) -> dict[str, Any]:
    """Wait briefly until tmux can address the target pane."""
    last_returncode = 1
    last_stderr = ""
    for attempt in range(1, attempts + 1):
        try:
            result = run(
                ["tmux", "list-panes", "-t", target, "-F", "#{pane_id}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "attempts": attempt, "error": str(exc)}

        last_returncode = _completed_returncode(result)
        last_stderr = _completed_stderr(result)[-1000:]
        if last_returncode == 0:
            return {"ok": True, "attempts": attempt}
        if attempt < attempts:
            sleep(interval_sec)

    return {
        "ok": False,
        "attempts": attempts,
        "returncode": last_returncode,
        "stderr": last_stderr,
    }


def _wait_for_tmux_prompt_ready(
    target: str,
    patterns: tuple[str, ...],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    sleep: Callable[[float], None],
    attempts: int = 30,
    interval_sec: float = 0.5,
) -> dict[str, Any]:
    """Wait until an interactive tmux pane shows a prompt-ready marker."""
    if not patterns:
        return {"ok": True, "skipped": True, "attempts": 0}

    last_returncode = 1
    last_stderr = ""
    last_excerpt = ""
    for attempt in range(1, attempts + 1):
        try:
            result = run(
                ["tmux", "capture-pane", "-p", "-t", target, "-S", "-80"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "attempts": attempt, "error": str(exc)}

        last_returncode = _completed_returncode(result)
        last_stderr = _completed_stderr(result)[-1000:]
        captured = _completed_stdout(result)
        last_excerpt = captured[-1000:]
        if last_returncode == 0:
            for pattern in patterns:
                if pattern in captured:
                    return {
                        "ok": True,
                        "attempts": attempt,
                        "matched_pattern": pattern,
                    }
        if attempt < attempts:
            sleep(interval_sec)

    return {
        "ok": False,
        "attempts": attempts,
        "returncode": last_returncode,
        "stderr": last_stderr,
        "last_excerpt": last_excerpt,
        "patterns": list(patterns),
    }


def _read_text_file(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return ""


def _monitor_tmux_job(
    store: JobStore,
    job_id: str,
    *,
    session_name: str,
    output_path: Path,
    transcript_path: Path,
    exit_status_path: Path,
    run: Callable[..., subprocess.CompletedProcess[str]],
    sleep: Callable[[float], None],
    poll_interval_sec: float,
    bootstrap_diagnostics: dict[str, Any] | None = None,
    timeout_sec: int | None = None,
    complete_on_output: bool = False,
    baseline_output_bytes: int = 0,
    profile: str | None = None,
) -> dict[str, Any]:
    """Wait for the tmux session to end, then persist a result."""
    deadline = time.monotonic() + timeout_sec if timeout_sec else None
    consecutive_complete_observations = 0
    required_complete_observations = 10 if (profile or "").casefold() == "codex" else 1
    while True:
        job = store.get_job(job_id)
        if job is not None and store._read_job_meta(job.path).get("status") == "stopped":
            cleanup = run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            message = "Tmux monitor stopped after job_stop"
            store.send_event(
                job_id,
                level="info",
                type="tmux_exited",
                message=message,
                data={
                    "tmux_session": session_name,
                    "stopped": True,
                    "tmux_cleanup": "killed"
                    if _completed_returncode(cleanup) == 0
                    else "kill_failed",
                },
            )
            return {"ok": False, "error": "job_stopped", "message": message}
        try:
            alive = run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            message = f"Failed to inspect tmux session: {exc}"
            store.send_event(
                job_id,
                level="error",
                type="error",
                message=message,
                data={"tmux_session": session_name},
            )
            store.set_result(job_id, ok=False, summary=message, artifacts=[])
            return {"ok": False, "error": "tmux_inspect_failed", "message": message}
        if _completed_returncode(alive) != 0:
            break
        if complete_on_output:
            output = _read_text_file(output_path)
            if interactive_tmux_session_resumed(output, profile=profile):
                artifacts = [str(path) for path in (transcript_path, output_path) if path.exists()]
                message = "Reasonix resumed an existing session; tmux dev jobs must start isolated sessions"
                store.send_event(
                    job_id,
                    level="error",
                    type="tmux_session_resumed",
                    message=message,
                    data={
                        "tmux_session": session_name,
                        "tmux_output_path": str(output_path),
                    },
                )
                store.set_result(job_id, ok=False, summary=message, artifacts=artifacts)
                cleanup = run(
                    ["tmux", "kill-session", "-t", session_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return {
                    "ok": False,
                    "error": "session_resumed",
                    "message": message,
                    "tmux_session": session_name,
                    "tmux_cleanup": "killed"
                    if _completed_returncode(cleanup) == 0
                    else "kill_failed",
                }
            output_complete = interactive_tmux_output_complete(
                output,
                baseline_bytes=baseline_output_bytes,
                profile=profile,
            )
            consecutive_complete_observations = (
                consecutive_complete_observations + 1 if output_complete else 0
            )
            if consecutive_complete_observations >= required_complete_observations:
                artifacts = [str(path) for path in (transcript_path, output_path) if path.exists()]
                summary = interactive_tmux_output_summary(output, profile=profile)
                store.send_event(
                    job_id,
                    level="info",
                    type="tmux_output_complete",
                    message="Interactive tmux output completed",
                    data={
                        "tmux_session": session_name,
                        "tmux_output_path": str(output_path),
                        "baseline_output_bytes": baseline_output_bytes,
                    },
                )
                store.set_result(job_id, ok=True, summary=summary, artifacts=artifacts)
                cleanup = run(
                    ["tmux", "kill-session", "-t", session_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                cleanup_ok = _completed_returncode(cleanup) == 0
                store.send_event(
                    job_id,
                    level="info" if cleanup_ok else "warn",
                    type="tmux_cleanup",
                    message=(
                        "Tmux session stopped after output completion"
                        if cleanup_ok
                        else "Tmux session cleanup after output completion failed"
                    ),
                    data={
                        "tmux_session": session_name,
                        "returncode": _completed_returncode(cleanup),
                        "stderr": _completed_stderr(cleanup)[-1000:],
                    },
                )
                return {
                    "ok": True,
                    "completion_reason": "interactive_output_complete",
                    "output": output,
                    "tmux_session": session_name,
                    "tmux_cleanup": "killed" if cleanup_ok else "kill_failed",
                }
        if deadline is not None and time.monotonic() >= deadline:
            run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            message = f"Timed out waiting for tmux session after {timeout_sec} seconds"
            store.send_event(
                job_id,
                level="error",
                type="error",
                message=message,
                data={"tmux_session": session_name, **(bootstrap_diagnostics or {})},
            )
            artifacts = [str(path) for path in (transcript_path, output_path) if path.exists()]
            store.set_result(job_id, ok=False, summary=message, artifacts=artifacts)
            return {"ok": False, "error": "timed_out", "message": message}
        sleep(poll_interval_sec)

    job = store.get_job(job_id)
    if job is not None:
        meta = store._read_job_meta(job.path)
        if meta.get("status") == "stopped":
            message = "Tmux session ended after job_stop"
            store.send_event(
                job_id,
                level="info",
                type="tmux_exited",
                message=message,
                data={"tmux_session": session_name, "stopped": True},
            )
            return {"ok": False, "error": "job_stopped", "message": message}

    exit_code: int | None = None
    try:
        if exit_status_path.exists():
            exit_code = int(exit_status_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        exit_code = None

    output = ""
    try:
        if output_path.exists():
            output = output_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""

    artifacts = [str(path) for path in (transcript_path, output_path) if path.exists()]
    if output:
        store.send_event(
            job_id,
            level="info",
            type="stdout",
            message=output[-4000:],
            data={"output_path": str(output_path)},
        )

    if exit_code is None:
        diagnostics = {
            "tmux_session": session_name,
            "tmux_output_path": str(output_path),
            "tmux_exit_status_path": str(exit_status_path),
            "output_path": str(output_path),
            **(bootstrap_diagnostics or {}),
        }
        prompt_delivery_failed = (
            diagnostics.get("pipe_pane_ok") is False or diagnostics.get("prompt_sent") is False
        )
        message = (
            "Tmux session ended before prompt delivery and without an exit status"
            if prompt_delivery_failed
            else "Tmux session ended without an exit status"
        )
        store.send_event(
            job_id,
            level="error",
            type="error",
            message=message,
            data=diagnostics,
        )
        store.set_result(job_id, ok=False, summary=message, artifacts=artifacts)
        return {"ok": False, "error": "missing_exit_status", "message": message}

    ok = exit_code == 0
    message = f"Tmux session exited with code {exit_code}"
    store.send_event(
        job_id,
        level="info" if ok else "error",
        type="tmux_exited",
        message=message,
        data={"tmux_session": session_name, "exit_code": exit_code},
    )
    summary = output[-4000:] if output else message
    store.set_result(job_id, ok=ok, summary=summary, artifacts=artifacts)
    return {"ok": ok, "exit_code": exit_code, "output": output}


def run_tmux_job(
    store: JobStore,
    job_id: str,
    req: dict[str, Any],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monitor: bool = True,
    poll_interval_sec: float = 1.0,
    timeout_sec: int | None = None,
    complete_on_output: bool = False,
) -> dict[str, Any]:
    """Start a provider-native dev job in a detached tmux session."""
    job = store.get_job(job_id)
    if job is None:
        return {"ok": False, "error": "job_not_found", "job_id": job_id}
    if store._read_job_meta(job.path).get("status") == "stopped":
        message = "Tmux provider execution skipped because job_stop was already requested"
        store.send_event(
            job_id,
            level="info",
            type="tmux_start_skipped",
            message=message,
            data={"tmux_session": _tmux_session_name(job_id), "stopped": True},
        )
        return {"ok": False, "error": "job_stopped", "message": message}

    candidates, plan_error = _tmux_candidates(req, job.path)
    if not candidates:
        message = plan_error or f"No tmux runner for profile={req.get('profile')}"
        store.send_event(
            job_id,
            level="error",
            type="error",
            message=message,
            data={"profile": req.get("profile"), "operation": req.get("operation")},
        )
        store.set_result(job_id, ok=False, summary=message, artifacts=[])
        return {"ok": False, "error": "provider_not_implemented", "message": message}

    session_name = _tmux_session_name(job_id)
    target = session_name
    transcript_path = job.path / "transcript.jsonl"
    output_path = job.path / "tmux-output.log"
    exit_status_path = job.path / "tmux-exit-status.txt"

    store.send_event(
        job_id,
        level="info",
        type="provider_started",
        message="Tmux provider execution started",
        data={"profile": req.get("profile"), "operation": req.get("operation")},
    )

    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        args = ["tmux", "new-session", "-d", "-s", session_name]
        if req.get("cwd"):
            args += ["-c", str(req["cwd"])]
        tmux_env = {**_tmux_inherited_env(), **candidate.env}
        shell_command = _tmux_shell_command(candidate, exit_status_path, output_path, env=tmux_env)
        for key, value in tmux_env.items():
            args += ["-e", f"{key}={value}"]
        args.append(shell_command)

        try:
            result = run(
                args,
                env={**os.environ, **tmux_env},
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            attempts.append({"candidate": candidate.name, "ok": False, "error": str(exc)})
            continue

        attempts.append(
            {
                "candidate": candidate.name,
                "ok": _completed_returncode(result) == 0,
                "exit_code": _completed_returncode(result),
                "stderr": _completed_stderr(result)[-1000:],
            }
        )
        if _completed_returncode(result) != 0:
            continue

        # ``job_stop`` may run after the worker starts but before tmux has
        # created its session.  Re-check immediately after creation so a
        # session cannot outlive a job that has already been stopped.
        current_job = store.get_job(job_id)
        if (
            current_job is not None
            and store._read_job_meta(current_job.path).get("status") == "stopped"
        ):
            cleanup = run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            message = "Tmux session removed because job_stop was requested during startup"
            store.send_event(
                job_id,
                level="info",
                type="tmux_start_cancelled",
                message=message,
                data={
                    "tmux_session": session_name,
                    "tmux_cleanup": "killed"
                    if _completed_returncode(cleanup) == 0
                    else "kill_failed",
                },
            )
            return {"ok": False, "error": "job_stopped", "message": message}

        target_ready = _wait_for_tmux_target(target, run=run, sleep=sleep)
        pipe_result = run(
            [
                "tmux",
                "pipe-pane",
                "-t",
                target,
                "-o",
                f"cat >> {shlex.quote(str(output_path))}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pipe_pane_ok = _completed_returncode(pipe_result) == 0
        prompt_sent = False
        prompt_ready_result: dict[str, Any] | None = None
        send_literal_result: Any | None = None
        load_buffer_result: Any | None = None
        paste_buffer_result: Any | None = None
        delete_buffer_result: Any | None = None
        send_enter_result: Any | None = None
        prompt_delivery_method: str | None = None
        prompt_buffer_name: str | None = None
        prompt_submit_delay_applied_sec = candidate.prompt_submit_delay_sec
        prompt_text = (
            candidate.prompt_text
            if candidate.prompt_text is not None
            else str(req.get("prompt") or "")
        )
        if (
            candidate.send_prompt
            and str(req.get("prompt") or "").strip()
            and target_ready.get("ok") is True
            and pipe_pane_ok
        ):
            if candidate.prompt_delay_sec > 0:
                sleep(candidate.prompt_delay_sec)
            prompt_ready_result = _wait_for_tmux_prompt_ready(
                target,
                candidate.prompt_ready_patterns,
                run=run,
                sleep=sleep,
            )
            if prompt_ready_result.get("ok") is True:
                if len(prompt_text.encode("utf-8")) > _TMUX_LITERAL_PROMPT_MAX_BYTES:
                    prompt_delivery_method = "paste_buffer"
                    prompt_buffer_name = _tmux_prompt_buffer_name(session_name)
                    try:
                        load_buffer_result = run(
                            [
                                "tmux",
                                "load-buffer",
                                "-b",
                                prompt_buffer_name,
                                "-",
                            ],
                            input=prompt_text,
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        load_buffer_result = subprocess.CompletedProcess([], 1, "", str(exc))
                    if _completed_returncode(load_buffer_result) == 0:
                        try:
                            paste_buffer_result = run(
                                [
                                    "tmux",
                                    "paste-buffer",
                                    "-b",
                                    prompt_buffer_name,
                                    "-t",
                                    target,
                                    "-d",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                        except (OSError, subprocess.SubprocessError) as exc:
                            paste_buffer_result = subprocess.CompletedProcess([], 1, "", str(exc))
                    delivery_ok = (
                        _completed_returncode(load_buffer_result) == 0
                        and paste_buffer_result is not None
                        and _completed_returncode(paste_buffer_result) == 0
                    )
                    if not delivery_ok:
                        try:
                            delete_buffer_result = run(
                                [
                                    "tmux",
                                    "delete-buffer",
                                    "-b",
                                    prompt_buffer_name,
                                ],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                        except (OSError, subprocess.SubprocessError) as exc:
                            delete_buffer_result = subprocess.CompletedProcess([], 1, "", str(exc))
                    prompt_submit_delay_applied_sec = max(
                        prompt_submit_delay_applied_sec,
                        _TMUX_PASTE_SETTLE_SEC,
                    )
                else:
                    prompt_delivery_method = "send_keys_literal"
                    send_literal_result = run(
                        ["tmux", "send-keys", "-t", target, "-l", prompt_text],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    delivery_ok = _completed_returncode(send_literal_result) == 0
                if delivery_ok and prompt_submit_delay_applied_sec > 0:
                    sleep(prompt_submit_delay_applied_sec)
                if delivery_ok:
                    send_enter_result = run(
                        ["tmux", "send-keys", "-t", target, "C-m"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                prompt_sent = (
                    delivery_ok
                    and send_enter_result is not None
                    and _completed_returncode(send_enter_result) == 0
                )

        metadata = {
            "tmux_session": session_name,
            "tmux_target": target,
            "tmux_transcript_path": str(transcript_path),
            "tmux_output_path": str(output_path),
            "tmux_exit_status_path": str(exit_status_path),
            "selected_candidate": candidate.name,
            "candidate_argv": candidate.argv,
            "interactive": candidate.redirect_output is False or candidate.send_prompt is True,
            "tmux_target_ready": target_ready.get("ok") is True,
            "tmux_target_ready_attempts": target_ready.get("attempts"),
            "pipe_pane_ok": pipe_pane_ok,
            "prompt_sent": prompt_sent,
            "prompt_delivery_method": prompt_delivery_method,
            "prompt_buffer_name": prompt_buffer_name,
            "prompt_bytes": len(prompt_text.encode("utf-8"))
            if candidate.send_prompt and str(req.get("prompt") or "").strip()
            else 0,
            "prompt_delay_sec": candidate.prompt_delay_sec,
            "prompt_submit_delay_sec": candidate.prompt_submit_delay_sec,
            "prompt_submit_delay_applied_sec": prompt_submit_delay_applied_sec,
            "prompt_ready": None
            if prompt_ready_result is None
            else prompt_ready_result.get("ok") is True,
            "prompt_ready_detail": prompt_ready_result,
        }
        bootstrap_diagnostics = {
            **metadata,
            "tmux_target_ready_detail": target_ready,
            "pipe_pane_returncode": _completed_returncode(pipe_result),
            "pipe_pane_stderr": _completed_stderr(pipe_result)[-1000:],
        }
        if prompt_ready_result is not None:
            bootstrap_diagnostics["prompt_ready_detail"] = prompt_ready_result
        if send_literal_result is not None:
            bootstrap_diagnostics.update(
                {
                    "send_keys_literal_returncode": _completed_returncode(send_literal_result),
                    "send_keys_literal_stderr": _completed_stderr(send_literal_result)[-1000:],
                }
            )
        if load_buffer_result is not None:
            bootstrap_diagnostics.update(
                {
                    "load_buffer_returncode": _completed_returncode(load_buffer_result),
                    "load_buffer_stderr": _completed_stderr(load_buffer_result)[-1000:],
                }
            )
        if paste_buffer_result is not None:
            bootstrap_diagnostics.update(
                {
                    "paste_buffer_returncode": _completed_returncode(paste_buffer_result),
                    "paste_buffer_stderr": _completed_stderr(paste_buffer_result)[-1000:],
                }
            )
        if delete_buffer_result is not None:
            bootstrap_diagnostics.update(
                {
                    "delete_buffer_returncode": _completed_returncode(delete_buffer_result),
                    "delete_buffer_stderr": _completed_stderr(delete_buffer_result)[-1000:],
                }
            )
        if send_enter_result is not None:
            bootstrap_diagnostics.update(
                {
                    "send_keys_enter_returncode": _completed_returncode(send_enter_result),
                    "send_keys_enter_stderr": _completed_stderr(send_enter_result)[-1000:],
                }
            )
        baseline_output_bytes = 0
        metadata["baseline_output_bytes"] = baseline_output_bytes
        bootstrap_diagnostics["baseline_output_bytes"] = baseline_output_bytes
        store.update_job_meta(job_id, metadata)
        store.send_event(
            job_id,
            level="info",
            type="tmux_started",
            message="Tmux provider session started",
            data={
                **bootstrap_diagnostics,
                "attempts": attempts,
            },
        )

        if (
            prompt_ready_result is not None
            and prompt_ready_result.get("ok") is True
            and not prompt_sent
        ):
            cleanup = run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            message = "Tmux prompt delivery failed after provider became ready"
            failure_data = {
                **bootstrap_diagnostics,
                "tmux_cleanup": "killed" if _completed_returncode(cleanup) == 0 else "kill_failed",
            }
            store.send_event(
                job_id,
                level="error",
                type="prompt_delivery_failed",
                message=message,
                data=failure_data,
            )
            store.set_result(job_id, ok=False, summary=message, artifacts=[])
            return {
                "ok": False,
                "error": "prompt_delivery_failed",
                "message": message,
                **metadata,
            }

        if not monitor:
            return {"ok": True, "selected_candidate": candidate.name, **metadata}

        return _monitor_tmux_job(
            store,
            job_id,
            session_name=session_name,
            output_path=output_path,
            transcript_path=transcript_path,
            exit_status_path=exit_status_path,
            run=run,
            sleep=sleep,
            poll_interval_sec=poll_interval_sec,
            bootstrap_diagnostics=bootstrap_diagnostics,
            timeout_sec=timeout_sec,
            complete_on_output=complete_on_output,
            baseline_output_bytes=baseline_output_bytes,
            profile=str(req.get("profile") or ""),
        )

    message = "All tmux provider candidates failed"
    store.send_event(
        job_id,
        level="error",
        type="error",
        message=message,
        data={"attempts": attempts},
    )
    store.set_result(job_id, ok=False, summary=message, artifacts=[])
    return {
        "ok": False,
        "error": "all_candidates_failed",
        "message": message,
        "attempts": attempts,
    }


def run_gui_job(
    store: JobStore,
    job_id: str,
    req: dict[str, Any],
    *,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Run a GUI job and persist events/result."""
    job = store.get_job(job_id)
    # A stop that raced startup must be observed before any browser focus,
    # prompt delivery, or session publication.
    if job is not None:
        meta_status = store.job_status(job_id) or "running"
        if meta_status != "running":
            store.send_event(
                job_id,
                level="info",
                type="cancelled",
                message="GUI provider start skipped: job is already terminal",
                data={"status": meta_status},
            )
            return {
                "ok": False,
                "error": "cancelled",
                "message": "Job was stopped before the GUI provider started",
                "attempts": [],
            }

    # Reuse the pre-registered event when present: re-registering inherits an
    # already-cancelled state, so an early stop is never lost.
    existing = run_handles.get(job_id)
    cancel = existing.cancel_event if existing is not None else threading.Event()

    def on_cancel() -> dict[str, Any]:
        session = sessions.active(job_id)
        data: dict[str, Any] = {"transport": "gui", "provider_stop": "requested"}
        if session is not None:
            data["gui_session"] = session.snapshot()
        return data

    run_handles.register(job_id, cancel_event=cancel, on_cancel=on_cancel)

    store.send_event(
        job_id,
        level="info",
        type="provider_started",
        message="GUI provider execution started",
        data={
            "profile": req.get("profile"),
            "operation": req.get("operation"),
            "timeout_sec": timeout_sec or CHATGPT_PRO_DEFAULT_TIMEOUT_SEC,
        },
    )

    def progress(event_type: str, message: str, data: dict[str, Any]) -> None:
        store.send_event(
            job_id,
            level="info",
            type=event_type,
            message=message,
            data=data,
        )

    try:
        result = run_gui_request(
            req,
            timeout_sec=timeout_sec,
            progress=progress,
            cancel=cancel,
            job_id=job_id,
            artifact_dir=job.path if job is not None else None,
        )
    finally:
        run_handles.release(job_id)

    if result.get("error") == "cancelled":
        store.send_event(
            job_id,
            level="info",
            type="cancelled",
            message=result.get("message") or "GUI provider cancelled",
            data={
                "provider_stop_confirmed": bool(result.get("provider_stop_confirmed")),
                "lifecycle": result.get("lifecycle", []),
                "artifacts": result.get("artifacts", []),
            },
        )
        # The durable stopped status written by job_stop stays terminal; the
        # late-result guard in set_result rejects any competing outcome.
        store.set_result(
            job_id,
            ok=False,
            summary=str(result.get("message") or "GUI provider cancelled"),
            artifacts=result.get("artifacts", []),
        )
        return result

    if result.get("output"):
        store.send_event(
            job_id,
            level="info",
            type="stdout",
            message=result["output"],
            data={"selected_candidate": result.get("selected_candidate")},
        )

    if result["ok"]:
        store.set_result(
            job_id,
            ok=True,
            summary=result.get("output", ""),
            artifacts=result.get("artifacts", []),
        )
        return result

    message = result.get("message") or result.get("error") or "GUI provider failed"
    store.send_event(
        job_id,
        level="error",
        type="error",
        message=message,
        data={
            "attempts": result.get("attempts", []),
            "artifacts": result.get("artifacts", []),
        },
    )
    store.set_result(job_id, ok=False, summary=message, artifacts=result.get("artifacts", []))
    return result


def start_print_job(
    store: JobStore,
    job_id: str,
    req: dict[str, Any],
    *,
    timeout_sec: int | None = None,
) -> threading.Thread:
    """Start a print job in the background and return the worker thread."""
    thread = threading.Thread(
        target=run_print_job,
        kwargs={
            "store": store,
            "job_id": job_id,
            "req": dict(req),
            "timeout_sec": timeout_sec,
        },
        name=f"agents-print-{job_id}",
        daemon=True,
    )
    thread.start()
    return thread


def start_gui_job(
    store: JobStore,
    job_id: str,
    req: dict[str, Any],
    *,
    timeout_sec: int | None = None,
) -> threading.Thread:
    """Start a GUI job in the background and return the worker thread."""
    # Register the cancellation handle before the job becomes observable, so a
    # stop that races the worker cannot slip through the window between the
    # worker's durable status check and its own registration.
    run_handles.register(job_id)
    thread = threading.Thread(
        target=run_gui_job,
        kwargs={
            "store": store,
            "job_id": job_id,
            "req": dict(req),
            "timeout_sec": timeout_sec,
        },
        name=f"agents-gui-{job_id}",
        daemon=True,
    )
    thread.start()
    return thread


def start_tmux_job(
    store: JobStore,
    job_id: str,
    req: dict[str, Any],
    *,
    poll_interval_sec: float = 1.0,
    timeout_sec: int | None = None,
    complete_on_output: bool = True,
) -> threading.Thread:
    """Start a tmux job in the background and return the worker thread."""
    job = store.get_job(job_id)
    if job is not None:
        # Store the deterministic session identity before exposing the job to
        # callers.  That makes job_stop effective even if it races the worker
        # before ``tmux new-session`` completes.
        store.update_job_meta(
            job_id,
            {
                "tmux_session": _tmux_session_name(job_id),
                "tmux_target": _tmux_session_name(job_id),
                "tmux_transcript_path": str(job.path / "transcript.jsonl"),
                "tmux_output_path": str(job.path / "tmux-output.log"),
                "tmux_exit_status_path": str(job.path / "tmux-exit-status.txt"),
            },
        )
    thread = threading.Thread(
        target=run_tmux_job,
        kwargs={
            "store": store,
            "job_id": job_id,
            "req": dict(req),
            "poll_interval_sec": poll_interval_sec,
            "timeout_sec": timeout_sec,
            "complete_on_output": complete_on_output,
        },
        name=f"agents-tmux-{job_id}",
        daemon=True,
    )
    thread.start()
    return thread
