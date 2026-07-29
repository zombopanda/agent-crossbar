"""GUI cancellation, session isolation, and job-local artifact evidence."""

from __future__ import annotations

import json
import threading
import time

import pytest

from agent_crossbar import runner as runner_module
from agent_crossbar.gui_lifecycle import sessions
from agent_crossbar.jobs import JobStore
from agent_crossbar.run_handles import run_handles

CANDIDATE = runner_module._CHATGPT_BROWSER_CANDIDATES[0]


@pytest.fixture(autouse=True)
def _clean_registries():
    run_handles.clear()
    sessions.clear()
    yield
    run_handles.clear()
    sessions.clear()


def composer_tree(value: str, *, running: bool = False) -> str:
    escaped = json.dumps(value)[1:-1]
    action = '[4] AXButton "Stop"' if running else "[3] AXButton (Send prompt)"
    return (
        f'AXWebArea "ChatGPT"\n[1] AXTextArea = "{escaped}" (Chat with ChatGPT)\n'
        f'[2] AXButton "Pro"\n{action}'
    )


class ScriptedCua:
    """Minimal CUA fake that streams forever until Stop is clicked."""

    def __init__(self, *, running: bool = True):
        self.state = {"composer": "Ask ChatGPT", "submitted": False, "running": running}
        self.clicks: list[int] = []
        self.stop_clicked = False

    def call(self, tool, payload, timeout_sec=None):
        if tool == "list_apps":
            return {"apps": [{"bundle_id": CANDIDATE.bundle_id, "pid": 123, "running": True}]}
        if tool == "list_windows":
            return {
                "windows": [
                    {
                        "pid": 123,
                        "window_id": 456,
                        "title": "ChatGPT",
                        "layer": 0,
                        "is_on_screen": True,
                    }
                ]
            }
        if tool == "get_window_state":
            return {
                "tree_markdown": composer_tree(
                    self.state["composer"],
                    running=self.state["submitted"] and self.state["running"],
                )
            }
        if tool == "page":
            return {"text": "signed in\nAsk ChatGPT"}
        if tool == "hotkey":
            return {"ok": True}
        if tool == "click":
            index = payload.get("element_index")
            self.clicks.append(index)
            if index == 4:
                self.stop_clicked = True
                self.state["running"] = False
            elif index == 3:
                self.state["submitted"] = True
            return {"ok": True}
        raise AssertionError(tool)

    def call_with_timeout(self, tool, payload, timeout_sec):
        return self.call(tool, payload)


def deliver_stub(cua):
    def deliver(*args, **kwargs):
        cua.state["composer"] = args[4] if len(args) > 4 else kwargs["prompt"]
        return True

    return deliver


def test_stop_before_start_never_touches_the_browser(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = store.create_job(profile="chatgpt_pro", operation="advice", transport="gui")
    store.stop_job(job.job_id)
    called: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "run_gui_request",
        lambda *_a, **_k: called.append("ran") or {"ok": True, "output": "x"},
    )

    result = runner_module.run_gui_job(store, job.job_id, {"profile": "chatgpt_pro"})

    assert called == []
    assert result["error"] == "cancelled"
    assert [event["type"] for event in job.events.read_since(0)][-1] == "cancelled"
    assert store.job_status(job.job_id) == "stopped"


def test_cancel_before_submit_returns_cancelled_without_submitting(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    cancel = threading.Event()
    cancel.set()
    submitted: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_run_chatgpt_browser_candidate",
        lambda *_a, **_k: submitted.append("submitted") or {"ok": True, "output": "x"},
    )

    result = runner_module.run_gui_request(
        {"profile": "chatgpt_pro", "operation": "advice", "prompt": "advise"},
        cua=object(),
        sleep=lambda _: None,
        cancel=cancel,
        timeout_sec=5,
    )

    assert submitted == []
    assert result["error"] == "cancelled"


def test_cancel_during_startup_is_observed_before_prompt_delivery(monkeypatch):
    cancel = threading.Event()
    cua = ScriptedCua()
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", deliver_stub(cua))

    def app(*_args, **_kwargs):
        cancel.set()
        return {"pid": 123}

    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", app)
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_a, **_k: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )

    result = runner_module._run_chatgpt_browser_candidate(
        CANDIDATE,
        {"prompt": "advise"},
        cua,
        lambda _: None,
        time.monotonic() + 5,
        "nonce",
        cancel=cancel,
    )

    assert result["error"] == "cancelled"
    assert result["provider_stop_confirmed"] is False
    assert cua.state["submitted"] is False
    assert result["diagnostics"]["prompt_submitted"] is False


def test_cancel_during_generation_clicks_stop_and_retires_session(monkeypatch):
    cua = ScriptedCua()
    cancel = threading.Event()
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_a, **_k: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_a, **_k: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", deliver_stub(cua))
    session, _ = sessions.acquire("job-1", CANDIDATE.key, CANDIDATE.name)

    def sleep(_seconds):
        # The stop lands while ChatGPT is still streaming.
        cancel.set()

    result = runner_module._run_chatgpt_browser_candidate(
        CANDIDATE,
        {"prompt": "advise"},
        cua,
        sleep,
        time.monotonic() + 5,
        "nonce",
        cancel=cancel,
        session=session,
    )

    assert result["error"] == "cancelled"
    assert result["provider_stop_confirmed"] is True
    assert cua.stop_clicked is True
    assert session.retired is True
    assert session.retired_reason == "cancelled_after_submit"
    assert result["diagnostics"]["prompt_submitted"] is True


def test_stop_job_cancels_a_registered_gui_run_handle(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="chatgpt_pro", operation="advice", transport="gui")
    cancel = threading.Event()
    run_handles.register(
        job.job_id,
        cancel_event=cancel,
        on_cancel=lambda: {"transport": "gui", "provider_stop": "requested"},
    )

    first = store.stop_job(job.job_id)
    second = store.stop_job(job.job_id)

    assert first["ok"] is True
    assert second["ok"] is True
    assert cancel.is_set() is True
    stops = [
        event
        for event in job.events.read_since(0)
        if event["type"] == "stopped" and "run_handle_stop" in event["data"]
    ]
    assert stops[0]["data"]["run_handle_stop"]["repeated"] is False
    assert stops[1]["data"]["run_handle_stop"]["repeated"] is True


def test_late_provider_success_never_overwrites_a_stopped_job(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = store.create_job(profile="chatgpt_pro", operation="advice", transport="gui")

    def late_success(req, **kwargs):
        store.stop_job(job.job_id)
        return {"ok": True, "output": "LATE", "selected_candidate": "ChatGPT Pro web via Helium"}

    monkeypatch.setattr(runner_module, "run_gui_request", late_success)

    runner_module.run_gui_job(store, job.job_id, {"profile": "chatgpt_pro"})

    assert store.job_status(job.job_id) == "stopped"
    result = store.get_result(job.job_id)
    assert result.get("status") == "stopped" or result.get("ok") is not True


def test_second_concurrent_turn_is_busy_and_never_touches_cua(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    sessions.acquire("other-job", CANDIDATE.key, CANDIDATE.name)
    touched: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_run_chatgpt_browser_candidate",
        lambda *_a, **_k: touched.append("cua") or {"ok": True, "output": "x"},
    )

    result = runner_module.run_gui_request(
        {"profile": "chatgpt_pro", "operation": "advice", "prompt": "advise"},
        cua=object(),
        sleep=lambda _: None,
        job_id="job-1",
        timeout_sec=5,
    )

    assert result["error"] == "busy"
    assert result["busy_reason"] == "busy"
    assert touched == []


def test_session_mismatch_fails_closed_without_typing(monkeypatch):
    cua = ScriptedCua()
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_a, **_k: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_a, **_k: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )
    typed: list[str] = []
    monkeypatch.setattr(
        runner_module, "_chatgpt_deliver_prompt", lambda *_a, **_k: typed.append("typed") or True
    )
    session, _ = sessions.acquire("job-1", CANDIDATE.key, CANDIDATE.name)
    session.bind(123, 999)

    # A window identity mismatch is only observable once the runner rebinds,
    # so pin the owned identity and refuse the rebind.
    monkeypatch.setattr(runner_module.ChatGptSession, "bind", lambda *_a, **_k: None)

    result = runner_module._run_chatgpt_browser_candidate(
        CANDIDATE,
        {"prompt": "advise"},
        cua,
        lambda _: None,
        time.monotonic() + 5,
        "nonce",
        session=session,
    )

    assert result["error"] == "session_mismatch"
    assert typed == []
    assert session.retired is True


def test_gui_diagnostics_are_written_under_the_job_directory(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "state")
    job = store.create_job(profile="chatgpt_pro", operation="advice", transport="gui")
    cua = ScriptedCua()
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_a, **_k: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_a, **_k: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )
    # Report a successful paste without changing the composer: the pre-Send
    # verification must fail closed and capture job-local evidence.
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", lambda *_a, **_k: True)
    monkeypatch.setattr(runner_module, "CuaDriverClient", lambda **_k: cua)

    result = runner_module.run_gui_job(
        store,
        job.job_id,
        {"profile": "chatgpt_pro", "operation": "advice", "prompt": "advise"},
        timeout_sec=5,
    )

    assert result["ok"] is False
    artifacts = result["artifacts"]
    assert artifacts, "a failed pre-submit verification must leave job-local evidence"
    for path in artifacts:
        assert str(job.path.resolve()) in path
    assert result["lifecycle"][-1] == "failed"


def test_context_scope_failure_is_reported_before_browser_work(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    touched: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_run_chatgpt_browser_candidate",
        lambda *_a, **_k: touched.append("cua") or {"ok": True, "output": "x"},
    )

    result = runner_module.run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "prompt": "advise",
            "cwd": str(tmp_path),
            "scope": {"paths": ["missing-dir"]},
        },
        cua=object(),
        sleep=lambda _: None,
        timeout_sec=5,
    )

    assert result["error"] == "context_path_missing"
    assert touched == []


def test_packed_context_reaches_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    (tmp_path / "README.md").write_text("project readme", encoding="utf-8")
    seen: dict[str, str] = {}

    def capture(candidate, req, cua, sleep, deadline, nonce, *_args, **kwargs):
        seen["context"] = kwargs.get("context_text", "")
        return {"ok": True, "output": "OK", "browser": candidate.name}

    monkeypatch.setattr(runner_module, "_run_chatgpt_browser_candidate", capture)

    result = runner_module.run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "prompt": "advise",
            "cwd": str(tmp_path),
            "scope": {"paths": ["README.md"]},
        },
        cua=object(),
        sleep=lambda _: None,
        timeout_sec=5,
    )

    assert result["ok"] is True
    assert "project readme" in seen["context"]
    assert result["context_summary"]["files_included"] == 1


def test_start_gui_job_registers_the_handle_before_the_worker_runs(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = store.create_job(profile="chatgpt_pro", operation="advice", transport="gui")
    started = threading.Event()
    observed: dict[str, bool] = {}

    def blocking_run_gui_request(req, **kwargs):
        started.set()
        cancel = kwargs["cancel"]
        observed["cancelled"] = cancel.wait(timeout=5)
        return {
            "ok": False,
            "error": "cancelled",
            "message": "stopped",
            "attempts": [],
            "artifacts": [],
        }

    monkeypatch.setattr(runner_module, "run_gui_request", blocking_run_gui_request)

    thread = runner_module.start_gui_job(store, job.job_id, {"profile": "chatgpt_pro"})
    assert started.wait(timeout=5)
    store.stop_job(job.job_id)
    thread.join(timeout=10)

    assert observed["cancelled"] is True
    assert store.job_status(job.job_id) == "stopped"


def test_requested_attachment_fails_closed_when_upload_is_unsupported(tmp_path, monkeypatch):
    """cua-driver exposes no usable ChatGPT upload route — refuse, never drop."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    (tmp_path / "diagram.png").write_bytes(b"\x89PNG payload")
    cua = ScriptedCua()
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_a, **_k: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_a, **_k: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", deliver_stub(cua))

    result = runner_module.run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "prompt": "advise",
            "cwd": str(tmp_path),
            "scope": {"attachments": ["diagram.png"]},
        },
        cua=cua,
        sleep=lambda _: None,
        timeout_sec=5,
    )

    assert result["ok"] is False
    assert result["attempts"][0]["error"] == "attachment_upload_unsupported"
    assert cua.state["submitted"] is False


def test_safari_upload_capability_is_refused_without_probing(monkeypatch):
    safari = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "safari"
    )

    class ExplodingCua:
        def call(self, tool, payload):
            raise AssertionError("Safari must not be probed for a CDP endpoint")

    supported, reason = runner_module._chatgpt_upload_capability(
        ExplodingCua(), safari, 1, 2, "session"
    )

    assert supported is False
    assert reason == "webkit_no_cdp_endpoint"


def test_chromium_upload_capability_reports_driver_refusal(monkeypatch):
    class RefusingCua:
        def call(self, tool, payload):
            assert tool == "get_browser_state"
            return {
                "status": "refused",
                "refusal": {"code": "browser_route_unavailable", "message": "not a browser"},
            }

    supported, reason = runner_module._chatgpt_upload_capability(
        RefusingCua(), CANDIDATE, 1, 2, "session"
    )

    assert supported is False
    assert reason == "browser_route_unavailable"


def test_stop_control_matches_the_live_chatgpt_label():
    """The live control is `Stop answering` — a fixed label list missed it."""
    for label in ("Stop", "Stop answering", "Stop generating", "Stop streaming"):
        tree = f'AXWebArea "ChatGPT"\n- [248] AXButton ({label}) [actions=[press]]'
        assert runner_module._find_chatgpt_web_stop_button(tree) == 248
        assert runner_module._chatgpt_response_is_running(tree) is True

    idle = 'AXWebArea "ChatGPT"\n- [229] AXButton (Copy response) [actions=[press]]'
    assert runner_module._find_chatgpt_web_stop_button(idle) is None
    assert runner_module._chatgpt_response_is_running(idle) is False
    assert runner_module._chatgpt_completion_action_visible(idle) is True


def test_prompt_delivery_prefers_a_background_ax_write(monkeypatch):
    """The default path must not steal focus or touch the clipboard."""
    monkeypatch.setattr(
        runner_module,
        "_read_text_clipboard",
        lambda: (_ for _ in ()).throw(AssertionError("clipboard must stay untouched")),
    )
    monkeypatch.setattr(
        runner_module,
        "_write_text_clipboard",
        lambda _text: (_ for _ in ()).throw(AssertionError("clipboard must stay untouched")),
    )
    composer = {"value": ""}
    calls: list[str] = []

    class BackgroundCua:
        def call(self, tool, payload):
            calls.append(tool)
            if tool == "set_value":
                composer["value"] = payload["value"]
                return {"ok": True}
            if tool == "get_window_state":
                escaped = json.dumps(composer["value"])[1:-1]
                return {
                    "tree_markdown": (
                        f'[1] AXTextArea = "{escaped}" (Chat with ChatGPT)\n'
                        "[3] AXButton (Send prompt)"
                    )
                }
            raise AssertionError(f"{tool} would move the user's focus")

    trace: dict[str, object] = {}
    delivered = runner_module._chatgpt_deliver_prompt(
        BackgroundCua(), 123, 456, 1, "exact prompt", diagnostics=trace
    )

    assert delivered is True
    assert trace["focus_mode"] == "background_ax"
    assert "hotkey" not in calls
    assert "click" not in calls
    assert composer["value"] == "exact prompt"


def test_failed_delivery_clears_its_own_composer_residue(monkeypatch):
    """A corrupted partial prompt must never be left in the user's draft."""
    cua = ScriptedCua()
    cleared: list[str] = []
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_a, **_k: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_a, **_k: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )

    def mangled_delivery(*args, **kwargs):
        # Simulate the pixel-typing path corrupting the prompt.
        cua.state["composer"] = "You arre the aChatGPT Pro advisor"
        return False

    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", mangled_delivery)
    original_call = cua.call

    def tracking_call(tool, payload, timeout_sec=None):
        if tool == "set_value":
            cleared.append(str(payload.get("value")))
            cua.state["composer"] = payload["value"]
            return {"ok": True}
        return original_call(tool, payload, timeout_sec)

    cua.call = tracking_call

    result = runner_module._run_chatgpt_browser_candidate(
        CANDIDATE,
        {"prompt": "advise"},
        cua,
        lambda _: None,
        time.monotonic() + 5,
        "nonce",
    )

    assert result["error"] == "prompt_insertion_failed"
    assert cleared == [""], "the turn must clear the text it wrote itself"
    assert result["diagnostics"]["residue_cleanup"] == {"cleared": True, "chars": 33}
    assert cua.state["submitted"] is False
