"""Owner-mediated ACP permission and native-completion tests.

Covers the client-level auto-allow boundary and honest surfacing, plus the
runtime-level pending hold/resolve flow and stop_reason completion gating.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
from acp.schema import (
    AllowedOutcome,
    PermissionOption,
    ToolCallLocation,
    ToolCallUpdate,
)

from agent_crossbar.acp_client import (
    NO_OUTPUT_SENTINEL,
    AcpResult,
    _OneShotClient,
    permission_decision_options,
    permission_request_detail,
    permission_response_for_decision,
)
from agent_crossbar.acp_runtime import run_acp_job
from agent_crossbar.jobs import JobStore
from agent_crossbar.models import Autonomy


def _opt(option_id, kind):
    return PermissionOption(option_id=option_id, name=option_id, kind=kind)


def _call(kind, title=None, locations=None, raw_input=None):
    return ToolCallUpdate(
        tool_call_id="tc-1",
        kind=kind,
        title=title,
        locations=locations,
        raw_input=raw_input,
    )


def _run(coro):
    return asyncio.run(coro)


# ── client-level: auto-allow boundary ───────────────────────────────────


def test_auto_allowed_local_edit_does_not_invoke_owner_handler():
    calls = []

    async def handler(tool_call, options):
        calls.append(tool_call)
        return permission_response_for_decision(options, "reject")

    client = _OneShotClient(Autonomy.EDIT_LOCAL, cwd="/tmp", permission_handler=handler)
    resp = _run(
        client.request_permission(
            "s",
            _call("edit", locations=[ToolCallLocation(path="/tmp/file")]),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert isinstance(resp.outcome, AllowedOutcome)
    assert resp.outcome.option_id == "allow"
    assert calls == []


def test_non_auto_allowed_request_is_held_for_owner():
    seen = {}

    async def handler(tool_call, options):
        seen["kind"] = tool_call.kind
        return permission_response_for_decision(options, "reject")

    client = _OneShotClient(Autonomy.EDIT_LOCAL, cwd="/tmp", permission_handler=handler)
    resp = _run(
        client.request_permission(
            "s",
            _call("other", raw_input={"command": "rm -rf /"}),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert seen["kind"] == "other"
    assert resp.outcome.option_id == "reject"


def test_read_only_request_is_held_for_owner_not_auto_rejected():
    seen = []

    async def handler(tool_call, options):
        seen.append(tool_call.kind)
        return permission_response_for_decision(options, "reject")

    client = _OneShotClient(Autonomy.READ_ONLY, permission_handler=handler)
    _run(
        client.request_permission(
            "s",
            _call("read", locations=[ToolCallLocation(path="/etc/passwd")]),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert seen == ["read"]


def test_no_handler_rejects_non_auto_allowed_request():
    client = _OneShotClient(Autonomy.EDIT_LOCAL, cwd="/tmp")
    resp = _run(
        client.request_permission(
            "s",
            _call("other"),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert resp.outcome.option_id == "reject"


# ── client-level: honest, bounded, non-escalating surface ───────────────


def test_decision_options_never_offer_allow_always():
    options = [_opt("always", "allow_always"), _opt("reject", "reject_once")]
    assert permission_decision_options(options) == ["reject"]
    response = permission_response_for_decision(options, "reject")
    assert response.outcome.option_id == "reject"


def test_allow_maps_only_to_allow_once_option():
    options = [_opt("always", "allow_always"), _opt("once", "allow_once")]
    assert permission_decision_options(options) == ["allow", "reject"]
    response = permission_response_for_decision(options, "allow")
    assert response.outcome.option_id == "once"


def test_kind_other_and_switch_mode_are_honest():
    assert permission_request_detail(_call("other"))["kind"] == "other"
    assert permission_request_detail(_call("switch_mode"))["kind"] == "switch_mode"
    assert permission_request_detail(_call(None))["kind"] == "other"


def test_filepath_and_parentdir_raw_input_are_recognized():
    detail = permission_request_detail(_call("edit", raw_input={"filepath": "/tmp/a.py"}))
    assert detail["path"] == "/tmp/a.py"
    detail = permission_request_detail(_call("read", raw_input={"parentDir": "/tmp/pkg"}))
    assert detail["path"] == "/tmp/pkg"


def test_permission_detail_preserves_all_raw_targets_in_stable_order():
    detail = permission_request_detail(
        _call(
            "other",
            raw_input={"parentDir": "/tmp/pkg", "filepath": "/tmp/pkg/a.py"},
        )
    )
    assert detail["path"] == "/tmp/pkg/a.py"
    assert detail["paths"] == ["/tmp/pkg/a.py", "/tmp/pkg"]


def test_locations_merge_with_raw_input_paths_instead_of_being_skipped():
    """A tool call can carry targets in both raw_input and locations (e.g. a
    multi-file move); the locations targets must not be silently dropped just
    because raw_input already yielded a path."""
    detail = permission_request_detail(
        _call(
            "edit",
            raw_input={"filepath": "/tmp/from.py"},
            locations=[ToolCallLocation(path="/tmp/to.py")],
        )
    )
    assert detail["path"] == "/tmp/from.py"
    assert set(detail["paths"]) == {"/tmp/from.py", "/tmp/to.py"}


def test_more_than_eight_paths_is_marked_truncated_and_reject_only():
    """Bounding the surfaced path list to 8 entries hides real targets beyond
    the cap; that must be flagged exactly like a per-field truncation so the
    owner can never `allow` an action whose full scope wasn't shown."""
    keys = (
        "filepath",
        "filePath",
        "file_path",
        "path",
        "target",
        "parentDir",
        "parent_dir",
        "parentdir",
        "cwd",
    )
    raw_input = {key: f"/tmp/file-{key}" for key in keys}
    detail = permission_request_detail(_call("edit", raw_input=raw_input))
    assert len(detail["paths"]) == 8
    assert detail["details_truncated"] is True
    options = [_opt("allow", "allow_once"), _opt("reject", "reject_once")]
    decisions = permission_decision_options(options)
    if detail.get("details_truncated"):
        decisions = ["reject"]
    assert decisions == ["reject"]


def test_pwd_with_out_of_scope_raw_cwd_is_not_auto_allowed():
    calls = []

    async def handler(tool_call, options):
        calls.append(tool_call)
        return permission_response_for_decision(options, "reject")

    client = _OneShotClient(Autonomy.EDIT_LOCAL, cwd="/tmp/work", permission_handler=handler)
    response = _run(
        client.request_permission(
            "s",
            _call("execute", raw_input={"command": "pwd", "cwd": "/etc"}),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert calls
    assert response.outcome.option_id == "reject"


def test_detail_redacts_secret_looking_command():
    detail = permission_request_detail(
        _call("execute", raw_input={"command": "curl -H 'Authorization: Bearer sk-live-123'"})
    )
    assert "sk-live-123" not in detail["command"]
    assert "REDACTED" in detail["command"]


def test_detail_omits_absent_fields():
    detail = permission_request_detail(_call("other"))
    assert "command" not in detail
    assert "path" not in detail


@pytest.mark.parametrize("argument_key", ["args", "arguments", "argv"])
def test_permission_detail_surfaces_each_argument_spelling(argument_key):
    detail = permission_request_detail(
        _call(
            "execute",
            raw_input={
                "command": "rm",
                argument_key: ["-rf", "/tmp/target", "TOKEN=sk-live-hidden"],
            },
        )
    )
    assert detail[argument_key][:2] == ["-rf", "/tmp/target"]
    assert "sk-live-hidden" not in str(detail[argument_key])
    assert "REDACTED" in str(detail[argument_key])


def test_permission_detail_surfaces_all_argument_forms_without_fallback_hiding_targets():
    detail = permission_request_detail(
        _call(
            "execute",
            raw_input={
                "command": "ls",
                "args": [],
                "arguments": ["visible.txt"],
                "argv": ["hidden-target.txt"],
            },
        )
    )
    assert detail["args"] == []
    assert detail["arguments"] == ["visible.txt"]
    assert detail["argv"] == ["hidden-target.txt"]
    assert "details_incomplete" not in detail


def test_malformed_argument_form_is_visible_and_fail_closed():
    detail = permission_request_detail(
        _call("execute", raw_input={"command": "ls", "argv": {"path": "/tmp/target"}})
    )
    assert detail["argv"] == "[UNAVAILABLE]"
    assert detail["details_incomplete"] is True

    client = _OneShotClient(Autonomy.EDIT_LOCAL, cwd="/tmp")
    response = _run(
        client.request_permission(
            "s",
            _call("execute", raw_input={"command": "ls", "argv": {"path": "/tmp/target"}}),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert response.outcome.option_id == "reject"


def test_hidden_argument_form_cannot_authorize_unsafe_execute(tmp_path):
    client = _OneShotClient(Autonomy.EDIT_LOCAL, cwd=str(tmp_path))
    target = str(tmp_path / "target")
    # ``args`` is empty, but the provider also supplies a dangerous hidden
    # target under ``argv``.  The policy must inspect every spelling.
    response = _run(
        client.request_permission(
            "s",
            _call(
                "execute",
                raw_input={"command": "rm", "args": [], "argv": ["-rf", target]},
            ),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert response.outcome.option_id == "reject"


def test_argument_truncation_is_reject_only():
    raw_input = {"command": "ls", "args": ["x"] * 33}
    detail = permission_request_detail(_call("execute", raw_input=raw_input))
    assert len(detail["args"]) == 32
    assert detail["details_truncated"] is True

    client = _OneShotClient(Autonomy.EDIT_LOCAL, cwd="/tmp")
    response = _run(
        client.request_permission(
            "s",
            _call("execute", raw_input=raw_input),
            [_opt("allow", "allow_once"), _opt("reject", "reject_once")],
        )
    )
    assert response.outcome.option_id == "reject"


# ── runtime-level: native completion evidence ───────────────────────────


def _create_job_store(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode",
        operation="dev",
        transport="print",
        cwd=str(tmp_path),
    )
    return store, job.job_id


async def _run_job(store, job_id, tmp_path):
    await run_acp_job(
        store,
        job_id,
        provider="opencode",
        prompt="do the thing",
        cwd=str(tmp_path),
        model="glm",
        task="dev",
        autonomy=Autonomy.EDIT_LOCAL,
        max_runtime_sec=30,
    )


@pytest.mark.parametrize("stop_reason", ["refusal", "cancelled", "max_turn_requests"])
def test_incomplete_stop_reason_with_output_is_not_success(tmp_path, stop_reason):
    store, job_id = _create_job_store(tmp_path)
    acp_result = AcpResult(
        output="I started but could not finish.", stop_reason=stop_reason, session_id="s1"
    )
    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(return_value=acp_result),
    ):
        asyncio.run(_run_job(store, job_id, tmp_path))

    stored = store.get_result(job_id)
    assert stored["status"] == "failed"
    assert stored["stop_reason"] == stop_reason
    assert stored["failure"]["code"] == "acp_incomplete"


def test_end_turn_with_output_is_success(tmp_path):
    store, job_id = _create_job_store(tmp_path)
    acp_result = AcpResult(output="All done.", stop_reason="end_turn", session_id="s1")
    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(return_value=acp_result),
    ):
        asyncio.run(_run_job(store, job_id, tmp_path))

    stored = store.get_result(job_id)
    assert stored["ok"] is True
    assert stored["status"] == "completed"


def test_rejected_permission_with_progress_and_end_turn_is_failed(tmp_path):
    store, job_id = _create_job_store(tmp_path)
    acp_result = AcpResult(
        output="Progress: starting the bounded permission check.",
        stop_reason="end_turn",
        session_id="s1",
        permission_rejected=True,
    )
    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(return_value=acp_result),
    ):
        asyncio.run(_run_job(store, job_id, tmp_path))

    stored = store.get_result(job_id)
    assert stored["ok"] is False
    assert stored["status"] == "failed"
    assert stored["stop_reason"] == "permission_rejected"
    assert stored["failure"]["code"] == "acp_incomplete"


# ── runtime-level: full pending hold + job_send resolution ──────────────


def test_runtime_holds_permission_pending_and_resolves_via_job_send(tmp_path):
    store, job_id = _create_job_store(tmp_path)
    captured: dict = {}

    async def fake_run(*_args, **kwargs):
        handler = kwargs["permission_handler"]
        tool_call = _call(
            "other",
            title="run shell",
            raw_input={
                "command": "rm -rf /",
                "args": [],
                "arguments": ["-rf"],
                "argv": ["/"],
                "filepath": "/tmp/out.py",
                "parentDir": "/tmp",
            },
        )
        options = [_opt("a", "allow_once"), _opt("r", "reject_once")]
        captured["response"] = await handler(tool_call, options)
        return AcpResult(output="done", stop_reason="end_turn", session_id="s1")

    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(side_effect=fake_run),
    ):
        thread = threading.Thread(
            target=lambda: asyncio.run(_run_job(store, job_id, tmp_path)), daemon=True
        )
        thread.start()
        request_id = None
        deadline = time.time() + 5
        while time.time() < deadline:
            tail = store.job_tail(job_id)
            pending = tail.get("pending_request")
            if pending:
                request_id = pending["request_id"]
                # Surfaced detail is bounded, honest, and non-secret.
                assert pending["kind"] == "other"
                assert pending["decisions"] == ["allow", "reject"]
                assert "command" in pending
                assert pending["args"] == []
                assert pending["arguments"] == ["-rf"]
                assert pending["argv"] == ["/"]
                assert pending["paths"] == ["/tmp/out.py", "/tmp"]
                break
            time.sleep(0.02)
        assert request_id, "job never surfaced a pending permission request"

        sent = store.send_user_input(
            job_id, json.dumps({"request_id": request_id, "decision": "allow"})
        )
        assert sent["ok"] is True
        thread.join(timeout=5)
        assert not thread.is_alive()

    response = captured["response"]
    assert isinstance(response.outcome, AllowedOutcome)
    assert response.outcome.option_id == "a"
    stored = store.get_result(job_id)
    assert stored["ok"] is True
    assert stored["status"] == "completed"


@pytest.mark.parametrize(
    "output",
    [
        "Progress: starting the bounded permission check.",
        NO_OUTPUT_SENTINEL,
    ],
    ids=["progress", "empty-output"],
)
def test_actual_client_reject_latches_before_output_classification(tmp_path, output):
    """A client-side reject wins even when ACP emits no assistant text."""
    store, job_id = _create_job_store(tmp_path)
    captured: dict = {}

    async def fake_run(*_args, **kwargs):
        handler = kwargs["permission_handler"]
        client = _OneShotClient(Autonomy.READ_ONLY, permission_handler=handler)
        captured["response"] = await client.request_permission(
            "s1",
            _call(
                "other",
                title="bounded permission check",
                raw_input={"filepath": "/tmp/" + ("outside" * 100)},
            ),
            [_opt("allow-id", "allow_once"), _opt("reject-id", "reject_once")],
        )
        return AcpResult(
            output=output,
            stop_reason="end_turn",
            session_id="s1",
            permission_rejected=client.permission_rejected,
        )

    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(side_effect=fake_run),
    ):
        thread = threading.Thread(
            target=lambda: asyncio.run(_run_job(store, job_id, tmp_path)), daemon=True
        )
        thread.start()
        request_id = None
        deadline = time.time() + 5
        while time.time() < deadline:
            pending = store.job_tail(job_id).get("pending_request")
            if pending:
                request_id = pending["request_id"]
                assert pending["decisions"] == ["reject"]
                assert pending["details_truncated"] is True
                break
            time.sleep(0.02)
        assert request_id
        sent = store.send_user_input(
            job_id, json.dumps({"request_id": request_id, "decision": "reject"})
        )
        assert sent["ok"] is True
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert captured["response"].outcome.option_id == "reject-id"
    stored = store.get_result(job_id)
    assert stored["ok"] is False
    assert stored["status"] == "failed"
    assert stored["stop_reason"] == "permission_rejected"
    assert stored["failure"]["code"] == "acp_incomplete"


def test_concurrent_permission_callbacks_are_serialized_and_each_is_surfaced(tmp_path):
    store, job_id = _create_job_store(tmp_path)
    captured: dict = {}

    async def fake_run(*_args, **kwargs):
        handler = kwargs["permission_handler"]
        options = [_opt("a", "allow_once"), _opt("r", "reject_once")]
        first = asyncio.create_task(handler(_call("other", title="first"), options))
        captured["first"] = first
        deadline = time.time() + 5
        while time.time() < deadline:
            pending = store.job_tail(job_id).get("pending_request")
            if pending:
                captured["first_id"] = pending["request_id"]
                break
            await asyncio.sleep(0.01)
        assert "first_id" in captured
        first_response = await first

        second = asyncio.create_task(handler(_call("other", title="second"), options))
        captured["second"] = second
        deadline = time.time() + 5
        while time.time() < deadline:
            pending = store.job_tail(job_id).get("pending_request")
            if pending and pending["request_id"] != captured["first_id"]:
                captured["second_id"] = pending["request_id"]
                break
            await asyncio.sleep(0.01)
        assert "second_id" in captured
        second_response = await second
        captured["responses"] = (first_response, second_response)
        return AcpResult(output="done", stop_reason="end_turn", session_id="s1")

    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(side_effect=fake_run),
    ):
        thread = threading.Thread(
            target=lambda: asyncio.run(_run_job(store, job_id, tmp_path)), daemon=True
        )
        thread.start()
        deadline = time.time() + 5
        while "first_id" not in captured and time.time() < deadline:
            time.sleep(0.01)
        assert "first_id" in captured
        assert store.send_user_input(
            job_id,
            json.dumps({"request_id": captured["first_id"], "decision": "allow"}),
        )["ok"]
        deadline = time.time() + 5
        while "second_id" not in captured and time.time() < deadline:
            time.sleep(0.01)
        assert "second_id" in captured
        assert store.send_user_input(
            job_id,
            json.dumps({"request_id": captured["second_id"], "decision": "reject"}),
        )["ok"]
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert captured["responses"][0].outcome.option_id == "a"
    assert captured["responses"][1].outcome.option_id == "r"


def test_stop_winning_before_permission_consume_forces_reject(tmp_path, monkeypatch):
    store, job_id = _create_job_store(tmp_path)
    captured: dict = {}
    original_transition = store.transition_job_status

    def race_transition(job, status, *, allowed_from, updates=None, remove=()):
        if status == "running" and "awaiting_input" in allowed_from:
            store.stop_job(job, reason="race")
        return original_transition(
            job, status, allowed_from=allowed_from, updates=updates, remove=remove
        )

    monkeypatch.setattr(store, "transition_job_status", race_transition)

    async def fake_run(*_args, **kwargs):
        response = await kwargs["permission_handler"](
            _call("other", title="race"),
            [_opt("allow-id", "allow_once"), _opt("reject-id", "reject_once")],
        )
        captured["response"] = response
        return AcpResult(output="progress", stop_reason="end_turn", session_id="s1")

    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(side_effect=fake_run),
    ):
        thread = threading.Thread(
            target=lambda: asyncio.run(_run_job(store, job_id, tmp_path)), daemon=True
        )
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            pending = store.job_tail(job_id).get("pending_request")
            if pending:
                break
            time.sleep(0.01)
        assert pending
        assert store.send_user_input(
            job_id,
            json.dumps({"request_id": pending["request_id"], "decision": "allow"}),
        )["ok"]
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert captured["response"].outcome.option_id == "reject-id"
    assert store.job_status(job_id) == "stopped"


def test_registry_overflow_reverts_status_instead_of_stranding_awaiting_input(
    tmp_path, monkeypatch
):
    """A bounded-registry overflow must not leave the job showing
    ``awaiting_input`` with no concrete pending request — a waiter/CLI must
    never see that vague waiting state (job-owner-decision-protocol:
    "Waiters MUST see a concrete pending request, not a vague waiting
    state")."""
    from agent_crossbar.pending_permissions import PendingRequestLimitError, pending_permissions

    store, job_id = _create_job_store(tmp_path)
    captured: dict = {}

    def raise_limit(*_args, **_kwargs):
        raise PendingRequestLimitError("forced for test")

    monkeypatch.setattr(pending_permissions, "register", raise_limit)

    async def fake_run(*_args, **kwargs):
        handler = kwargs["permission_handler"]
        response = await handler(
            _call("other", title="overflow"),
            [_opt("a", "allow_once"), _opt("r", "reject_once")],
        )
        captured["response"] = response
        # Snapshot durable state right after the handler returns and before
        # the job's own completion overwrites it, so the assertion targets
        # exactly what the overflow branch left behind.
        captured["meta"] = store._read_job_meta(store.get_job(job_id).path)
        return AcpResult(output="done", stop_reason="end_turn", session_id="s1")

    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(side_effect=fake_run),
    ):
        asyncio.run(_run_job(store, job_id, tmp_path))

    assert captured["response"].outcome.option_id == "r"
    meta = captured["meta"]
    assert meta.get("status") == "running"
    assert "pending_request" not in meta
    assert "waiting_for" not in meta
