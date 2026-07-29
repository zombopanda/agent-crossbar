import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    DeniedOutcome,
    PermissionOption,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    TextContentBlock,
    ToolCallUpdate,
)

from agent_crossbar.acp_client import (
    AcpLaunchError,
    AcpProtocolError,
    AcpProviderUnavailableError,
    AcpResult,
    AcpTimeoutError,
    _OneShotClient,
    classify_provider_failure,
    run_acp_prompt,
)
from agent_crossbar.models import Autonomy

# --- helpers -----------------------------------------------------------


def _opt(option_id, kind):
    return PermissionOption(option_id=option_id, name=option_id, kind=kind)


def _call(kind, title=None):
    return ToolCallUpdate(tool_call_id="tc-1", kind=kind, title=title)


def _assert_selected(response, id):
    assert isinstance(response.outcome, AllowedOutcome)
    assert response.outcome.outcome == "selected"
    assert response.outcome.option_id == id


def _assert_cancelled(response):
    assert isinstance(response.outcome, DeniedOutcome)
    assert response.outcome.outcome == "cancelled"


# --- _Conn -------------------------------------------------------------


class _Conn:
    def __init__(
        self,
        texts=None,
        stop_reason="end_turn",
        session_id="session-1",
        hang=False,
        hang_before_prompt=False,
        protocol_error=None,
        protocol_error_after_prompt=None,
        model="test-model",
    ):
        self.texts = texts or []
        self.stop_reason = stop_reason
        self.session_id = session_id
        self.hang = hang
        self.hang_before_prompt = hang_before_prompt
        self.protocol_error = protocol_error
        self.protocol_error_after_prompt = protocol_error_after_prompt
        self.model = model
        self.client = None
        self.prompt_cancelled = False

    async def initialize(self, protocol_version, client_capabilities=None, **kwargs):
        if self.protocol_error is not None:
            raise self.protocol_error
        return SimpleNamespace(protocol_version=protocol_version)

    async def new_session(self, cwd, **kwargs):
        if self.hang_before_prompt:
            await asyncio.Event().wait()
        return SimpleNamespace(
            session_id=self.session_id,
            config_options=[
                SessionConfigOptionSelect(
                    type="select",
                    id="model",
                    name="Model",
                    description=None,
                    category="model",
                    current_value=self.model,
                    options=[
                        SessionConfigSelectOption(
                            value=self.model,
                            name=self.model,
                            description=None,
                        )
                    ],
                )
            ],
        )

    async def set_config_option(self, config_id, session_id, value, **kwargs):
        self.model = value
        return SimpleNamespace(
            config_options=[
                SessionConfigOptionSelect(
                    type="select",
                    id="model",
                    name="Model",
                    description=None,
                    category="model",
                    current_value=value,
                    options=[
                        SessionConfigSelectOption(
                            value=value,
                            name=value,
                            description=None,
                        )
                    ],
                )
            ]
        )

    async def prompt(self, session_id, prompt, **kwargs):
        if self.hang:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.prompt_cancelled = True
                raise
        if self.protocol_error_after_prompt is not None:
            raise self.protocol_error_after_prompt
        for text in self.texts:
            chunk = AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text=text),
            )
            await self.client.session_update(session_id, chunk)
        return SimpleNamespace(stop_reason=self.stop_reason)


# --- _spawn ------------------------------------------------------------


def _spawn(conn, state):
    @asynccontextmanager
    async def _ctx(to_client, command, *args, **kwargs):
        conn.client = to_client(conn) if callable(to_client) else to_client
        try:
            yield conn, SimpleNamespace(pid=1)
        finally:
            state["cleaned"] = True

    return _ctx


# --- _run --------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


# A. permission reject for READ_ONLY / PROPOSE_PATCH across all kinds
@pytest.mark.parametrize("autonomy", [Autonomy.READ_ONLY, Autonomy.PROPOSE_PATCH])
@pytest.mark.parametrize(
    "kind",
    [
        "read",
        "edit",
        "delete",
        "move",
        "search",
        "execute",
        "think",
        "fetch",
        "switch_mode",
        "other",
        None,
    ],
)
def test_permission_reject_non_edit(autonomy, kind):
    client = _OneShotClient(autonomy)
    resp = _run(
        client.request_permission(
            "s",
            _call(kind),
            [
                _opt("allow", "allow_once"),
                _opt("reject", "reject_once"),
            ],
        )
    )
    _assert_selected(resp, "reject")


# B. EDIT_LOCAL: non-edit kinds → reject
@pytest.mark.parametrize(
    "kind",
    ["read", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other", None],
)
def test_edit_local_non_edit_kinds_reject(kind):
    client = _OneShotClient(Autonomy.EDIT_LOCAL)
    resp = _run(
        client.request_permission(
            "s",
            _call(kind),
            [
                _opt("allow", "allow_once"),
                _opt("reject", "reject_once"),
            ],
        )
    )
    _assert_selected(resp, "reject")


def test_edit_local_edit_allows():
    client = _OneShotClient(Autonomy.EDIT_LOCAL)
    resp = _run(
        client.request_permission(
            "s",
            _call("edit"),
            [
                _opt("allow", "allow_once"),
                _opt("reject", "reject_once"),
            ],
        )
    )
    _assert_selected(resp, "allow")


def test_edit_local_edit_allow_always_and_reject():
    client = _OneShotClient(Autonomy.EDIT_LOCAL)
    resp = _run(
        client.request_permission(
            "s",
            _call("edit"),
            [
                _opt("allow_always", "allow_always"),
                _opt("reject", "reject_once"),
            ],
        )
    )
    _assert_selected(resp, "reject")


def test_edit_local_deny_no_reject_option():
    client = _OneShotClient(Autonomy.EDIT_LOCAL)
    resp = _run(
        client.request_permission(
            "s",
            _call("read"),
            [
                _opt("allow", "allow_once"),
                _opt("always", "allow_always"),
            ],
        )
    )
    _assert_cancelled(resp)


def test_edit_local_read_titled_edit_file():
    client = _OneShotClient(Autonomy.EDIT_LOCAL)
    resp = _run(
        client.request_permission(
            "s",
            _call("read", title="Edit file"),
            [
                _opt("allow", "allow_once"),
                _opt("reject", "reject_once"),
            ],
        )
    )
    _assert_selected(resp, "reject")


# C. successful run
def test_run_acp_prompt_success():
    conn = _Conn(texts=["hello ", "world"], stop_reason="end_turn", session_id="session-42")
    state = {}
    deltas: list[str] = []
    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        _spawn(conn, state),
    ):
        result = _run(
            run_acp_prompt(
                ["fake"],
                "safe",
                "/tmp",
                autonomy=Autonomy.EDIT_LOCAL,
                model="test-model",
                on_text_delta=deltas.append,
            )
        )
    assert isinstance(result, AcpResult)
    assert result.output == "hello world"
    assert result.stop_reason == "end_turn"
    assert result.session_id == "session-42"
    assert deltas == ["hello ", "world"]
    assert state.get("cleaned") is True


# D. invalid autonomy before spawn
def test_invalid_autonomy_no_spawn():
    secret = "TOP-SECRET"
    spawn_mock = mock.MagicMock()
    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        spawn_mock,
    ):
        with pytest.raises(AcpProtocolError) as exc:
            _run(
                run_acp_prompt(
                    ["fake"],
                    secret,
                    "/tmp",
                    autonomy="invalid",
                    model="test-model",
                )
            )
    assert secret not in str(exc.value)
    spawn_mock.assert_not_called()
    # Never spawned a process — this is a prompt-delivery-stage failure,
    # not a generic "protocol" bucket.
    assert exc.value.stage == "prompt_delivery"


# E. launch error — FileNotFoundError mapping
def test_launch_error_maps():
    secret = "TOP-SECRET"

    @asynccontextmanager
    async def _missing(*args, **kwargs):
        raise FileNotFoundError("missing")
        yield

    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        _missing,
    ):
        with pytest.raises(AcpLaunchError) as exc:
            _run(
                run_acp_prompt(
                    ["fake"],
                    secret,
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="test-model",
                )
            )
    assert secret not in str(exc.value)


# F. protocol error maps
def test_protocol_error_maps():
    secret = "TOP-SECRET"
    state = {}
    conn = _Conn(protocol_error=RuntimeError("handshake failed"))
    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        _spawn(conn, state),
    ):
        with pytest.raises(AcpProtocolError) as exc:
            _run(
                run_acp_prompt(
                    ["fake"],
                    secret,
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="test-model",
                )
            )
    assert "handshake failed" in str(exc.value)
    assert secret not in str(exc.value)
    assert state.get("cleaned") is True
    # Handshake (initialize) fails before the prompt is ever dispatched.
    assert exc.value.stage == "prompt_delivery"


# F2. protocol error raised after the prompt was already dispatched — a
# later-provider failure must classify as execution, not prompt_delivery.
def test_protocol_error_after_prompt_dispatch_marks_execution_stage():
    secret = "TOP-SECRET"
    state = {}
    conn = _Conn(protocol_error_after_prompt=RuntimeError("stream corrupted"))
    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        _spawn(conn, state),
    ):
        with pytest.raises(AcpProtocolError) as exc:
            _run(
                run_acp_prompt(
                    ["fake"],
                    secret,
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="test-model",
                )
            )
    assert secret not in str(exc.value)
    assert exc.value.stage == "execution"


# G. timeout cleanup
def test_timeout_cleanup():
    secret = "TOP-SECRET"
    state = {}
    conn = _Conn(hang=True)
    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        _spawn(conn, state),
    ):
        with pytest.raises(AcpTimeoutError) as exc:
            _run(
                run_acp_prompt(
                    ["fake"],
                    secret,
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    timeout=0.01,
                    model="test-model",
                )
            )
    assert secret not in str(exc.value)
    assert state.get("cleaned") is True
    assert conn.prompt_cancelled is True
    # Prompt was already dispatched (hang happens inside prompt()) — the
    # timeout is a legitimate execution-stage timeout, not a delivery failure.
    assert exc.value.stage == "execution"


# H. timeout before the prompt was ever dispatched must be diagnosable
def test_timeout_before_prompt_sent_marks_prompt_delivery_stage():
    """A timeout during initialize/session/new must not collapse into an
    undifferentiated execution timeout — the prompt was never delivered.
    """
    secret = "TOP-SECRET"
    state = {}
    conn = _Conn(hang_before_prompt=True)
    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        _spawn(conn, state),
    ):
        with pytest.raises(AcpTimeoutError) as exc:
            _run(
                run_acp_prompt(
                    ["fake"],
                    secret,
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    timeout=0.01,
                    model="test-model",
                )
            )
    assert secret not in str(exc.value)
    assert state.get("cleaned") is True
    assert exc.value.stage == "prompt_delivery"


# H. frozen AcpResult
def test_acp_result_frozen():
    result = AcpResult(output="test", stop_reason="end_turn", session_id="s1")
    with pytest.raises(Exception):
        result.output = "mutated"


# I. model selection via config_option
# ---------------------------------------------------------------------------


def _config_with_category(
    option_id="model",
    category="model",
    values=None,
    groups=None,
    current_value="",
):
    """Build a SessionConfigOptionSelect for testing."""
    opts = [SessionConfigSelectOption(value=v, name=v) for v in (values or [])] if values else []
    grps = (
        [
            SessionConfigSelectGroup(
                group=g["id"],
                name=g["name"],
                options=[SessionConfigSelectOption(value=v, name=v) for v in g["options"]],
            )
            for g in groups
        ]
        if groups
        else []
    )
    return SessionConfigOptionSelect(
        id=option_id,
        name=option_id,
        category=category,
        type="select",
        current_value=current_value,
        options=opts or grps,
    )


class _ConnWithConfig:
    """A _Conn that also returns config_options and tracks set_config_option.

    Parameters
    ----------
    set_config_response_options : list or None, optional
        When set, returned directly as the ``config_options`` in the
        ``SetSessionConfigOptionResponse``.  When *None* (default) the
        response is built from ``config_options`` with the matching option's
        ``current_value`` patched to the requested value — this is the
        expected happy-path behaviour that an ACP agent confirms the value.
    """

    def __init__(
        self,
        config_options=None,
        texts=None,
        stop_reason="end_turn",
        session_id="session-1",
        hang=False,
        protocol_error=None,
        set_config_error=None,
        set_config_response_options=None,
    ):
        self.texts = texts or []
        self.stop_reason = stop_reason
        self.session_id = session_id
        self.hang = hang
        self.protocol_error = protocol_error
        self.set_config_error = set_config_error
        self.client = None
        self._config_options = config_options
        self._set_config_response_options = set_config_response_options
        self.set_config_option_calls = []

    async def initialize(self, protocol_version, client_capabilities=None, **kwargs):
        if self.protocol_error is not None:
            raise self.protocol_error
        return SimpleNamespace(protocol_version=protocol_version)

    async def new_session(self, cwd, **kwargs):
        return SimpleNamespace(
            session_id=self.session_id,
            config_options=self._config_options,
        )

    async def set_config_option(self, config_id, session_id, value, **kwargs):
        from acp.schema import (
            SessionConfigOptionSelect,
            SetSessionConfigOptionResponse,
        )

        self.set_config_option_calls.append((config_id, session_id, value))
        if self.set_config_error:
            raise self.set_config_error

        # Override path — caller controls the exact response (e.g. for error testing)
        if self._set_config_response_options is not None:
            return SetSessionConfigOptionResponse(config_options=self._set_config_response_options)

        # Happy path — patch the matching option's current_value
        patched: list[Any] = []
        for opt in self._config_options or []:
            if isinstance(opt, SessionConfigOptionSelect) and opt.id == config_id:
                patched.append(opt.model_copy(update={"current_value": value}))
            else:
                patched.append(opt)
        return SetSessionConfigOptionResponse(config_options=patched)

    async def prompt(self, session_id, prompt, **kwargs):
        if self.hang:
            await asyncio.Event().wait()
        for text in self.texts:
            chunk = AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text=text),
            )
            await self.client.session_update(session_id, chunk)
        return SimpleNamespace(stop_reason=self.stop_reason)


def _spawn_with_config(conn):
    @asynccontextmanager
    async def _ctx(to_client, command, *args, **kwargs):
        conn.client = to_client(conn) if callable(to_client) else to_client
        try:
            yield conn, SimpleNamespace(pid=1)
        finally:
            pass

    return _ctx


class TestRunAcpPromptModelSelection:
    """TDD RED: tests for model config-option selection in run_acp_prompt."""

    def test_model_none_is_rejected(self):
        """Internal ACP calls cannot bypass the required-model contract."""
        model_opt = _config_with_category(values=["opencode-go/deepseek-v4-flash"])
        conn = _ConnWithConfig(config_options=[model_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError, match="Requested model"):
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "hello",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model=None,  # type: ignore[arg-type]
                    )
                )

    def test_model_calls_set_config_option_via_category(self):
        """Finds config option by category=='model' and calls set_config_option."""
        model_opt = _config_with_category(
            option_id="model",
            category="model",
            values=["opencode-go/deepseek-v4-flash", "opencode-go/deepseek-v4-pro"],
        )
        # Also add a non-model option to test filtering
        other_opt = _config_with_category(
            option_id="effort",
            category="effort",
            values=["low", "medium", "high"],
        )
        conn = _ConnWithConfig(config_options=[other_opt, model_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            result = _run(
                run_acp_prompt(
                    ["fake"],
                    "hello",
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="opencode-go/deepseek-v4-flash",
                )
            )
        assert isinstance(result, AcpResult)
        assert len(conn.set_config_option_calls) == 1
        config_id, sess_id, value = conn.set_config_option_calls[0]
        assert config_id == "model"
        assert sess_id == "session-1"
        assert value == "opencode-go/deepseek-v4-flash"

    def test_model_fallback_to_id_when_category_missing(self):
        """Falls back to id=='model' when no config option has category=='model'."""
        model_opt = _config_with_category(
            option_id="model",
            category="",  # no category
            values=["gpt-5", "gpt-4"],
        )
        conn = _ConnWithConfig(config_options=[model_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            result = _run(
                run_acp_prompt(
                    ["fake"],
                    "hello",
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="gpt-5",
                )
            )
        assert isinstance(result, AcpResult)
        assert len(conn.set_config_option_calls) == 1
        assert conn.set_config_option_calls[0][0] == "model"

    def test_model_value_in_grouped_options(self):
        """Finds value among grouped options inside a SessionConfigSelectGroup."""
        model_opt = _config_with_category(
            category="model",
            groups=[
                {"id": "fast", "name": "Fast models", "options": ["opencode-go/deepseek-v4-flash"]},
                {"id": "pro", "name": "Pro models", "options": ["opencode-go/deepseek-v4-pro"]},
            ],
        )
        conn = _ConnWithConfig(config_options=[model_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            result = _run(
                run_acp_prompt(
                    ["fake"],
                    "hello",
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="opencode-go/deepseek-v4-pro",
                )
            )
        assert isinstance(result, AcpResult)
        assert len(conn.set_config_option_calls) == 1
        assert conn.set_config_option_calls[0][2] == "opencode-go/deepseek-v4-pro"

    def test_effort_uses_thought_level_config_option_after_model_selection(self):
        """OpenCode-style effort is set through ACP, after model selection."""
        model_opt = _config_with_category(
            option_id="model",
            category="model",
            values=["opencode-go/deepseek-v4-flash"],
        )
        effort_opt = _config_with_category(
            option_id="effort",
            category="thought_level",
            values=["low", "high"],
            current_value="low",
        )
        conn = _ConnWithConfig(config_options=[model_opt, effort_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            result = _run(
                run_acp_prompt(
                    ["fake"],
                    "hello",
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="opencode-go/deepseek-v4-flash",
                    effort="high",
                )
            )
        assert isinstance(result, AcpResult)
        assert conn.set_config_option_calls == [
            ("model", "session-1", "opencode-go/deepseek-v4-flash"),
            ("effort", "session-1", "high"),
        ]

    def test_codex_reasoning_effort_uses_thought_level_config_option(self):
        """Codex ACP exposes `reasoning_effort`, not OpenCode's `effort` id."""
        model_opt = _config_with_category(
            option_id="model",
            category="model",
            values=["gpt-5.6-sol"],
        )
        reasoning_opt = _config_with_category(
            option_id="reasoning_effort",
            category="thought_level",
            values=["low", "medium", "high", "max"],
            current_value="high",
        )
        conn = _ConnWithConfig(config_options=[model_opt, reasoning_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            result = _run(
                run_acp_prompt(
                    ["fake"],
                    "hello",
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="gpt-5.6-sol",
                    effort="max",
                )
            )
        assert isinstance(result, AcpResult)
        assert conn.set_config_option_calls == [
            ("model", "session-1", "gpt-5.6-sol"),
            ("reasoning_effort", "session-1", "max"),
        ]

    def test_effort_without_advertised_selector_is_rejected_before_prompt(self):
        """Explicit effort must never be silently ignored by an ACP agent."""
        model_opt = _config_with_category(
            category="model",
            values=["opencode-go/deepseek-v4-flash"],
        )
        conn = _ConnWithConfig(config_options=[model_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError, match="No effort config option available") as exc:
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "safe",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model="opencode-go/deepseek-v4-flash",
                        effort="high",
                    )
                )
        assert exc.value.stage == "prompt_delivery"

    def test_model_option_absent_raises_protocol_error(self):
        """No config option with category==model or id==model -> AcpProtocolError."""
        conn = _ConnWithConfig(config_options=[])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError, match="No model config option available") as exc:
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "secret",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model="gpt-5",
                    )
                )
        assert exc.value.stage == "prompt_delivery"

    def test_model_value_not_available_raises_protocol_error(self):
        """Requested model not in available options -> AcpProtocolError."""
        model_opt = _config_with_category(
            values=["opencode-go/deepseek-v4-flash"],
        )
        conn = _ConnWithConfig(config_options=[model_opt])
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError, match="Requested model.*not available") as exc:
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "secret",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model="nonexistent-model",
                    )
                )
        assert exc.value.stage == "prompt_delivery"

    def test_set_config_option_failure_raises_protocol_error(self):
        """set_config_option raising an error -> AcpProtocolError."""
        model_opt = _config_with_category(
            values=["opencode-go/deepseek-v4-flash"],
        )
        conn = _ConnWithConfig(
            config_options=[model_opt],
            set_config_error=RuntimeError("connection lost"),
        )
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError, match="Failed to set model config") as exc:
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "secret",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model="opencode-go/deepseek-v4-flash",
                    )
                )
        assert exc.value.stage == "prompt_delivery"

    # ── Response validation: set_config_option response must confirm model ──
    # (RED tests for Issue 1 & 2)

    def test_set_config_option_response_mismatched_current_value_raises(self):
        """Response with current_value != requested model → AcpProtocolError before prompt."""
        model_opt = _config_with_category(
            values=["opencode-go/deepseek-v4-flash"],
            current_value="",
        )
        conn = _ConnWithConfig(
            config_options=[model_opt],
            # Return the unpatched option so current_value stays ""
            set_config_response_options=[model_opt],
        )
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError, match="Agent rejected model") as exc:
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "safe",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model="opencode-go/deepseek-v4-flash",
                    )
                )
        assert exc.value.stage == "prompt_delivery"

    def test_set_config_option_response_no_model_option_raises(self):
        """Response without model option → AcpProtocolError before prompt."""
        model_opt = _config_with_category(
            values=["opencode-go/deepseek-v4-flash"],
        )
        conn = _ConnWithConfig(
            config_options=[model_opt],
            set_config_response_options=[],  # empty — no model option in response
        )
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError, match="Agent rejected model") as exc:
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "safe",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model="opencode-go/deepseek-v4-flash",
                    )
                )
        assert exc.value.stage == "prompt_delivery"

    def test_set_config_option_exception_does_not_leak_secret(self):
        """Exception text from set_config_option is not in AcpProtocolError message.

        The original exception is preserved as __cause__ but the message is
        stable — no credentials / secrets are interpolated.
        """
        model_opt = _config_with_category(
            values=["opencode-go/deepseek-v4-flash"],
        )
        secret = "sk-proj-SECRET-KEY-12345"
        conn = _ConnWithConfig(
            config_options=[model_opt],
            set_config_error=RuntimeError(secret),
        )
        with mock.patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _spawn_with_config(conn),
        ):
            with pytest.raises(AcpProtocolError) as exc_info:
                _run(
                    run_acp_prompt(
                        ["fake"],
                        "safe",
                        "/tmp",
                        autonomy=Autonomy.EDIT_LOCAL,
                        model="opencode-go/deepseek-v4-flash",
                    )
                )
        # The secret MUST NOT appear in the public error message
        assert secret not in str(exc_info.value)
        # The original exception is preserved as __cause__
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert secret in str(exc_info.value.__cause__)


# J. model and effort forwarded through run_acp_job
class TestRunAcpJobForwardsConfiguration:
    def test_model_and_effort_passed_to_run_acp_prompt(self, tmp_path):
        """run_acp_job forwards requested ACP session configuration."""
        from agent_crossbar.acp_runtime import run_acp_job
        from agent_crossbar.jobs import JobStore

        store = JobStore(tmp_path)
        job = store.create_job(
            profile="opencode", operation="dev", transport="print", cwd=str(tmp_path)
        )

        with mock.patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(
                return_value=AcpResult(output="ok", stop_reason="end_turn", session_id="s1")
            ),
        ) as mock_run:
            asyncio.run(
                run_acp_job(
                    store,
                    job.job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="opencode-go/deepseek-v4-flash",
                    effort="high",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=30,
                )
            )

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "opencode-go/deepseek-v4-flash"
        assert kwargs["effort"] == "high"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("429 Too Many Requests", "provider_limit_exhausted"),
        ("Weekly usage limit reached", "provider_limit_exhausted"),
        ("insufficient_quota", "provider_limit_exhausted"),
        ("Internal error: No provider available", "provider_unavailable"),
    ],
)
def test_provider_failure_classification_is_actionable_and_sanitized(text, code):
    classified = classify_provider_failure(text)
    assert classified is not None
    assert classified[0] == code
    assert text not in classified[1]


def test_provider_failure_exception_carries_machine_readable_code():
    exc = AcpProviderUnavailableError(
        "provider_limit_exhausted",
        "The selected provider has exhausted its quota or rate limit",
    )
    assert exc.code == "provider_limit_exhausted"
    assert exc.stage == "prompt_delivery"


def test_stderr_quota_failure_interrupts_hung_acp_startup():
    conn = _Conn(hang_before_prompt=True)

    class _QuotaStderr:
        async def readline(self):
            return b"429 weekly usage limit reached\n"

    stderr = _QuotaStderr()

    @asynccontextmanager
    async def _spawn_with_quota(to_client, command, *args, **kwargs):
        conn.client = to_client(conn) if callable(to_client) else to_client
        yield conn, SimpleNamespace(pid=123, stderr=stderr)

    with mock.patch(
        "agent_crossbar.acp_client.spawn_agent_process",
        _spawn_with_quota,
    ):
        with pytest.raises(AcpProviderUnavailableError) as exc:
            _run(
                run_acp_prompt(
                    ["fake"],
                    "safe",
                    "/tmp",
                    autonomy=Autonomy.EDIT_LOCAL,
                    model="test-model",
                    startup_timeout=5,
                )
            )
    assert exc.value.code == "provider_limit_exhausted"
    assert exc.value.stage == "prompt_delivery"
