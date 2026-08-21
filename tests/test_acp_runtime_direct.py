"""Tests for acp_runtime — build_acp_agent_command and run_acp_job.

TDD RED step — run_acp_job does not exist yet, imports will fail.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch

import pytest
from acp.schema import (
    AgentMessageChunk,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    TextContentBlock,
)

# ── production target ────────────────────────────────────────────────
from agent_crossbar.acp_client import (
    AcpError,
    AcpLaunchError,
    AcpProtocolError,
    AcpProviderUnavailableError,
    AcpResult,
    AcpTimeoutError,
)
from agent_crossbar.acp_runtime import build_acp_agent_command, run_acp_job
from agent_crossbar.envelope import FAILURE_STAGES
from agent_crossbar.jobs import JobStore
from agent_crossbar.models import Autonomy

# ── helpers ──────────────────────────────────────────────────────────

SECRET = "ssh-ed25519 AAA... bogus key"


def test_run_acp_job_requires_explicit_model():
    model = inspect.signature(run_acp_job).parameters["model"]
    assert model.default is inspect.Parameter.empty


def _create_job_store(tmp_path) -> tuple[JobStore, str]:
    """Return (store, job_id) with a fresh job already stored."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode",
        operation="dev",
        transport="print",
        sensitivity="normal",
        cwd=str(tmp_path),
    )
    return store, job.job_id


def _read_store_events(store: JobStore, job_id: str) -> list[dict]:
    events_file = store.get_job(job_id).path / "events.jsonl"
    if not events_file.exists():
        return []
    return [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]


def _read_store_meta(store: JobStore, job_id: str) -> dict:
    meta_file = store.get_job(job_id).path / "meta.json"
    return json.loads(meta_file.read_text())


class _ChunkingAcpConnection:
    """Protocol-shaped fake that emits real AgentMessageChunk notifications."""

    def __init__(self, texts: list[str], model: str) -> None:
        self.texts = texts
        self.model = model
        self.client = None

    async def initialize(self, protocol_version, **_kwargs):
        return SimpleNamespace(protocol_version=protocol_version)

    async def new_session(self, **_kwargs):
        return SimpleNamespace(
            session_id="protocol-fake-1",
            config_options=[self._model_config(self.model)],
        )

    async def set_config_option(self, value, **_kwargs):
        self.model = value
        return SimpleNamespace(config_options=[self._model_config(value)])

    async def prompt(self, session_id, **_kwargs):
        assert self.client is not None
        for text in self.texts:
            await self.client.session_update(
                session_id,
                AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text=text),
                ),
            )
        return SimpleNamespace(stop_reason="end_turn")

    @staticmethod
    def _model_config(model: str) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            type="select",
            id="model",
            name="Model",
            description=None,
            category="model",
            current_value=model,
            options=[SessionConfigSelectOption(value=model, name=model, description=None)],
        )


def _chunking_acp_spawn(connection: _ChunkingAcpConnection):
    @asynccontextmanager
    async def _spawn(client, *_args, **_kwargs):
        connection.client = client
        yield connection, SimpleNamespace(pid=17)

    return _spawn


# ── command builder ──────────────────────────────────────────────────


class TestBuildAcpAgentCommand:
    def test_opencode(self):
        assert build_acp_agent_command("opencode") == ["opencode", "acp"]

    def test_codex(self):
        assert build_acp_agent_command("codex") == [
            "pnpm",
            "dlx",
            "@agentclientprotocol/codex-acp@1.1.7",
        ]

    def test_unknown_provider_raises_valueerror(self):
        with pytest.raises(ValueError):
            build_acp_agent_command("claude-code")


# ── run_acp_job ──────────────────────────────────────────────────────


class TestRunAcpJobSuccess:
    def test_happy_path(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        acp_result = AcpResult(
            output="done",
            stop_reason="end_turn",
            session_id="native-1",
        )

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ) as mock_run:
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        # ── assert run_acp_prompt called correctly (model forwarded) ──
        mock_run.assert_awaited_once_with(
            ["opencode", "acp"],
            SECRET,
            str(tmp_path),
            timeout=12,
            autonomy=Autonomy.EDIT_LOCAL,
            model="glm",
            effort=None,
            on_process_start=ANY,
            on_text_delta=ANY,
            on_execution_heartbeat=ANY,
        )

        # ── assert store result ──
        stored = store.get_result(job_id)
        assert stored["ok"] is True
        assert stored["status"] == "completed"
        assert stored["summary"] == "done"
        assert stored["output"] == "done"
        assert stored["stop_reason"] == "end_turn"
        assert stored["resolved"]["backend"] == "acp"
        assert stored["technical"]["native_session_id"] == "native-1"

        # ── secret absent from events and meta ──
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text

    def test_execution_heartbeat_is_persisted_without_native_working_claim(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        async def fake_run(*_args, **kwargs):
            kwargs["on_execution_heartbeat"](
                {
                    "process_alive": True,
                    "prompt_active": True,
                    "elapsed_sec": 31,
                    "last_protocol_activity_sec": 31,
                }
            )
            return AcpResult(output="done", stop_reason="end_turn", session_id="native-1")

        with patch("agent_crossbar.acp_runtime.run_acp_prompt", new=fake_run):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="safe",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    max_runtime_sec=30,
                )
            )

        heartbeat = [
            event
            for event in _read_store_events(store, job_id)
            if event["type"] == "execution_heartbeat"
        ]
        assert len(heartbeat) == 1
        assert heartbeat[0]["data"]["process_alive"] is True
        assert "native_state" not in heartbeat[0]["data"]


class TestRunAcpJobTimeout:
    def test_timeout_sets_failure_status(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=AcpTimeoutError("ACP prompt timed out after 1.0s")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["stop_reason"] == "timeout"
        assert stored["failure"]["code"] == "acp_timeout"
        assert stored["failure"]["retryable"] is True
        assert stored["failure"]["next_action"] == "check_provider_limits_or_retry_with_free_model"
        assert stored["failure"]["diagnostics"]["max_runtime_sec"] == 12
        assert "quota" in stored["output"].lower()
        assert "free model" in stored["output"].lower()

        # ── secret absent everywhere ──
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text
        if "error" in stored:
            assert SECRET not in json.dumps(stored["error"])
        if "failure" in stored:
            assert SECRET not in json.dumps(stored["failure"])


class TestRunAcpJobPromptDeliveryTimeout:
    def test_timeout_before_prompt_delivery_is_not_generic_timeout(self, tmp_path):
        """A prompt-delivery-stage timeout must not collapse into the
        generic acp_timeout/execution classification — it is diagnosable
        as a distinct stage with its own code and next action.
        """
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(
                side_effect=AcpTimeoutError(
                    "ACP prompt timed out after 12.0s", stage="prompt_delivery"
                )
            ),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["stage"] == "prompt_delivery"
        assert stored["failure"]["code"] != "acp_timeout"
        assert stored["failure"]["retryable"] is True
        assert stored["failure"]["next_action"]

        # ── secret absent everywhere ──
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text


class TestRunAcpJobLaunchError:
    def test_missing_provider_binary(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=AcpLaunchError("Provider binary not found: opencode")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["stage"] == "launch"
        assert stored["failure"]["code"] == "acp_launch_error"
        assert (
            "install" in stored["failure"]["next_action"].lower()
            or "check" in stored["failure"]["next_action"].lower()
        )

        # prompt absent
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text


class TestRunAcpJobProtocolError:
    def test_handshake_failure(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(
                side_effect=AcpProtocolError("handshake failed", stage="prompt_delivery")
            ),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["code"] == "acp_protocol_error"
        assert "handshake" in stored["failure"]["diagnostics"]["error"].lower()
        # The exception's own stage must be forwarded, not collapsed into a
        # hardcoded (and out-of-taxonomy) "protocol" bucket.
        assert stored["failure"]["stage"] == "prompt_delivery"
        assert stored["failure"]["stage"] in FAILURE_STAGES

        # prompt absent
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text

    def test_protocol_failure_after_prompt_dispatch_keeps_execution_stage(self, tmp_path):
        """A protocol error raised after the prompt was already sent to the
        agent is a later-provider failure — it must classify as
        execution, not collapse to the same bucket as a pre-dispatch
        handshake failure.
        """
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=AcpProtocolError("stream corrupted", stage="execution")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["failure"]["stage"] == "execution"
        assert stored["failure"]["stage"] in FAILURE_STAGES


class TestRunAcpJobInvalidAutonomyStage:
    def test_invalid_autonomy_is_preflight_not_protocol(self, tmp_path):
        """Invalid autonomy is rejected before any provider interaction —
        it belongs in the 'preflight' bucket, not the removed 'protocol'
        bucket (which was never one of the six allowed stages)."""
        store, job_id = _create_job_store(tmp_path)

        asyncio.run(
            run_acp_job(
                store,
                job_id,
                provider="opencode",
                prompt=SECRET,
                cwd=str(tmp_path),
                task="dev",
                model="glm",
                effort=None,
                autonomy="not-a-real-autonomy-value",
                max_runtime_sec=12,
            )
        )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["stage"] == "preflight"
        assert stored["failure"]["stage"] in FAILURE_STAGES


class TestSafeErrorRedaction:
    """_safe_error must redact secrets beyond just the raw prompt text —
    a provider exception can embed live credentials from its own
    environment or stderr, not only the prompt we sent it."""

    def test_bearer_token_in_exception_message_is_redacted(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        leaked = "sk-live-should-not-leak-1234567890"

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(
                side_effect=Exception(f"upstream call failed: Authorization: Bearer {leaked}")
            ),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert leaked not in stored["summary"]
        assert leaked not in stored["output"]
        assert leaked not in json.dumps(stored["failure"])

    def test_key_value_secret_in_exception_message_is_redacted(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        leaked = "abcDEF1234567890"

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=Exception(f"config error: OPENAI_API_KEY={leaked}")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert leaked not in stored["summary"]
        assert leaked not in stored["output"]
        assert leaked not in json.dumps(stored["failure"])


class TestRunAcpJobEvents:
    def test_protocol_shaped_acp_chunks_reach_job_tail(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        connection = _ChunkingAcpConnection(["streamed ", "text"], model="glm")

        with patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _chunking_acp_spawn(connection),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        events = _read_store_events(store, job_id)
        assert [event["data"]["text"] for event in events if event["type"] == "log_delta"] == [
            "streamed ",
            "text",
        ]
        assert store.get_result(job_id)["output"] == "streamed text"

    def test_success_events_stream_each_acp_text_delta(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        async def _streaming_acp_prompt(*_args, **kwargs):
            on_text_delta = kwargs["on_text_delta"]
            on_text_delta("first ")
            on_text_delta("second")
            return AcpResult(
                output="first second",
                stop_reason="end_turn",
                session_id="native-1",
            )

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=_streaming_acp_prompt),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        events = _read_store_events(store, job_id)
        acp_events = [
            event
            for event in events
            if event["type"] in {"acp_command", "log_delta", "acp_completed"}
        ]
        assert [event["type"] for event in acp_events] == [
            "acp_command",
            "log_delta",
            "log_delta",
            "acp_completed",
        ]
        assert [event["data"]["text"] for event in acp_events if event["type"] == "log_delta"] == [
            "first ",
            "second",
        ]

    def test_success_events_contain_acp_types(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        acp_result = AcpResult(
            output="done",
            stop_reason="end_turn",
            session_id="native-1",
        )

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        events = _read_store_events(store, job_id)
        event_types = {e["type"] for e in events}
        assert "acp_command" in event_types
        assert "acp_completed" in event_types

        # no "acpx" strings anywhere in events
        for event in events:
            json_str = json.dumps(event)
            assert "acpx" not in json_str


# ── Exhaustive stage-taxonomy regression (review fix #1) ──────────────
#
# Every failure branch of run_acp_job must emit one of the six allowed
# envelope failure stages. This is more than a per-branch spot check: it
# drives EVERY branch through the real run_acp_job and asserts each
# resulting stage is in FAILURE_STAGES — and explicitly rejects the old
# out-of-taxonomy "protocol" bucket. build_result_envelope also enforces
# this at runtime (raises ValueError on an unknown stage), so this test
# additionally proves no branch tries to publish a value outside the six.


def _run_job_and_get_failure_stage(tmp_path, *, autonomy, side_effect) -> str:
    store, job_id = _create_job_store(tmp_path)
    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(side_effect=side_effect),
    ):
        asyncio.run(
            run_acp_job(
                store,
                job_id,
                provider="opencode",
                prompt=SECRET,
                cwd=str(tmp_path),
                task="dev",
                model="glm",
                effort=None,
                autonomy=autonomy,
                max_runtime_sec=12,
            )
        )
    stored = store.get_result(job_id)
    assert stored["status"] == "failed"
    return stored["failure"]["stage"]


class TestExhaustiveFailureStageTaxonomy:
    @pytest.mark.parametrize(
        ("label", "autonomy", "side_effect"),
        [
            ("invalid_autonomy", "not-a-real-value", None),
            ("timeout_execution_default", Autonomy.EDIT_LOCAL, AcpTimeoutError("timed out")),
            (
                "timeout_prompt_delivery",
                Autonomy.EDIT_LOCAL,
                AcpTimeoutError("timed out", stage="prompt_delivery"),
            ),
            (
                "launch_error",
                Autonomy.EDIT_LOCAL,
                AcpLaunchError("binary not found"),
            ),
            (
                "protocol_error_prompt_delivery",
                Autonomy.EDIT_LOCAL,
                AcpProtocolError("handshake failed", stage="prompt_delivery"),
            ),
            (
                "protocol_error_execution",
                Autonomy.EDIT_LOCAL,
                AcpProtocolError("stream corrupted", stage="execution"),
            ),
            ("generic_acp_error", Autonomy.EDIT_LOCAL, AcpError("unexpected acp failure")),
            ("generic_exception", Autonomy.EDIT_LOCAL, RuntimeError("boom")),
        ],
    )
    def test_every_failure_branch_emits_an_allowed_stage(
        self, tmp_path, label, autonomy, side_effect
    ):
        stage = _run_job_and_get_failure_stage(tmp_path, autonomy=autonomy, side_effect=side_effect)
        assert stage in FAILURE_STAGES, (
            f"{label}: stage {stage!r} is outside the six allowed failure stages "
            f"{sorted(FAILURE_STAGES)}"
        )
        # The specific bug reported in review: the old code emitted this
        # literal value, which was never one of the six allowed stages.
        assert stage != "protocol", f"{label}: regressed to the removed 'protocol' stage"


# ── safe_acp_termination tests ──────────────────────────────────────────


def test_provider_limit_failure_persists_terminal_actionable_envelope(tmp_path):
    store, job_id = _create_job_store(tmp_path)
    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(
            side_effect=AcpProviderUnavailableError(
                "provider_limit_exhausted",
                "The selected provider has exhausted its quota or rate limit",
            )
        ),
    ):
        asyncio.run(
            run_acp_job(
                store,
                job_id,
                provider="opencode",
                prompt="safe",
                cwd=str(tmp_path),
                model="opencode-go/deepseek-v4-flash",
                max_runtime_sec=60,
            )
        )

    result = store.get_result(job_id)
    assert result is not None
    assert result["ok"] is False
    assert result["envelope"]["status"] == "failed"
    assert result["envelope"]["failure"]["code"] == "provider_limit_exhausted"
    assert result["envelope"]["failure"]["retryable"] is True


class TestSafeAcpTermination:
    def test_no_pid_in_meta(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        result = safe_acp_termination({})
        assert result["terminated"] is False
        assert result["reason"] == "no_acp_pid_in_meta"
        assert result["pid"] is None

    def test_invalid_pid_type(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        result = safe_acp_termination({"acp_pid": "not-a-number"})
        assert result["terminated"] is False
        assert "invalid_acp_pid" in result["reason"]
        assert result["pid"] is None

    def test_nonexistent_pid(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        result = safe_acp_termination({"acp_pid": 99999999})
        assert result["terminated"] is True
        assert result["reason"] == "process_already_gone"

    def test_idempotent_double_call(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        meta = {"acp_pid": 99999999}
        r1 = safe_acp_termination(meta)
        r2 = safe_acp_termination(meta)
        assert r1 == r2
        assert r1["terminated"] is True

    def test_importable_from_acp_runtime(self):
        """safe_acp_termination must be importable — no ImportError."""
        from agent_crossbar.acp_runtime import safe_acp_termination  # noqa: F811

        assert callable(safe_acp_termination)
