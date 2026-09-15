"""Tests for the provider-neutral result envelope builder."""

from __future__ import annotations

import pytest

from agent_crossbar.envelope import (
    DIAGNOSTICS_MAX_BYTES,
    FAILURE_STAGES,
    build_result_envelope,
    map_adapter_status,
    truncate_diagnostics,
)

# ── status mapping ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("adapter_status", "expected"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("waiting", "waiting"),
    ],
)
def test_map_adapter_status_passes_through_terminal_states(
    adapter_status: str, expected: str
) -> None:
    assert map_adapter_status(adapter_status) == expected


def test_map_adapter_status_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown adapter status"):
        map_adapter_status("running")


# ── diagnostics truncation ────────────────────────────────────────────────


def test_diagnostics_truncation_bounds_to_2kib() -> None:
    big = "x" * 5000
    result = truncate_diagnostics({"detail": big, "safe": "ok"})
    raw = result["detail"]
    assert isinstance(raw, str)
    assert len(raw.encode("utf-8")) <= DIAGNOSTICS_MAX_BYTES
    assert result["safe"] == "ok"


def test_diagnostics_truncation_strips_raw_environment() -> None:
    result = truncate_diagnostics({"env": {"SECRET_KEY": "abc", "PATH": "/usr/bin"}, "ok": True})
    assert "env" not in result
    assert result["ok"] is True


def test_diagnostics_truncation_strips_secrets_value() -> None:
    result = truncate_diagnostics(
        {"token": "sk-abc123", "config": {"api_key": "secret"}, "message": "hi"}
    )
    # secrets fields should be stripped
    assert "token" not in result
    # nested secrets
    assert "api_key" not in result.get("config", {})
    assert result["message"] == "hi"


def test_diagnostics_truncation_redacts_secret_values_embedded_in_raw_text() -> None:
    """Raw provider output can embed a live secret VALUE under an innocuous key.

    Key-based stripping alone misses this — the key is "output", not a
    secret-looking name — so the raw text must be scanned for secret
    patterns too, not just filtered by key name.
    """
    leaked = (
        "Connecting to provider...\n"
        "ANTHROPIC_API_KEY=sk-ant-oat01-abcdefghijklmnopqrstuvwxyz\n"
        "authentication failed"
    )
    result = truncate_diagnostics({"output": leaked})
    assert "sk-ant-oat01-abcdefghijklmnopqrstuvwxyz" not in result["output"]


# ── envelope builder ───────────────────────────────────────────────────────


def test_envelope_has_schema_version_1() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
    )
    assert env["schema_version"] == "1"


def test_envelope_includes_all_required_top_level_keys() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
    )
    required = {
        "schema_version",
        "status",
        "stop_reason",
        "output",
        "timing",
        "requested",
        "resolved",
        "process",
        "failure",
        "usage",
        "changes",
        "artifacts",
        "technical",
    }
    assert required.issubset(set(env.keys())), f"Missing keys: {required - set(env.keys())}"


def test_envelope_timing_includes_all_fields() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
        started_at="2025-01-01T00:00:01Z",
        finished_at="2025-01-01T00:01:00Z",
        queue_ms=500,
        run_ms=59000,
        total_ms=59500,
    )
    timing = env["timing"]
    assert timing["created_at"] == "2025-01-01T00:00:00Z"
    assert timing["started_at"] == "2025-01-01T00:00:01Z"
    assert timing["finished_at"] == "2025-01-01T00:01:00Z"
    assert timing["queue_ms"] == 500
    assert timing["run_ms"] == 59000
    assert timing["total_ms"] == 59500


def test_envelope_derives_total_ms_from_created_and_finished_timestamps() -> None:
    env = build_result_envelope(
        status="failed",
        stop_reason="timeout",
        output="timed out",
        created_at="2025-01-01T00:00:00Z",
        started_at="2025-01-01T00:00:01Z",
        finished_at="2025-01-01T00:00:16Z",
    )

    assert env["timing"]["queue_ms"] == 0
    assert env["timing"]["run_ms"] == 15000
    assert env["timing"]["total_ms"] == 16000


def test_envelope_requested_and_resolved_never_invent_unavailable() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
        requested={"profile": "claude", "task": "ask", "interactive": False},
        resolved={"profile": "claude", "task": "ask", "interactive": False, "backend": "claude_bg"},
    )
    requested = env["requested"]
    resolved = env["resolved"]
    # Fields not provided should be null, not invented
    assert requested.get("model") is None
    assert requested.get("effort") is None
    assert requested.get("cwd") is None
    assert resolved.get("model") is None
    assert resolved.get("effort") is None
    assert resolved.get("cwd") is None


def test_envelope_failure_is_null_on_success() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
    )
    assert env["failure"] is None


def test_envelope_failure_block_has_required_fields() -> None:
    env = build_result_envelope(
        status="failed",
        stop_reason="max_runtime_exceeded",
        output="timeout",
        created_at="2025-01-01T00:00:00Z",
        failure={
            "stage": "execution",
            "code": "max_runtime_exceeded",
            "retryable": True,
            "next_action": "retry_with_higher_timeout",
            "diagnostics": {"layer": "timeout", "limit_sec": 600},
        },
    )
    failure = env["failure"]
    assert failure["stage"] == "execution"
    assert failure["code"] == "max_runtime_exceeded"
    assert failure["retryable"] is True
    assert failure["next_action"] == "retry_with_higher_timeout"
    assert "layer" in failure["diagnostics"]


# ── failure schema validation (fail fast on programmer error) ─────────────


def _valid_failure(**overrides) -> dict:
    base = {
        "stage": "execution",
        "code": "some_code",
        "retryable": True,
        "next_action": "inspect_logs",
    }
    base.update(overrides)
    return base


def test_failure_stages_is_the_six_allowed_values() -> None:
    assert FAILURE_STAGES == {
        "preflight",
        "launch",
        "auth",
        "prompt_delivery",
        "execution",
        "finalization",
    }


@pytest.mark.parametrize("stage", sorted(FAILURE_STAGES))
def test_build_result_envelope_accepts_every_allowed_stage(stage: str) -> None:
    env = build_result_envelope(
        status="failed",
        stop_reason="x",
        output="x",
        created_at="2025-01-01T00:00:00Z",
        failure=_valid_failure(stage=stage),
    )
    assert env["failure"]["stage"] == stage


def test_build_result_envelope_rejects_stage_outside_taxonomy() -> None:
    with pytest.raises(ValueError, match="stage"):
        build_result_envelope(
            status="failed",
            stop_reason="x",
            output="x",
            created_at="2025-01-01T00:00:00Z",
            failure=_valid_failure(stage="protocol"),
        )


def test_build_result_envelope_rejects_empty_code() -> None:
    with pytest.raises(ValueError, match="code"):
        build_result_envelope(
            status="failed",
            stop_reason="x",
            output="x",
            created_at="2025-01-01T00:00:00Z",
            failure=_valid_failure(code=""),
        )


def test_build_result_envelope_rejects_non_string_code() -> None:
    with pytest.raises(ValueError, match="code"):
        build_result_envelope(
            status="failed",
            stop_reason="x",
            output="x",
            created_at="2025-01-01T00:00:00Z",
            failure=_valid_failure(code=None),
        )


def test_build_result_envelope_rejects_non_bool_retryable() -> None:
    with pytest.raises(ValueError, match="retryable"):
        build_result_envelope(
            status="failed",
            stop_reason="x",
            output="x",
            created_at="2025-01-01T00:00:00Z",
            failure=_valid_failure(retryable="yes"),
        )


def test_build_result_envelope_rejects_empty_next_action() -> None:
    with pytest.raises(ValueError, match="next_action"):
        build_result_envelope(
            status="failed",
            stop_reason="x",
            output="x",
            created_at="2025-01-01T00:00:00Z",
            failure=_valid_failure(next_action=""),
        )


def test_envelope_usage_defaults_to_unavailable() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
    )
    assert env["usage"]["available"] is False
    assert env["usage"]["status"] == "unavailable"
    assert env["usage"]["source"] is None
    assert env["usage"]["reason"]
    assert env["usage"]["total_tokens"] is None


def test_envelope_usage_preserves_native_when_provided() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
        usage={"available": True, "input_tokens": 150, "output_tokens": 50},
    )
    assert env["usage"]["available"] is True
    assert env["usage"]["input_tokens"] == 150


def test_envelope_technical_has_lifecycle_events_and_session_ids() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
        technical={
            "lifecycle_events": 15,
            "native_session_id": "e2accc98",
            "native_full_session_id": "e2accc98-9fd6-4813-a6e6-3cb0d134de46",
        },
    )
    tech = env["technical"]
    assert tech["lifecycle_events"] == 15
    assert tech["native_session_id"] == "e2accc98"
    assert tech["native_full_session_id"] == "e2accc98-9fd6-4813-a6e6-3cb0d134de46"
    assert tech["turns"] is None
    assert tech["tool_calls"] is None


def test_envelope_process_fields() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
        exit_code=0,
    )
    assert env["process"]["exit_code"] == 0
    assert env["process"]["signal"] is None


def test_envelope_changes_and_artifacts_are_empty_lists() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="hello",
        created_at="2025-01-01T00:00:00Z",
    )
    assert env["changes"] == []
    assert env["artifacts"] == []


def test_envelope_preserves_summary_for_backward_compat() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="full output here",
        created_at="2025-01-01T00:00:00Z",
        summary="legacy summary",
    )
    assert env["summary"] == "legacy summary"


def test_envelope_defaults_summary_from_output_when_not_provided() -> None:
    env = build_result_envelope(
        status="completed",
        stop_reason="done",
        output="output becomes summary",
        created_at="2025-01-01T00:00:00Z",
    )
    assert env["summary"] == "output becomes summary"


# ── Compound secret-key detection (review fix #5) ─────────────────────────


class TestCompoundSecretKeyDetection:
    """Compound keys like openai_api_key, anthropic_api_key, agent_token
    must be redacted, not just simple keys like 'api_key' or 'token'."""

    def test_compound_key_openai_api_key_redacted(self):
        """openai_api_key must be stripped from diagnostics."""
        result = truncate_diagnostics(
            {
                "openai_api_key": "sk-abc123",
                "message": "hi",
            }
        )
        assert "openai_api_key" not in result
        assert result["message"] == "hi"

    def test_compound_key_anthropic_api_key_redacted(self):
        """anthropic_api_key must be stripped from diagnostics."""
        result = truncate_diagnostics(
            {
                "anthropic_api_key": "sk-ant-xxx",
                "message": "hi",
            }
        )
        assert "anthropic_api_key" not in result
        assert result["message"] == "hi"

    def test_compound_key_agent_token_redacted(self):
        """agent_token must be stripped from diagnostics."""
        result = truncate_diagnostics(
            {
                "agent_token": "secret-token-123",
                "message": "hi",
            }
        )
        assert "agent_token" not in result
        assert result["message"] == "hi"

    def test_compound_key_access_token_redacted(self):
        """access_token must be stripped from diagnostics."""
        result = truncate_diagnostics(
            {
                "access_token": "ghp_xxx",
                "ok": True,
            }
        )
        assert "access_token" not in result
        assert result["ok"] is True

    def test_compound_key_refresh_token_redacted(self):
        """refresh_token must be stripped."""
        result = truncate_diagnostics(
            {
                "refresh_token": "rt-xxx",
                "ok": True,
            }
        )
        assert "refresh_token" not in result

    def test_compound_key_in_nested_dict_redacted(self):
        """Compound keys in nested dicts must also be stripped."""
        result = truncate_diagnostics(
            {
                "config": {
                    "openai_api_key": "sk-nested",
                    "anthropic_api_key": "sk-ant-nested",
                    "safe_field": "visible",
                },
                "message": "hi",
            }
        )
        config = result.get("config", {})
        assert "openai_api_key" not in config
        assert "anthropic_api_key" not in config
        assert config.get("safe_field") == "visible"

    def test_uppercase_compound_key_redacted(self):
        """Uppercase compound keys like OPENAI_API_KEY must be redacted."""
        result = truncate_diagnostics(
            {
                "OPENAI_API_KEY": "sk-upper",
                "message": "hi",
            }
        )
        assert "OPENAI_API_KEY" not in result

    def test_mixed_case_compound_key_redacted(self):
        """Mixed case like Openai_Api_Key must be redacted."""
        result = truncate_diagnostics(
            {
                "Openai_Api_Key": "sk-mixed",
                "message": "hi",
            }
        )
        assert "Openai_Api_Key" not in result
