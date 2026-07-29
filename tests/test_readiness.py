"""Tests for provider readiness probes, profile_health upgrade, and doctor CLI."""

from __future__ import annotations

import json
import time

import pytest

from agent_crossbar.adapters.claude import RunResult
from agent_crossbar.readiness import (
    PROBE_CACHE_TTL_SECONDS,
    ProbeResult,
    ReadinessResult,
    _cache_key,
    check_chatgpt_pro_readiness,
    check_claude_readiness,
    check_codex_readiness,
    check_opencode_readiness,
    check_reasonix_readiness,
    probe_profile,
    readiness_cache,
    refresh_readiness,
)

# ── ReadinessResult contract ────────────────────────────────────────────────


class TestReadinessResultContract:
    """Every ReadinessResult must conform to the shared schema."""

    REQUIRED_FIELDS = {
        "profile",
        "state",
        "support_tier",
        "authenticated",
        "error_code",
        "remediation",
        "evidence",
        "probe_version",
        "timestamp",
    }
    VALID_STATES = {
        "ready",
        "needs_auth",
        "missing_binary",
        "unsupported_os",
        "misconfigured",
        "degraded",
    }

    def test_all_required_fields_present(self):
        """Every ReadinessResult must have all required fields."""
        for profile in ("codex", "claude", "opencode", "reasonix", "chatgpt_pro"):
            result = ReadinessResult(
                profile=profile,
                state="ready",
                support_tier="supported",
                authenticated=True,
                error_code=None,
                remediation=None,
                evidence=None,
                probe_version=1,
                timestamp=time.time(),
            )
            present = {
                f for f in self.REQUIRED_FIELDS if getattr(result, f, _MISSING) is not _MISSING
            }
            assert present == self.REQUIRED_FIELDS, f"Missing: {self.REQUIRED_FIELDS - present}"

    def test_state_must_be_valid(self):
        """State must be one of the six valid states."""
        result = ReadinessResult(
            profile="codex",
            state="ready",
            support_tier="supported",
            authenticated=True,
            error_code=None,
            remediation=None,
            evidence=None,
            probe_version=1,
            timestamp=time.time(),
        )
        assert result.state in self.VALID_STATES

    def test_registered_is_not_a_valid_state(self):
        """'registered' is not a valid readiness state."""
        with pytest.raises(ValueError, match="registered"):
            ReadinessResult(
                profile="codex",
                state="registered",
                support_tier="supported",
                authenticated=True,
                error_code=None,
                remediation=None,
                evidence=None,
                probe_version=1,
                timestamp=time.time(),
            )

    def test_unknown_state_is_rejected(self):
        """Arbitrary states are rejected."""
        with pytest.raises(ValueError):
            ReadinessResult(
                profile="codex",
                state="unknown_garbage",
                support_tier="supported",
                authenticated=True,
                error_code=None,
                remediation=None,
                evidence=None,
                probe_version=1,
                timestamp=time.time(),
            )

    def test_support_tier_must_be_valid(self):
        """Support tier must be supported or experimental."""
        for tier in ("supported", "experimental"):
            result = ReadinessResult(
                profile="codex",
                state="ready",
                support_tier=tier,
                authenticated=True,
                error_code=None,
                remediation=None,
                evidence=None,
                probe_version=1,
                timestamp=time.time(),
            )
            assert result.support_tier == tier
        with pytest.raises(ValueError):
            ReadinessResult(
                profile="codex",
                state="ready",
                support_tier="legacy",
                authenticated=True,
                error_code=None,
                remediation=None,
                evidence=None,
                probe_version=1,
                timestamp=time.time(),
            )

    def test_ready_state_does_not_require_error_fields(self):
        """Ready state may have null error_code and remediation."""
        result = ReadinessResult(
            profile="codex",
            state="ready",
            support_tier="supported",
            authenticated=True,
            error_code=None,
            remediation=None,
            evidence="All checks passed",
            probe_version=1,
            timestamp=time.time(),
        )
        assert result.state == "ready"
        assert result.error_code is None
        assert result.remediation is None

    def test_non_ready_state_has_error_code_and_remediation(self):
        """Non-ready states should carry error_code and remediation."""
        result = ReadinessResult(
            profile="codex",
            state="missing_binary",
            support_tier="supported",
            authenticated=False,
            error_code="codex_missing",
            remediation="Install the Codex CLI.",
            evidence=None,
            probe_version=1,
            timestamp=time.time(),
        )
        assert result.error_code == "codex_missing"
        assert result.remediation is not None

    def test_to_dict_includes_all_fields(self):
        """to_dict serializes all fields."""
        result = ReadinessResult(
            profile="claude",
            state="ready",
            support_tier="supported",
            authenticated=True,
            auth_mode="subscription",
            billing_mode="subscription_quota",
            error_code=None,
            remediation=None,
            evidence="authMethod=oauth",
            probe_version=2,
            timestamp=1700000000.0,
        )
        d = result.to_dict()
        assert d["profile"] == "claude"
        assert d["state"] == "ready"
        assert d["support_tier"] == "supported"
        assert d["authenticated"] is True
        assert d["auth_mode"] == "subscription"
        assert d["billing_mode"] == "subscription_quota"
        assert d["error_code"] is None
        assert d["remediation"] is None
        assert d["probe_version"] == 2
        assert d["timestamp"] == 1700000000.0


# ── Probe contract ──────────────────────────────────────────────────────────


class TestProbeResult:
    """ProbeResult is the intermediate type returned by probe functions."""

    def test_probe_result_from_file_not_found(self):
        """ProbeResult wraps FileNotFoundError as missing_binary."""
        pr = ProbeResult.from_file_not_found("codex", FileNotFoundError("codex"))
        assert pr.state == "missing_binary"
        assert pr.error_code == "codex_missing"
        assert pr.remediation is not None
        assert "codex" in pr.remediation.lower()

    def test_probe_result_from_exception(self):
        """ProbeResult wraps generic exceptions as probe_failed."""
        pr = ProbeResult.from_exception("codex", RuntimeError("boom"))
        assert pr.state == "misconfigured"
        assert pr.error_code == "codex_probe_failed"
        assert "boom" in pr.remediation

    def test_probe_result_from_broken_binary(self):
        """ProbeResult for a broken binary."""
        pr = ProbeResult.from_process_error("reasonix", "reasonix_broken", "reinstall")
        assert pr.state == "misconfigured"
        assert pr.error_code == "reasonix_broken"
        assert pr.remediation == "reinstall"

    def test_probe_result_to_readiness(self):
        """ProbeResult can be converted to a full ReadinessResult."""
        pr = ProbeResult(
            state="ready",
            authenticated=True,
            auth_mode="api_key",
            billing_mode="api",
            error_code=None,
            remediation=None,
            evidence="ok",
            version="1.2.3",
        )
        rr = pr.to_readiness("codex", "supported")
        assert rr.profile == "codex"
        assert rr.state == "ready"
        assert rr.support_tier == "supported"
        assert rr.auth_mode == "api_key"
        assert rr.billing_mode == "api"
        assert rr.version == "1.2.3"

    def test_to_readiness_sanitizes_bearer_token_in_evidence(self):
        """A raw exception carrying a Bearer token must never survive into ReadinessResult."""
        exc = RuntimeError("request failed: Authorization: Bearer sk-live-abc123XYZ")
        pr = ProbeResult.from_exception("codex", exc)
        rr = pr.to_readiness("codex", "supported")
        assert "sk-live-abc123XYZ" not in (rr.evidence or "")
        assert "sk-live-abc123XYZ" not in (rr.remediation or "")
        assert "[REDACTED]" in (rr.evidence or "") or "[REDACTED]" in (rr.remediation or "")

    def test_to_readiness_sanitizes_query_string_secret_in_evidence(self):
        """A query-string-shaped secret (api_key=...) must never survive into ReadinessResult."""
        exc = RuntimeError(
            "GET https://api.example.com/v1?api_key=sk-xxxx-super-secret&foo=bar failed"
        )
        pr = ProbeResult.from_exception("codex", exc)
        rr = pr.to_readiness("codex", "supported")
        assert "sk-xxxx-super-secret" not in (rr.evidence or "")
        assert "sk-xxxx-super-secret" not in (rr.remediation or "")

    def test_to_readiness_sanitizes_key_value_secret_in_evidence(self):
        """A KEY=VALUE-shaped secret (token=...) must never survive into ReadinessResult."""
        exc = RuntimeError("probe failed: token=super-secret-value-1234")
        pr = ProbeResult.from_exception("codex", exc)
        rr = pr.to_readiness("codex", "supported")
        assert "super-secret-value-1234" not in (rr.evidence or "")
        assert "super-secret-value-1234" not in (rr.remediation or "")


# ── Claude readiness probe ──────────────────────────────────────────────────


class TestClaudeReadiness:
    def test_missing_binary(self):
        """Claude probe detects missing binary."""
        runner = _FakeRunner(FileNotFoundError("claude"))
        result = check_claude_readiness(runner)
        assert result.state == "missing_binary"
        assert result.error_code == "claude_missing"
        assert result.authenticated is False

    def test_auth_probe_failed(self):
        """Claude probe detects failed auth status."""
        runner = _FakeRunner(RunResult(1, "", "auth error"))
        result = check_claude_readiness(runner)
        assert result.state == "needs_auth"
        assert result.authenticated is False
        assert result.error_code == "auth_probe_failed"

    def test_not_authenticated(self):
        """Claude probe detects not-logged-in state."""
        runner = _FakeRunner(RunResult(0, json.dumps({"loggedIn": False}), ""))
        result = check_claude_readiness(runner)
        assert result.state == "needs_auth"
        assert result.error_code == "not_authenticated"
        assert "claude auth login" in result.remediation.lower()

    def test_authenticated_oauth(self):
        """Claude probe detects OAuth-authenticated state."""
        runner = _FakeRunner(
            RunResult(
                0,
                json.dumps(
                    {
                        "loggedIn": True,
                        "authMethod": "oauth",
                        "subscriptionType": "pro",
                        "apiProvider": "anthropic",
                    }
                ),
                "",
            )
        )
        result = check_claude_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is True
        assert result.auth_mode == "subscription"
        assert result.billing_mode == "subscription_quota"
        assert result.subscription_type == "pro"

    def test_authenticated_api_key(self):
        """Claude probe detects API-key-authenticated state."""
        runner = _FakeRunner(
            RunResult(
                0,
                json.dumps(
                    {
                        "loggedIn": True,
                        "authMethod": "api_key",
                        "apiProvider": "anthropic",
                    }
                ),
                "",
            )
        )
        result = check_claude_readiness(runner)
        assert result.state == "ready"
        assert result.auth_mode == "api_key"
        assert result.billing_mode == "api"


# ── Codex readiness probe ───────────────────────────────────────────────────


class TestCodexReadiness:
    def test_codex_binary_missing(self):
        """Codex probe detects missing binary."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": FileNotFoundError("codex"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "missing_binary"
        assert result.error_code == "codex_missing"

    def test_codex_broken(self):
        """Codex probe detects broken binary."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(1, "", "broken"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "misconfigured"
        assert result.error_code == "codex_broken"

    def test_pnpm_missing_for_acp(self):
        """Codex probe detects missing pnpm for ACP."""
        calls = {
            "pnpm --version": FileNotFoundError("pnpm"),
            "codex --version": RunResult(0, "1.2.3", ""),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "missing_binary"
        assert result.error_code == "pnpm_missing"

    def test_codex_ready(self):
        """Codex probe passes when binary + login are both ok.
        Note: codex login status writes to stderr, not stdout."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(0, "", "Logged in using ChatGPT\n"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is True
        assert "codex=1.2.3" in (result.evidence or "")

    def test_codex_login_status_not_logged_in(self):
        """Codex probe returns needs_auth when login status shows no session."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(0, "Not logged in\n", ""),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "needs_auth"
        assert result.error_code == "codex_not_authenticated"
        assert result.authenticated is False

    def test_codex_login_status_fails(self):
        """Codex probe returns needs_auth when login status exits non-zero."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(1, "", "error: no credentials\n"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "needs_auth"
        assert result.error_code == "codex_login_status_failed"


# ── OpenCode readiness probe ────────────────────────────────────────────────


class TestOpenCodeReadiness:
    def test_opencode_binary_missing(self):
        """OpenCode probe detects missing binary — no pnpm probe needed."""
        calls = {
            "opencode --version": FileNotFoundError("opencode"),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "missing_binary"
        assert result.error_code == "opencode_missing"

    def test_opencode_ready(self):
        """OpenCode probe passes when binary + auth list succeed — no pnpm probe."""
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": RunResult(
                0, "●  OpenAI \x1b[90moauth\n●  OpenRouter \x1b[90mapi\n  2 credentials\n", ""
            ),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is True
        assert result.auth_mode == "providers_configured"
        assert result.billing_mode == "provider_dependent"
        assert "opencode=1.0.0" in (result.evidence or "")

    def test_opencode_auth_list_empty(self):
        """Empty auth list → free defaults may still work, but authenticated
        must be honest (False) — never inferred from binary presence alone."""
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": RunResult(0, "\n\n", ""),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is False
        assert result.auth_mode == "free_defaults"
        assert "unauthenticated free defaults" in (result.evidence or "")

    def test_opencode_auth_list_fails(self):
        """Auth list exits non-zero → the probe itself is unproven, so this must
        NOT be reported as ready. A non-zero exit disproves nothing about
        credentials, but it also does not prove the empty-list free-defaults
        case — only a successful (exit 0) empty list may be ready."""
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": RunResult(1, "", "error"),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "degraded"
        assert result.authenticated is False
        assert result.error_code == "opencode_auth_list_failed"
        assert result.remediation is not None
        assert "opencode auth list" in result.remediation.lower()

    def test_opencode_auth_list_subcommand_missing(self):
        """Regression: FileNotFoundError from 'auth list' must NOT infer
        authenticated=True from binary presence alone — this is a probe
        failure, reported honestly as degraded/auth-unverified."""
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": FileNotFoundError("opencode"),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "degraded"
        assert result.authenticated is False
        assert result.error_code == "opencode_auth_unverified"

    def test_opencode_auth_list_probe_error(self):
        """Regression: an arbitrary exception from 'auth list' must NOT infer
        authenticated=True from binary presence alone — this is a probe
        failure, reported honestly as degraded/auth-unverified."""
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": RuntimeError("boom"),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "degraded"
        assert result.authenticated is False
        assert result.error_code == "opencode_auth_unverified"


# ── Reasonix readiness probe ───────────────────────────────────────────────


class TestReasonixReadiness:
    def test_reasonix_missing_binary(self):
        """Reasonix probe detects missing binary."""
        runner = _FakeRunner(FileNotFoundError("reasonix"))
        result = check_reasonix_readiness(runner)
        assert result.state == "missing_binary"
        assert result.error_code == "reasonix_missing"

    def test_reasonix_never_claims_ready_from_version_alone(self):
        """Regression: binary presence alone must NEVER produce ready/authenticated.

        Reasonix has no doctor output here (only --version succeeds) —
        the probe must report degraded, not ready.
        """
        calls = {
            "reasonix --version": RunResult(0, "0.53.2", ""),
            "reasonix doctor --json": FileNotFoundError("doctor subcommand unavailable"),
        }
        runner = _SequenceRunner(calls)
        result = check_reasonix_readiness(runner)
        assert result.state != "ready", "must not claim ready without proving auth"
        assert result.authenticated is False
        assert result.error_code is not None
        assert result.remediation is not None
        rr = result.to_readiness("reasonix", "experimental")
        assert rr.support_tier == "experimental"

    def test_reasonix_ready_when_doctor_proves_api_key_and_reach(self):
        """`reasonix doctor --json` proving api-key + api-reach → truthful ready."""
        doctor_json = json.dumps(
            {
                "version": "0.53.2",
                "summary": {"ok": 9, "warn": 0, "fail": 0},
                "checks": [
                    {"id": "api-key", "status": "ok", "message": "set via env DEEPSEEK_API_KEY"},
                    {
                        "id": "api-reach",
                        "status": "ok",
                        "message": "/models ok — 2 models (deepseek-v4-flash, deepseek-v4-pro)",
                    },
                ],
            }
        )
        calls = {
            "reasonix --version": RunResult(0, "0.53.2", ""),
            "reasonix doctor --json": RunResult(0, doctor_json, ""),
        }
        runner = _SequenceRunner(calls)
        result = check_reasonix_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is True
        assert result.auth_mode == "api_key"
        assert result.billing_mode == "api"
        # Evidence must never contain the actual key value
        assert "DEEPSEEK_API_KEY" in (result.evidence or "")
        assert "sk-" not in (result.evidence or "")

    def test_reasonix_needs_auth_when_api_key_missing(self):
        """`reasonix doctor --json` reporting a failed api-key check → needs_auth."""
        doctor_json = json.dumps(
            {
                "checks": [
                    {"id": "api-key", "status": "fail", "message": "not set"},
                    {"id": "api-reach", "status": "fail", "message": "no key to test"},
                ]
            }
        )
        calls = {
            "reasonix --version": RunResult(0, "0.53.2", ""),
            "reasonix doctor --json": RunResult(0, doctor_json, ""),
        }
        runner = _SequenceRunner(calls)
        result = check_reasonix_readiness(runner)
        assert result.state == "needs_auth"
        assert result.authenticated is False
        assert result.error_code == "reasonix_not_authenticated"
        assert "reasonix setup" in (result.remediation or "").lower()

    def test_reasonix_needs_auth_when_api_unreachable(self):
        """api-key ok but api-reach fails → needs_auth, not ready."""
        doctor_json = json.dumps(
            {
                "checks": [
                    {"id": "api-key", "status": "ok", "message": "set via env DEEPSEEK_API_KEY"},
                    {"id": "api-reach", "status": "fail", "message": "401 unauthorized"},
                ]
            }
        )
        calls = {
            "reasonix --version": RunResult(0, "0.53.2", ""),
            "reasonix doctor --json": RunResult(0, doctor_json, ""),
        }
        runner = _SequenceRunner(calls)
        result = check_reasonix_readiness(runner)
        assert result.state == "needs_auth"
        assert result.authenticated is False
        assert result.error_code == "reasonix_api_unreachable"

    def test_reasonix_degraded_when_doctor_output_unparseable(self):
        """Malformed doctor JSON → honest degraded, never ready."""
        calls = {
            "reasonix --version": RunResult(0, "0.53.2", ""),
            "reasonix doctor --json": RunResult(0, "not json", ""),
        }
        runner = _SequenceRunner(calls)
        result = check_reasonix_readiness(runner)
        assert result.state == "degraded"
        assert result.authenticated is False
        assert result.error_code is not None
        assert result.remediation is not None

    def test_reasonix_matrix_billing_mode_matches_proven_readiness(self):
        """Regression: the static support matrix must not diverge from what
        check_reasonix_readiness actually proves. Reasonix authenticates via
        an api-key (DeepSeek), never free defaults — the matrix's
        billing_mode must say so."""
        from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

        calls = {
            "reasonix --version": RunResult(0, "0.53.2", ""),
            "reasonix doctor --json": RunResult(
                0,
                json.dumps(
                    {
                        "checks": [
                            {"id": "api-key", "status": "ok", "message": "configured"},
                            {"id": "api-reach", "status": "ok", "message": "reachable"},
                        ]
                    }
                ),
                "",
            ),
        }
        runner = _SequenceRunner(calls)
        result = check_reasonix_readiness(runner)
        assert result.state == "ready"
        assert result.billing_mode == "api"
        assert PROVIDER_SUPPORT_MATRIX["reasonix"]["billing_mode"] == result.billing_mode


# ── ChatGPT Pro readiness probe ─────────────────────────────────────────────


class TestChatGPTProReadiness:
    def test_chatgpt_pro_not_on_macos(self):
        """ChatGPT Pro probe reports unsupported_os on non-macOS."""
        import platform

        if platform.system() == "Darwin":
            pytest.skip("This test is for non-macOS only")
        result = check_chatgpt_pro_readiness()
        assert result.state == "unsupported_os"
        assert result.error_code == "unsupported_os"

    def test_chatgpt_pro_is_experimental(self):
        """ChatGPT Pro is experimental tier."""
        # We test the tier assignment, not the actual app detection.
        # The probe now returns degraded (cannot safely prove auth).
        pr = ProbeResult(
            state="degraded",
            authenticated=False,
            error_code="chatgpt_pro_manual_gate",
            remediation="Manual gate required",
            evidence="macOS detected",
        )
        rr = pr.to_readiness("chatgpt_pro", "experimental")
        assert rr.support_tier == "experimental"
        assert rr.authenticated is False


# ── Probe profile dispatcher ────────────────────────────────────────────────


class TestProbeProfile:
    def test_probe_profile_dispatches_correctly(self, tmp_path, monkeypatch):
        """probe_profile dispatches to the correct provider probe."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        # Test with a sequence runner proving reasonix's doctor-based auth
        calls = {
            "reasonix --version": RunResult(0, "0.53.2", ""),
            "reasonix doctor --json": RunResult(
                0,
                json.dumps(
                    {
                        "checks": [
                            {"id": "api-key", "status": "ok", "message": "set"},
                            {"id": "api-reach", "status": "ok", "message": "ok"},
                        ]
                    }
                ),
                "",
            ),
        }
        result = probe_profile("reasonix", _runner=_SequenceRunner(calls))
        assert result.profile == "reasonix"
        assert result.state == "ready"

    def test_probe_profile_unknown(self, tmp_path, monkeypatch):
        """probe_profile rejects unknown profiles."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Unknown profile"):
            probe_profile("nonexistent")


# ── Cache behavior ──────────────────────────────────────────────────────────


class TestReadinessCache:
    def test_cache_stores_and_retrieves(self, tmp_path, monkeypatch):
        """Cache stores ReadinessResult and retrieves it within TTL."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        rr = ReadinessResult(
            profile="codex",
            state="ready",
            support_tier="supported",
            authenticated=True,
            error_code=None,
            remediation=None,
            evidence="ok",
            probe_version=1,
            timestamp=time.time(),
        )
        cache_key = _cache_key("codex")
        readiness_cache.put(cache_key, rr)

        cached = readiness_cache.get(cache_key)
        assert cached is not None
        assert cached.state == "ready"
        assert cached.profile == "codex"

    def test_cache_expires_after_ttl(self, tmp_path, monkeypatch):
        """Cache returns None after TTL expiry."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        rr = ReadinessResult(
            profile="codex",
            state="ready",
            support_tier="supported",
            authenticated=True,
            error_code=None,
            remediation=None,
            evidence="ok",
            probe_version=1,
            timestamp=time.time() - PROBE_CACHE_TTL_SECONDS - 10,
        )
        cache_key = _cache_key("codex")
        readiness_cache.put(cache_key, rr)

        cached = readiness_cache.get(cache_key)
        assert cached is None  # Expired

    def test_cache_clear(self, tmp_path, monkeypatch):
        """Cache can be cleared."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        rr = ReadinessResult(
            profile="codex",
            state="ready",
            support_tier="supported",
            authenticated=True,
            error_code=None,
            remediation=None,
            evidence="ok",
            probe_version=1,
            timestamp=time.time(),
        )
        cache_key = _cache_key("codex")
        readiness_cache.put(cache_key, rr)
        readiness_cache.clear()
        assert readiness_cache.get(cache_key) is None

    def test_refresh_readiness_bypasses_cache(self, tmp_path, monkeypatch):
        """refresh_readiness always runs probes, ignoring cache."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        # Populate cache with a stale ready state
        rr = ReadinessResult(
            profile="codex",
            state="ready",
            support_tier="supported",
            authenticated=True,
            error_code=None,
            remediation=None,
            evidence="ok",
            probe_version=1,
            timestamp=time.time(),
        )
        cache_key = _cache_key("codex")
        readiness_cache.put(cache_key, rr)

        # refresh_readiness should ignore cache and run the probe
        # (the probe will fail because our fake runner simulates missing binary)
        runner = _FakeRunner(FileNotFoundError("codex"))
        result = refresh_readiness("codex", runner)
        assert result.state == "missing_binary"  # Fresh probe result, not cached "ready"


# ── Helpers ─────────────────────────────────────────────────────────────────

_MISSING = object()


class _FakeRunner:
    """Runner that returns a fixed result or raises a fixed exception."""

    def __init__(self, result_or_exc):
        self._result_or_exc = result_or_exc

    def run(self, args, *, timeout=None, cwd=None, env=None):
        if isinstance(self._result_or_exc, BaseException):
            raise self._result_or_exc
        return self._result_or_exc


class _SequenceRunner:
    """Runner that returns different results based on the command."""

    def __init__(self, calls: dict[str, object]):
        self._calls = calls
        self._counts = {}

    def run(self, args, *, timeout=None, cwd=None, env=None):
        cmd = " ".join(args)
        # Try exact match first, then prefix match
        for key, value in self._calls.items():
            if cmd == key or cmd.startswith(key):
                self._counts[key] = self._counts.get(key, 0) + 1
                if isinstance(value, BaseException):
                    raise value
                return value
        raise RuntimeError(f"Unexpected call: {cmd}")


# ── Preflight integration tests ──────────────────────────────────────────────


class TestPreflightInAgentStart:
    """Preflight must reject non-ready providers before any state mutation."""

    def test_missing_binary_rejected_before_job_creation(self, tmp_path, monkeypatch):
        """agent_start must reject when readiness probe says missing_binary."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        # All provider probes will fail with FileNotFoundError via our mock
        # We need to mock probe_profile to return a non-ready state
        import agent_crossbar.readiness as rmod
        from agent_crossbar.server import agent_start

        def fake_probe(profile, _runner=None, use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return ReadinessResult(
                profile=profile,
                state="missing_binary",
                support_tier="supported",
                authenticated=False,
                error_code=f"{profile}_missing",
                remediation=f"Install {profile}.",
                probe_version=1,
                timestamp=time.time(),
            )

        monkeypatch.setattr(rmod, "probe_profile", fake_probe)

        result = agent_start(profile="codex", prompt="test", model="gpt-5.6-sol", task="ask")
        assert result["ok"] is False
        assert result["error"] == "codex_missing"
        assert result["job_created"] is False
        # No job directory should exist
        jobs_dir = tmp_path / "jobs"
        assert not jobs_dir.exists() or not any(jobs_dir.iterdir())

    def test_ready_provider_proceeds_to_job_creation(self, tmp_path, monkeypatch):
        """agent_start creates a job when readiness says ready."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        import agent_crossbar.readiness as rmod
        from agent_crossbar.server import agent_start

        def fake_probe(profile, _runner=None, use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return ReadinessResult(
                profile=profile,
                state="ready",
                support_tier="supported",
                authenticated=True,
                probe_version=1,
                timestamp=time.time(),
            )

        monkeypatch.setattr(rmod, "probe_profile", fake_probe)

        def fake_start_print(store, job_id, req, **kwargs):
            store.set_result(job_id, True, summary="ok")

        monkeypatch.setattr(
            "agent_crossbar.server.start_print_job",
            fake_start_print,
        )

        # reasonix supports advice operation (task=ask)
        result = agent_start(
            profile="reasonix",
            prompt="test",
            model="deepseek-v4-flash",
            task="ask",
        )
        assert result["ok"] is True
        assert "job_id" in result

    def test_preflight_uses_cache_not_fresh_probe_every_time(self, tmp_path, monkeypatch):
        """Subsequent agent_start calls within TTL should use cached probe."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        import agent_crossbar.readiness as rmod

        call_count = 0

        def counting_check(runner=None):
            nonlocal call_count
            call_count += 1
            from agent_crossbar.readiness import ProbeResult

            return ProbeResult(
                state="ready",
                authenticated=True,
                evidence=f"call #{call_count}",
            )

        monkeypatch.setattr(rmod, "check_reasonix_readiness", counting_check)

        def fake_start_print(store, job_id, req, **kwargs):
            store.set_result(job_id, True, summary="ok")

        monkeypatch.setattr(
            "agent_crossbar.server.start_print_job",
            fake_start_print,
        )

        from agent_crossbar.server import agent_start

        # First call — should run the probe (via probe_profile → check_reasonix_readiness)
        rmod.readiness_cache.clear()
        result1 = agent_start(
            profile="reasonix",
            prompt="test",
            model="deepseek-v4-flash",
            task="ask",
        )
        assert result1["ok"] is True
        assert call_count == 1

        # Second call within TTL — should use cache, not probe again
        result2 = agent_start(
            profile="reasonix",
            prompt="test2",
            model="deepseek-v4-flash",
            task="ask",
        )
        assert result2["ok"] is True
        assert call_count == 1  # Still 1, cache hit

        # Verify the cache by checking probe_profile directly
        from agent_crossbar.readiness import _cache_key, readiness_cache

        cached = readiness_cache.get(_cache_key("reasonix"))
        assert cached is not None
        assert cached.state == "ready"


# ── Doctor CLI tests ─────────────────────────────────────────────────────────


class TestDoctorCLI:
    """agent-crossbar doctor CLI entrypoint tests."""

    def test_doctor_json_output(self, tmp_path, monkeypatch, capsys):
        """doctor --json prints valid JSON with profiles array."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        import agent_crossbar.readiness as rmod

        def fake_probe_all(use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return {
                "codex": ReadinessResult(
                    profile="codex",
                    state="ready",
                    support_tier="supported",
                    authenticated=True,
                    probe_version=1,
                    timestamp=time.time(),
                ),
            }

        monkeypatch.setattr(rmod, "probe_all_profiles", fake_probe_all)

        from agent_crossbar.cli import doctor_cmd

        doctor_cmd(json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "profiles" in data
        assert len(data["profiles"]) == 1
        assert data["profiles"][0]["profile"] == "codex"
        assert data["profiles"][0]["state"] == "ready"

    def test_doctor_text_output(self, tmp_path, monkeypatch, capsys):
        """doctor without --json prints human-readable text."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        import agent_crossbar.readiness as rmod

        def fake_probe_all(use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return {
                "claude": ReadinessResult(
                    profile="claude",
                    state="ready",
                    support_tier="supported",
                    authenticated=True,
                    auth_mode="subscription",
                    billing_mode="subscription_quota",
                    probe_version=1,
                    timestamp=time.time(),
                ),
            }

        monkeypatch.setattr(rmod, "probe_all_profiles", fake_probe_all)

        from agent_crossbar.cli import doctor_cmd

        doctor_cmd(json_output=False)

        captured = capsys.readouterr()
        assert "claude" in captured.out
        assert "ready" in captured.out
        assert "subscription" in captured.out

    def test_doctor_single_profile(self, tmp_path, monkeypatch, capsys):
        """doctor --profile NAME filters to one provider."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        import agent_crossbar.readiness as rmod

        def fake_probe(profile, _runner=None, use_cache=True):
            import time

            from agent_crossbar.readiness import ReadinessResult

            return ReadinessResult(
                profile=profile,
                state="ready",
                support_tier="supported",
                authenticated=True,
                probe_version=1,
                timestamp=time.time(),
            )

        monkeypatch.setattr(rmod, "probe_profile", fake_probe)

        from agent_crossbar.cli import doctor_cmd

        doctor_cmd(json_output=True, profile="codex")

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["profiles"]) == 1
        assert data["profiles"][0]["profile"] == "codex"

    def test_doctor_unknown_profile(self, tmp_path, monkeypatch, capsys):
        """doctor with unknown profile exits with error."""
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

        from agent_crossbar.cli import doctor_cmd

        with pytest.raises(SystemExit) as exc_info:
            doctor_cmd(profile="nonexistent")
        assert exc_info.value.code == 1


# ── New tests for review fixes ────────────────────────────────────────────────


class FakeCuaRunner:
    """Deterministic stand-in for the read-only cua-driver probe calls."""

    def __init__(self, apps=None, windows=None, trees=None, missing=False, fail_tool=None):
        self.apps = apps if apps is not None else []
        self.windows = windows if windows is not None else {}
        self.trees = trees if trees is not None else {}
        self.missing = missing
        self.fail_tool = fail_tool
        self.tools: list[str] = []

    def run(self, args, timeout=None, cwd=None, env=None):
        import subprocess as _sp

        if self.missing:
            raise FileNotFoundError(2, "No such file or directory", args[0])
        tool = args[2]
        payload = json.loads(args[3])
        self.tools.append(tool)
        if tool == self.fail_tool:
            return _sp.CompletedProcess(args, 1, "", "cua-driver refused")
        if tool == "list_apps":
            body = {"apps": self.apps}
        elif tool == "list_windows":
            body = {"windows": self.windows.get(payload.get("pid"), [])}
        elif tool == "get_window_state":
            body = {"tree_markdown": self.trees.get(payload.get("window_id"), "")}
        elif tool == "check_permissions":
            body = {"accessibility": True, "screen_recording": True}
        else:  # pragma: no cover - a mutating tool would be a contract break
            raise AssertionError(f"readiness probe must not call {tool}")
        return _sp.CompletedProcess(args, 0, json.dumps(body), "")


SIGNED_IN_TREE = (
    'AXWindow "ChatGPT"\n[1] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n[2] AXButton "Pro"'
)
SIGNED_OUT_TREE = (
    'AXWindow "ChatGPT"\n'
    '[1] AXTextField = "chatgpt.com" (Address and search bar)\n'
    'AXWebArea "ChatGPT"\n'
    '[2] AXButton "Log in"\n[3] AXButton "Sign up"'
)
# A backgrounded window exposes only its own chrome plus history menu items —
# never a page. It must never be read as a signed-out ChatGPT.
BACKGROUNDED_TREE = (
    'AXWindow "ChatGPT" [actions=[raise]]\n'
    "AXToolbar\n"
    '[449] AXMenuItem "Notion - Log in"\n'
    '[572] AXMenuItem "Error processing form - Log in - PyPI"'
)


def helium_runner(**kwargs):
    return FakeCuaRunner(
        apps=[{"bundle_id": "net.imput.helium", "pid": 123, "running": True}],
        windows={123: [{"pid": 123, "window_id": 456, "title": "ChatGPT"}]},
        **kwargs,
    )


class TestChatGPTProBrowserReadiness:
    """Readiness inspects the browser surface, non-mutating and truthfully."""

    @pytest.fixture(autouse=True)
    def _force_macos(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")

    def test_signed_in_pro_browser_is_ready(self):
        runner = helium_runner(trees={456: SIGNED_IN_TREE})

        result = check_chatgpt_pro_readiness(runner)

        assert result.state == "ready"
        assert result.authenticated is True
        assert result.auth_mode == "browser_session"
        assert result.billing_mode == "subscription_quota"
        assert result.subscription_type == "chatgpt_pro"
        assert "browser=Helium" in (result.evidence or "")
        assert "composer=present" in (result.evidence or "")

    def test_probe_only_uses_read_only_tools(self):
        runner = helium_runner(trees={456: SIGNED_IN_TREE})

        check_chatgpt_pro_readiness(runner)

        assert set(runner.tools) <= {
            "list_apps",
            "list_windows",
            "get_window_state",
            "check_permissions",
        }

    def test_signed_out_browser_needs_auth(self):
        runner = helium_runner(trees={456: SIGNED_OUT_TREE})

        result = check_chatgpt_pro_readiness(runner)

        assert result.state == "needs_auth"
        assert result.authenticated is False
        assert result.error_code == "chatgpt_pro_not_authenticated"
        assert "sign in" in (result.remediation or "").lower()

    def test_no_supported_browser_running(self):
        result = check_chatgpt_pro_readiness(FakeCuaRunner(apps=[]))

        assert result.state == "degraded"
        assert result.error_code == "chatgpt_pro_browser_not_running"
        assert "chatgpt.com" in (result.remediation or "")
        assert "desktop app" not in (result.remediation or "").lower()

    def test_browser_without_chatgpt_window(self):
        runner = FakeCuaRunner(
            apps=[{"bundle_id": "com.google.Chrome", "pid": 7, "running": True}],
            windows={7: [{"pid": 7, "window_id": 1, "title": "Gitea"}]},
        )

        result = check_chatgpt_pro_readiness(runner)

        assert result.state == "degraded"
        assert result.error_code == "chatgpt_pro_window_not_found"
        assert result.authenticated is False

    def test_backgrounded_window_is_unverified_not_signed_out(self):
        """History entries in a backgrounded window must not read as signed out."""
        runner = helium_runner(trees={456: BACKGROUNDED_TREE})

        result = check_chatgpt_pro_readiness(runner)

        assert result.state == "degraded"
        assert result.error_code == "chatgpt_pro_readiness_unverified"
        assert result.authenticated is False

    def test_unreadable_composer_is_unverified_not_ready(self):
        runner = helium_runner(trees={456: 'AXWindow "ChatGPT"'})

        result = check_chatgpt_pro_readiness(runner)

        assert result.state == "degraded"
        assert result.error_code == "chatgpt_pro_readiness_unverified"
        assert "Helium" in (result.evidence or "")

    def test_missing_cua_driver_reports_missing_binary(self):
        result = check_chatgpt_pro_readiness(FakeCuaRunner(missing=True))

        assert result.state == "missing_binary"
        assert result.error_code == "chatgpt_pro_cua_driver_missing"

    def test_failed_probe_is_degraded_with_remediation(self):
        result = check_chatgpt_pro_readiness(FakeCuaRunner(fail_tool="list_apps"))

        assert result.state == "degraded"
        assert result.error_code == "chatgpt_pro_browser_probe_failed"
        assert result.authenticated is False

    def test_readiness_is_cached_within_the_ttl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
        readiness_cache.clear()
        runner = helium_runner(trees={456: SIGNED_IN_TREE})

        first = probe_profile("chatgpt_pro", runner, use_cache=True)
        calls_after_first = len(runner.tools)
        second = probe_profile("chatgpt_pro", runner, use_cache=True)

        assert first.state == "ready"
        assert second.timestamp == first.timestamp
        assert len(runner.tools) == calls_after_first
        readiness_cache.clear()

    def test_registration_alone_is_not_ready(self):
        result = check_chatgpt_pro_readiness(FakeCuaRunner(apps=[]))

        assert result.state != "ready"
        assert result.authenticated is False


class TestChatGPTProDegraded:
    """ChatGPT Pro must never claim readiness it cannot prove."""

    def test_chatgpt_pro_returns_unsupported_os_on_linux(self, monkeypatch):
        """On non-macOS, chatgpt_pro must return unsupported_os."""
        with monkeypatch.context() as m:
            m.setattr("platform.system", lambda: "Linux")
            result = check_chatgpt_pro_readiness()
            assert result.state == "unsupported_os"
            assert result.authenticated is False

    def test_chatgpt_pro_never_claims_ready_without_evidence(self, monkeypatch):
        """chatgpt_pro must never return ready without live browser evidence."""
        for fake_os in ("Darwin", "Linux", "Windows"):
            with monkeypatch.context() as m:
                m.setattr("platform.system", lambda: fake_os)
                result = check_chatgpt_pro_readiness(FakeCuaRunner(apps=[]))
                assert result.state != "ready", (
                    f"chatgpt_pro returned 'ready' on {fake_os} — must never claim ready"
                )
                assert result.authenticated is False, (
                    f"chatgpt_pro returned authenticated=True on {fake_os}"
                )


class TestProbeAllProfilesNoSilentOmit:
    """probe_all_profiles must return misconfigured for crashing probes."""

    def test_crashing_probe_returns_misconfigured(self, monkeypatch):
        """A probe that raises Exception must produce a misconfigured result."""
        import agent_crossbar.readiness as rmod

        def crashing_probe(profile, _runner=None, use_cache=True):
            raise RuntimeError("simulated probe crash")

        monkeypatch.setattr(rmod, "probe_profile", crashing_probe)

        results = rmod.probe_all_profiles()
        # Must have results for all canonical profiles
        from agent_crossbar.profiles import list_profiles

        assert set(results.keys()) == set(list_profiles()), (
            f"probe_all_profiles silently omitted profiles: "
            f"{set(list_profiles()) - set(results.keys())}"
        )
        # Every result must be misconfigured
        for profile, rr in results.items():
            assert rr.state == "misconfigured", (
                f"Profile '{profile}': expected 'misconfigured', got '{rr.state}'"
            )
            assert rr.error_code == f"{profile}_probe_crashed"
            assert "simulated probe crash" in (rr.remediation or "")


class TestCodexAuthReadiness:
    """Codex readiness uses codex login status — truthful, read-only auth probe."""

    def test_codex_login_status_parses_logged_in(self):
        """'Logged in using ChatGPT' on stderr → ready, authenticated."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(0, "", "Logged in using ChatGPT\n"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is True
        assert "codex login status" in (result.evidence or "").lower() or "Logged in" in (
            result.evidence or ""
        )

    def test_codex_login_status_chatgpt_reports_subscription_billing(self):
        """Logged in via ChatGPT must report truthful subscription auth/billing, never null."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(0, "", "Logged in using ChatGPT\n"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "ready"
        assert result.auth_mode is not None, "auth_mode must not be null when ready"
        assert result.billing_mode is not None, "billing_mode must not be null when ready"
        assert "chatgpt" in result.auth_mode.lower()
        assert result.billing_mode == "subscription_quota"

    def test_codex_login_status_api_key_reports_api_billing(self):
        """Logged in via an API key must report api auth/billing, not subscription."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(0, "", "Logged in using an API key\n"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "ready"
        assert result.auth_mode == "api_key"
        assert result.billing_mode == "api"

    def test_codex_login_status_unknown_method_reports_explicit_unknown(self):
        """An unrecognized (but logged-in) status string must be explicit, never null."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(0, "", "Logged in\n"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "ready"
        assert result.auth_mode == "unknown"
        assert result.billing_mode == "unknown"

    def test_codex_login_status_not_logged_in_returns_needs_auth(self):
        """'Not logged in' output → needs_auth with stable remediation."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(0, "Not logged in\n", ""),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "needs_auth"
        assert result.error_code == "codex_not_authenticated"
        assert "codex login" in (result.remediation or "").lower()
        assert result.authenticated is False

    def test_codex_login_status_nonzero_returns_needs_auth(self):
        """Non-zero exit → needs_auth, not misconfigured."""
        calls = {
            "pnpm --version": RunResult(0, "9.0.0", ""),
            "codex --version": RunResult(0, "1.2.3\n", ""),
            "codex login status": RunResult(1, "", "fatal error"),
        }
        runner = _SequenceRunner(calls)
        result = check_codex_readiness(runner)
        assert result.state == "needs_auth"
        assert result.error_code == "codex_login_status_failed"


class TestOpenCodeAuthReadiness:
    """OpenCode readiness uses opencode auth list — truthful, read-only auth probe. No pnpm probe needed."""

    def test_opencode_auth_list_shows_credentials(self):
        """Credentials in auth list → ready, providers_configured."""
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": RunResult(
                0,
                "\u001b[0m\n●  OpenAI \u001b[90moauth\n●  OpenRouter \u001b[90mapi\n  2 credentials\n",
                "",
            ),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is True
        assert result.auth_mode == "providers_configured"
        assert result.billing_mode is not None, "billing_mode must not be null when ready"
        assert result.billing_mode == "provider_dependent"

    def test_opencode_auth_list_empty_still_ready(self):
        """Empty auth list → still ready (free defaults), but authenticated
        stays honest (False) — never inferred from binary presence alone."""
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": RunResult(0, "\n\n", ""),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "ready"
        assert result.authenticated is False
        assert result.auth_mode == "free_defaults"
        assert result.billing_mode == "free_defaults"

    def test_opencode_auth_list_fails_reports_degraded(self):
        """Auth list command fails (non-zero exit) → degraded, not ready.

        A non-zero exit proves nothing — it doesn't confirm credentials and
        it doesn't prove the empty-list free-defaults case either. Only a
        successful (exit 0) empty list may be ready.
        """
        calls = {
            "opencode --version": RunResult(0, "1.0.0\n", ""),
            "opencode auth list": RunResult(1, "", "command not found"),
        }
        runner = _SequenceRunner(calls)
        result = check_opencode_readiness(runner)
        assert result.state == "degraded"
        assert result.authenticated is False
        assert result.error_code == "opencode_auth_list_failed"
        assert result.remediation is not None


class TestReadinessNoMutation:
    """Doctor/profile_health probes must be genuinely read-only — no pnpm list, no pnpm dlx."""

    def test_codex_probe_uses_codex_login_status(self):
        """Codex probe uses 'codex login status' — not pnpm list or pnpm dlx."""
        import inspect

        source = inspect.getsource(check_codex_readiness)
        code_lines = [line for line in source.split("\n") if not line.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        # Must NOT call pnpm list or pnpm dlx
        assert '"pnpm", "list"' not in code_only, (
            "check_codex_readiness must not run pnpm list (unreliable for dlx-cached packages)."
        )
        assert "'pnpm', 'list'" not in code_only, "check_codex_readiness must not run pnpm list."
        assert '"pnpm", "dlx"' not in code_only, (
            "check_codex_readiness must not run pnpm dlx (network/download in doctor)."
        )
        assert "'pnpm', 'dlx'" not in code_only, "check_codex_readiness must not run pnpm dlx."
        # Must use codex login status
        assert (
            '"codex", "login", "status"' in code_only or "'codex', 'login', 'status'" in code_only
        ), "check_codex_readiness must use 'codex login status' for read-only auth inspection"

    def test_opencode_probe_uses_opencode_auth_list(self):
        """OpenCode probe uses 'opencode auth list' — not pnpm list or pnpm dlx."""
        import inspect

        source = inspect.getsource(check_opencode_readiness)
        code_lines = [line for line in source.split("\n") if not line.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert '"pnpm", "list"' not in code_only, (
            "check_opencode_readiness must not run pnpm list (unreliable for dlx-cached packages)."
        )
        assert "'pnpm', 'list'" not in code_only, "check_opencode_readiness must not run pnpm list."
        assert '"pnpm", "dlx"' not in code_only, (
            "check_opencode_readiness must not run pnpm dlx (network/download in doctor)."
        )
        assert "'pnpm', 'dlx'" not in code_only, "check_opencode_readiness must not run pnpm dlx."
        assert (
            '"opencode", "auth", "list"' in code_only or "'opencode', 'auth', 'list'" in code_only
        ), "check_opencode_readiness must use 'opencode auth list' for read-only auth inspection"

    def test_readiness_never_calls_pnpm_list_regression(self):
        """Regression: 'dlx previously succeeded but pnpm list has no dependency'.
        Neither codex nor opencode readiness must ever call 'pnpm list'."""
        import inspect

        for probe_fn in (check_codex_readiness, check_opencode_readiness):
            source = inspect.getsource(probe_fn)
            code_lines = [line for line in source.split("\n") if not line.strip().startswith("#")]
            code_only = "\n".join(code_lines)
            assert '"pnpm", "list"' not in code_only, (
                f"{probe_fn.__name__} must never run pnpm list. "
                "pnpm dlx-cached packages are invisible to pnpm list."
            )
            assert "'pnpm', 'list'" not in code_only, (
                f"{probe_fn.__name__} must never run pnpm list."
            )

    def test_opencode_readiness_does_not_probe_pnpm_version(self):
        """check_opencode_readiness must NOT probe ``pnpm --version``.

        OpenCode native ACP does not require pnpm. Only codex readiness
        may probe pnpm (for its pinned codex-acp bridge).
        """
        import inspect

        source = inspect.getsource(check_opencode_readiness)
        code_lines = [line for line in source.split("\n") if not line.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert '"pnpm", "--version"' not in code_only, (
            "check_opencode_readiness must not probe pnpm --version. "
            "OpenCode native ACP does not require pnpm."
        )
        assert "'pnpm', '--version'" not in code_only, (
            "check_opencode_readiness must not probe pnpm --version."
        )

    def test_codex_readiness_still_probes_pnpm_version(self):
        """check_codex_readiness MUST still probe ``pnpm --version``.

        Codex ACP uses the pinned codex-acp package via pnpm dlx. The
        pnpm probe is essential for Codex readiness, only OpenCode drops it.
        """
        import inspect

        source = inspect.getsource(check_codex_readiness)
        assert '"pnpm", "--version"' in source or "'pnpm', '--version'" in source, (
            "check_codex_readiness must still probe pnpm --version. "
            "Codex ACP requires pnpm for its pinned codex-acp bridge."
        )
