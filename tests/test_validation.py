"""Validation tests for Task 2: Profile Registry and Validation."""

from unittest.mock import patch

import pytest

from agent_crossbar.adapters.base import ModelCatalog, ModelInfo
from agent_crossbar.models import Autonomy, Sensitivity, Transport
from agent_crossbar.profiles import (
    allowed_models,
    list_profiles,
    resolve_profile,
)
from agent_crossbar.validation import validate_start_request


def test_profiles_are_canonical_only():
    profiles = list_profiles()
    assert sorted(profiles) == ["chatgpt_pro", "claude", "codex", "opencode", "reasonix"]


def test_unknown_profile_rejected_without_job_creation(tmp_path):
    req = {
        "operation": "review",
        "profile": "unknown_profile",
        "transport": "print",
        "autonomy": "read_only",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "gpt-5.6-sol",
    }
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "invalid_profile"
    assert result["job_created"] is False
    assert not (tmp_path / "jobs").exists()


def test_required_autonomy_and_sensitivity(tmp_path):
    req = {
        "operation": "review",
        "profile": "reasonix",
        "transport": "print",
        "prompt": "x",
        "model": "deepseek-v4-flash",
    }
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_required_field"


# ---------------------------------------------------------------------------
# Enum completeness tests — each new value would have failed before the fix
# ---------------------------------------------------------------------------


def _make_base_req(**overrides):
    req = {
        "operation": "review",
        "profile": "codex",
        "transport": "print",
        "autonomy": "read_only",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "gpt-5.6-sol",
    }
    req.update(overrides)
    return req


# -- Autonomy: propose_patch and edit_local must be accepted ---------------


def test_autonomy_propose_patch_accepted(tmp_path):
    req = _make_base_req(autonomy="propose_patch")
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is True
    assert result["error"] is None


def test_autonomy_edit_local_accepted(tmp_path):
    req = _make_base_req(
        operation="dev", transport="tmux", autonomy="edit_local", profile="reasonix", model="deepseek-v4-pro"
    )
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is True
    assert result["error"] is None


def test_autonomy_all_enum_values_exist():
    """Autonomy must have exactly these three values (full_local removed per approved contract)."""
    assert {m.value for m in Autonomy} == {
        "read_only",
        "propose_patch",
        "edit_local",
    }


# -- Transport: gui must be accepted ---------------------------------------


def test_transport_gui_accepted(tmp_path):
    req = _make_base_req(profile="chatgpt_pro", transport="gui", operation="advice", model="gpt-5")
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is True
    assert result["error"] is None


def test_transport_all_enum_values_exist():
    """Transport must have exactly these four values."""
    assert {m.value for m in Transport} == {
        "auto",
        "print",
        "tmux",
        "gui",
    }


# -- Sensitivity: private must be accepted ---------------------------------


def test_sensitivity_private_accepted(tmp_path):
    req = _make_base_req(sensitivity="private")
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is True
    assert result["error"] is None


def test_sensitivity_all_enum_values_exist():
    """Sensitivity must have exactly these three values."""
    assert {m.value for m in Sensitivity} == {
        "normal",
        "private",
        "secret",
    }


# ---------------------------------------------------------------------------
# Alias resolution tests
# ---------------------------------------------------------------------------


def test_alias_deepseek_resolves_to_reasonix(tmp_path):
    """deepseek is an alias for reasonix; validation must succeed and resolve."""
    req = _make_base_req(profile="deepseek", model="deepseek-v4-flash")
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is True
    assert result["profile"] == "reasonix"


@pytest.mark.parametrize(
    ("profile", "model"),
    [
        ("claude", "claude-opus-5"),
        ("opus", "claude-opus-5"),
        ("fable", "claude-fable-5"),
    ],
)
def test_claude_profile_names_resolve_with_expected_model(tmp_path, profile, model):
    catalog = ModelCatalog(
        models=("claude-opus-5", "claude-fable-5"),
        default_model="claude-opus-5",
        native_efforts=("low", "medium", "high"),
        source="test",
    )
    import agent_crossbar.discovery as _disc

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        result = validate_start_request(
            _make_base_req(profile=profile, transport="print", model=model),
            state_root=tmp_path,
        )

    assert result["ok"] is True
    assert result["profile"] == "claude"
    assert result["model"] == model


def test_codex_models_efforts_defaults_light_alias_and_xhigh(tmp_path):
    from agent_crossbar.profiles import (
        CODEX_DEFAULT_EFFORT,
        CODEX_EFFORT_ALIASES,
        CODEX_EFFORTS,
    )

    assert allowed_models("codex") == ["gpt-5.6-sol", "gpt-5.6-terra"]
    assert CODEX_EFFORTS == ["low", "medium", "high", "max"]
    assert CODEX_EFFORT_ALIASES == {"light": "low"}
    assert CODEX_DEFAULT_EFFORT == "medium"

    default = validate_start_request(
        _make_base_req(profile="codex", transport="print", model="gpt-5.6-sol"),
        state_root=tmp_path,
    )
    assert default["ok"] is True
    assert default["model"] == "gpt-5.6-sol"
    assert default["effort"] == "medium"

    light = validate_start_request(
        _make_base_req(profile="codex", transport="print", model="gpt-5.6-sol", effort="light"),
        state_root=tmp_path,
    )
    assert light["ok"] is True
    assert light["effort"] == "low"

    # xhigh is intentionally excluded from the public contract
    xhigh = validate_start_request(
        _make_base_req(profile="codex", transport="print", model="gpt-5.6-terra", effort="xhigh"),
        state_root=tmp_path,
    )
    assert xhigh["ok"] is False
    assert xhigh["error"] == "invalid_effort"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"model": "gpt-5.5"}, "invalid_model"),
        ({"model": "gpt-5.6-sol", "effort": "ultra"}, "invalid_effort"),
    ],
)
def test_codex_rejects_unknown_model_or_effort(tmp_path, overrides, error):
    result = validate_start_request(
        _make_base_req(profile="codex", transport="print", **overrides),
        state_root=tmp_path,
    )
    assert result["ok"] is False
    assert result["error"] == error
    assert result["job_created"] is False


def test_opencode_profile_exposes_all_opencode_go_models_and_defaults(tmp_path):
    expected_models = [
        "opencode/deepseek-v4-flash-free",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "glm-5.1",
        "glm-5.2",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "mimo-v2.5",
        "mimo-v2.5-pro",
        "minimax-m2.7",
        "minimax-m3",
    ]

    assert allowed_models("opencode") == expected_models

    # Mock discovery to avoid hitting the live CLI
    catalog = ModelCatalog(
        models=("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"),
        default_model="opencode-go/glm-5.2",
        native_efforts=("low", "medium", "high"),
        source="test",
        model_info=(
            ModelInfo(id="opencode-go/glm-5.2", supported_efforts=("low", "medium", "high")),
            ModelInfo(id="opencode-go/kimi-k2.7-code", supported_efforts=("medium", "high")),
        ),
    )

    import agent_crossbar.discovery as _disc

    claude_catalog = ModelCatalog(
        models=("claude-sonnet-5",),
        default_model="claude-sonnet-5",
        native_efforts=("low", "medium", "high"),
        source="test",
    )

    def discover(state_root, profile, *, refresh=False):
        return claude_catalog if profile == "claude" else catalog

    with patch.object(_disc, "discover_profile_models", side_effect=discover):
        for operation, overrides in (
            ("review", {"transport": "print", "autonomy": "read_only", "model": "kimi-k2.7-code"}),
            ("advice", {"transport": "print", "autonomy": "read_only", "model": "kimi-k2.7-code"}),
            ("text", {"transport": "print", "autonomy": "read_only", "model": "kimi-k2.7-code"}),
            ("dev", {"transport": "print", "autonomy": "edit_local", "model": "kimi-k2.7-code"}),
        ):
            result = validate_start_request(
                _make_base_req(profile="opencode", operation=operation, **overrides),
                state_root=tmp_path,
            )
            assert result["ok"] is True, operation
            assert result["profile"] == "opencode"
            assert result["model"] == "opencode-go/kimi-k2.7-code"

        explicit = validate_start_request(
            _make_base_req(
                profile="opencode",
                operation="dev",
                transport="print",
                autonomy="edit_local",
                model="kimi-k2.7-code",
            ),
            state_root=tmp_path,
        )
        assert explicit["ok"] is True
        assert explicit["model"] == "opencode-go/kimi-k2.7-code"

        invalid = validate_start_request(
            _make_base_req(profile="opencode", operation="review", model="unknown-model"),
            state_root=tmp_path,
        )
        assert invalid["ok"] is False
        assert invalid["error"] == "invalid_model"


def test_resolve_profile_deepseek_alias():
    ok, resolved = resolve_profile("deepseek")
    assert ok is True
    assert resolved == "reasonix"


@pytest.mark.parametrize("profile", ["opus", "fable"])
def test_resolve_profile_accepts_claude_aliases(profile):
    ok, resolved = resolve_profile(profile)
    assert ok is True
    assert resolved == "claude"


# ---------------------------------------------------------------------------
# list_profiles must remain canonical-only (no aliases leaked)
# ---------------------------------------------------------------------------


def test_list_profiles_excludes_aliases():
    """Aliases deepseek, opus, and fable must NOT appear in list_profiles output."""
    profiles = list_profiles()
    assert "deepseek" not in profiles
    assert "opus" not in profiles
    assert "fable" not in profiles
    assert "minimax" not in profiles
    assert "claude" in profiles
    assert "opencode" in profiles
    # Only the five supported canonical profiles
    assert len(profiles) == 5

# ── Model required tests ───────────────────────────────────────────────────

def test_model_omitted_fails_validation(tmp_path):
    """Omitted model must fail before any provider process is created."""
    req = _make_base_req(profile="reasonix", transport="print")
    req.pop("model", None)
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_model"
    assert result["job_created"] is False


def test_model_null_fails_validation(tmp_path):
    """Null model must fail before any provider process is created."""
    req = _make_base_req(profile="reasonix", transport="print", model=None)
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_model"
    assert result["job_created"] is False


def test_model_blank_fails_validation(tmp_path):
    """Blank model must fail before any provider process is created."""
    req = _make_base_req(profile="reasonix", transport="print", model="")
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_model"
    assert result["job_created"] is False


def test_model_blank_whitespace_fails_validation(tmp_path):
    """Whitespace-only model must fail before any provider process is created."""
    req = _make_base_req(profile="reasonix", transport="print", model="   ")
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_model"
    assert result["job_created"] is False


def test_codex_omitted_model_fails(tmp_path):
    """Codex with omitted model must fail — no silent default."""
    req = _make_base_req(profile="codex", transport="print")
    req.pop("model", None)
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_model"


def test_claude_omitted_model_fails(tmp_path):
    """Claude with omitted model must fail — no silent default."""
    req = _make_base_req(profile="claude", transport="print")
    req.pop("model", None)
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_model"


def test_opencode_omitted_model_fails(tmp_path):
    """OpenCode with omitted model must fail — no silent default from catalog."""
    req = _make_base_req(profile="opencode", transport="print")
    req.pop("model", None)
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is False
    assert result["error"] == "missing_model"


def test_model_is_explicitly_passed_for_all_profiles(tmp_path):
    """When model IS provided, all profiles must accept it."""
    from agent_crossbar.adapters.base import ModelCatalog, ModelInfo

    catalog = ModelCatalog(
        models=("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"),
        default_model="opencode-go/glm-5.2",
        native_efforts=("low", "medium", "high"),
        source="test",
        model_info=(
            ModelInfo(id="opencode-go/glm-5.2", supported_efforts=("low", "medium", "high")),
            ModelInfo(id="opencode-go/kimi-k2.7-code", supported_efforts=("medium", "high")),
        ),
    )

    import agent_crossbar.discovery as _disc

    claude_catalog = ModelCatalog(
        models=("claude-sonnet-5",),
        default_model="claude-sonnet-5",
        native_efforts=("low", "medium", "high"),
        source="test",
    )

    def discover(state_root, profile, *, refresh=False):
        return claude_catalog if profile == "claude" else catalog

    with patch.object(_disc, "discover_profile_models", side_effect=discover):
        # All profiles with explicit model should work
        tests = [
            {"profile": "reasonix", "model": "deepseek-v4-flash"},
            {"profile": "codex", "model": "gpt-5.6-sol"},
            {"profile": "claude", "model": "claude-sonnet-5"},
            {"profile": "opencode", "model": "kimi-k2.7-code"},
        ]
        for t in tests:
            req = _make_base_req(**t)
            result = validate_start_request(req, state_root=tmp_path)
            assert result["ok"] is True, f"Failed for {t}: {result}"
            assert result["model"] is not None, f"Model not set for {t}"
