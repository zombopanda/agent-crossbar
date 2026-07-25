"""Profile registry and operation validation tests for the flat, provider-neutral schema."""

import pytest

from agent_crossbar.adapters.base import ModelCatalog
from agent_crossbar.server import profiles_list
from agent_crossbar.validation import validate_start_request


@pytest.fixture(autouse=True)
def _no_real_provider_execution(monkeypatch):
    """Capability tests inspect validation, not live provider CLIs."""

    def fake_start_print_job(store, job_id, req, **kwargs):
        return store.set_result(job_id, True, summary="PROVIDER_OK\n")

    monkeypatch.setattr(
        "agent_crossbar.server.start_print_job",
        fake_start_print_job,
    )


@pytest.fixture(autouse=True)
def _no_live_model_discovery(tmp_path, monkeypatch):
    """profiles_list now attempts bounded live discovery on a cache miss.

    Isolate the state dir (never touch the real ``~/.local/state`` cache)
    and default discovery to a fast, deterministic failure so these tests
    exercise validation logic against the static registry, not whatever
    codex/opencode/claude happen to be installed on the test machine.
    """
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("AGENT_HARNESS_STATE_DIR", raising=False)

    import agent_crossbar.discovery as _disc

    def _boom(state_root, profile, *, refresh=False):
        if profile == "claude":
            return ModelCatalog(
                models=(
                    "claude-opus-5",
                    "claude-sonnet-5",
                    "claude-fable-5",
                    "claude-haiku-4-5",
                ),
                default_model="claude-opus-5",
                native_efforts=("low", "medium", "high"),
                source="test",
            )
        raise RuntimeError(f"no live {profile} probe in unit tests")

    monkeypatch.setattr(_disc, "discover_profile_models", _boom)


def _base_request(**overrides):
    req = {
        "operation": "review",
        "profile": "codex",
        "transport": "print",
        "autonomy": "read_only",
        "sensitivity": "normal",
        "prompt": "x",
    }
    req.update(overrides)
    req.setdefault(
        "model",
        {
            "codex": "gpt-5.6-sol",
            "claude": "claude-sonnet-5",
            "opus": "claude-opus-5",
            "fable": "claude-fable-5",
            "opencode": "opencode/deepseek-v4-flash-free",
            "reasonix": "deepseek-v4-flash",
            "chatgpt_pro": "manual",
        }[req["profile"]],
    )
    return req


def test_unsupported_operations_are_rejected(tmp_path):
    """chatgpt_pro only supports review and advice — text and dev are rejected."""
    for profile, operation in (
        ("chatgpt_pro", "text"),
        ("chatgpt_pro", "dev"),
    ):
        result = validate_start_request(
            _base_request(profile=profile, operation=operation),
            state_root=tmp_path,
        )
        assert result["ok"] is False
        assert result["error"] == "unsupported_operation"
        assert result["job_created"] is False


def test_tmux_transport_rejected_for_non_interactive_profiles(tmp_path):
    """Profiles with interactive=False must reject tmux transport."""
    cases = (
        _base_request(profile="codex", operation="review", transport="tmux"),
        _base_request(profile="opencode", operation="review", transport="tmux"),
    )
    for req in cases:
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is False, f"{req['profile']} + tmux should fail"
        assert result["error"] == "unsupported_transport"


def test_review_with_edit_local_autonomy_is_accepted(tmp_path):
    """Current minimal contract does not gate on autonomy×operation — edit_local is accepted."""
    result = validate_start_request(
        _base_request(profile="codex", operation="review", autonomy="edit_local"),
        state_root=tmp_path,
    )
    assert result["ok"] is True
    assert result["error"] is None


def test_advice_requires_prompt(tmp_path):
    result = validate_start_request(
        _base_request(operation="advice", profile="reasonix", transport="print", prompt=""),
        state_root=tmp_path,
    )
    assert result["ok"] is False
    assert result["error"] == "missing_required_field"


def test_reasonix_review_requires_explicit_flash_model(tmp_path):
    result = validate_start_request(
        _base_request(
            profile="reasonix",
            operation="review",
            model="deepseek-v4-flash",
        ),
        state_root=tmp_path,
    )
    assert result["ok"] is True
    assert result["model"] == "deepseek-v4-flash"


def test_reasonix_review_rejects_unknown_model(tmp_path):
    result = validate_start_request(
        _base_request(profile="reasonix", operation="review", model="deepseek-v4-ultra"),
        state_root=tmp_path,
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_model"


def test_reasonix_dev_validates_with_tmux_and_edit_local(tmp_path):
    """reasonix (interactive=True) supports dev with tmux transport."""
    result = validate_start_request(
        _base_request(
            profile="reasonix",
            operation="dev",
            transport="tmux",
            autonomy="edit_local",
        ),
        state_root=tmp_path,
    )
    assert result["ok"] is True


def test_flat_profile_entries_have_required_fields(tmp_path):
    """Every profile entry has the minimal provider-neutral fields."""
    details = profiles_list(client_name="codex")["profile_details"]
    required = {"models", "operations", "interactive"}
    assert len(details) == 5  # reasonix, codex, claude, opencode, chatgpt_pro
    for name, entry in details.items():
        missing = required - set(entry.keys())
        assert not missing, f"{name} missing fields: {missing}"
        assert isinstance(entry["models"], list), f"{name} models not list"
        assert isinstance(entry["operations"], list), f"{name} operations not list"
        assert isinstance(entry["interactive"], bool), f"{name} interactive not bool"
        if name in ("codex", "claude", "opencode", "reasonix"):
            assert isinstance(entry["aliases"], list), f"{name} aliases missing"


def test_claude_aliases_and_models(tmp_path):
    """Claude profile exposes aliases and models via flat schema."""
    profiles = profiles_list(client_name="codex")["profile_details"]

    assert profiles["claude"]["aliases"] == ["opus", "fable"]
    assert profiles["claude"]["models"] == [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    ]
    assert profiles["claude"]["operations"] == ["review", "advice", "dev"]
    assert profiles["claude"]["interactive"] is True

    # Aliases resolve to canonical claude profile
    for profile in ("claude", "opus", "fable"):
        result = validate_start_request(
            _base_request(profile=profile, operation="review", transport="print"),
            state_root=tmp_path,
        )
        assert result["ok"] is True, profile
        assert result["profile"] == "claude"


def test_opencode_flat_schema(tmp_path):
    """OpenCode profile uses flat schema — no nested capabilities."""
    profiles = profiles_list(client_name="codex")["profile_details"]
    opencode = profiles["opencode"]
    assert opencode["models"] == [
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
    assert opencode["operations"] == ["review", "text", "advice", "dev"]
    assert opencode["interactive"] is False


def test_all_profile_entries_have_support_tier(tmp_path):
    """Every profile entry must include support_tier field."""
    details = profiles_list(client_name="codex")["profile_details"]
    for name, entry in details.items():
        assert "support_tier" in entry, f"{name} missing support_tier"
        assert entry["support_tier"] in ("supported", "experimental"), (
            f"{name} support_tier must be 'supported' or 'experimental', got {entry['support_tier']!r}"
        )


def test_claude_profile_entry_support_tier(tmp_path):
    """Claude profile must declare support_tier='supported'."""
    profiles = profiles_list(client_name="codex")["profile_details"]
    assert profiles["claude"]["support_tier"] == "supported"


def test_provider_support_matrix_exists():
    """PROVIDER_SUPPORT_MATRIX must be defined and cover all canonical profiles."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX, list_profiles

    assert isinstance(PROVIDER_SUPPORT_MATRIX, dict)
    assert set(PROVIDER_SUPPORT_MATRIX.keys()) == set(list_profiles()), (
        f"Matrix missing profiles: {set(list_profiles()) - set(PROVIDER_SUPPORT_MATRIX.keys())}"
    )


_REQUIRED_MATRIX_FIELDS = frozenset(
    {
        "support_tier",
        "os",
        "operations",
        "interaction_modes",
        "effort_support",
        "billing_mode",
    }
)


def test_provider_support_matrix_has_required_fields():
    """Every matrix entry must have all required fields."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    for provider, entry in PROVIDER_SUPPORT_MATRIX.items():
        missing = _REQUIRED_MATRIX_FIELDS - set(entry.keys())
        assert not missing, f"{provider} matrix missing fields: {missing}"
        assert entry["support_tier"] in ("supported", "experimental"), (
            f"{provider} support_tier must be 'supported' or 'experimental'"
        )
        assert isinstance(entry["os"], list), f"{provider} os must be list"
        assert isinstance(entry["operations"], list), f"{provider} operations must be list"


def test_claude_matrix_reports_claude_bg_backend():
    """Claude matrix must truthfully report claude_bg backend and no print/tmux."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    claude = PROVIDER_SUPPORT_MATRIX["claude"]
    assert "backend" in claude, "claude matrix must declare backend"
    assert claude["backend"] == "claude_bg", (
        f"Claude backend must be 'claude_bg', got {claude['backend']!r}"
    )
    # Must not claim print or tmux support
    assert "print" not in claude.get("interaction_modes", []), (
        "Claude must not claim print interaction mode"
    )
    assert "tmux" not in claude.get("interaction_modes", []), (
        "Claude must not claim tmux interaction mode"
    )


def test_claude_matrix_job_send_supported():
    """Claude's native background attach path truthfully supports job_send."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    assert PROVIDER_SUPPORT_MATRIX["claude"].get("job_send_supported") is True, (
        "Claude claude_bg must declare job_send_supported=True"
    )


def test_claude_matrix_no_print_capability():
    """Claude matrix must not include 'print' in interaction_modes."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    modes = PROVIDER_SUPPORT_MATRIX["claude"].get("interaction_modes", [])
    assert "print" not in modes, "Claude must not claim print as an interaction mode"
    check_all = frozenset(modes)
    assert "noninteractive" in check_all, "Claude must declare noninteractive interaction mode"


def test_reasonix_matrix_is_experimental():
    """Reasonix matrix must declare experimental support_tier."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    assert PROVIDER_SUPPORT_MATRIX["reasonix"]["support_tier"] == "experimental"


def test_chatgpt_pro_matrix_is_macos_only():
    """ChatGPT Pro matrix must declare darwin-only OS support."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    assert "darwin" in PROVIDER_SUPPORT_MATRIX["chatgpt_pro"]["os"]
    assert "linux" not in PROVIDER_SUPPORT_MATRIX["chatgpt_pro"]["os"]
    assert PROVIDER_SUPPORT_MATRIX["chatgpt_pro"]["support_tier"] == "experimental"


def test_matrix_entries_owned_by_provider_modules():
    """profiles/__init__.py must assemble the matrix from each provider's own
    build_matrix_entry(), not hardcode a duplicated central matrix."""
    from agent_crossbar.profiles import (
        PROVIDER_SUPPORT_MATRIX,
        chatgpt_pro,
        claude,
        codex,
        opencode,
        reasonix,
    )

    expected = {
        "reasonix": reasonix.build_matrix_entry(),
        "codex": codex.build_matrix_entry(),
        "claude": claude.build_matrix_entry(),
        "opencode": opencode.build_matrix_entry(),
        "chatgpt_pro": chatgpt_pro.build_matrix_entry(),
    }
    assert PROVIDER_SUPPORT_MATRIX == expected


def test_matrix_interaction_modes_are_provider_neutral():
    """interaction_modes must only ever contain provider-neutral labels, never
    a raw transport/backend name (print, tmux, acp, claude_bg, gui)."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    allowed = {"noninteractive", "interactive"}
    transport_labels = {"print", "tmux", "acp", "claude_bg", "gui"}
    for provider, entry in PROVIDER_SUPPORT_MATRIX.items():
        modes = set(entry["interaction_modes"])
        assert modes <= allowed, f"{provider} interaction_modes not provider-neutral: {modes}"
        assert not (modes & transport_labels), (
            f"{provider} interaction_modes leaks a transport label: {modes}"
        )


def test_matrix_backend_field_matches_provider_transport():
    """The internal (non-public) matrix backend field must reflect the real transport."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    expected_backends = {
        "claude": "claude_bg",
        "codex": "acp",
        "opencode": "acp",
        "reasonix": "tmux",
        "chatgpt_pro": "gui",
    }
    for provider, backend in expected_backends.items():
        assert PROVIDER_SUPPORT_MATRIX[provider]["backend"] == backend


def test_matrix_uses_billing_mode_not_billing():
    """billing_mode is the one stable field name; a bare 'billing' key must never appear."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    for provider, entry in PROVIDER_SUPPORT_MATRIX.items():
        assert "billing" not in entry, f"{provider} matrix must use billing_mode, not billing"
        assert "billing_mode" in entry, f"{provider} matrix missing billing_mode"


def test_reasonix_matrix_supports_both_interaction_modes():
    """Reasonix truthfully supports both noninteractive (print) and interactive (tmux)."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    entry = PROVIDER_SUPPORT_MATRIX["reasonix"]
    assert set(entry["interaction_modes"]) == {"noninteractive", "interactive"}
    assert entry["job_send_supported"] is True


def test_single_shot_profiles_do_not_support_job_send():
    """Codex, OpenCode, and ChatGPT Pro are single-shot: noninteractive only."""
    from agent_crossbar.profiles import PROVIDER_SUPPORT_MATRIX

    for provider in ("codex", "opencode", "chatgpt_pro"):
        entry = PROVIDER_SUPPORT_MATRIX[provider]
        assert entry["interaction_modes"] == ["noninteractive"], provider
        assert entry["job_send_supported"] is False, provider


def test_profiles_list_has_no_legacy_fields(tmp_path, monkeypatch):
    """profiles_list must not expose budget, cloud_backed, capabilities, or transports."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    result = profiles_list(client_name="codex")
    assert result["ok"] is True
    legacy = {
        "budget",
        "cloud_backed",
        "capabilities",
        "transports",
        "blocked_operations",
        "enforcement",
        "context_gathering",
        "denied_path_filtering",
        "manifest_accuracy",
    }
    for name, entry in result["profile_details"].items():
        overlap = legacy & set(entry.keys())
        assert not overlap, f"{name} has legacy fields: {overlap}"
