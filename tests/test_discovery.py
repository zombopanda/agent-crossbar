"""Tests for live model discovery: parsers, process protocol, cache, profile_health, effort rejection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent_crossbar.adapters.base import ModelCatalog, ModelInfo
from agent_crossbar.adapters.codex import (
    CodexAdapter,
    _max_effort_for_model,
    parse_model_list_response,
)
from agent_crossbar.adapters.opencode import OpencodeAdapter, parse_models_output
from agent_crossbar.discovery import (
    _PROFILES_WITH_DISCOVERY,
    EFFORT_LEGEND,
    _catalog_to_dict,
    _compact_model_info,
    _compact_model_info_with_fallback,
    _dict_to_catalog,
    _get_cli_version,
    _group_model_info_by_provider,
    build_profile_health_entry,
    cached_models_for_listing,
    cached_profile_health_entry,
    discover_profile_models,
    live_profile_registry,
)
from agent_crossbar.discovery_runner import (
    DiscoveryRun,
    PopenCodexSession,
    _read_jsonl_line,
    discover_codex_models,
)
from agent_crossbar.model_cache import (
    cache_get_or_fetch,
    write_cache,
)

# ── Fake runner for deterministic process-boundary tests ────────────────────


class FakeDiscoveryProcess:
    """Deterministic DiscoveryProcess that returns canned responses."""

    def __init__(self, results: list[DiscoveryRun] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict] = []

    def run(self, args, *, timeout=None, cwd=None):
        self.calls.append({"args": list(args), "timeout": timeout, "cwd": cwd})
        if not self.results:
            raise AssertionError(f"Unexpected subprocess call: {args}")
        return self.results.pop(0)


# ── Codex parser tests ──────────────────────────────────────────────────────


def test_codex_parser_extracts_per_model_efforts() -> None:
    result = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                        {"reasoningEffort": "xhigh"},
                    ],
                },
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "hidden": False,
                    "isDefault": False,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
                {
                    "id": "hidden-model",
                    "model": "hidden-model",
                    "hidden": True,
                    "isDefault": False,
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [],
                },
            ]
        }
    )

    assert result.models == ("gpt-5.6-sol", "gpt-5.6-terra")
    assert result.default_model == "gpt-5.6-sol"
    assert result.native_efforts == ("low", "medium", "high", "xhigh")

    # Per-model info
    assert len(result.model_info) == 2  # hidden excluded
    sol_info = result.model_info[0]
    assert sol_info.id == "gpt-5.6-sol"
    assert sol_info.supported_efforts == ("low", "medium", "high", "xhigh")
    assert sol_info.default_effort == "medium"

    terra_info = result.model_info[1]
    assert terra_info.id == "gpt-5.6-terra"
    assert terra_info.supported_efforts == ("medium", "high")
    assert terra_info.default_effort == "high"


def test_max_effort_for_model_returns_highest_priority() -> None:
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "medium",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                        {"reasoningEffort": "xhigh"},
                    ],
                },
            ]
        }
    )

    assert _max_effort_for_model(catalog, "gpt-5.6-sol") == "xhigh"
    assert _max_effort_for_model(catalog, "nonexistent") is None


def test_resolve_effort_max_to_strongest_available_high_only() -> None:
    """max resolves to 'high' when that is the strongest native effort advertised."""
    adapter = CodexAdapter()
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
            ]
        }
    )

    # max → high (strongest available, no native max or xhigh)
    assert adapter.resolve_effort("max", catalog, "gpt-5.6-terra") == "high"
    # Without catalog, falls back to static effort_map (max → "max")
    assert adapter.resolve_effort("max", None, None) == "max"


def test_codex_validate_effort_rejects_unsupported() -> None:
    adapter = CodexAdapter()
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
            ]
        }
    )

    # "low" is not in supported efforts for terra
    err = adapter.validate_effort_for_model("low", catalog, "gpt-5.6-terra")
    assert err is not None
    assert "low" in err

    # "medium" is supported
    err = adapter.validate_effort_for_model("medium", catalog, "gpt-5.6-terra")
    assert err is None

    # No catalog = no validation (graceful)
    err = adapter.validate_effort_for_model("xhigh", None, "gpt-5.6-terra")
    assert err is None


# ── OpenCode parser tests ────────────────────────────────────────────────────


def test_opencode_parser_extracts_slash_models() -> None:
    result = parse_models_output(
        "opencode-go/glm-5.2\nanthropic/claude-opus-4-8\nopencode-go/deepseek-v4-pro\n"
    )

    assert result == [
        "opencode-go/glm-5.2",
        "anthropic/claude-opus-4-8",
        "opencode-go/deepseek-v4-pro",
    ]


def test_opencode_parser_accepts_colon_tags() -> None:
    """parse_models_output must accept model IDs with colon tags (:free, :thinking)."""
    result = parse_models_output(
        "openrouter/cohere/north-mini-code:free\n"
        "openrouter/qwen/qwen-plus-2025-07-28:thinking\n"
        "openrouter/meta-llama/llama-4-maverick:free\n"
    )

    assert result == [
        "openrouter/cohere/north-mini-code:free",
        "openrouter/qwen/qwen-plus-2025-07-28:thinking",
        "openrouter/meta-llama/llama-4-maverick:free",
    ]


def test_opencode_parser_rejects_urls_on_bare_lines() -> None:
    """parse_models_output must reject bare URLs (://) even though : is now allowed."""
    result = parse_models_output(
        "opencode-go/glm-5.2\n"
        "https://api.openai.com/v1/models/gpt-4\n"
        "http://localhost:11434/v1/models/llama3\n"
        "opencode-go/kimi-k2.7-code\n"
    )

    assert result == [
        "opencode-go/glm-5.2",
        "opencode-go/kimi-k2.7-code",
    ], f"URLs must be rejected, got: {result}"


def test_opencode_parser_rejects_malformed_ids() -> None:
    """parse_models_output must reject malformed IDs: no slash, empty segments, whitespace, quotes."""
    result = parse_models_output(
        "opencode-go/glm-5.2\n"
        "no-slash-at-all\n"
        "provider/\n"
        "/model-only\n"
        "provider//empty-segment\n"
        '{"json":"impostor"}\n'
        "provider/model with spaces\n"
        '  "quoted/thing": true\n'
    )

    assert result == ["opencode-go/glm-5.2"], f"Malformed IDs must be rejected, got: {result}"


def test_opencode_discover_models_returns_catalog() -> None:
    runner = FakeDiscoveryProcess(
        [DiscoveryRun(0, "opencode-go/glm-5.2\nopencode-go/kimi-k2.7-code\n", "")]
    )
    adapter = OpencodeAdapter()
    catalog = adapter.discover_models(runner)

    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code")
    assert catalog.default_model == "opencode-go/glm-5.2"
    assert catalog.source == "opencode models --verbose"


def test_opencode_discover_models_handles_error() -> None:
    runner = FakeDiscoveryProcess(
        [
            DiscoveryRun(1, "", "opencode: command not found"),  # --verbose fails
            DiscoveryRun(1, "", "opencode: command not found"),  # fallback also fails
        ]
    )
    adapter = OpencodeAdapter()
    catalog = adapter.discover_models(runner)

    assert catalog.models == ()
    assert catalog.error is not None
    assert "exited 1" in catalog.error


# ── OpenCode qualified-default selection tests ──────────────────────────────
# Regression: live discovery listed "opencode/big-pickle" as the first CLI
# row and the old code picked model_ids[0] unconditionally, so the launched
# default was an unqualified provider/model that the ACP transport rejected
# ("No provider available"). The configured qualified default
# OPENCODE_DEFAULT_MODEL must win whenever it is
# present in the live list, regardless of row order.


def test_opencode_verbose_parser_prefers_qualified_default_when_not_first() -> None:
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    OPENCODE_DEFAULT_MODEL = (
        "opencode/deepseek-v4-flash-free"  # moved internal, define inline for tests
    )

    qualified_default = OPENCODE_DEFAULT_MODEL
    output = (
        "opencode/big-pickle\n"
        '{"variants":{"low":{},"medium":{}}}\n'
        f"{qualified_default}\n"
        '{"variants":{"low":{},"medium":{},"high":{}}}\n'
        "opencode-go/kimi-k2.7-code\n"
        '{"variants":{"medium":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    # All live models must still be returned, in discovery order.
    assert catalog.models == (
        "opencode/big-pickle",
        qualified_default,
        "opencode-go/kimi-k2.7-code",
    )
    # But the qualified default wins even though it wasn't the first row.
    assert catalog.default_model == qualified_default


def test_opencode_verbose_parser_falls_back_deterministically_when_qualified_default_absent() -> (
    None
):
    """Without the qualified default present, fall back to the first discovered
    model — deterministic, but not a claim that it is a qualified/supported
    default."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode/big-pickle\n"
        '{"variants":{"low":{}}}\n'
        "opencode-go/kimi-k2.7-code\n"
        '{"variants":{"medium":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode/big-pickle", "opencode-go/kimi-k2.7-code")
    assert catalog.default_model == "opencode/big-pickle"


def test_opencode_nonverbose_fallback_prefers_qualified_default_when_not_first() -> None:
    """The non-verbose ``opencode models`` fallback path must apply the same
    qualified-default preference as the verbose parser."""
    OPENCODE_DEFAULT_MODEL = (
        "opencode/deepseek-v4-flash-free"  # moved internal, define inline for tests
    )

    qualified_default = OPENCODE_DEFAULT_MODEL
    runner = FakeDiscoveryProcess(
        [
            DiscoveryRun(1, "", "opencode: --verbose not supported"),  # --verbose fails
            DiscoveryRun(0, f"opencode/big-pickle\n{qualified_default}\n", ""),  # fallback succeeds
        ]
    )
    adapter = OpencodeAdapter()
    catalog = adapter.discover_models(runner)

    assert catalog.models == ("opencode/big-pickle", qualified_default)
    assert catalog.default_model == qualified_default


# ── ModelCatalog round-trip tests ────────────────────────────────────────────


def test_catalog_dict_roundtrip() -> None:
    original = ModelCatalog(
        models=("gpt-5.6-sol", "gpt-5.6-terra"),
        default_model="gpt-5.6-sol",
        native_efforts=("low", "medium", "high", "xhigh"),
        source="live",
        model_info=(
            ModelInfo(
                id="gpt-5.6-sol",
                supported_efforts=("low", "medium", "high", "xhigh"),
                default_effort="medium",
            ),
            ModelInfo(
                id="gpt-5.6-terra", supported_efforts=("medium", "high"), default_effort="high"
            ),
        ),
        cli_version="0.145.0",
        fetched_at=1234567890.0,
        stale=False,
    )

    d = _catalog_to_dict(original)
    restored = _dict_to_catalog(d, source="live", version="0.145.0", fetched_at=1234567890.0)

    assert restored.models == original.models
    assert restored.default_model == original.default_model
    assert restored.native_efforts == original.native_efforts
    assert len(restored.model_info) == 2
    assert restored.model_info[0].id == "gpt-5.6-sol"
    assert restored.model_info[0].supported_efforts == ("low", "medium", "high", "xhigh")
    assert restored.model_info[0].default_effort == "medium"


# ── Cache tests ─────────────────────────────────────────────────────────────


def test_cache_hit_returns_fresh_data(tmp_path: Path) -> None:
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.145.0", data)

    result, from_cache = cache_get_or_fetch(
        tmp_path, "codex", "0.145.0", lambda: {"models": [], "source": "fresh"}
    )

    assert from_cache is True
    assert result["models"] == ["gpt-5.6-sol"]


def test_cache_version_mismatch_bypasses_cache(tmp_path: Path) -> None:
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.144.0", data)

    fresh = {"models": ["gpt-5.6-terra"], "source": "fresh"}
    result, from_cache = cache_get_or_fetch(tmp_path, "codex", "0.145.0", lambda: fresh)

    assert from_cache is False
    assert result["models"] == ["gpt-5.6-terra"]


def test_cache_refresh_bypasses_cache(tmp_path: Path) -> None:
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.145.0", data)

    fresh = {"models": ["gpt-5.6-terra"], "source": "fresh"}
    result, from_cache = cache_get_or_fetch(
        tmp_path, "codex", "0.145.0", lambda: fresh, refresh=True
    )

    assert from_cache is False
    assert result["models"] == ["gpt-5.6-terra"]


def test_cache_expired_ttl_bypasses_cache(tmp_path: Path) -> None:
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.145.0", data)

    # Manually age the cache
    path = tmp_path / "model_cache" / "codex.json"
    entry = json.loads(path.read_text())
    entry["fetched_at"] = time.time() - 7200  # 2 hours ago
    path.write_text(json.dumps(entry))

    fresh = {"models": ["gpt-5.6-terra"], "source": "fresh"}
    result, from_cache = cache_get_or_fetch(
        tmp_path, "codex", "0.145.0", lambda: fresh, ttl_seconds=3600
    )

    assert from_cache is False
    assert result["models"] == ["gpt-5.6-terra"]


def test_cache_last_good_preserved_on_write_failure(tmp_path: Path) -> None:
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.145.0", data)

    path = tmp_path / "model_cache" / "codex.json"
    original_content = path.read_text()

    # Patch _atomic_write to simulate a write failure, then verify last-good
    # is restored.  The original chmod trick was racy on macOS (owner override).
    import agent_crossbar.model_cache as mc

    _orig_atomic = mc._atomic_write

    def _failing_atomic(p: Path, d: dict[str, Any]) -> None:
        raise OSError("simulated disk full")

    mc._atomic_write = _failing_atomic
    try:
        with pytest.raises(OSError, match="simulated disk full"):
            write_cache(tmp_path, "codex", "0.145.0", {"models": ["bad"], "source": "bad"})
    finally:
        mc._atomic_write = _orig_atomic

    # The original file should still be intact
    assert path.read_text() == original_content


def test_cache_fetched_at_propagated(tmp_path: Path) -> None:
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.145.0", data)

    result, from_cache = cache_get_or_fetch(
        tmp_path, "codex", "0.145.0", lambda: {"models": [], "source": "fresh"}
    )

    assert from_cache is True
    assert "fetched_at" in result
    assert isinstance(result["fetched_at"], float)


# ── profile_health shape tests ──────────────────────────────────────────────


def test_profile_health_codex_live_shape(tmp_path: Path) -> None:
    """profile_health for codex returns structured discovery data."""
    # Pre-populate cache; pin _get_cli_version so the cache key matches
    import agent_crossbar.discovery as _disc

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "0.145.0"
    try:
        data = {
            "models": ["gpt-5.6-sol", "gpt-5.6-terra"],
            "default_model": "gpt-5.6-sol",
            "native_efforts": ["low", "medium", "high", "xhigh"],
            "model_info": [
                {
                    "id": "gpt-5.6-sol",
                    "supported_efforts": ["low", "medium", "high", "xhigh"],
                    "default_effort": "medium",
                },
                {
                    "id": "gpt-5.6-terra",
                    "supported_efforts": ["medium", "high"],
                    "default_effort": "high",
                },
            ],
            "source": "codex app-server model/list",
        }
        write_cache(tmp_path, "codex", "0.145.0", data)

        catalog = discover_profile_models(tmp_path, "codex")
        entry = build_profile_health_entry(tmp_path, "codex", catalog)

        assert entry["name"] == "codex"
        assert entry["runtime_checked"] is True
        assert entry["discovery_available"] is True
        assert entry["default_model"] == "gpt-5.6-sol"
        assert entry["native_efforts"] == ["low", "medium", "high", "xhigh"]
        # plain (no "/") ids group under the "" provider key
        assert list(entry["model_info"].keys()) == [""]
        assert len(entry["model_info"][""]) == 2
        assert entry["model_info"][""][0]["id"] == "gpt-5.6-sol"
        assert entry["model_info"][""][0]["e"] == [1, 2, 3, 5]  # low, medium, high, xhigh
        assert entry["model_info"][""][0]["d"] == 2  # medium
        assert entry["stale"] is False  # fresh TTL-valid cache
        assert entry["cache_hit"] is True
        assert entry["source"] == "codex app-server model/list"  # preserves provider source
    finally:
        _disc._get_cli_version = _orig_version


def test_profile_health_claude_live_shape(tmp_path: Path) -> None:
    """Claude profile now supports live model discovery via /model picker."""
    import agent_crossbar.discovery as _disc

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "2.1.211"
    try:
        data = {
            "models": ["claude-opus-4-8", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5"],
            "default_model": "claude-opus-4-8",
            "native_efforts": ["low", "medium", "high"],
            "model_info": [
                {
                    "id": "claude-opus-4-8",
                    "supported_efforts": ["low", "medium", "high"],
                    "default_effort": "medium",
                },
                {
                    "id": "claude-sonnet-5",
                    "supported_efforts": ["medium", "high"],
                    "default_effort": "medium",
                },
                {
                    "id": "claude-fable-5",
                    "supported_efforts": ["low", "medium"],
                    "default_effort": "low",
                },
                {"id": "claude-haiku-4-5", "supported_efforts": ["low"], "default_effort": "low"},
            ],
            "source": "claude interactive /model picker",
        }
        write_cache(tmp_path, "claude", "2.1.211", data)

        catalog = discover_profile_models(tmp_path, "claude")
        entry = build_profile_health_entry(tmp_path, "claude", catalog)

        assert entry["name"] == "claude"
        assert entry["runtime_checked"] is True
        assert entry["discovery_available"] is True
        assert entry["default_model"] == "claude-opus-4-8"
        assert entry["native_efforts"] == ["low", "medium", "high"]
        assert list(entry["model_info"].keys()) == [""]
        assert len(entry["model_info"][""]) == 4
        assert entry["source"] == "claude interactive /model picker"
        assert entry["stale"] is False
        assert entry["cache_hit"] is True
    finally:
        _disc._get_cli_version = _orig_version


def test_profile_health_reasonix_returns_honest_unavailable(tmp_path: Path) -> None:
    """Reasonix profile has discovery_available: False with honest error."""
    catalog = discover_profile_models(tmp_path, "reasonix")
    entry = build_profile_health_entry(tmp_path, "reasonix", catalog)

    assert entry["discovery_available"] is False
    assert "does not support live model discovery" in entry["error"]


def test_profile_health_chatgpt_pro_returns_honest_unavailable(tmp_path: Path) -> None:
    """chatgpt_pro profile has discovery_available: False."""
    catalog = discover_profile_models(tmp_path, "chatgpt_pro")
    entry = build_profile_health_entry(tmp_path, "chatgpt_pro", catalog)

    assert entry["discovery_available"] is False


def test_profile_health_opencode_live_shape(tmp_path: Path, monkeypatch) -> None:
    """profile_health for opencode returns structured discovery data."""
    monkeypatch.setattr(
        "agent_crossbar.discovery._get_cli_version",
        lambda _profile: "1.18.4",
    )
    data = {
        "models": ["opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"],
        "default_model": "opencode-go/glm-5.2",
        "native_efforts": [],
        "model_info": [
            {"id": "opencode-go/glm-5.2", "supported_efforts": [], "default_effort": None},
            {"id": "opencode-go/kimi-k2.7-code", "supported_efforts": [], "default_effort": None},
        ],
        "source": "opencode models",
    }
    write_cache(tmp_path, "opencode", "1.18.4", data)

    catalog = discover_profile_models(tmp_path, "opencode")
    entry = build_profile_health_entry(tmp_path, "opencode", catalog)

    assert entry["name"] == "opencode"
    assert entry["runtime_checked"] is True
    assert entry["discovery_available"] is True
    # "opencode-go/" prefix groups the ids and is stripped from each entry
    assert list(entry["model_info"].keys()) == ["opencode-go"]
    assert entry["model_info"]["opencode-go"] == [{"id": "glm-5.2"}, {"id": "kimi-k2.7-code"}]
    assert entry["stale"] is False  # fresh TTL-valid cache
    assert entry["cache_hit"] is True
    assert entry["source"] == "opencode models"


def test_compact_model_info_encodes_efforts_as_legend_codes() -> None:
    """model_info entries drop spelled-out effort names in favor of small
    int codes resolvable via EFFORT_LEGEND — this is what keeps
    profile_health from ballooning to tens of KB per catalog with
    hundreds of models."""
    raw = [
        {"id": "model-a", "supported_efforts": ["low", "high", "max"], "default_effort": "high"},
        {"id": "model-b", "supported_efforts": [], "default_effort": None},
    ]

    compact = _compact_model_info(raw)

    assert compact == [
        {"id": "model-a", "e": [1, 3, 4], "d": 3},
        {"id": "model-b"},
    ]
    # every code used above resolves back to its name via the legend
    assert EFFORT_LEGEND["1"] == "low"
    assert EFFORT_LEGEND["3"] == "high"
    assert EFFORT_LEGEND["4"] == "max"
    # a model with no effort variants gets neither "e" nor "d" at all
    assert "e" not in compact[1]
    assert "d" not in compact[1]


def test_compact_model_info_accepts_modelinfo_objects() -> None:
    """The same encoder must handle live ModelInfo objects from a fresh
    ModelCatalog, not just raw cache dicts."""
    mi = ModelInfo(id="model-a", supported_efforts=("medium", "ultra"), default_effort="medium")

    compact = _compact_model_info([mi])

    assert compact == [{"id": "model-a", "e": [2, 6], "d": 2}]


def test_group_model_info_by_provider_strips_shared_prefix() -> None:
    """Ids sharing a ``provider/`` prefix (repeated hundreds of times in a
    large catalog like opencode's) are grouped and the prefix is dropped
    from each entry; plain ids without a ``/`` land under the "" group."""
    compact = [
        {"id": "opencode-go/deepseek-v4-flash", "e": [3, 4]},
        {"id": "opencode-go/glm-5.1"},
        {"id": "google-vertex/gemini-3.5-flash"},
        {"id": "haiku"},
    ]

    grouped = _group_model_info_by_provider(compact)

    assert grouped == {
        "opencode-go": [
            {"id": "deepseek-v4-flash", "e": [3, 4]},
            {"id": "glm-5.1"},
        ],
        "google-vertex": [{"id": "gemini-3.5-flash"}],
        "": [{"id": "haiku"}],
    }


def test_compact_model_info_with_fallback_recovers_ids_when_model_info_empty() -> None:
    """A cache record with a populated ``models`` list but empty/missing
    ``model_info`` (older entry, or a fetch that didn't capture per-model
    effort metadata) must not silently lose its model ids — it falls back
    to bare ``{"id": ...}`` entries instead of vanishing from profile_health."""
    compact = _compact_model_info_with_fallback([], ["opencode-go/glm-5.1", "opencode-go/kimi-k3"])

    assert compact == [{"id": "opencode-go/glm-5.1"}, {"id": "opencode-go/kimi-k3"}]


def test_compact_model_info_with_fallback_prefers_real_model_info() -> None:
    """When model_info is actually populated, the fallback must not kick in
    and override it with id-only entries."""
    raw = [{"id": "model-a", "supported_efforts": ["high"], "default_effort": "high"}]

    compact = _compact_model_info_with_fallback(raw, ["model-a"])

    assert compact == [{"id": "model-a", "e": [3], "d": 3}]


def test_compact_model_info_with_fallback_stays_empty_when_no_ids_at_all() -> None:
    """No model_info and no models list at all → still an empty list, not a
    spurious entry."""
    assert _compact_model_info_with_fallback([], []) == []


# ── Model discovery provenance tests ────────────────────────────────────────


def test_fresh_live_provenance(tmp_path: Path) -> None:
    """Fresh live fetch: cache_hit=False, stale=False, source=provider."""
    import agent_crossbar.discovery as _disc

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "0.145.0"

    # Patch cache_get_or_fetch at the discovery module's reference, not model_cache
    def _fake_get_or_fetch(sr, p, v, f, refresh=False, ttl_seconds=3600):
        return {
            "models": ["gpt-5.6-terra"],
            "default_model": "gpt-5.6-terra",
            "native_efforts": ["medium", "high"],
            "model_info": [
                {
                    "id": "gpt-5.6-terra",
                    "supported_efforts": ["medium", "high"],
                    "default_effort": "high",
                }
            ],
            "source": "codex app-server model/list",
        }, False

    _orig_get_or_fetch = _disc.cache_get_or_fetch
    _disc.cache_get_or_fetch = _fake_get_or_fetch
    try:
        catalog = discover_profile_models(tmp_path, "codex")
        entry = build_profile_health_entry(tmp_path, "codex", catalog)

        assert entry["cache_hit"] is False
        assert entry["stale"] is False
        assert entry["source"] == "codex app-server model/list"
        assert entry["error"] is None
    finally:
        _disc.cache_get_or_fetch = _orig_get_or_fetch
        _disc._get_cli_version = _orig_version


def test_fresh_cache_hit_provenance(tmp_path: Path) -> None:
    """Fresh TTL-valid cache: cache_hit=True, stale=False, source=provider."""
    import agent_crossbar.discovery as _disc

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "0.145.0"
    try:
        data = {
            "models": ["gpt-5.6-sol"],
            "default_model": "gpt-5.6-sol",
            "native_efforts": ["low", "medium", "high"],
            "model_info": [
                {
                    "id": "gpt-5.6-sol",
                    "supported_efforts": ["low", "medium", "high"],
                    "default_effort": "medium",
                }
            ],
            "source": "codex app-server model/list",
        }
        write_cache(tmp_path, "codex", "0.145.0", data)

        catalog = discover_profile_models(tmp_path, "codex")
        entry = build_profile_health_entry(tmp_path, "codex", catalog)

        assert entry["cache_hit"] is True
        assert entry["stale"] is False
        assert entry["source"] == "codex app-server model/list"
        assert entry["error"] is None
    finally:
        _disc._get_cli_version = _orig_version


def test_last_good_fallback_provenance(tmp_path: Path) -> None:
    """Last-good fallback after failed fetch: cache_hit=True, stale=True, error set, source preserved."""
    last_good_result = {
        "models": ["gpt-5.6-sol"],
        "default_model": "gpt-5.6-sol",
        "native_efforts": ["low", "medium", "high"],
        "model_info": [
            {
                "id": "gpt-5.6-sol",
                "supported_efforts": ["low", "medium", "high"],
                "default_effort": "medium",
            }
        ],
        "source": "codex app-server model/list",
        "fetched_at": time.time() - 7200,
        "stale": True,
        "error": "codex: connection refused",
    }

    with (
        patch("agent_crossbar.discovery.cache_get_or_fetch", return_value=(last_good_result, True)),
        patch("agent_crossbar.discovery._get_cli_version", return_value="0.145.0"),
    ):
        catalog = discover_profile_models(tmp_path, "codex")
        entry = build_profile_health_entry(tmp_path, "codex", catalog)

        assert entry["cache_hit"] is True
        assert entry["stale"] is True
        assert entry["source"] == "codex app-server model/list"
        assert entry["error"] == "codex: connection refused"


# ── Pagination / JSON-lines protocol tests ───────────────────────────────────


def test_jsonl_line_reader_single_line() -> None:
    buffer = bytearray(b'{"key": "value"}\n')
    line, rest = _read_jsonl_line(buffer)
    assert line == '{"key": "value"}'
    assert len(rest) == 0


def test_jsonl_line_reader_multiple_lines() -> None:
    buffer = bytearray(b'{"a": 1}\n{"b": 2}\n')
    line1, rest = _read_jsonl_line(buffer)
    assert line1 == '{"a": 1}'
    line2, rest = _read_jsonl_line(rest)
    assert line2 == '{"b": 2}'
    assert len(rest) == 0


def test_jsonl_line_reader_incomplete_line() -> None:
    buffer = bytearray(b'{"key":')
    line, rest = _read_jsonl_line(buffer)
    assert line is None
    assert rest == buffer  # unchanged


def test_jsonl_line_reader_empty() -> None:
    line, rest = _read_jsonl_line(bytearray(b""))
    assert line is None
    assert len(rest) == 0


# ── Codex discover_models with FakeRunner ────────────────────────────────────


def test_codex_discover_models_with_fake_runner() -> None:
    """Test CodexAdapter.discover_models end-to-end with a FakeDiscoveryProcess."""
    runner = FakeDiscoveryProcess(
        [DiscoveryRun(0, "opencode-go/glm-5.2\nopencode-go/kimi-k2.7-code\n", "")]
    )
    adapter = OpencodeAdapter()
    catalog = adapter.discover_models(runner)

    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code")


def test_codex_discover_models_error_catalog() -> None:
    runner = FakeDiscoveryProcess(
        [
            DiscoveryRun(1, "", "command not found"),  # --verbose fails
            DiscoveryRun(1, "", "command not found"),  # fallback fails
        ]
    )
    adapter = OpencodeAdapter()
    catalog = adapter.discover_models(runner)

    assert catalog.models == ()
    assert catalog.error is not None


# ── PROFILES_WITH_DISCOVERY constant ─────────────────────────────────────────


def test_discovery_profiles_set() -> None:
    assert _PROFILES_WITH_DISCOVERY == {"codex", "opencode", "claude"}


# ── Claude adapter discover_models with fake probe ────────────────────────────


def test_claude_discover_models_probe_error_returns_honest_empty() -> None:
    """Claude discover_models returns honest empty catalog on probe error."""
    from agent_crossbar.adapters.claude import ClaudeAdapter
    from agent_crossbar.adapters.claude_model_probe import ProbeResult

    class FakeProbe:
        def probe(self) -> ProbeResult:
            return ProbeResult(output=None, error="timeout: probe failed")

    adapter = ClaudeAdapter()
    catalog = adapter.discover_models(probe=FakeProbe(), help_output="")

    assert catalog.models == ()
    assert catalog.error is not None
    assert "timeout" in catalog.error
    assert catalog.source == "claude interactive /model picker"


def test_claude_discover_models_auth_prompt_returns_empty() -> None:
    """Claude discover_models returns empty catalog on auth prompt (never fakes)."""
    from agent_crossbar.adapters.claude import ClaudeAdapter
    from agent_crossbar.adapters.claude_model_probe import ProbeResult

    class FakeProbe:
        def probe(self) -> ProbeResult:
            return ProbeResult(
                output="Please run `claude auth login` first\n> ",
                error=None,
            )

    adapter = ClaudeAdapter()
    catalog = adapter.discover_models(probe=FakeProbe(), help_output="")

    assert catalog.models == ()
    assert catalog.error is not None
    assert "auth" in catalog.error.lower()


# ── Effort rejection: no job created for provably unsupported model effort ──


def test_validation_rejects_effort_unsupported_by_model(tmp_path: Path) -> None:
    """validate_start_request rejects a codex job when the model doesn't support the effort."""
    # Pin _get_cli_version so the cache key matches
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "0.145.0"
    try:
        # Pre-populate cache so discovery returns terra with only medium+high
        data = {
            "models": ["gpt-5.6-sol", "gpt-5.6-terra"],
            "default_model": "gpt-5.6-sol",
            "native_efforts": ["low", "medium", "high", "xhigh"],
            "model_info": [
                {
                    "id": "gpt-5.6-sol",
                    "supported_efforts": ["low", "medium", "high", "xhigh"],
                    "default_effort": "medium",
                },
                {
                    "id": "gpt-5.6-terra",
                    "supported_efforts": ["medium", "high"],
                    "default_effort": "high",
                },
            ],
            "source": "codex app-server model/list",
        }
        write_cache(tmp_path, "codex", "0.145.0", data)

        # Request terra with "low" effort — terra only supports medium+high
        req = {
            "operation": "review",
            "profile": "codex",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "gpt-5.6-terra",
            "effort": "low",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is False
        assert result["error"] == "unsupported_effort_for_model"
        assert result["job_created"] is False
        assert "low" in result["message"]
    finally:
        _disc._get_cli_version = _orig_version


def test_validation_allows_effort_supported_by_model(tmp_path: Path) -> None:
    """validate_start_request allows a codex job when the model supports the effort."""
    from agent_crossbar.validation import validate_start_request

    data = {
        "models": ["gpt-5.6-sol", "gpt-5.6-terra"],
        "default_model": "gpt-5.6-sol",
        "native_efforts": ["low", "medium", "high", "xhigh"],
        "model_info": [
            {
                "id": "gpt-5.6-sol",
                "supported_efforts": ["low", "medium", "high", "xhigh"],
                "default_effort": "medium",
            },
        ],
        "source": "codex app-server model/list",
    }
    write_cache(tmp_path, "codex", "0.145.0", data)

    # Request sol with "high" — sol supports it
    req = {
        "operation": "review",
        "profile": "codex",
        "transport": "print",
        "autonomy": "read_only",
        "external_context": "allowed",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "gpt-5.6-sol",
        "effort": "high",
    }
    result = validate_start_request(req, state_root=tmp_path)
    assert result["ok"] is True
    assert result["effort"] == "high"


def test_validation_fails_when_codex_discovery_fails(tmp_path: Path) -> None:
    """Codex validation must not fall back to a static model allowlist."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    req = {
        "operation": "review",
        "profile": "codex",
        "transport": "print",
        "autonomy": "read_only",
        "external_context": "allowed",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "gpt-5.6-sol",
        "effort": "medium",
    }
    with patch.object(_disc, "discover_profile_models", side_effect=RuntimeError("CLI not found")):
        result = validate_start_request(req, state_root=tmp_path)

    assert result["ok"] is False
    assert result["error"] == "discovery_error"
    assert result["job_created"] is False


# ── CLI version detection ───────────────────────────────────────────────────


def test_get_cli_version_unknown_for_non_discovery_profiles() -> None:
    """Non-discovery profiles return 'unknown'; discovery profiles get real CLI version."""
    assert _get_cli_version("reasonix") == "unknown"
    # Claude is now a discovery profile — returns real version when claude is installed,
    # or "unknown" when not.  Don't assert a fixed value.


# ═══════════════════════════════════════════════════════════════════════════════
# RED tests — Bug 1+2: CodexSession protocol + live request/response loop
# ═══════════════════════════════════════════════════════════════════════════════


class FakeCodexSession:
    """Deterministic CodexSession that replays canned JSON-Lines responses."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines: list[str] = list(lines or [])
        self.sent: list[str] = []
        self.terminated: bool = False
        self._closed: bool = False

    def send(self, line: str) -> None:
        if self._closed:
            raise BrokenPipeError("session closed")
        self.sent.append(line)

    def read_line(self, timeout: float | None = None) -> str | None:  # noqa: ARG002
        if not self._lines:
            return None
        return self._lines.pop(0)

    def terminate(self) -> None:
        self.terminated = True
        self._closed = True


def test_codex_session_send_read_terminate() -> None:
    """CodexSession fake records sent lines and replays canned responses."""
    session = FakeCodexSession(
        [
            '{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}',
            '{"jsonrpc":"2.0","id":2,"result":{"data":[],"nextCursor":null}}',
        ]
    )
    session.send('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
    assert len(session.sent) == 1

    line = session.read_line()
    assert line is not None
    assert '"id":1' in line

    session.terminate()
    assert session.terminated is True

    # Sending after terminate raises
    with pytest.raises(BrokenPipeError):
        session.send("boom")


def test_discover_codex_models_uses_injected_session() -> None:
    """discover_codex_models sends init + model/list and collects models."""

    session = FakeCodexSession(
        [
            '{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}',
            '{"jsonrpc":"2.0","id":2,"result":{"data":[{"id":"m1","model":"m1"}],"nextCursor":null}}',
        ]
    )
    result = discover_codex_models(session)

    assert result["ok"] is True
    assert len(result["models"]) == 1
    assert result["models"][0]["id"] == "m1"
    assert session.terminated is True


def test_discover_codex_models_two_page_pagination() -> None:
    """Arbitrary pagination until nextCursor is null."""

    session = FakeCodexSession(
        [
            # initialize response
            '{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}',
            # first page with nextCursor
            '{"jsonrpc":"2.0","id":2,"result":{"data":[{"id":"m1"}],"nextCursor":"abc"}}',
            # second page (no more)
            '{"jsonrpc":"2.0","id":3,"result":{"data":[{"id":"m2"}],"nextCursor":null}}',
        ]
    )
    result = discover_codex_models(session)

    assert result["ok"] is True
    assert len(result["models"]) == 2
    assert result["models"][0]["id"] == "m1"
    assert result["models"][1]["id"] == "m2"

    # Verify unique request ids were used
    ids = [json.loads(s)["id"] for s in session.sent if json.loads(s).get("method") == "model/list"]
    assert ids == [2, 3]


def test_discover_codex_models_ignores_unrelated_notifications() -> None:
    """JSON-RPC notifications (no id) are skipped."""

    session = FakeCodexSession(
        [
            '{"jsonrpc":"2.0","method":"notifications/status","params":{"text":"loading..."}}',
            '{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}',
            '{"jsonrpc":"2.0","method":"notifications/progress","params":{"pct":50}}',
            '{"jsonrpc":"2.0","id":2,"result":{"data":[{"id":"m1"}],"nextCursor":null}}',
        ]
    )
    result = discover_codex_models(session)

    assert result["ok"] is True
    assert len(result["models"]) == 1


def test_discover_codex_models_propagates_jsonrpc_error() -> None:
    """model/list returning a JSON-RPC error is propagated."""

    session = FakeCodexSession(
        [
            '{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}',
            '{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found"}}',
        ]
    )
    result = discover_codex_models(session)

    assert result["ok"] is False
    assert "Method not found" in result.get("error", "")


def test_discover_codex_models_terminates_on_timeout() -> None:
    """Timeout triggers terminate and returns error."""

    # A session that never returns data — should hit timeout
    session = FakeCodexSession([])  # empty -> read_line returns None forever
    result = discover_codex_models(session, timeout=0.05)

    assert result["ok"] is False
    assert "timeout" in result.get("error", "").lower()
    assert session.terminated is True


def test_discover_codex_models_always_terminates_in_finally() -> None:
    """Even when an exception occurs mid-loop, session is terminated."""

    class ExplodingSession(FakeCodexSession):
        def read_line(self, timeout: float | None = None) -> str | None:
            raise RuntimeError("simulated crash")

    session = ExplodingSession(["any"])
    with pytest.raises(RuntimeError, match="simulated crash"):
        discover_codex_models(session)

    assert session.terminated is True


def test_initialized_notification_method_is_exact() -> None:
    """discover_codex_models sends 'initialized' (not 'notifications/initialized')."""

    session = FakeCodexSession(
        [
            '{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}',
            '{"jsonrpc":"2.0","id":2,"result":{"data":[],"nextCursor":null}}',
        ]
    )
    discover_codex_models(session)

    # Find the notification in sent messages
    notifications = [
        json.loads(s)
        for s in session.sent
        if json.loads(s).get("method") is not None and "id" not in json.loads(s)
    ]
    assert len(notifications) >= 1, f"Expected at least one notification, got sent={session.sent}"
    # The only notification sent must be exactly 'initialized'
    methods = {n["method"] for n in notifications}
    assert methods == {"initialized"}, f"Expected notification method 'initialized', got {methods}"


def test_popen_codex_session_is_importable() -> None:
    """PopenCodexSession exists (production implementation, no spawn in test)."""
    assert PopenCodexSession is not None
    # Verify the class has the expected protocol methods
    assert hasattr(PopenCodexSession, "send")
    assert hasattr(PopenCodexSession, "read_line")
    assert hasattr(PopenCodexSession, "terminate")


def test_popen_read_line_buffers_multiple_lines_in_one_chunk() -> None:
    """read_line must retain leftover bytes across calls — two lines in one chunk."""
    from unittest.mock import MagicMock, patch

    mock_stdout = MagicMock()
    mock_stdout.read1.return_value = b'{"a":1}\n{"b":2}\n'

    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = MagicMock()

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("select.select", return_value=([mock_stdout], [], [])):
            session = PopenCodexSession(["codex", "app-server"])

            line1 = session.read_line(timeout=1.0)
            assert line1 == '{"a":1}', f"First line: {line1!r}"

            # After first call, mock stdout returns EOF
            mock_stdout.read1.return_value = b""
            line2 = session.read_line(timeout=1.0)
            assert line2 == '{"b":2}', f"Second line (buffered): {line2!r}"

            session.terminate()


# ═══════════════════════════════════════════════════════════════════════════════
# RED tests — Bug 3: model_cache never overwrites last-good on fetch failure
# ═══════════════════════════════════════════════════════════════════════════════


def test_cache_returns_last_good_on_fresh_fetch_failure(tmp_path: Path) -> None:
    """Fresh fetch fails → return last-good cached data marked stale with error."""
    # Pre-populate good cache
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.145.0", data)

    # Fetcher raises
    def failing_fetcher() -> dict[str, Any]:
        raise RuntimeError("codex not installed")

    # refresh=True forces fetcher call → failure → last-good fallback
    result, from_cache = cache_get_or_fetch(
        tmp_path,
        "codex",
        "0.145.0",
        failing_fetcher,
        refresh=True,
    )

    assert from_cache is True
    assert result["models"] == ["gpt-5.6-sol"]
    assert result.get("stale") is True
    assert "codex not installed" in str(result.get("error", ""))


def test_cache_returns_last_good_on_version_change_failure(tmp_path: Path) -> None:
    """Version changed → fresh fetch fails → return stale last-good."""
    # Pre-populate with old version
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.144.0", data)

    # Fetcher raises (version mismatch means cache miss → fetch → fail)
    def failing_fetcher() -> dict[str, Any]:
        raise RuntimeError("network down")

    result, from_cache = cache_get_or_fetch(
        tmp_path,
        "codex",
        "0.145.0",
        failing_fetcher,
    )

    # Should return the stale v0.144.0 data
    assert from_cache is True
    assert result["models"] == ["gpt-5.6-sol"]
    assert result.get("stale") is True
    assert "network down" in str(result.get("error", ""))


def test_cache_propagates_failure_when_no_last_good(tmp_path: Path) -> None:
    """No cached data at all → fetcher failure propagates."""

    def failing_fetcher() -> dict[str, Any]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cache_get_or_fetch(tmp_path, "codex", "0.145.0", failing_fetcher)


def test_cache_refresh_failure_returns_stale_last_good(tmp_path: Path) -> None:
    """refresh=True with a failing fetcher returns stale last-good."""
    data = {"models": ["gpt-5.6-sol"], "source": "live"}
    write_cache(tmp_path, "codex", "0.145.0", data)

    def failing_fetcher() -> dict[str, Any]:
        raise RuntimeError("refresh failed")

    result, from_cache = cache_get_or_fetch(
        tmp_path,
        "codex",
        "0.145.0",
        failing_fetcher,
        refresh=True,
    )

    assert from_cache is True
    assert result["models"] == ["gpt-5.6-sol"]
    assert result.get("stale") is True
    assert "refresh failed" in str(result.get("error", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# RED tests — Bug 4: Codex max effort resolves native max, validation error
# ═══════════════════════════════════════════════════════════════════════════════


def test_max_effort_priority_max_is_highest() -> None:
    """When 'max' is in native efforts, it beats xhigh."""
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "super-model",
                    "model": "super-model",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                        {"reasoningEffort": "xhigh"},
                        {"reasoningEffort": "max"},
                    ],
                },
            ]
        }
    )

    # max should be the highest priority
    assert _max_effort_for_model(catalog, "super-model") == "max"


def test_resolve_effort_max_to_native_max_when_advertised() -> None:
    """CodexAdapter.resolve_effort('max', ...) returns 'max' when native max exists."""
    adapter = CodexAdapter()
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                        {"reasoningEffort": "xhigh"},
                        {"reasoningEffort": "max"},
                    ],
                },
            ]
        }
    )

    # Max should resolve to native "max", not "xhigh"
    assert adapter.resolve_effort("max", catalog, "gpt-5.6-sol") == "max"


def test_validate_effort_max_passes_when_high_is_strongest() -> None:
    """max resolves to 'high' when that is strongest available — passes validation."""
    adapter = CodexAdapter()
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-5.6-terra",
                    "model": "gpt-5.6-terra",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                    ],
                },
            ]
        }
    )

    err = adapter.validate_effort_for_model("max", catalog, "gpt-5.6-terra")
    assert err is None


def test_effort_max_resolves_max_when_no_catalog() -> None:
    """Without a catalog, 'max' falls back to static effort_map (max→max)."""
    adapter = CodexAdapter()
    # No catalog → static fallback (effort_map now maps max → "max")
    assert adapter.resolve_effort("max", None, None) == "max"


def test_resolve_effort_max_to_xhigh_when_xhigh_is_strongest() -> None:
    """max resolves to 'xhigh' when xhigh is strongest and native max is absent."""
    adapter = CodexAdapter()
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-5.6-sol",
                    "model": "gpt-5.6-sol",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                        {"reasoningEffort": "medium"},
                        {"reasoningEffort": "high"},
                        {"reasoningEffort": "xhigh"},
                    ],
                },
            ]
        }
    )
    assert adapter.resolve_effort("max", catalog, "gpt-5.6-sol") == "xhigh"
    err = adapter.validate_effort_for_model("max", catalog, "gpt-5.6-sol")
    assert err is None


def test_resolve_effort_max_to_low_when_low_is_only_effort() -> None:
    """max resolves to 'low' when that is the only native effort advertised."""
    adapter = CodexAdapter()
    catalog = parse_model_list_response(
        {
            "data": [
                {
                    "id": "gpt-mini",
                    "model": "gpt-mini",
                    "hidden": False,
                    "isDefault": True,
                    "defaultReasoningEffort": "low",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                    ],
                },
            ]
        }
    )
    assert adapter.resolve_effort("max", catalog, "gpt-mini") == "low"
    err = adapter.validate_effort_for_model("max", catalog, "gpt-mini")
    assert err is None


# ═══════════════════════════════════════════════════════════════════════════════
# RED tests — Bug 5: Claude discover_models does NOT call subprocess (verify)
# ═══════════════════════════════════════════════════════════════════════════════


def test_claude_discover_models_uses_cached_data_without_live_probe(tmp_path: Path) -> None:
    """Claude discover_profile_models returns cached data when available.

    The live PTY probe is never run in unit tests — this verifies the
    cache path works correctly for claude.
    """
    import agent_crossbar.discovery as _disc

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "2.1.211"
    try:
        data = {
            "models": ["claude-opus-4-8", "claude-sonnet-5"],
            "default_model": "claude-opus-4-8",
            "native_efforts": ["low", "medium", "high"],
            "model_info": [
                {
                    "id": "claude-opus-4-8",
                    "supported_efforts": ["low", "medium", "high"],
                    "default_effort": "medium",
                },
            ],
            "source": "claude interactive /model picker",
        }
        write_cache(tmp_path, "claude", "2.1.211", data)

        catalog = discover_profile_models(tmp_path, "claude")
        assert catalog.models == ("claude-opus-4-8", "claude-sonnet-5")
        assert catalog.default_model == "claude-opus-4-8"
        assert catalog.source == "claude interactive /model picker"
        assert catalog.cache_hit is True
    finally:
        _disc._get_cli_version = _orig_version


# ── Task 3.6: OpenCode verbose model discovery parser (TDD) ──


def test_opencode_verbose_parser_handles_multiple_json_blocks():
    """Parse opencode models --verbose output with multiple model+JSON blocks."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        '{"variants":{"low":{"max_tokens":4096},"medium":{"max_tokens":8192},"high":{"max_tokens":16384}}}\n'
        "opencode-go/kimi-k2.7-code\n"
        '{"variants":{"medium":{"max_tokens":8192},"high":{"max_tokens":32768}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code")
    assert catalog.default_model == "opencode-go/glm-5.2"
    assert set(catalog.native_efforts) == {"low", "medium", "high"}
    assert len(catalog.model_info) == 2

    glm_info = catalog.model_info[0]
    assert glm_info.id == "opencode-go/glm-5.2"
    assert set(glm_info.supported_efforts) == {"low", "medium", "high"}
    assert glm_info.default_effort is None  # dict key order is not a default marker

    kimi_info = catalog.model_info[1]
    assert kimi_info.id == "opencode-go/kimi-k2.7-code"
    assert set(kimi_info.supported_efforts) == {"medium", "high"}
    assert kimi_info.default_effort is None


def test_opencode_verbose_parser_handles_empty_variants():
    """Model with variants:{} should have empty supported_efforts."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = 'opencode-go/glm-5.2\n{"variants":{}}\n'

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2",)
    assert catalog.native_efforts == ()
    assert len(catalog.model_info) == 1
    assert catalog.model_info[0].supported_efforts == ()
    assert catalog.model_info[0].default_effort is None


def test_opencode_verbose_parser_handles_malformed_block():
    """Malformed JSON block should not crash; model still listed, no efforts."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        "not-valid-json\n"
        "opencode-go/kimi-k2.7-code\n"
        '{"variants":{"medium":{"max_tokens":8192}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code")
    # glm-5.2 had malformed JSON → no efforts
    glm_info = catalog.model_info[0]
    assert glm_info.id == "opencode-go/glm-5.2"
    assert glm_info.supported_efforts == ()
    assert glm_info.default_effort is None
    # kimi should parse fine
    kimi_info = catalog.model_info[1]
    assert kimi_info.id == "opencode-go/kimi-k2.7-code"
    assert kimi_info.supported_efforts == ("medium",)


def test_opencode_verbose_parser_handles_max_effort():
    """Parse model with 'max' variant."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = 'opencode-go/glm-5.2\n{"variants":{"low":{},"medium":{},"high":{},"max":{}}}\n'

    catalog = parse_opencode_verbose_output(output)
    assert "max" in catalog.native_efforts
    assert "max" in catalog.model_info[0].supported_efforts


# ── Task 3.6 correction: OpenCode verbose parser must not inflate model count ──


def test_opencode_verbose_parser_rejects_urls_as_models():
    """Pretty-printed JSON with URLs (containing /) must NOT be treated as model IDs.

    Reproduces the 1931-vs-420 bug: the real output has multi-line JSON
    objects containing ``"base_url": "https://api.openai.com/v1/..."``;
    every /-containing substring was grabbed as a model ID.
    """
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        "{\n"
        '  "variants": {\n'
        '    "low": {"max_tokens": 4096},\n'
        '    "medium": {"max_tokens": 8192}\n'
        "  },\n"
        '  "base_url": "https://api.openai.com/v1/models/glm-5.2",\n'
        '  "provider": "openai"\n'
        "}\n"
        "opencode-go/kimi-k2.7-code\n"
        "{\n"
        '  "variants": {\n'
        '    "medium": {"max_tokens": 8192},\n'
        '    "high": {"max_tokens": 32768}\n'
        "  }\n"
        "}\n"
    )

    catalog = parse_opencode_verbose_output(output)

    # Must NOT treat URL substrings as model IDs — exactly 2 models
    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"), (
        f"Expected 2 models, got {len(catalog.models)}: {catalog.models}"
    )
    assert len(catalog.model_info) == 2


def test_opencode_verbose_parser_handles_multiline_json_accumulation():
    """Multi-line pretty JSON must be accumulated until balanced braces."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        "{\n"
        '  "variants": {\n'
        '    "low": {"max_tokens": 4096},\n'
        '    "medium": {"max_tokens": 8192},\n'
        '    "high": {"max_tokens": 16384}\n'
        "  }\n"
        "}\n"
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2",)
    assert set(catalog.model_info[0].supported_efforts) == {"low", "medium", "high"}
    assert len(catalog.model_info) == 1


def test_opencode_verbose_parser_filters_non_public_variant_keys():
    """Variant keys like 'xhigh', 'turbo', 'extended' must be filtered out.

    supported_efforts must only contain values from PUBLIC_EFFORTS = (low, medium, high, max).
    """
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        'opencode-go/glm-5.2\n{"variants":{"low":{},"medium":{},"xhigh":{},"turbo":{},"max":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    supported = set(catalog.model_info[0].supported_efforts)
    assert supported == {"low", "medium", "max"}, f"Expected only PUBLIC_EFFORTS, got {supported}"
    assert "xhigh" not in supported
    assert "turbo" not in supported
    assert set(catalog.native_efforts) == {"low", "medium", "max"}


def test_opencode_verbose_parser_default_effort_is_none_unless_explicit():
    """default_effort must be None unless metadata explicitly marks a default.

    The old parser incorrectly used the first variant dict key as
    default_effort.  Order in a dict is not a default marker.
    """
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = 'opencode-go/glm-5.2\n{"variants":{"medium":{},"low":{},"high":{}}}\n'

    catalog = parse_opencode_verbose_output(output)

    # Must not assume first key ("medium") is the default
    assert catalog.model_info[0].default_effort is None, (
        f"default_effort should be None, got {catalog.model_info[0].default_effort}"
    )


def test_opencode_verbose_parser_handles_nested_braces_in_json():
    """JSON values with nested braces (dicts inside variants) must not confuse
    the brace-balanced accumulator."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        "{\n"
        '  "variants": {\n'
        '    "low": {"max_tokens": 4096, "temperature": 0.7},\n'
        '    "high": {"max_tokens": 16384, "tools": ["read", "write"]}\n'
        "  }\n"
        "}\n"
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2",)
    assert set(catalog.model_info[0].supported_efforts) == {"low", "high"}


def test_opencode_verbose_parser_dedupes_duplicate_model_ids():
    """Duplicate model ID lines must only produce one ModelInfo, stable order."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        '{"variants":{"low":{}}}\n'
        "opencode-go/kimi-k2.7-code\n"
        '{"variants":{"medium":{}}}\n'
        "opencode-go/glm-5.2\n"
        '{"variants":{"low":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"), (
        f"Expected 2 deduped models, got {len(catalog.models)}: {catalog.models}"
    )
    assert len(catalog.model_info) == 2


def test_opencode_verbose_parser_truncated_json_produces_error():
    """A truncated/malformed JSON block must produce an honest catalog error,
    not silently claim clean discovery."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        '{"variants":{"low":{"max_tokens":4096},"medium":{"ma'  # truncated mid-key
    )

    catalog = parse_opencode_verbose_output(output)

    # Model should still be listed (partial data is better than none)
    assert catalog.models == ("opencode-go/glm-5.2",)
    # But the catalog must carry an error signal about the malformed JSON
    assert catalog.error is not None, (
        "Truncated JSON must produce a catalog error, not silent success"
    )
    assert "glm-5.2" in catalog.error


def test_opencode_verbose_parser_handles_model_without_json_block():
    """A model ID line followed by another model ID (no JSON) should still be
    listed with empty efforts."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = 'opencode-go/glm-5.2\nopencode-go/kimi-k2.7-code\n{"variants":{"medium":{}}}\n'

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code")
    # glm-5.2: no JSON block found → empty supported_efforts
    assert catalog.model_info[0].id == "opencode-go/glm-5.2"
    assert catalog.model_info[0].supported_efforts == ()
    # kimi: has JSON
    assert catalog.model_info[1].id == "opencode-go/kimi-k2.7-code"
    assert catalog.model_info[1].supported_efforts == ("medium",)


# ── Parser regression: accept real OpenCode model IDs with nested /, @, ~ ──


def test_opencode_verbose_parser_accepts_nested_slash_ids():
    """Model IDs with nested slashes (provider/sub/model) must be accepted."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "google-vertex/deepseek-ai/deepseek-v3.1-maas\n"
        '{"variants":{"low":{},"medium":{},"high":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("google-vertex/deepseek-ai/deepseek-v3.1-maas",), (
        f"Expected 1 model, got {len(catalog.models)}: {catalog.models}"
    )
    assert catalog.default_model == "google-vertex/deepseek-ai/deepseek-v3.1-maas"
    assert set(catalog.model_info[0].supported_efforts) == {"low", "medium", "high"}


def test_opencode_verbose_parser_accepts_at_sign_in_id():
    """Model IDs with @ (e.g. version suffix) must be accepted."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        'google-vertex/claude-opus-4-8@default\n{"variants":{"medium":{},"high":{},"max":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("google-vertex/claude-opus-4-8@default",), (
        f"Expected 1 model, got {len(catalog.models)}: {catalog.models}"
    )
    assert set(catalog.model_info[0].supported_efforts) == {"medium", "high", "max"}


def test_opencode_verbose_parser_accepts_tilde_prefix():
    """Model IDs with ~ prefix in a segment must be accepted."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = 'openrouter/~anthropic/claude-opus-latest\n{"variants":{"low":{},"high":{}}}\n'

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("openrouter/~anthropic/claude-opus-latest",), (
        f"Expected 1 model, got {len(catalog.models)}: {catalog.models}"
    )
    assert set(catalog.model_info[0].supported_efforts) == {"low", "high"}


def test_opencode_verbose_parser_synthetic_count_parity():
    """A synthetic list of diverse real-world IDs must achieve count parity
    (every valid ID line is captured, no false negatives)."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    ids = [
        "google-vertex/claude-opus-4-8@default",
        "google-vertex/deepseek-ai/deepseek-v3.1-maas",
        "lmstudio/mistralai/devstral-small-2-2512",
        "openrouter/~anthropic/claude-opus-latest",
        "opencode-go/glm-5.2",
        "opencode-go/kimi-k2.7-code",
        "opencode-go/deepseek-v4-flash",
        "provider-x/sub-y/sub-z/model-v2.5",
        "acme/~experimental/gpt-5@rc1",
        "cloud-vendor/team-7/model@2026-03-15",
    ]

    # Build synthetic verbose output: each ID followed by a minimal JSON block
    lines: list[str] = []
    for mid in ids:
        lines.append(mid)
        lines.append('{"variants":{"medium":{}}}')
    output = "\n".join(lines)

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == tuple(ids), (
        f"Expected {len(ids)} models, got {len(catalog.models)}: {catalog.models}"
    )
    assert len(catalog.model_info) == len(ids)
    # The configured free default is absent, so fallback is deterministic.
    assert catalog.default_model == ids[0]


def test_opencode_verbose_parser_still_rejects_urls_and_json():
    """Broadened regex must NOT regress: URLs (://) and JSON punctuation
    must still be rejected as model IDs."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        "{\n"
        '  "variants": {\n'
        '    "low": {"max_tokens": 4096},\n'
        '    "medium": {"max_tokens": 8192}\n'
        "  },\n"
        '  "base_url": "https://api.openai.com/v1/models/glm-5.2",\n'
        '  "provider": "openai"\n'
        "}\n"
        "opencode-go/kimi-k2.7-code\n"
        "{\n"
        '  "variants": {\n'
        '    "medium": {"max_tokens": 8192}\n'
        "  }\n"
        "}\n"
        # JSON-like lines that must NOT be treated as model IDs
        '{"impostor":"looks-like-json/but-is-not-a-model"}\n'
        "[1, 2, 3]\n"
        '  "quoted/string/with/slashes": true\n'
    )

    catalog = parse_opencode_verbose_output(output)

    # Only the 2 real model IDs must be captured
    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"), (
        f"Expected 2 models, got {len(catalog.models)}: {catalog.models}"
    )
    assert len(catalog.model_info) == 2


def test_opencode_verbose_parser_accepts_colon_tags():
    """parse_opencode_verbose_output must accept model IDs with colon tags (:free, :thinking)."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "openrouter/cohere/north-mini-code:free\n"
        '{"variants":{"low":{},"medium":{}}}\n'
        "openrouter/qwen/qwen-plus-2025-07-28:thinking\n"
        '{"variants":{"low":{},"high":{},"max":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == (
        "openrouter/cohere/north-mini-code:free",
        "openrouter/qwen/qwen-plus-2025-07-28:thinking",
    ), f"Expected 2 colon-tagged models, got {len(catalog.models)}: {catalog.models}"
    assert set(catalog.model_info[0].supported_efforts) == {"low", "medium"}
    assert set(catalog.model_info[1].supported_efforts) == {"low", "high", "max"}


def test_opencode_verbose_parser_still_rejects_url_bare_lines():
    """Even with : in the regex, bare URL lines must not be treated as model IDs."""
    from agent_crossbar.adapters.opencode import parse_opencode_verbose_output

    output = (
        "opencode-go/glm-5.2\n"
        '{"variants":{"low":{}}}\n'
        "https://api.openai.com/v1/models/impostor\n"
        "opencode-go/kimi-k2.7-code\n"
        '{"variants":{"medium":{}}}\n'
    )

    catalog = parse_opencode_verbose_output(output)

    assert catalog.models == ("opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"), (
        f"URL bare line must be rejected, got {len(catalog.models)}: {catalog.models}"
    )


# ── Task 3.6: Claude --help effort parsing (TDD) ──


def test_claude_help_effort_parser_extracts_public_values():
    """Bounded parse of claude --help extracts effort values."""
    from agent_crossbar.adapters.claude import parse_claude_help_efforts

    help_output = (
        "Usage: claude [options] [prompt]\n"
        "  --effort <effort>  Reasoning effort (low, medium, high, max)\n"
        "  --model <model>    Model to use\n"
    )

    efforts = parse_claude_help_efforts(help_output)
    assert set(efforts) == {"low", "medium", "high", "max"}


def test_claude_help_effort_parser_excludes_non_public():
    """Bounded parse excludes xhigh from public efforts."""
    from agent_crossbar.adapters.claude import parse_claude_help_efforts

    help_output = (
        "Usage: claude [options] [prompt]\n"
        "  --effort <effort>  Reasoning effort (low, medium, high, xhigh, max)\n"
    )

    efforts = parse_claude_help_efforts(help_output)
    assert "xhigh" not in efforts
    assert set(efforts) == {"low", "medium", "high", "max"}


def test_claude_help_effort_parser_no_match_returns_empty():
    """When --effort line is not found, return empty tuple."""
    from agent_crossbar.adapters.claude import parse_claude_help_efforts

    help_output = "Usage: claude [options]\n  --model <model>\n"

    efforts = parse_claude_help_efforts(help_output)
    assert efforts == ()


def test_claude_help_effort_parser_handles_wrapped_help_line():
    """Parse the real two-line --effort help output from Claude >=2.1.211.

    The real CLI renders::

        --effort <level>  Effort level for the current session
                          (low, medium, high, xhigh, max)

    The regex must match across the line break and exclude xhigh from
    public values.
    """
    from agent_crossbar.adapters.claude import parse_claude_help_efforts

    help_output = (
        "Usage: claude [options] [prompt]\n"
        "  --effort <level>  Effort level for the current session\n"
        "                    (low, medium, high, xhigh, max)\n"
        "  --model <model>   Model to use\n"
    )

    efforts = parse_claude_help_efforts(help_output)
    assert set(efforts) == {"low", "medium", "high", "max"}, (
        f"Expected {{low, medium, high, max}}, got {set(efforts)}"
    )
    assert "xhigh" not in efforts


# ── Task 3.6: No job dir / child on unsupported effort (TDD) ──


# ── Helper: build a ModelCatalog from a compact dict for test brevity ──


def _make_catalog(
    models: list[str],
    default_model: str | None = None,
    model_info: list[dict] | None = None,
    *,
    error: str | None = None,
    cache_hit: bool = False,
) -> ModelCatalog:
    """Build a ModelCatalog from test data."""
    if model_info is None:
        model_info = [
            {"id": m, "supported_efforts": ["low", "medium", "high"], "default_effort": None}
            for m in models
        ]
    return ModelCatalog(
        models=tuple(models),
        default_model=default_model or (models[0] if models else None),
        native_efforts=tuple(
            sorted({e for mi in model_info for e in mi.get("supported_efforts", [])})
        ),
        source="opencode models --verbose",
        error=error,
        model_info=tuple(
            ModelInfo(
                id=mi["id"],
                supported_efforts=tuple(mi.get("supported_efforts", [])),
                default_effort=mi.get("default_effort"),
            )
            for mi in model_info
        ),
        cache_hit=cache_hit,
    )


def test_codex_preflight_accepts_model_advertised_by_live_discovery(tmp_path: Path) -> None:
    """Validation must accept a model that the same live discovery publishes."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(models=["gpt-5.6-luna"])
    req = {
        "operation": "review",
        "profile": "codex",
        "transport": "print",
        "autonomy": "read_only",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "gpt-5.6-luna",
        "effort": "medium",
    }

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        result = validate_start_request(req, state_root=tmp_path)

    assert result["ok"] is True
    assert result["model"] == "gpt-5.6-luna"


def test_codex_preflight_rejects_model_absent_from_live_discovery(tmp_path: Path) -> None:
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(models=["gpt-5.6-luna"])
    req = {
        "operation": "review",
        "profile": "codex",
        "transport": "print",
        "autonomy": "read_only",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "gpt-5.6-sol",
        "effort": "medium",
    }

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        result = validate_start_request(req, state_root=tmp_path)

    assert result["ok"] is False
    assert result["error"] == "invalid_model"


# ── OpenCode preflight: discover_profile_models integration ──


def test_opencode_validation_rejects_unsupported_effort(tmp_path):
    """validate_start_request rejects opencode job when model doesn't support effort."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(
        models=["opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"],
        default_model="opencode-go/glm-5.2",
        model_info=[
            {
                "id": "opencode-go/glm-5.2",
                "supported_efforts": ["low", "medium", "high"],
                "default_effort": "medium",
            },
            {
                "id": "opencode-go/kimi-k2.7-code",
                "supported_efforts": ["medium", "high"],
                "default_effort": "medium",
            },
        ],
        cache_hit=True,
    )

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "kimi-k2.7-code",
            "effort": "low",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is False
        assert result["error"] == "unsupported_effort_for_model"
        assert result["job_created"] is False
        assert "low" in result["message"]


def test_opencode_validation_allows_supported_effort(tmp_path):
    """validate_start_request allows opencode job when model supports effort."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(
        models=["opencode-go/glm-5.2"],
        model_info=[
            {
                "id": "opencode-go/glm-5.2",
                "supported_efforts": ["low", "medium", "high"],
                "default_effort": "medium",
            },
        ],
        cache_hit=True,
    )

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "glm-5.2",
            "effort": "high",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is True
        assert result["job_created"] is True
        assert result["effort"] == "high"


def test_codex_max_effort_resolves_to_strongest_from_cache(tmp_path):
    """Codex max effort resolves to strongest native effort from cached catalog."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "0.145.0"
    try:
        data = {
            "models": ["gpt-5.6-terra"],
            "default_model": "gpt-5.6-terra",
            "native_efforts": ["medium", "high"],
            "model_info": [
                {
                    "id": "gpt-5.6-terra",
                    "supported_efforts": ["medium", "high"],
                    "default_effort": "high",
                },
            ],
            "source": "codex app-server model/list",
        }
        write_cache(tmp_path, "codex", "0.145.0", data)

        req = {
            "operation": "review",
            "profile": "codex",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "gpt-5.6-terra",
            "effort": "max",
        }
        result = validate_start_request(req, state_root=tmp_path)
        # max is accepted — internally resolves to "high" (strongest available)
        assert result["ok"] is True
        assert result["effort"] == "max"
        assert result["job_created"] is True
    finally:
        _disc._get_cli_version = _orig_version


# ── Task 3.6 correction: validate_effort_for_model must reject empty supported_efforts ──


def test_opencode_validate_effort_rejects_empty_supported_efforts():
    """When model_info exists but supported_efforts is empty, effort must be rejected.

    No advertised variant means effort cannot be proven or mapped —
    validate_effort_for_model must not silently allow.
    """
    from agent_crossbar.adapters.base import ModelCatalog, ModelInfo
    from agent_crossbar.adapters.opencode import adapter as opencode_adapter

    catalog = ModelCatalog(
        models=("opencode-go/glm-5.2",),
        default_model="opencode-go/glm-5.2",
        native_efforts=(),
        source="test",
        model_info=(
            ModelInfo(id="opencode-go/glm-5.2", supported_efforts=(), default_effort=None),
        ),
    )

    err = opencode_adapter.validate_effort_for_model("low", catalog, "opencode-go/glm-5.2")
    assert err is not None, "Must reject when supported_efforts is empty"
    assert "no advertised" in err.lower() or "cannot" in err.lower()


def test_opencode_validate_effort_rejects_on_catalog_error():
    """When catalog carries an error, effort validation must produce a distinct
    discovery/preflight error, not silently allow."""
    from agent_crossbar.adapters.base import ModelCatalog
    from agent_crossbar.adapters.opencode import adapter as opencode_adapter

    catalog = ModelCatalog(
        models=(),
        default_model=None,
        native_efforts=(),
        source="test",
        error="opencode models exited 1: connection refused",
    )

    err = opencode_adapter.validate_effort_for_model("low", catalog, "opencode-go/glm-5.2")
    assert err is not None, "Must reject when catalog has an error"
    assert "error" in err.lower() or "discovery" in err.lower() or "unavailable" in err.lower()


def test_opencode_validate_effort_rejects_when_no_model_info():
    """When catalog has no model_info at all (but model is listed in models),
    effort cannot be validated — must reject."""
    from agent_crossbar.adapters.base import ModelCatalog
    from agent_crossbar.adapters.opencode import adapter as opencode_adapter

    catalog = ModelCatalog(
        models=("opencode-go/glm-5.2",),
        default_model="opencode-go/glm-5.2",
        native_efforts=(),
        source="test",
        model_info=(),  # no per-model data
    )

    err = opencode_adapter.validate_effort_for_model("low", catalog, "opencode-go/glm-5.2")
    assert err is not None, "Must reject when no model_info exists for the model"
    assert "no capability" in err.lower() or "cannot" in err.lower()


# ── OpenCode preflight: discover_profile_models is the authoritative source ──


def test_opencode_validation_accepts_catalog_model_not_in_static_list(tmp_path):
    """A model discovered via live catalog (not in the static OPENCODE_MODELS list)
    must be accepted.  The static list is not authoritative — the live CLI is."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(
        models=["opencode-go/new-hotness-v1", "opencode-go/glm-5.2"],
        default_model="opencode-go/new-hotness-v1",
        cache_hit=True,
    )

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "new-hotness-v1",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is True, f"Expected ok=True for catalog-discovered model, got {result}"
        assert result["job_created"] is True
        assert result["model"] == "opencode-go/new-hotness-v1"


def test_opencode_validation_never_defaults_to_catalog_model(tmp_path):
    """A discovered selected/default model never replaces the required input."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(
        models=["opencode-go/kimi-k2.7-code", "opencode-go/glm-5.2"],
        default_model="opencode-go/kimi-k2.7-code",
        cache_hit=True,
    )

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is False
        assert result["error"] == "missing_model"
        assert result["job_created"] is False


def test_opencode_validation_fails_preflight_on_discovery_error(tmp_path):
    """When catalog discovery fails (error field set), validation must fail
    preflight with a stable structured error — no job created."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(
        models=[],
        default_model=None,
        model_info=[],
        error="opencode exited 1: command not found",
    )

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "glm-5.2",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is False, f"Expected ok=False on discovery error, got {result}"
        assert result["job_created"] is False
        assert "discovery" in result["error"].lower()


def test_opencode_validation_fails_preflight_on_empty_catalog(tmp_path):
    """When catalog has no models (empty list), validation must fail preflight."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(models=[], default_model=None, model_info=[])

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "glm-5.2",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is False, f"Expected ok=False for empty catalog, got {result}"
        assert result["job_created"] is False


# ── OpenCode preflight: discover_profile_models integration (new) ──


def test_opencode_preflight_discovers_on_first_call(tmp_path):
    """First call with no cache must trigger discovery via discover_profile_models
    and use the returned catalog for model resolution."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(
        models=["google-vertex/claude-opus-4-8@default", "opencode-go/glm-5.2"],
        default_model="google-vertex/claude-opus-4-8@default",
        cache_hit=False,  # first call, no cache
    )

    with patch.object(_disc, "discover_profile_models", return_value=catalog) as mock_disc:
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "claude-opus-4-8@default",  # short name
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is True, f"Expected ok=True, got {result}"
        assert result["job_created"] is True
        assert result["model"] == "google-vertex/claude-opus-4-8@default"
        # Verify discover_profile_models was called with correct args
        mock_disc.assert_called_once()
        args, _ = mock_disc.call_args
        assert args[1] == "opencode"


def test_opencode_preflight_uses_cache_path(tmp_path):
    """When cache is fresh, discover_profile_models returns cache_hit=True.
    Validation must still resolve models correctly from cached data."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(
        models=["opencode-go/glm-5.2", "opencode-go/kimi-k2.7-code"],
        default_model="opencode-go/glm-5.2",
        cache_hit=True,
    )

    with patch.object(_disc, "discover_profile_models", return_value=catalog) as mock_disc:
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "opencode-go/glm-5.2",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is True, f"Expected ok=True, got {result}"
        assert result["job_created"] is True
        assert result["model"] == "opencode-go/glm-5.2"
        mock_disc.assert_called_once()


def test_opencode_preflight_fails_on_discovery_exception(tmp_path):
    """When discover_profile_models raises an exception, validation must fail
    preflight — no job created, no fallback to static allowlist."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    with patch.object(
        _disc, "discover_profile_models", side_effect=RuntimeError("opencode CLI not found")
    ):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "glm-5.2",
        }
        result = validate_start_request(req, state_root=tmp_path)
        assert result["ok"] is False, f"Expected ok=False on exception, got {result}"
        assert result["job_created"] is False
        assert result["error"] == "discovery_error"
        assert "CLI not found" in result["message"]


def test_opencode_preflight_fails_on_missing_state_root():
    """When state_root is None, opencode preflight must fail immediately
    — no discovery attempt at all."""
    from agent_crossbar.validation import validate_start_request

    req = {
        "operation": "review",
        "profile": "opencode",
        "transport": "print",
        "autonomy": "read_only",
        "external_context": "allowed",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "glm-5.2",
    }
    result = validate_start_request(req, state_root=None)
    assert result["ok"] is False, f"Expected ok=False without state_root, got {result}"
    assert result["job_created"] is False
    assert result["error"] == "discovery_error"
    assert "state_root" in result["message"].lower()


def test_opencode_preflight_no_static_fallback():
    """Prove the static allowlist is NOT used as a fallback.

    When discover_profile_models returns an empty catalog (no models,
    no error), validation must fail — it must NOT silently fall through
    to the static OPENCODE_MODELS list.
    """
    from pathlib import Path

    import agent_crossbar.discovery as _disc
    from agent_crossbar.validation import validate_start_request

    catalog = _make_catalog(models=[], default_model=None, model_info=[])

    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        req = {
            "operation": "review",
            "profile": "opencode",
            "transport": "print",
            "autonomy": "read_only",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "x",
            "model": "opencode-go/glm-5.2",
        }
        result = validate_start_request(req, state_root=Path("/tmp/fake"))
        assert result["ok"] is False, (
            f"Must fail when catalog is empty — no static fallback allowed. Got {result}"
        )
        assert result["job_created"] is False
        assert result["error"] == "discovery_error"


# ── Task 4.2: dynamic profiles_list (cache-only, static fallback) ──────────


def test_cached_models_for_listing_invokes_discovery_on_clean_cache(tmp_path: Path) -> None:
    """A clean cache (no prior discovery) must attempt real discover_profile_models —
    profiles_list never silently skips discovery just because there's no cache yet."""
    import agent_crossbar.discovery as _disc

    catalog = ModelCatalog(
        models=("gpt-live-1", "gpt-live-2"),
        default_model="gpt-live-1",
        native_efforts=(),
        source="codex app-server model/list",
    )
    with patch.object(_disc, "discover_profile_models", return_value=catalog) as mocked:
        models, default_model = cached_models_for_listing(
            tmp_path, "codex", ["gpt-5.6-sol", "gpt-5.6-terra"], "gpt-5.6-sol"
        )
    mocked.assert_called_once_with(tmp_path, "codex")
    assert models == ["gpt-live-1", "gpt-live-2"]
    assert default_model == "gpt-live-1"


def test_cached_models_for_listing_falls_back_when_probe_fails(tmp_path: Path) -> None:
    """A failed live probe (no cache to fall back on either) → static fallback, never empty."""
    import agent_crossbar.discovery as _disc

    def _boom(state_root, profile, *, refresh=False):
        raise RuntimeError("no live cli available")

    with patch.object(_disc, "discover_profile_models", _boom):
        models, default_model = cached_models_for_listing(
            tmp_path, "codex", ["gpt-5.6-sol", "gpt-5.6-terra"], "gpt-5.6-sol"
        )
    assert models == ["gpt-5.6-sol", "gpt-5.6-terra"]
    assert default_model == "gpt-5.6-sol"


def test_cached_models_for_listing_uses_fresh_cache(tmp_path: Path) -> None:
    """A fresh, matching-version cache overrides the static fallback."""
    import agent_crossbar.discovery as _disc

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "0.999.0"
    try:
        write_cache(
            tmp_path,
            "codex",
            "0.999.0",
            {
                "models": ["gpt-live-1", "gpt-live-2"],
                "default_model": "gpt-live-1",
                "source": "codex app-server model/list",
            },
        )
        models, default_model = cached_models_for_listing(
            tmp_path, "codex", ["gpt-5.6-sol", "gpt-5.6-terra"], "gpt-5.6-sol"
        )
        assert models == ["gpt-live-1", "gpt-live-2"]
        assert default_model == "gpt-live-1"
    finally:
        _disc._get_cli_version = _orig_version


def test_cached_models_for_listing_falls_back_on_cached_error(tmp_path: Path) -> None:
    """A cached entry that itself carries a discovery error is not surfaced as usable models."""
    import agent_crossbar.discovery as _disc

    _orig_version = _disc._get_cli_version
    _disc._get_cli_version = lambda p: "0.999.0"
    try:
        write_cache(
            tmp_path,
            "codex",
            "0.999.0",
            {"models": [], "default_model": None, "error": "codex not reachable"},
        )
        models, default_model = cached_models_for_listing(
            tmp_path, "codex", ["gpt-5.6-sol", "gpt-5.6-terra"], "gpt-5.6-sol"
        )
        assert models == ["gpt-5.6-sol", "gpt-5.6-terra"]
        assert default_model == "gpt-5.6-sol"
    finally:
        _disc._get_cli_version = _orig_version


def test_cached_models_for_listing_ignores_non_discovery_profiles(tmp_path: Path) -> None:
    """reasonix/chatgpt_pro are not live-discovery profiles — always static."""
    models, default_model = cached_models_for_listing(
        tmp_path, "reasonix", ["deepseek-v4-flash", "deepseek-v4-pro"], "deepseek-v4-flash"
    )
    assert models == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert default_model == "deepseek-v4-flash"


def test_live_profile_registry_overlays_discovered_models(tmp_path: Path) -> None:
    """live_profile_registry overlays discovered models onto the static registry
    without adding or removing any entry keys."""
    import agent_crossbar.discovery as _disc

    def _fake_discover(state_root, profile, *, refresh=False):
        if profile == "opencode":
            return ModelCatalog(
                models=("opencode-go/live-model",),
                default_model="opencode-go/live-model",
                native_efforts=(),
                source="opencode models --verbose",
            )
        raise RuntimeError(f"no live {profile} probe in unit tests")

    with patch.object(_disc, "discover_profile_models", side_effect=_fake_discover):
        registry = live_profile_registry(tmp_path)

    assert registry["opencode"]["models"] == ["opencode-go/live-model"]
    assert "default_model" not in registry["opencode"]  # removed from public API
    # codex/claude probes failed → static fallback preserved
    from agent_crossbar.profiles import profile_registry

    static = profile_registry()
    assert registry["codex"]["models"] == static["codex"]["models"]
    assert registry["claude"]["models"] == static["claude"]["models"]
    # Untouched non-discovery profiles keep their static values
    assert registry["reasonix"]["models"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    # Schema stays minimal — no new keys introduced
    allowed = {"aliases", "models", "operations", "interactive", "support_tier"}
    for entry in registry.values():
        assert set(entry.keys()) == allowed


def test_live_profile_registry_falls_back_when_discovery_fails(tmp_path: Path) -> None:
    """When every live probe fails, live_profile_registry matches the static registry."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.profiles import profile_registry

    with patch.object(_disc, "discover_profile_models", side_effect=RuntimeError("boom")):
        registry = live_profile_registry(tmp_path)

    static = profile_registry()
    assert registry["codex"]["models"] == static["codex"]["models"]
    assert registry["claude"]["models"] == static["claude"]["models"]


# ── Task 4.2: cache-only profile_health model surfacing ─────────────────────


def test_cached_profile_health_entry_no_cache_is_honest_not_empty_silently(tmp_path: Path) -> None:
    """No cache yet → discovery_available True, empty models, explicit error explaining why."""
    entry = cached_profile_health_entry(tmp_path, "codex")
    assert entry["discovery_available"] is True
    assert entry["model_info"] == {}
    assert entry["cache_hit"] is False
    assert entry["error"] is not None


def test_cached_profile_health_entry_uses_any_cache_even_stale(tmp_path: Path) -> None:
    """A stale cache entry is still surfaced (marked stale), never hidden."""
    write_cache(
        tmp_path,
        "codex",
        "0.100.0",
        {
            "models": ["gpt-old-1"],
            "default_model": "gpt-old-1",
            "model_info": [{"id": "gpt-old-1", "supported_efforts": [], "default_effort": None}],
            "source": "codex app-server model/list",
        },
    )
    # Age it well past the model-cache TTL
    import json as _json

    path = tmp_path / "model_cache" / "codex.json"
    envelope = _json.loads(path.read_text())
    envelope["fetched_at"] = time.time() - 999999
    path.write_text(_json.dumps(envelope))

    entry = cached_profile_health_entry(tmp_path, "codex")
    assert entry["cache_hit"] is True
    assert entry["stale"] is True
    assert entry["model_info"] == {"": [{"id": "gpt-old-1"}]}


def test_cached_profile_health_entry_never_spawns_subprocess_for_discovery(tmp_path: Path) -> None:
    """cached_profile_health_entry must never invoke the live discovery fetchers."""
    import agent_crossbar.discovery as _disc

    def _boom(*args, **kwargs):
        raise AssertionError("cached_profile_health_entry must not call discover_profile_models")

    with patch.object(_disc, "discover_profile_models", _boom):
        entry = cached_profile_health_entry(tmp_path, "codex")
        assert entry["discovery_available"] is True


def test_cached_profile_health_entry_reasonix_honest_unavailable(tmp_path: Path) -> None:
    """Non-discovery profiles keep the same honest 'unavailable' shape."""
    entry = cached_profile_health_entry(tmp_path, "reasonix")
    assert entry["discovery_available"] is False
    assert "does not support live model discovery" in entry["error"]


# ── Bug: discovery fallback discards the actual failure ─────────────────────


def test_cached_models_for_listing_records_diagnostic_on_exception(tmp_path: Path) -> None:
    """A live-discovery exception must be preserved as a sanitized diagnostic,
    not just silently swallowed in favor of the static fallback."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()

    def _boom(state_root, profile, *, refresh=False):
        raise RuntimeError("codex crashed: connection refused")

    with patch.object(_disc, "discover_profile_models", _boom):
        models, default_model = cached_models_for_listing(
            tmp_path, "codex", ["gpt-5.6-sol"], "gpt-5.6-sol"
        )

    assert models == ["gpt-5.6-sol"]
    diagnostic = discovery_diagnostics.get("codex")
    assert diagnostic is not None
    assert "connection refused" in diagnostic


def test_cached_models_for_listing_records_diagnostic_on_catalog_error(tmp_path: Path) -> None:
    """A catalog carrying a discovery error must also be preserved as a diagnostic."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()

    catalog = ModelCatalog(
        models=(),
        default_model=None,
        native_efforts=(),
        source="codex",
        error="codex not reachable: timeout",
    )
    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        cached_models_for_listing(tmp_path, "codex", ["gpt-5.6-sol"], "gpt-5.6-sol")

    diagnostic = discovery_diagnostics.get("codex")
    assert diagnostic is not None
    assert "codex not reachable" in diagnostic


def test_cached_models_for_listing_sanitizes_secret_in_diagnostic(tmp_path: Path) -> None:
    """A secret-shaped string in a discovery exception must not survive into the
    stored diagnostic — the diagnostic is surfaced via profile_health."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()

    def _boom(state_root, profile, *, refresh=False):
        raise RuntimeError("auth failed: Authorization: Bearer sk-live-abc123XYZ")

    with patch.object(_disc, "discover_profile_models", _boom):
        cached_models_for_listing(tmp_path, "codex", ["gpt-5.6-sol"], "gpt-5.6-sol")

    diagnostic = discovery_diagnostics.get("codex")
    assert diagnostic is not None
    assert "sk-live-abc123XYZ" not in diagnostic


def test_cached_models_for_listing_clears_diagnostic_on_success(tmp_path: Path) -> None:
    """A subsequent successful discovery must clear the previously recorded failure."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()
    discovery_diagnostics.record("codex", "stale failure")

    catalog = ModelCatalog(
        models=("gpt-live-1",),
        default_model="gpt-live-1",
        native_efforts=(),
        source="codex",
    )
    with patch.object(_disc, "discover_profile_models", return_value=catalog):
        cached_models_for_listing(tmp_path, "codex", ["gpt-5.6-sol"], "gpt-5.6-sol")

    assert discovery_diagnostics.get("codex") is None


def test_cached_profile_health_entry_surfaces_last_discovery_diagnostic(
    tmp_path: Path,
) -> None:
    """profile_health (via cached_profile_health_entry) must surface the last
    recorded discovery diagnostic when there is no cache at all, instead of
    the uninformative generic 'no cached discovery yet' message alone."""
    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()
    discovery_diagnostics.record("codex", "codex crashed: connection refused")

    entry = cached_profile_health_entry(tmp_path, "codex")
    assert entry["discovery_available"] is True
    assert entry["model_info"] == {}
    assert entry["cache_hit"] is False
    assert "connection refused" in entry["error"]

    discovery_diagnostics.clear()


def test_cached_profile_health_entry_never_spawns_subprocess_when_reading_diagnostic(
    tmp_path: Path,
) -> None:
    """Surfacing the stored diagnostic must not trigger a live discovery probe."""
    import agent_crossbar.discovery as _disc
    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()
    discovery_diagnostics.record("codex", "some prior failure")

    def _boom(*args, **kwargs):
        raise AssertionError("cached_profile_health_entry must not call discover_profile_models")

    with patch.object(_disc, "discover_profile_models", _boom):
        entry = cached_profile_health_entry(tmp_path, "codex")
        assert entry["discovery_available"] is True

    discovery_diagnostics.clear()


def test_cached_profile_health_entry_prefers_newer_diagnostic_over_stale_cache_error(
    tmp_path: Path,
) -> None:
    """A good cache with no error must not hide a *later* recorded discovery
    failure behind stale ``data["error"]`` (which is ``None`` here) — when a
    newer diagnostic exists, it must win and the entry must be marked stale."""
    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()
    write_cache(
        tmp_path,
        "codex",
        "0.100.0",
        {
            "models": ["gpt-good-1"],
            "default_model": "gpt-good-1",
            "model_info": [{"id": "gpt-good-1", "supported_efforts": [], "default_effort": None}],
            "source": "codex app-server model/list",
        },
    )
    discovery_diagnostics.record("codex", "codex crashed: connection refused")

    entry = cached_profile_health_entry(tmp_path, "codex")

    assert entry["cache_hit"] is True
    assert entry["model_info"] == {"": [{"id": "gpt-good-1"}]}
    assert entry["stale"] is True
    assert "connection refused" in entry["error"]

    discovery_diagnostics.clear()


def test_discovery_diagnostics_thread_safe_concurrent_record_and_get() -> None:
    """Concurrent record()/get() calls across profiles must not corrupt state
    (live_profile_registry probes profiles concurrently via a thread pool)."""
    from concurrent.futures import ThreadPoolExecutor

    from agent_crossbar.discovery import discovery_diagnostics

    discovery_diagnostics.clear()
    profiles = [f"profile-{i}" for i in range(20)]

    def _record(name: str) -> None:
        discovery_diagnostics.record(name, f"failure for {name}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_record, profiles))

    for name in profiles:
        assert discovery_diagnostics.get(name) == f"failure for {name}"

    discovery_diagnostics.clear()
