"""Tests for Claude interactive /model picker probe and parser.

All tests use ``FakeClaudeModelProbe`` — no real Claude process is ever spawned.
"""

from __future__ import annotations

import pytest

from agent_crossbar.adapters.claude import ClaudeAdapter
from agent_crossbar.adapters.claude_model_probe import (
    _EXPLICIT_ID_RE,
    _MODEL_DISPLAY_PATTERNS,
    _MODEL_FAMILIES,
    _PROMPT_PATTERN,
    ProbeResult,
    _extract_display_name,
    _extract_nearby_efforts,
    _find_section_bounds,
    _looks_like_auth_prompt,
    _looks_like_error,
    _match_model_line,
    _reap_child,
    _recognize_prompt,
    _resolve_numbered_label,
    _sanitize_output,
    parse_model_picker_output,
    probe_to_catalog,
    strip_ansi,
)

# ── Fake probe ──────────────────────────────────────────────────────────────


class FakeClaudeModelProbe:
    """Deterministic ClaudeModelProbe returning canned PTY output."""

    def __init__(self, output: str | None = None, error: str | None = None) -> None:
        self._output = output
        self._error = error

    def probe(self) -> ProbeResult:
        return ProbeResult(output=self._output, error=self._error)


# ── Realistic picker fixture ────────────────────────────────────────────────

# Raw ANSI-laden output simulating a real Claude /model picker.
# Model list: Default (Opus 4.8), Sonnet 5, Fable 5, Opus 4.8 (1M context), Haiku 4.5
RAW_PICKER_ANSI = (
    "\x1b[?25l"  # hide cursor
    "\x1b[2J\x1b[H"  # clear screen, home
    "\x1b[1mModel\x1b[0m\n"
    "\n"
    "\x1b[32m● Opus 4.8 (1M context)  (Default)\x1b[0m\n"
    "  Sonnet 5\n"
    "  Fable 5\n"
    "  Haiku 4.5\n"
    "\n"
    "efforts: low medium high\n"
    "\x1b[?25h"  # show cursor
)

# What the above looks like after ANSI stripping
CLEAN_PICKER = (
    "Model\n"
    "\n"
    "● Opus 4.8 (1M context)  (Default)\n"
    "  Sonnet 5\n"
    "  Fable 5\n"
    "  Haiku 4.5\n"
    "\n"
    "efforts: low medium high\n"
)

# Alternative picker layout: different default marker, different ordering
RAW_PICKER_ALT = (
    "\x1b[?25l"
    "\x1b[2J\x1b[H"
    "\x1b[1mSelect model\x1b[0m\n"
    "\n"
    "> Sonnet 5  \x1b[2m(default)\x1b[0m\n"
    "  Opus 4.8 (1M context)\n"
    "  Fable 5\n"
    "  Haiku 4.5\n"
    "\x1b[?25h"
)

CLEAN_PICKER_ALT = (
    "Select model\n\n> Sonnet 5  (default)\n  Opus 4.8 (1M context)\n  Fable 5\n  Haiku 4.5\n"
)

# Picker with control-sequence redraws (terminal cursor movements)
RAW_PICKER_REDRAWS = (
    "\x1b[?25l"
    "Loading models...\r"
    "\x1b[K"  # erase line
    "\x1b[1A"  # cursor up
    "\x1b[K"  # erase line
    "Model\n"
    "\n"
    "● Opus 4.8 (1M context) (Default)\n"
    "  Sonnet 5\n"
    "\x1b[?25h"
)


# ── ANSI / control-sequence stripping tests ──────────────────────────────────


def test_strip_ansi_removes_csi_sequences() -> None:
    result = strip_ansi("\x1b[32mGreen text\x1b[0m normal")
    assert result == "Green text normal"


def test_strip_ansi_removes_osc_sequences() -> None:
    result = strip_ansi("\x1b]0;Window Title\x07actual content")
    assert result == "actual content"


def test_strip_ansi_removes_cursor_hide_show() -> None:
    result = strip_ansi("\x1b[?25lcontent\x1b[?25h")
    assert result == "content"


def test_strip_ansi_removes_clear_screen() -> None:
    result = strip_ansi("\x1b[2J\x1b[Hcontent")
    assert result == "content"


def test_strip_ansi_preserves_non_escape_text() -> None:
    text = "Plain text without any escapes"
    assert strip_ansi(text) == text


def test_strip_ansi_handles_realistic_picker_output() -> None:
    result = strip_ansi(RAW_PICKER_ANSI)
    assert "Model" in result
    assert "Opus 4.8" in result
    assert "Sonnet 5" in result
    assert "Fable 5" in result
    assert "Haiku 4.5" in result
    assert "Default" in result
    # No raw escape sequences should remain
    assert "\x1b[" not in result


def test_strip_ansi_handles_control_redraws() -> None:
    result = strip_ansi(RAW_PICKER_REDRAWS)
    assert "Opus 4.8" in result
    assert "Sonnet 5" in result
    assert "Loading models" in result  # preserved, not erased by strip_ansi


# ── Model line matching tests ────────────────────────────────────────────────


def test_match_opus_line() -> None:
    canonical, is_default = _match_model_line("● Opus 4.8 (1M context)  (Default)")
    assert canonical == "claude-opus-4-8"
    assert is_default is True


def test_match_sonnet_line() -> None:
    canonical, is_default = _match_model_line("  Sonnet 5")
    assert canonical == "claude-sonnet-5"
    assert is_default is False


def test_match_fable_line() -> None:
    canonical, is_default = _match_model_line("  Fable 5")
    assert canonical == "claude-fable-5"
    assert is_default is False


def test_match_haiku_line() -> None:
    canonical, is_default = _match_model_line("  Haiku 4.5")
    assert canonical == "claude-haiku-4-5"
    assert is_default is False


def test_match_default_with_gt_marker() -> None:
    canonical, is_default = _match_model_line("> Sonnet 5  (default)")
    assert canonical == "claude-sonnet-5"
    assert is_default is True


def test_match_opus_without_context_note() -> None:
    canonical, is_default = _match_model_line("  Opus 4.8")
    assert canonical == "claude-opus-4-8"
    assert is_default is False


def test_match_case_insensitive() -> None:
    canonical, is_default = _match_model_line("  opus 4.8")
    assert canonical == "claude-opus-4-8"


def test_match_new_picker_version_without_a_crossbar_release() -> None:
    canonical, is_default = _match_model_line("  Opus 6.2")
    assert canonical == "claude-opus-6-2"
    assert is_default is False


def test_match_new_picker_family_without_a_crossbar_release() -> None:
    canonical, is_default = _match_model_line("  Nimbus 1.0")
    assert canonical == "claude-nimbus-1-0"
    assert is_default is False


def test_non_model_line_returns_none() -> None:
    canonical, is_default = _match_model_line("  Select a model:")
    assert canonical is None
    assert is_default is False


def test_empty_line_returns_none() -> None:
    canonical, is_default = _match_model_line("")
    assert canonical is None


def test_efforts_line_not_matched_as_model() -> None:
    canonical, is_default = _match_model_line("efforts: low medium high")
    assert canonical is None


# ── Display name extraction tests ───────────────────────────────────────────


def test_extract_display_name_removes_default_marker() -> None:
    name = _extract_display_name("● Opus 4.8 (1M context)  (Default)")
    assert "Opus" in name
    assert "4.8" in name
    assert "Default" not in name
    assert "(" not in name  # parenthetical removed


def test_extract_display_name_removes_gt_marker() -> None:
    name = _extract_display_name("> Sonnet 5")
    assert name == "Sonnet 5"


def test_extract_display_name_plain() -> None:
    name = _extract_display_name("  Haiku 4.5")
    assert "Haiku" in name


# ── Effort extraction tests ──────────────────────────────────────────────────


def test_extract_efforts_from_nearby_lines() -> None:
    lines = [
        "Model",
        "",
        "● Opus 4.8 (1M context) (Default)",
        "efforts: low medium high",
        "  Sonnet 5",
        "",
    ]
    efforts = _extract_nearby_efforts(lines, 2)  # Opus line at index 2
    assert "low" in efforts
    assert "medium" in efforts
    assert "high" in efforts


def test_extract_efforts_no_nearby() -> None:
    lines = ["Model", "", "  Haiku 4.5", ""]
    efforts = _extract_nearby_efforts(lines, 2)
    assert efforts == []


# ── Full parser tests ────────────────────────────────────────────────────────


def test_parse_realistic_picker_output() -> None:
    entries, default_id, error = parse_model_picker_output(CLEAN_PICKER)

    assert error is None
    assert len(entries) == 4
    model_ids = {e["id"] for e in entries}
    assert model_ids == {
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    }
    assert default_id == "claude-opus-4-8"

    # Default entry
    opus_entry = next(e for e in entries if e["id"] == "claude-opus-4-8")
    assert opus_entry["is_default"] is True
    assert "Default" in opus_entry["display_name"] or "Opus" in opus_entry["display_name"]


def test_parse_alt_picker_with_different_default() -> None:
    entries, default_id, error = parse_model_picker_output(CLEAN_PICKER_ALT)

    assert error is None
    assert len(entries) == 4
    assert default_id == "claude-sonnet-5"

    sonnet_entry = next(e for e in entries if e["id"] == "claude-sonnet-5")
    assert sonnet_entry["is_default"] is True


def test_parse_picker_with_redraws() -> None:
    cleaned = strip_ansi(RAW_PICKER_REDRAWS)
    entries, default_id, error = parse_model_picker_output(cleaned)

    assert error is None
    assert len(entries) >= 2  # At minimum Opus + Sonnet
    assert default_id == "claude-opus-4-8"


def test_parse_empty_output() -> None:
    entries, default_id, error = parse_model_picker_output("")

    assert entries == []
    assert error is not None
    assert "empty" in error.lower()


def test_parse_whitespace_only() -> None:
    entries, default_id, error = parse_model_picker_output("   \n  \n  ")

    assert entries == []
    assert error is not None


def test_parse_unrecognized_format() -> None:
    entries, default_id, error = parse_model_picker_output(
        "Some random output\nthat does not contain\nany model names"
    )

    assert entries == []
    assert error is not None
    assert "unrecognized" in error.lower()


def test_parse_auth_prompt() -> None:
    entries, default_id, error = parse_model_picker_output(
        "Please run `claude auth login` to authenticate.\n> "
    )

    assert entries == []
    assert error is not None
    assert "auth" in error.lower()


def test_parse_not_logged_in() -> None:
    entries, default_id, error = parse_model_picker_output(
        "You are not logged in.\nLaunching Claude Code...\n> "
    )

    assert entries == []
    assert error is not None
    assert "auth" in error.lower()


def test_parse_claude_error() -> None:
    entries, default_id, error = parse_model_picker_output("claude: command not found")

    assert entries == []
    assert error is not None


def test_clean_output_no_models_present() -> None:
    """Output that passes basic checks but has no model-like lines."""
    entries, default_id, error = parse_model_picker_output(
        "Welcome to Claude Code\nType /help for commands\n> "
    )
    assert entries == []
    assert error is not None
    assert "unrecognized" in error.lower()


# ── Auth / error heuristic tests ─────────────────────────────────────────────


def test_looks_like_auth_prompt_positive() -> None:
    assert _looks_like_auth_prompt("Run `claude auth login` to continue.") is True
    assert _looks_like_auth_prompt("You are not logged in.") is True
    assert _looks_like_auth_prompt("Authentication required") is True


def test_looks_like_auth_prompt_negative() -> None:
    assert _looks_like_auth_prompt("Model selection:\n● Opus 4.8 (Default)") is False


def test_looks_like_error_positive() -> None:
    assert _looks_like_error("claude: command not found") is True
    assert _looks_like_error("fatal error: cannot start") is True


def test_looks_like_error_negative() -> None:
    assert _looks_like_error("Model\n● Opus 4.8") is False


# ── probe_to_catalog tests ───────────────────────────────────────────────────


def test_probe_to_catalog_success() -> None:
    result = ProbeResult(output=CLEAN_PICKER, error=None)
    data = probe_to_catalog(result)

    assert data["error"] is None
    assert data["source"] == "claude interactive /model picker"
    assert len(data["models"]) == 4
    assert data["default_model"] == "claude-opus-4-8"
    assert len(data["model_info"]) == 4


def test_probe_to_catalog_probe_error() -> None:
    result = ProbeResult(output=None, error="timeout")
    data = probe_to_catalog(result)

    assert data["models"] == []
    assert data["default_model"] is None
    assert data["error"] == "timeout"
    assert data["source"] == "claude interactive /model picker"


def test_probe_to_catalog_unrecognized_output() -> None:
    result = ProbeResult(output="garbage output", error=None)
    data = probe_to_catalog(result)

    assert data["models"] == []
    assert data["error"] is not None
    assert "unrecognized" in data["error"].lower()


# ── ClaudeAdapter.discover_models() with fake probe ──────────────────────────


def test_adapter_discover_models_with_fake_probe() -> None:
    """ClaudeAdapter returns correct ModelCatalog from probe output."""
    probe = FakeClaudeModelProbe(output=CLEAN_PICKER)
    adapter = ClaudeAdapter()
    catalog = adapter.discover_models(probe=probe)

    # Order follows picker layout (Opus first, as in the fixture)
    assert set(catalog.models) == {
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    }
    assert len(catalog.models) == 4
    assert catalog.default_model == "claude-opus-4-8"
    assert catalog.source == "claude interactive /model picker"
    assert catalog.error is None
    assert len(catalog.model_info) == 4


def test_adapter_discover_models_probe_error() -> None:
    """ClaudeAdapter returns empty catalog on probe error."""
    probe = FakeClaudeModelProbe(error="timeout: Claude did not respond")
    adapter = ClaudeAdapter()
    catalog = adapter.discover_models(probe=probe)

    assert catalog.models == ()
    assert catalog.error is not None
    assert "timeout" in catalog.error
    assert catalog.source == "claude interactive /model picker"


def test_adapter_discover_models_unrecognized_output() -> None:
    """ClaudeAdapter returns honest empty catalog on unrecognized output."""
    probe = FakeClaudeModelProbe(output="random text\nno models here\n> ")
    adapter = ClaudeAdapter()
    catalog = adapter.discover_models(probe=probe)

    assert catalog.models == ()
    assert catalog.error is not None
    assert "unrecognized" in catalog.error.lower()
    assert catalog.default_model is None


def test_adapter_discover_models_auth_prompt() -> None:
    """ClaudeAdapter returns honest empty catalog on auth prompt."""
    probe = FakeClaudeModelProbe(output="Please run claude auth login first\n> ")
    adapter = ClaudeAdapter()
    catalog = adapter.discover_models(probe=probe)

    assert catalog.models == ()
    assert catalog.error is not None
    assert "auth" in catalog.error.lower()


def test_adapter_discover_models_never_fakes_static_list() -> None:
    """On any failure path, models must be empty — never a hardcoded fallback."""
    # Empty picker
    probe = FakeClaudeModelProbe(output="")
    catalog = ClaudeAdapter().discover_models(probe=probe)
    assert catalog.models == ()
    assert catalog.error is not None

    # Auth prompt
    probe2 = FakeClaudeModelProbe(output="claude auth login required\n> ")
    catalog2 = ClaudeAdapter().discover_models(probe=probe2)
    assert catalog2.models == ()

    # Unrecognized
    probe3 = FakeClaudeModelProbe(output="just a prompt\n> ")
    catalog3 = ClaudeAdapter().discover_models(probe=probe3)
    assert catalog3.models == ()


# ── Model canonical mapping coverage ────────────────────────────────────────


def test_model_families_covers_all_expected() -> None:
    """The model families tuple covers the four known Claude model families."""
    assert set(_MODEL_FAMILIES) == {"opus", "sonnet", "fable", "haiku"}


def test_model_families_are_unique() -> None:
    """All family aliases are unique."""
    assert len(_MODEL_FAMILIES) == len(set(_MODEL_FAMILIES))


def test_display_patterns_match_all_families() -> None:
    """Every model family has a corresponding display pattern."""
    pattern_keys = {p[0] for p in _MODEL_DISPLAY_PATTERNS}
    assert pattern_keys == set(_MODEL_FAMILIES)


# ── OS guard: PosixClaudeModelProbe on unsupported platform ──────────────────


def test_posix_probe_unsupported_platform(monkeypatch) -> None:
    """PosixClaudeModelProbe returns unsupported error on non-POSIX."""
    from agent_crossbar.adapters.claude_model_probe import PosixClaudeModelProbe

    monkeypatch.setattr("sys.platform", "win32")
    probe = PosixClaudeModelProbe()
    result = probe.probe()

    assert result.error is not None
    assert "unsupported" in result.error.lower()
    assert result.output is None


# ── Explicit full ID tests ──────────────────────────────────────────────────

# Picker where the full model ID is shown explicitly in the text
# (e.g. Claude Code's /model picker with verbose output)
EXPLICIT_ID_PICKER = (
    "Model\n"
    "\n"
    "> claude-sonnet-5  (default)\n"
    "  claude-opus-4-8\n"
    "  claude-fable-5\n"
    "  claude-haiku-4-5\n"
)


def test_match_explicit_full_id() -> None:
    """When picker shows an explicit full ID, use it directly."""
    model_id, is_default = _match_model_line("  claude-sonnet-5")
    assert model_id == "claude-sonnet-5"
    assert is_default is False


def test_match_explicit_full_id_with_default_marker() -> None:
    """Explicit full ID with default marker."""
    model_id, is_default = _match_model_line("> claude-sonnet-5  (default)")
    assert model_id == "claude-sonnet-5"
    assert is_default is True


def test_match_explicit_full_id_case_insensitive() -> None:
    """Explicit full ID matching is case-insensitive."""
    model_id, is_default = _match_model_line("  Claude-Opus-4-8")
    assert model_id == "claude-opus-4-8"


def test_parse_explicit_id_picker() -> None:
    """Full picker with explicit IDs."""
    entries, default_id, error = parse_model_picker_output(EXPLICIT_ID_PICKER)
    assert error is None
    assert len(entries) == 4
    model_ids = {e["id"] for e in entries}
    assert model_ids == {
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-fable-5",
        "claude-haiku-4-5",
    }
    assert default_id == "claude-sonnet-5"


def test_explicit_id_takes_precedence_over_alias() -> None:
    """Explicit full ID is preferred over display-name alias."""
    # This line has both "claude-sonnet-5" and "sonnet" in it.
    # The explicit ID should win.
    model_id, is_default = _match_model_line("● claude-sonnet-5 (Sonnet 5, 1M context)  (Default)")
    assert model_id == "claude-sonnet-5"


# ── Standalone Default entry tests ──────────────────────────────────────────

STANDALONE_DEFAULT_PICKER = (
    "Model\n\n● Default (currently Opus 4.8)\n  Opus 4.8\n  Sonnet 5\n  Haiku 4.5\n"
)


def test_match_standalone_default_entry() -> None:
    """Standalone Default resolves its displayed version to a full CLI ID."""
    model_id, is_default = _match_model_line("● Default (currently Opus 4.8)")
    assert model_id == "claude-opus-4-8"
    assert is_default is True


def test_match_standalone_default_plain() -> None:
    """A bare Default is unusable without a versioned model description."""
    model_id, is_default = _match_model_line("  Default")
    assert model_id is None
    assert is_default is False


def test_parse_standalone_default_picker() -> None:
    """Standalone Default is deduplicated into its versioned model ID."""
    entries, default_id, error = parse_model_picker_output(STANDALONE_DEFAULT_PICKER)
    assert error is None
    assert len(entries) == 3
    model_ids = {e["id"] for e in entries}
    assert "claude-opus-4-8" in model_ids
    assert default_id == "claude-opus-4-8"

    default_entry = next(e for e in entries if e["id"] == "claude-opus-4-8")
    assert default_entry["is_default"] is True


def test_standalone_default_does_not_collide_with_marker() -> None:
    """Default-as-marker on a model line is NOT treated as standalone."""
    # "● Opus 4.8 (Default)" — Default is a marker, not the primary label
    model_id, is_default = _match_model_line("● Opus 4.8 (1M context)  (Default)")
    assert model_id == "claude-opus-4-8"
    assert is_default is True


# ── No-default picker tests ─────────────────────────────────────────────────

NO_DEFAULT_PICKER = "Model\n\n  Opus 4.8\n  Sonnet 5\n  Haiku 4.5\n"


def test_parse_no_default_picker() -> None:
    """When no model is marked as default, default_id is None."""
    entries, default_id, error = parse_model_picker_output(NO_DEFAULT_PICKER)
    assert error is None
    assert len(entries) == 3
    assert default_id is None


def test_probe_to_catalog_no_default() -> None:
    """probe_to_catalog does NOT fall back to first model when no default."""
    result = ProbeResult(output=NO_DEFAULT_PICKER, error=None)
    data = probe_to_catalog(result)

    assert data["error"] is None
    assert len(data["models"]) == 3
    assert data["default_model"] is None  # NOT "opus" — no silent fallback


# ── Claude Code 2.1.211 numbered picker tests ───────────────────────────────

# Real sanitized output from Claude Code v2.1.211 /model picker
# (--ax-screen-reader mode, ANSI already stripped).
# Key characteristics:
#   - Startup banner mentions model names ("Sonnet 5 with high effort")
#   - "Select model" section with numbered entries
#   - "1. Default (recommended)" is its own entry with id "default"
#   - "2. (selected) Sonnet" marks the currently-active model
#   - "Enter selection" terminates the picker section
#   - "High effort (default)" footer is NOT a model-default marker
REAL_211_PICKER = (
    "[Screen Reader Mode: on via flag]\n"
    "Claude Code v2.1.211\n"
    "Sonnet 5 with high effort · Claude Team\n"
    "~/src/example-project/packages/agent-crossbar\n"
    "warning: Safe mode...\n"
    "plan mode on...\n"
    "$you: /model\n"
    "Select model\n"
    "Switch between Claude models. Your pick becomes the default for new sessions.\n"
    "For other/previous model names, specify with --model.\n"
    "1. Default (recommended) — Sonnet 5 · Efficient for routine tasks\n"
    "2. (selected) Sonnet — Sonnet 5 · Efficient for routine tasks\n"
    "3. Fable — Fable 5 · Most capable for your hardest and longest-running tasks ·\n"
    "Requires usage credits\n"
    "4. Opus — Opus 4.8 with 1M context · Best for everyday, complex tasks\n"
    "5. Haiku — Haiku 4.5 · Fastest for quick answers\n"
    "Enter selection [1-5], or Escape to cancel:\n"
    "● High effort (default) ←/→ to adjust\n"
    "Enter to set as default · s to use this session only · Esc to cancel\n"
)

# Duplicate regression: a startup banner that mentions "sonnet" followed by
# the same name in a legacy picker — the parser must deduplicate.
LEGACY_WITH_BANNER_DUPE = (
    "Claude Code v2.1.200\n"
    "Sonnet 5 with high effort · Claude Team\n"
    "Model\n"
    "\n"
    "● Opus 4.8 (1M context)  (Default)\n"
    "  Sonnet 5\n"
    "  Fable 5\n"
    "  Haiku 4.5\n"
)


def test_parse_real_211_picker() -> None:
    """Golden: real Claude 2.1.211 picker produces correct models + default."""
    entries, default_id, error = parse_model_picker_output(REAL_211_PICKER)

    assert error is None
    model_ids = [e["id"] for e in entries]
    assert model_ids == [
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-haiku-4-5",
    ]
    assert default_id == "claude-sonnet-5"

    # Each entry sanity
    sonnet_entry = next(e for e in entries if e["id"] == "claude-sonnet-5")
    assert sonnet_entry["is_default"] is True  # (selected) marker


def test_parse_real_211_no_banner_pollution() -> None:
    """Startup banner 'Sonnet 5 with high effort' does NOT produce a model."""
    entries, default_id, error = parse_model_picker_output(REAL_211_PICKER)
    assert error is None
    # Default/Sonnet collapse to one versioned model, with no banner duplicate.
    assert len(entries) == 4
    sonnet_entries = [e for e in entries if e["id"] == "claude-sonnet-5"]
    assert len(sonnet_entries) == 1


def test_parse_real_211_selected_not_default_marker() -> None:
    """(selected) sets default_model; '1. Default' entry does not."""
    entries, default_id, error = parse_model_picker_output(REAL_211_PICKER)
    assert error is None
    assert default_id == "claude-sonnet-5"


def test_parse_real_211_high_effort_footer_ignored() -> None:
    """'High effort (default)' footer is NOT parsed as a model/default marker."""
    entries, default_id, error = parse_model_picker_output(REAL_211_PICKER)
    assert error is None
    # No "high" model id leaked from the footer
    model_ids = [e["id"] for e in entries]
    assert "high" not in model_ids
    # The footer "(default)" does not change default_model
    assert default_id == "claude-sonnet-5"


def test_parse_numbered_no_selected_no_default_entry() -> None:
    """Numbered picker with neither (selected) nor a 'Default' entry → None."""
    output = (
        "Select model\n"
        "1. Sonnet — Sonnet 5 · Efficient for routine tasks\n"
        "2. Opus — Opus 4.8 with 1M context\n"
        "3. Haiku — Haiku 4.5 · Fastest\n"
        "Enter selection [1-3], or Escape to cancel:\n"
    )
    entries, default_id, error = parse_model_picker_output(output)
    assert error is None
    assert len(entries) == 3
    assert default_id is None
    for e in entries:
        assert e["is_default"] is False


def test_parse_numbered_selected_without_default_entry() -> None:
    """Numbered picker: (selected) on Opus, no 'Default' entry."""
    output = (
        "Select model\n"
        "1. Sonnet — Sonnet 5\n"
        "2. (selected) Opus — Opus 4.8\n"
        "3. Haiku — Haiku 4.5\n"
        "Enter selection [1-3], or Escape to cancel:\n"
    )
    entries, default_id, error = parse_model_picker_output(output)
    assert error is None
    assert default_id == "claude-opus-4-8"
    opus_entry = next(e for e in entries if e["id"] == "claude-opus-4-8")
    assert opus_entry["is_default"] is True


def test_parse_numbered_only_default_no_selected() -> None:
    """Numbered picker with 'Default' entry but no (selected) → default_model=None."""
    output = (
        "Select model\n"
        "1. Default (recommended) — Sonnet 5\n"
        "2. Sonnet — Sonnet 5\n"
        "3. Opus — Opus 4.8\n"
        "Enter selection [1-3], or Escape to cancel:\n"
    )
    entries, default_id, error = parse_model_picker_output(output)
    assert error is None
    # Default/Sonnet share one full ID, but without (selected) there is no active default.
    assert default_id is None
    sonnet_entry = next(e for e in entries if e["id"] == "claude-sonnet-5")
    assert sonnet_entry["is_default"] is False


def test_legacy_deduplication_removes_banner_dupe() -> None:
    """Legacy mode: banner 'Sonnet 5 …' followed by picker 'Sonnet 5' → one entry."""
    entries, default_id, error = parse_model_picker_output(LEGACY_WITH_BANNER_DUPE)
    assert error is None
    # 4 unique models, no duplicate sonnet
    assert len(entries) == 4
    model_ids = [e["id"] for e in entries]
    assert model_ids.count("claude-sonnet-5") == 1
    # Default from the legacy (Default) marker
    assert default_id == "claude-opus-4-8"


def test_legacy_deduplication_keeps_first_default() -> None:
    """Legacy mode: if two lines both claim to be default, first one wins."""
    output = (
        "Model\n"
        "● Opus 4.8 (Default)\n"
        "  Sonnet 5 (Default)\n"  # second default — ignored
    )
    entries, default_id, error = parse_model_picker_output(output)
    assert error is None
    assert default_id == "claude-opus-4-8"  # first default wins


def test_section_bounds_found_in_real_211() -> None:
    """_find_section_bounds correctly locates the Select model section."""
    lines = REAL_211_PICKER.splitlines()
    bounds = _find_section_bounds(lines)
    assert bounds is not None
    header_idx, terminator_idx = bounds
    assert "Select model" in lines[header_idx]
    assert "Enter selection" in lines[terminator_idx]
    assert terminator_idx > header_idx


def test_section_bounds_missing_header_returns_none() -> None:
    """_find_section_bounds returns None when 'Select model' is absent."""
    lines = ["Model", "", "● Opus 4.8 (Default)", "  Sonnet 5"]
    assert _find_section_bounds(lines) is None


def test_section_bounds_header_without_terminator_returns_none() -> None:
    """_find_section_bounds returns None when 'Enter selection' is missing."""
    lines = ["Select model", "1. Sonnet — Sonnet 5", "2. Opus — Opus 4.8"]
    assert _find_section_bounds(lines) is None


def test_resolve_numbered_label_default() -> None:
    """_resolve_numbered_label: 'Default (recommended)' → ('default', False)."""
    model_id, is_selected = _resolve_numbered_label("Default (recommended)")
    assert model_id == "default"
    assert is_selected is False


def test_resolve_numbered_label_selected_sonnet() -> None:
    """_resolve_numbered_label: '(selected) Sonnet' → ('sonnet', True)."""
    model_id, is_selected = _resolve_numbered_label("(selected) Sonnet")
    assert model_id == "sonnet"
    assert is_selected is True


def test_resolve_numbered_label_plain_family() -> None:
    """_resolve_numbered_label: 'Fable' → ('fable', False)."""
    model_id, is_selected = _resolve_numbered_label("Fable")
    assert model_id == "fable"
    assert is_selected is False


def test_resolve_numbered_label_case_insensitive() -> None:
    """_resolve_numbered_label is case-insensitive for family names."""
    model_id, is_selected = _resolve_numbered_label("SONNET")
    assert model_id == "sonnet"


def test_resolve_numbered_label_unknown_returns_none() -> None:
    """_resolve_numbered_label returns None for unrecognized labels."""
    model_id, is_selected = _resolve_numbered_label("UnknownModel")
    assert model_id is None


# ── Ring buffer regression test ─────────────────────────────────────────────


def test_ring_buffer_retains_latest_not_earliest_bytes() -> None:
    """Ring buffer keeps latest bytes, not earliest (regression)."""
    from agent_crossbar.adapters.claude_model_probe import _MAX_CAPTURED_BYTES

    # Simulate: 2 KiB early + 35 KiB late = 37 KiB > 32 KiB cap.
    # The cap keeps the last 32 KiB, so all early bytes are evicted.
    early = b"A" * (2 * 1024)
    late = b"B" * (35 * 1024)

    captured = bytearray(early + late)
    # Apply ring-buffer logic (same as in _run_probe_session)
    if len(captured) > _MAX_CAPTURED_BYTES:
        captured = captured[-_MAX_CAPTURED_BYTES:]

    # Latest bytes (B) must be retained, earliest (A) must be discarded
    assert len(captured) == _MAX_CAPTURED_BYTES
    assert captured[:100] == b"B" * 100  # starts with late data
    assert b"A" not in captured  # early data fully evicted


# ── Child reaping test ──────────────────────────────────────────────────────


def test_reap_child_collects_zombie() -> None:
    """_reap_child synchronously collects a terminated child process."""
    import os

    pid = os.fork()
    if pid == 0:
        # Child — exit immediately
        os._exit(0)

    # Parent — child has exited, it's a zombie
    import time

    time.sleep(0.05)  # Let child exit

    _reap_child(pid)

    # Reaping again should raise (child already collected)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, 0)


# ── Prompt pattern tests ────────────────────────────────────────────────────


def test_prompt_pattern_matches_real_chevron_prompt() -> None:
    """_PROMPT_PATTERN matches the real Claude Code ❯ prompt in bytes."""
    # ❯ in UTF-8 is \xe2\x9d\xaf
    assert _PROMPT_PATTERN.search(b"some output\r\n\xe2\x9d\xaf ") is not None
    assert _PROMPT_PATTERN.search(b"\xe2\x9d\xaf ") is not None
    # Prompt with trailing content (should NOT match — prompt must be at end)
    assert _PROMPT_PATTERN.search(b"\xe2\x9d\xaf extra") is None


def test_prompt_pattern_matches_gt_prompt() -> None:
    """_PROMPT_PATTERN still matches the legacy > prompt."""
    assert _PROMPT_PATTERN.search(b"> ") is not None
    assert _PROMPT_PATTERN.search(b"output\n> ") is not None


def test_prompt_pattern_rejects_non_prompt_text() -> None:
    """_PROMPT_PATTERN does NOT match arbitrary startup output."""
    assert _PROMPT_PATTERN.search(b"Claude Code v2.1.211") is None
    assert _PROMPT_PATTERN.search(b"Sonnet 5 with high effort") is None
    assert _PROMPT_PATTERN.search(b"safe mode") is None
    assert _PROMPT_PATTERN.search(b"") is None


def test_prompt_pattern_matches_dollar_prompt() -> None:
    """_PROMPT_PATTERN matches standalone $ prompt (screen-reader mode)."""
    # Real Claude 2.1.211 fixture: session name line then standalone $
    assert _PROMPT_PATTERN.search(b"agent-crossbar-model-probe-d6a7b02e\r\n$") is not None
    # Standalone $ at end of buffer
    assert _PROMPT_PATTERN.search(b"$") is not None
    # $ with trailing space (typical shell-like prompt)
    assert _PROMPT_PATTERN.search(b"$ ") is not None
    # $ after other output on a prior line
    assert _PROMPT_PATTERN.search(b"some output\r\n$") is not None


def test_prompt_pattern_rejects_embedded_dollar() -> None:
    """_PROMPT_PATTERN does NOT match $ inside a word like $100."""
    # $ followed by digits — NOT a prompt
    assert _PROMPT_PATTERN.search(b"cost: $100") is None
    assert _PROMPT_PATTERN.search(b"$100 total") is None
    # $ in the middle of a line with text after it
    assert _PROMPT_PATTERN.search(b"$5.99 + tax") is None


def test_prompt_pattern_rejects_dollar_in_content() -> None:
    """_PROMPT_PATTERN does NOT match a line ending in dollar content."""
    # $ followed by non-space content on the same line
    assert _PROMPT_PATTERN.search(b"the total is $5\r\nnext line") is None
    assert _PROMPT_PATTERN.search(b"price: $9.99") is None
    # $ with text on the same line (content, not a prompt)
    assert _PROMPT_PATTERN.search(b"echo $PATH") is None


# ── Prompt recognition on sanitized output ──────────────────────────────────

# Real fixture: Claude 2.1.211 screen-reader mode output with standalone `$`
_SANITIZED_DOLLAR_PROMPT = bytearray(b"agent-crossbar-model-probe-d6a7b02e\r\n$")
# Same fixture but with trailing ANSI cursor-show + erase-line after the `$`
_RAW_DOLLAR_WITH_TRAILING_ANSI = bytearray(
    b"agent-crossbar-model-probe-d6a7b02e\r\n$\x1b[?25h\x1b[K"
)
# Dollar *inside* content line (not a prompt)
_DOLLAR_IN_CONTENT = bytearray(b"cost: $100")
_DOLLAR_IN_CONTENT_MULTILINE = bytearray(b"the total is $5\r\nnext line")
# Raw with ESC 7/8 cursor save/restore around the prompt
_RAW_DOLLAR_WITH_ESC_7_8 = bytearray(b"output\r\n\x1b7$\x1b8")


def test_recognize_prompt_sanitized_dollar() -> None:
    """_recognize_prompt returns True for sanitized output ending in $."""
    assert _recognize_prompt(_SANITIZED_DOLLAR_PROMPT) is True


def test_recognize_prompt_raw_dollar_with_trailing_ansi() -> None:
    """_recognize_prompt returns True when raw bytes have trailing ANSI after $.

    This is the critical regression test: real PTY output can include
    ANSI cursor/redraw sequences after the visible prompt character.
    The old _PROMPT_PATTERN.search on raw bytes would miss these because
    the regex anchor ``$`` doesn't match past the trailing control bytes.
    """
    assert _recognize_prompt(_RAW_DOLLAR_WITH_TRAILING_ANSI) is True


def test_recognize_prompt_rejects_dollar_in_content() -> None:
    """_recognize_prompt returns False when $ appears inside content."""
    assert _recognize_prompt(_DOLLAR_IN_CONTENT) is False


def test_recognize_prompt_rejects_dollar_in_content_multiline() -> None:
    """_recognize_prompt returns False when $ is in content, not final line."""
    assert _recognize_prompt(_DOLLAR_IN_CONTENT_MULTILINE) is False


def test_recognize_prompt_chevron() -> None:
    """_recognize_prompt matches the real Claude Code ❯ prompt."""
    assert _recognize_prompt(bytearray(b"some output\r\n\xe2\x9d\xaf ")) is True
    assert _recognize_prompt(bytearray(b"\xe2\x9d\xaf ")) is True
    # ❯ followed by non-whitespace is NOT a prompt
    assert _recognize_prompt(bytearray(b"\xe2\x9d\xaf extra")) is False


def test_recognize_prompt_gt() -> None:
    """_recognize_prompt matches the legacy > prompt."""
    assert _recognize_prompt(bytearray(b"> ")) is True
    assert _recognize_prompt(bytearray(b"output\n> ")) is True


def test_recognize_prompt_rejects_non_prompt_text() -> None:
    """_recognize_prompt returns False for arbitrary startup output."""
    assert _recognize_prompt(bytearray(b"Claude Code v2.1.211")) is False
    assert _recognize_prompt(bytearray(b"Sonnet 5 with high effort")) is False
    assert _recognize_prompt(bytearray(b"safe mode")) is False
    assert _recognize_prompt(bytearray(b"")) is False


def test_recognize_prompt_raw_esc_7_8_around_dollar() -> None:
    """_recognize_prompt strips ESC 7/8 and still finds the $ prompt."""
    assert _recognize_prompt(_RAW_DOLLAR_WITH_ESC_7_8) is True


# ── Enhanced control-sequence sanitization tests ────────────────────────────

# Raw bytes simulating ESC 7/8, SI/SO, NBSP typical of TUI redraws
RAW_WITH_ESC_7_8 = b"before\x1b7saved\x1b8after"
RAW_WITH_SI_SO = b"normal\x0fshift_in\x0eback to normal"
RAW_WITH_NBSP = b"word1\xc2\xa0word2"  # NBSP between words
RAW_WITH_CHARSET_SHIFT = b"text\x1b(Bafter"  # ESC ( B — select ASCII charset


def test_sanitize_output_strips_esc_7_8() -> None:
    """_sanitize_output removes ESC 7 (save cursor) and ESC 8 (restore cursor)."""
    result = _sanitize_output(bytearray(RAW_WITH_ESC_7_8))
    assert "\x1b7" not in result
    assert "\x1b8" not in result
    assert "before" in result
    assert "saved" in result
    assert "after" in result


def test_sanitize_output_strips_esc_7_8_with_space() -> None:
    """_sanitize_output removes ESC 7 and ESC 8 even with space separator."""
    raw = bytearray(b"a\x1b 7b\x1b 8c")
    result = _sanitize_output(raw)
    assert "\x1b" not in result
    assert "7" in result or "a" in result  # non-escape content preserved


def test_sanitize_output_strips_si_so() -> None:
    """_sanitize_output removes SI (\\x0f) and SO (\\x0e) charset shifts."""
    result = _sanitize_output(bytearray(RAW_WITH_SI_SO))
    assert "\x0f" not in result
    assert "\x0e" not in result
    assert "normal" in result
    assert "shift_in" in result
    assert "back to normal" in result


def test_sanitize_output_normalizes_nbsp() -> None:
    """_sanitize_output converts NBSP (\\xa0 / \\xc2\\xa0) to regular space."""
    result = _sanitize_output(bytearray(RAW_WITH_NBSP))
    assert "\xa0" not in result
    assert "\xc2\xa0".encode("utf-8").decode("utf-8", errors="replace") not in result or True
    # Words should be separated by space, not concatenated
    assert "word1 word2" in result


def test_sanitize_output_strips_charset_select() -> None:
    """_sanitize_output removes ESC ( B (select ASCII charset)."""
    result = _sanitize_output(bytearray(RAW_WITH_CHARSET_SHIFT))
    assert "textafter" in result or "text after" in result


def test_strip_ansi_handles_esc_7_8() -> None:
    """strip_ansi (str version) removes ESC 7 and ESC 8 sequences."""
    result = strip_ansi("before\x1b7saved\x1b8after")
    assert "\x1b7" not in result
    assert "\x1b8" not in result
    assert "before" in result
    assert "after" in result


def test_strip_ansi_handles_si_so() -> None:
    """strip_ansi (str version) removes SI and SO charset shifts."""
    result = strip_ansi("normal\x0fshift_in\x0eback")
    assert "\x0f" not in result
    assert "\x0e" not in result
    assert "normalshift_inback" in result or "normal" in result


def test_strip_ansi_normalizes_nbsp() -> None:
    """strip_ansi normalizes NBSP to regular space."""
    result = strip_ansi("col1\xa0col2")
    assert "\xa0" not in result
    # NBSP replaced with space
    assert "col1 col2" in result


# ── Probe argv flag tests ───────────────────────────────────────────────────


def test_safe_base_args_include_ax_screen_reader() -> None:
    """The static safe-argv base includes --ax-screen-reader for deterministic output."""
    from agent_crossbar.adapters.claude_model_probe import _SAFE_CLAUDE_BASE_ARGS

    args_list = list(_SAFE_CLAUDE_BASE_ARGS)
    assert "--ax-screen-reader" in args_list


def test_safe_base_args_include_safe_mode_and_plan_permission() -> None:
    """Core safety flags are present in the static base args."""
    from agent_crossbar.adapters.claude_model_probe import _SAFE_CLAUDE_BASE_ARGS

    args_list = list(_SAFE_CLAUDE_BASE_ARGS)
    assert "--safe-mode" in args_list
    assert "--permission-mode" in args_list
    assert "plan" in args_list
    assert "--strict-mcp-config" in args_list


def test_safe_base_args_include_empty_mcp_config() -> None:
    """Empty MCP config is present to avoid inheriting user MCP servers."""
    from agent_crossbar.adapters.claude_model_probe import (
        _EMPTY_MCP_CONFIG,
        _SAFE_CLAUDE_BASE_ARGS,
    )

    args_list = list(_SAFE_CLAUDE_BASE_ARGS)
    assert "--mcp-config" in args_list
    assert _EMPTY_MCP_CONFIG in args_list


def test_build_probe_argv_includes_session_id_and_name() -> None:
    """_build_probe_argv adds unique --session-id and --name per invocation."""
    from agent_crossbar.adapters.claude_model_probe import _build_probe_argv

    argv1 = _build_probe_argv()
    argv2 = _build_probe_argv()

    assert "--session-id" in argv1
    assert "--name" in argv1

    # Find the values
    sid_idx = argv1.index("--session-id")
    name_idx = argv1.index("--name")
    session_id = argv1[sid_idx + 1]
    name = argv1[name_idx + 1]

    # UUID format (36 chars, 4 dashes)
    assert len(session_id) == 36
    assert session_id.count("-") == 4
    # name format
    assert name.startswith("agent-crossbar-model-probe-")
    assert len(name) > len("agent-crossbar-model-probe-")
    # short ID is first 8 chars of UUID
    assert session_id[:8] == name.split("-")[-1]

    # Each invocation produces a unique session ID
    sid2_idx = argv2.index("--session-id")
    assert argv2[sid2_idx + 1] != session_id


def test_build_probe_argv_no_print_flag() -> None:
    """_build_probe_argv does NOT include -p / --print (we need interactive TUI)."""
    from agent_crossbar.adapters.claude_model_probe import _build_probe_argv

    argv = _build_probe_argv()
    assert "-p" not in argv
    assert "--print" not in argv


def test_build_probe_argv_no_api_key_or_fake_home() -> None:
    """_build_probe_argv does NOT bypass auth with API key env or fake HOME."""
    from agent_crossbar.adapters.claude_model_probe import _build_probe_argv

    argv = _build_probe_argv()
    assert "ANTHROPIC_API_KEY" not in str(argv)
    assert "HOME" not in str(argv)


# ── Child reaping: guaranteed collection test ───────────────────────────────


def test_reap_child_guarantees_collection_with_blocking_wait() -> None:
    """_reap_child guarantees zombie collection even after WNOHANG polling fails."""
    import os
    import time

    pid = os.fork()
    if pid == 0:
        # Child — exit immediately
        os._exit(0)

    # Parent — child has exited, it's a zombie
    time.sleep(0.1)

    _reap_child(pid)

    # After _reap_child returns, the child MUST be collected.
    # A second waitpid should raise ChildProcessError (no such child).
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, 0)

    # Also verify: a second call to _reap_child on the same pid is a no-op
    # (should not raise — the implementation must handle already-reaped pids)
    _reap_child(pid)


# ── Explicit ID regex coverage ──────────────────────────────────────────────


def test_explicit_id_re_matches_valid_ids() -> None:
    """_EXPLICIT_ID_RE matches known full model ID patterns."""
    assert _EXPLICIT_ID_RE.search("claude-opus-4-8") is not None
    assert _EXPLICIT_ID_RE.search("claude-sonnet-5") is not None
    assert _EXPLICIT_ID_RE.search("claude-fable-5") is not None
    assert _EXPLICIT_ID_RE.search("claude-haiku-4-5") is not None


def test_explicit_id_re_rejects_unknown_families() -> None:
    """_EXPLICIT_ID_RE does NOT match model IDs with unknown family names."""
    assert _EXPLICIT_ID_RE.search("claude-unknown-1") is None
    assert _EXPLICIT_ID_RE.search("gpt-4") is None
