"""Canonical profile registry — one module per provider.

Each profile lives in its own bounded module under ``agent_crossbar.profiles.*``.
This package aggregates them into a single registry with a minimal, provider-neutral
public API.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent_crossbar.models import CANONICAL_PROFILES, PROFILE_ALIASES
from agent_crossbar.profiles import chatgpt_pro, claude, codex, opencode, reasonix

# Per-profile constants (re-exported for backward compatibility)
REASONIX_MODELS = reasonix.REASONIX_MODELS
CODEX_MODELS = codex.CODEX_MODELS
CODEX_EFFORTS = codex.CODEX_EFFORTS
CODEX_EFFORT_ALIASES = codex.CODEX_EFFORT_ALIASES
CODEX_DEFAULT_EFFORT = codex.CODEX_DEFAULT_EFFORT
OPENCODE_PROVIDER_ID = opencode.OPENCODE_PROVIDER_ID
OPENCODE_MODELS = opencode.OPENCODE_MODELS

# ── Qualified support matrix ──────────────────────────────────────────────
# Single machine-readable source of truth for provider capabilities.
# Used by profiles_list, readiness probes, doctor, and server validation.
# Every combination claimed as supported MUST pass a live provider gate
# before release.  Each provider module owns its own matrix entry via
# ``build_matrix_entry()``; this module only assembles them — do not
# hardcode a duplicated central matrix here.

PROVIDER_SUPPORT_MATRIX: dict[str, dict[str, Any]] = {
    "reasonix": reasonix.build_matrix_entry(),
    "codex": codex.build_matrix_entry(),
    "claude": claude.build_matrix_entry(),
    "opencode": opencode.build_matrix_entry(),
    "chatgpt_pro": chatgpt_pro.build_matrix_entry(),
}

PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
    "reasonix": reasonix.build_entry(),
    "codex": codex.build_entry(),
    "claude": claude.build_entry(),
    "opencode": opencode.build_entry(),
    "chatgpt_pro": chatgpt_pro.build_entry(),
}


def list_profiles() -> list[str]:
    """Return sorted list of canonical profile names (no aliases)."""
    return sorted(CANONICAL_PROFILES)


def profile_registry() -> dict[str, dict[str, Any]]:
    """Return a copy of the structured canonical profile registry."""
    return deepcopy(PROFILE_REGISTRY)


def allowed_models(profile: str) -> list[str]:
    """Return explicit model ids for profiles with selectable models."""
    return list(PROFILE_REGISTRY[profile].get("models", []))


def profile_operations(profile: str) -> list[str]:
    """Return the list of supported operations for a profile."""
    return list(PROFILE_REGISTRY[profile].get("operations", []))


def profile_interactive(profile: str) -> bool:
    """Return whether the profile supports interactive (tmux) mode."""
    return bool(PROFILE_REGISTRY[profile].get("interactive", False))


def resolve_profile(name: str) -> tuple[bool, str]:
    """Resolve a profile name, following aliases.

    Returns (ok, resolved_name). If the name is unknown, ok is False.
    """
    if name in CANONICAL_PROFILES:
        return True, name
    if name in PROFILE_ALIASES:
        return True, PROFILE_ALIASES[name]
    return False, name
