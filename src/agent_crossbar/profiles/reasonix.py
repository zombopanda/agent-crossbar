"""Reasonix profile constants and entry."""

from __future__ import annotations

# Single source of truth for this provider's support tier — the adapter
# module re-exports this constant rather than hardcoding its own literal.
SUPPORT_TIER = "experimental"

# Static validation allowlist only — NOT a live-discovery catalog. Live
# Reasonix models are namespaced per-provider (e.g.
# "deepseek-flash/deepseek-v4-flash") and are surfaced exclusively through
# ``discovery.discover_profile_models``/``live_profile_registry``, which
# never falls back to this list when a live probe fails or is stale. This
# list backs ``validate_start_request``'s static allowlist path only.
# Reasonix models are discovered from ``reasonix doctor --json`` at runtime.
# Keep this compatibility export empty so static profile listings cannot
# advertise stale IDs or act as a validation fallback.
REASONIX_MODELS: list[str] = []


def build_entry() -> dict:
    return {
        "aliases": ["deepseek"],
        "models": [],
        "operations": ["review", "text", "advice", "dev"],
        "interactive": True,
        "support_tier": SUPPORT_TIER,
    }


def build_matrix_entry() -> dict:
    entry = build_entry()
    return {
        "support_tier": entry["support_tier"],
        "os": ["darwin", "linux"],
        "operations": entry["operations"],
        "backend": "tmux",
        "interaction_modes": ["noninteractive", "interactive"],
        "effort_support": True,
        "billing_mode": "api",
        "job_send_supported": True,
    }
