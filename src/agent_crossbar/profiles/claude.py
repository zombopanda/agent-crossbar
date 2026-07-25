"""Claude profile constants and entry."""

from __future__ import annotations

# Single source of truth for this provider's support tier — the adapter
# module re-exports this constant rather than hardcoding its own literal.
SUPPORT_TIER = "supported"

def build_entry() -> dict:
    return {
        "aliases": ["opus", "fable"],
        # Claude models are discovered from its live /model picker. A static
        # fallback would make the public model surface stale after a CLI/model
        # rollout, so an unavailable probe is represented by an empty list.
        "models": [],
        "operations": ["review", "advice", "dev"],
        "interactive": True,
        "support_tier": SUPPORT_TIER,
    }


def build_matrix_entry() -> dict:
    entry = build_entry()
    return {
        "support_tier": entry["support_tier"],
        "os": ["darwin", "linux"],
        "operations": entry["operations"],
        "backend": "claude_bg",
        "interaction_modes": ["noninteractive", "interactive"],
        "effort_support": True,
        "billing_mode": "subscription_quota",
        "job_send_supported": True,
    }
