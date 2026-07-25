"""Claude profile constants and entry."""

from __future__ import annotations

# Single source of truth for this provider's support tier — the adapter
# module re-exports this constant rather than hardcoding its own literal.
SUPPORT_TIER = "supported"

CLAUDE_MODEL_IDS = {
    "opus": "claude-opus-5",
    "fable": "claude-fable-5",
    "sonnet": "claude-sonnet-4-5",
    "haiku": "claude-haiku-4-5",
}
CLAUDE_MODELS = list(CLAUDE_MODEL_IDS)


def build_entry() -> dict:
    return {
        "aliases": ["opus", "fable"],
        "models": CLAUDE_MODELS,
        "operations": ["review", "advice", "dev"],
        "interactive": False,
        "support_tier": SUPPORT_TIER,
    }


def build_matrix_entry() -> dict:
    entry = build_entry()
    return {
        "support_tier": entry["support_tier"],
        "os": ["darwin", "linux"],
        "operations": entry["operations"],
        "backend": "claude_bg",
        "interaction_modes": ["noninteractive"],
        "effort_support": True,
        "billing_mode": "subscription_quota",
        "job_send_supported": False,
    }
