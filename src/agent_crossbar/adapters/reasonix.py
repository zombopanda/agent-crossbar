"""Experimental Reasonix interactive CLI adapter."""

from __future__ import annotations

from ..profiles.reasonix import SUPPORT_TIER
from .base import StaticAdapter


def _readiness_probe(runner=None):
    from ..readiness import check_reasonix_readiness

    return check_reasonix_readiness(runner)


class ReasonixAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="reasonix",
            support_tier=SUPPORT_TIER,
            backend="print",
            supports_interactive=True,
            effort_map={"low": "low", "medium": "medium", "high": "high", "max": "max"},
            live_model_discovery=True,
            fuzzy_model_suffix_match=True,
            readiness_probe=_readiness_probe,
        )


adapter = ReasonixAdapter()
