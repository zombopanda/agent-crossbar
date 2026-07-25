"""Shared provider adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

PUBLIC_EFFORTS = ("low", "medium", "high", "max")


@dataclass(frozen=True)
class ModelInfo:
    """Per-model discovered capabilities."""

    id: str
    supported_efforts: tuple[str, ...] = ()
    default_effort: str | None = None


@dataclass(frozen=True)
class ModelCatalog:
    models: tuple[str, ...]
    default_model: str | None
    native_efforts: tuple[str, ...]
    source: str
    error: str | None = None
    # Per-model effort metadata when available from live discovery
    model_info: tuple[ModelInfo, ...] = ()
    cli_version: str | None = None
    fetched_at: float | None = None
    stale: bool = False
    cache_hit: bool = False

    def effort_for_model(self, model_id: str) -> tuple[str, ...]:
        """Return supported native efforts for *model_id*, or empty tuple."""
        for info in self.model_info:
            if info.id == model_id:
                return info.supported_efforts
        return ()

    def default_effort_for_model(self, model_id: str) -> str | None:
        """Return the default native effort for *model_id*, or None."""
        for info in self.model_info:
            if info.id == model_id:
                return info.default_effort
        return None


class ProviderAdapter(Protocol):
    name: str
    support_tier: str
    backend: str
    supports_interactive: bool
    effort_map: Mapping[str, str]

    def map_effort(self, effort: str) -> str: ...


class LifecycleAdapter(Protocol):
    """Narrow protocol for adapters that own agent lifecycle.

    Metadata-only adapters (chatgpt_pro, codex, opencode, reasonix) do
    NOT implement this — they have no launch / status / normalize cycle.
    """

    def status(self, runner: Any, session_id: str) -> dict[str, Any]: ...
    def get_logs(self, runner: Any, session_id: str) -> str: ...
    def normalize_result(self, entry: dict[str, Any], logs: str) -> Any: ...


def normalize_effort(effort: str, mapping: Mapping[str, str]) -> str:
    if effort not in PUBLIC_EFFORTS:
        raise ValueError(f"Unknown effort '{effort}'")
    try:
        return mapping[effort]
    except KeyError as exc:
        raise ValueError(f"Effort '{effort}' is not supported") from exc


@dataclass(frozen=True)
class StaticAdapter:
    name: str
    support_tier: str
    backend: str
    supports_interactive: bool
    effort_map: Mapping[str, str]
    # Whether this ACP agent advertises an effort/thought-level selector that
    # can be set through session/set_config_option.
    supports_acp_effort: bool = False

    def map_effort(self, effort: str) -> str:
        return normalize_effort(effort, self.effort_map)
