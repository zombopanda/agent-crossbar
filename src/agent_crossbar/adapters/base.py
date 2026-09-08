"""Shared provider adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

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
    default_transport: str
    native_lifecycle: bool
    live_model_discovery: bool
    review_warning: str | None

    def map_effort(self, effort: str) -> str: ...

    def resolve_model_id(
        self, requested: str, catalog: ModelCatalog
    ) -> tuple[str | None, tuple[str, str] | None]: ...

    def validate_effort(
        self,
        effort: str | None,
        catalog: ModelCatalog | None,
        model_id: str | None,
    ) -> tuple[str | None, str | None, tuple[str, str] | None]: ...


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
    # Public-request transport defaults are adapter metadata.  Core routing
    # uses this value without identifying a provider by name.
    default_transport: str = "print"
    native_lifecycle: bool = False
    readiness_probe: Callable[[Any], Any] | None = None
    # Whether this ACP agent advertises an effort/thought-level selector that
    # can be set through session/set_config_option.
    supports_acp_effort: bool = False
    # The ACP session-mode value (category=="mode") this provider requires
    # for "dev" tasks, e.g. OpenCode's "build" mode — None means no
    # provider-specific mode requirement. The generic ACP client requires
    # live advertisement and acceptance whenever this is non-None.
    dev_acp_mode: str | None = None
    # Optional adapter-owned ACP readiness probe. Core lifecycle code invokes
    # this hook without branching on provider names.
    acp_readiness: Callable[[Any], dict[str, Any]] | None = None
    live_model_discovery: bool = False
    default_effort: str | None = None
    # Whether live-discovered model ids are provider-qualified
    # (``<provider>/<model>``) and a unique unqualified suffix should also
    # resolve, e.g. OpenCode Go and Reasonix. Exact-id-only adapters
    # (Codex, Claude) leave this False.
    fuzzy_model_suffix_match: bool = False
    # Public effort value aliases accepted in addition to PUBLIC_EFFORTS,
    # e.g. Codex's "light" -> "low".
    effort_aliases: Mapping[str, str] = field(default_factory=dict)
    # Optional warning emitted by the adapter for a validated operation.  The
    # core records this metadata without knowing which provider needs it.
    review_warning: str | None = None

    def map_effort(self, effort: str) -> str:
        return normalize_effort(effort, self.effort_map)

    def resolve_model_id(
        self, requested: str, catalog: "ModelCatalog"
    ) -> tuple[str | None, tuple[str, str] | None]:
        """Resolve *requested* against a live-discovered catalog.

        Returns ``(model_id, None)`` on success or ``(None, (error, message))``
        on failure. The base behavior requires an exact match; adapters with
        provider-qualified ids (``fuzzy_model_suffix_match``) also accept a
        unique unqualified suffix.
        """
        if requested in catalog.models:
            return requested, None
        if self.fuzzy_model_suffix_match:
            matches = [
                candidate
                for candidate in catalog.models
                if "/" in candidate and candidate.split("/", 1)[1] == requested
            ]
            if len(matches) > 1:
                return None, (
                    "ambiguous_model",
                    f"Model '{requested}' matches multiple {self.name} providers "
                    f"({', '.join(sorted(matches))}); pass a fully qualified id.",
                )
            if matches:
                return matches[0], None
        return None, (
            "invalid_model",
            f"Model '{requested}' is not available in {self.name} "
            f"(discovered: {', '.join(catalog.models)})",
        )

    def validate_effort(
        self,
        effort: str | None,
        catalog: "ModelCatalog | None",
        model_id: str | None,
    ) -> tuple[str | None, str | None, tuple[str, str] | None]:
        """Resolve/validate a public *effort* against live discovery data.

        Returns ``(normalized_effort, resolved_effort, None)`` on success or
        ``(None, None, (error, message))`` on failure. The base behavior
        performs no effort handling at all — adapters without a native
        effort/model-capability relationship (Claude, Reasonix) leave the
        public effort field untouched. Adapters with per-model effort
        discovery (Codex, OpenCode) override this.
        """
        return None, None, None
