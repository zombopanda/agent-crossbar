"""Model-discovery orchestrator — ties cache, runner, and adapters for health checks."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .adapters.base import ModelCatalog, ModelInfo
from .adapters.codex import CodexAdapter
from .adapters.opencode import OpencodeAdapter
from .discovery_runner import DiscoveryProcess, DiscoveryRun
from .envelope import sanitize_diagnostic_text
from .model_cache import CACHE_TTL_SECONDS, cache_get_or_fetch, read_cache_any

# Effort-level names repeat once per model in ``model_info`` — with catalogs
# running into the hundreds of models, spelling them out inflates
# ``profile_health`` for no benefit over a small int code. Callers resolve
# codes back to names via the ``EFFORT_LEGEND`` included in the response.
EFFORT_CODES: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "max": 4,
    "xhigh": 5,
    "ultra": 6,
}
EFFORT_LEGEND: dict[str, str] = {str(code): name for name, code in EFFORT_CODES.items()}


def _encode_efforts(efforts: Any) -> list[int | str]:
    return [EFFORT_CODES.get(e, e) for e in efforts]


def _compact_model_info(model_info: Any) -> list[dict[str, Any]]:
    """Encode model-info entries with effort names replaced by legend codes.

    Accepts either ``ModelInfo`` objects (from a live ``ModelCatalog``) or
    raw cache dicts (``{"id", "supported_efforts", "default_effort"}``).
    ``e`` is omitted entirely when a model has no effort variants — the
    common case for a catalog with hundreds of chat-only models.
    """
    compact = []
    for mi in model_info:
        if isinstance(mi, ModelInfo):
            model_id, efforts, default = mi.id, mi.supported_efforts, mi.default_effort
        else:
            model_id = mi["id"]
            efforts = mi.get("supported_efforts", [])
            default = mi.get("default_effort")
        entry: dict[str, Any] = {"id": model_id}
        if efforts:
            entry["e"] = _encode_efforts(efforts)
        if default is not None:
            entry["d"] = EFFORT_CODES.get(default, default)
        compact.append(entry)
    return compact


def _compact_model_info_with_fallback(model_info: Any, model_ids: Any) -> list[dict[str, Any]]:
    """``_compact_model_info`` with a fallback to bare ``{"id": ...}`` entries.

    Some cache records (older entries, or a catalog fetch that populated
    ``models`` but not per-model effort metadata) carry a populated
    ``models``/``catalog.models`` list alongside an empty ``model_info``.
    Without this fallback those model ids would silently vanish from
    ``profile_health`` instead of surfacing (effort-less, but present).
    """
    compact = _compact_model_info(model_info)
    if not compact and model_ids:
        compact = [{"id": model_id} for model_id in model_ids]
    return compact


def _group_model_info_by_provider(compact: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group compact model-info entries by their ``provider/`` id prefix.

    Ids without a ``/`` (e.g. plain ``claude``/``codex`` model names) are
    grouped under the empty-string key. Grouping removes the repeated
    provider prefix (``opencode-go/``, ``google-vertex/``, ...) from every
    model id, which otherwise repeats hundreds of times in a large catalog.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in compact:
        provider, sep, suffix = entry["id"].partition("/")
        item = dict(entry)
        item["id"] = suffix if sep else provider
        groups.setdefault(provider if sep else "", []).append(item)
    return groups


class _DiscoveryDiagnostics:
    """Thread-safe bounded store of the last sanitized live-discovery failure per profile.

    ``cached_models_for_listing`` falls back to static models on any live-discovery
    failure — this store preserves *why* it fell back so ``profile_health`` can
    surface an honest diagnostic instead of a generic "nothing cached yet"
    message. Cleared per-profile on the next successful discovery.
    """

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}
        self._lock = threading.Lock()

    def record(self, profile: str, message: str) -> None:
        sanitized = sanitize_diagnostic_text(message)
        with self._lock:
            self._entries[profile] = sanitized

    def clear_profile(self, profile: str) -> None:
        with self._lock:
            self._entries.pop(profile, None)

    def get(self, profile: str) -> str | None:
        with self._lock:
            return self._entries.get(profile)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# Module-level store, mirroring the readiness_cache pattern in readiness.py
discovery_diagnostics = _DiscoveryDiagnostics()


def _catalog_to_dict(catalog: ModelCatalog) -> dict[str, Any]:
    """Serialize a ModelCatalog to a plain dict for the cache."""
    return {
        "models": list(catalog.models),
        "default_model": catalog.default_model,
        "native_efforts": list(catalog.native_efforts),
        "model_info": [
            {
                "id": mi.id,
                "supported_efforts": list(mi.supported_efforts),
                "default_effort": mi.default_effort,
            }
            for mi in catalog.model_info
        ],
        "error": catalog.error,
        "source": catalog.source,
    }


def _dict_to_catalog(
    data: dict[str, Any],
    source: str,
    version: str | None,
    fetched_at: float | None,
    *,
    cache_hit: bool = False,
) -> ModelCatalog:
    """Deserialize a cache dict back to a ModelCatalog."""
    model_info_raw = data.get("model_info", [])
    return ModelCatalog(
        models=tuple(data.get("models", [])),
        default_model=data.get("default_model"),
        native_efforts=tuple(data.get("native_efforts", [])),
        source=data.get("source", source),
        error=data.get("error"),
        model_info=tuple(
            ModelInfo(
                id=mi["id"],
                supported_efforts=tuple(mi.get("supported_efforts", [])),
                default_effort=mi.get("default_effort"),
            )
            for mi in model_info_raw
        ),
        cli_version=version,
        fetched_at=fetched_at,
        stale=bool(data.get("stale", False)),
        cache_hit=cache_hit,
    )


class _BoundedSubprocessRunner:
    """An argv-only subprocess runner implementing DiscoveryProcess."""

    def run(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> DiscoveryRun:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
        return DiscoveryRun(completed.returncode, completed.stdout, completed.stderr)


def _get_cli_version(profile: str) -> str:
    """Try to determine the CLI version for cache-key purposes."""
    runner = _BoundedSubprocessRunner()
    binary_map = {
        "codex": "codex",
        "opencode": "opencode",
        "claude": "claude",
        "reasonix": "reasonix",
    }
    binary = binary_map.get(profile)
    if binary is None:
        return "unknown"
    try:
        result = runner.run([binary, "--version"], timeout=10)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


_PROFILES_WITH_DISCOVERY = frozenset({"codex", "opencode", "claude", "reasonix"})


def _parse_reasonix_doctor_providers(payload: dict[str, Any]) -> list[str]:
    """Extract validated, deduplicated qualified model ids from a doctor payload.

    ``reasonix doctor --json`` reports live providers as a top-level
    ``providers`` array: ``[{"name": str, "models": [str, ...],
    "key_present": bool}, ...]``. Qualified ids are built as
    ``<provider name>/<model>`` — preserving the provider namespace so two
    providers can never collide on a bare model name (e.g.
    ``deepseek-flash/deepseek-v4-flash``, ``deepseek-pro/deepseek-v4-pro``).

    ``key_present`` is intentionally never consulted here: it proves only
    that a credential is configured, never that the provider (or any of
    its models) is actually reachable. Readiness's own doctor-based
    ``api-key``/``api-reach`` checks are the sole source of truth for
    reachability — discovery must not upgrade that claim from
    ``key_present`` alone.

    Malformed entries — a non-dict provider, a missing/blank ``name``, a
    ``models`` field that isn't a list, or a non-string/blank model — are
    skipped rather than raised, so a doctor schema drift degrades to fewer
    models instead of crashing discovery. If the top-level ``providers``
    field itself is missing or not a list, this yields no models.
    """
    providers = payload.get("providers")
    if not isinstance(providers, list):
        return []

    qualified: list[str] = []
    seen: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        name = provider.get("name")
        models = provider.get("models")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, str) or not model.strip():
                continue
            qid = f"{name}/{model}"
            if qid not in seen:
                seen.add(qid)
                qualified.append(qid)
    return qualified


def _fetch_reasonix(runner: DiscoveryProcess) -> dict[str, Any]:
    """Discover Reasonix models via ``reasonix doctor --json`` — no static fallback.

    Reads the doctor payload's top-level ``providers`` array and builds a
    qualified, deduplicated catalog (``<provider name>/<model>``) — never a
    single hardcoded ``reasonix/`` prefix, since Reasonix fans out to
    multiple named providers (e.g. ``deepseek-flash``, ``deepseek-pro``).
    This is fully independent of ``check_reasonix_readiness``'s own doctor
    invocation: it never inspects ``key_present`` and never reports
    authentication — a configured key is not proof of live reachability.
    When ``providers`` is missing, malformed, or empty after validation,
    this returns an honest ``error`` — never a guess.
    """
    import json

    source = "reasonix doctor --json"
    try:
        result = runner.run(["reasonix", "doctor", "--json"], timeout=20)
    except Exception as exc:
        return _catalog_to_dict(
            ModelCatalog(
                models=(),
                default_model=None,
                native_efforts=(),
                source=source,
                error=f"reasonix doctor probe failed: {exc}",
            )
        )
    if result.returncode != 0:
        return _catalog_to_dict(
            ModelCatalog(
                models=(),
                default_model=None,
                native_efforts=(),
                source=source,
                error=f"reasonix doctor exited {result.returncode}",
            )
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return _catalog_to_dict(
            ModelCatalog(
                models=(),
                default_model=None,
                native_efforts=(),
                source=source,
                error="reasonix doctor returned non-JSON output",
            )
        )
    if not isinstance(payload, dict):
        return _catalog_to_dict(
            ModelCatalog(
                models=(),
                default_model=None,
                native_efforts=(),
                source=source,
                error="reasonix doctor returned a non-object JSON payload",
            )
        )
    qualified_ids = _parse_reasonix_doctor_providers(payload)
    if not qualified_ids:
        return _catalog_to_dict(
            ModelCatalog(
                models=(),
                default_model=None,
                native_efforts=(),
                source=source,
                error="reasonix doctor --json did not report any providers with models "
                "(providers missing, malformed, or empty)",
            )
        )
    qualified = tuple(qualified_ids)
    return _catalog_to_dict(
        ModelCatalog(
            models=qualified,
            default_model=qualified[0],
            native_efforts=(),
            source=source,
        )
    )


def _fetch_codex(runner: DiscoveryProcess) -> dict[str, Any]:
    adapter = CodexAdapter()
    catalog = adapter.discover_models(runner)
    return _catalog_to_dict(catalog)


def _fetch_opencode(runner: DiscoveryProcess) -> dict[str, Any]:
    adapter = OpencodeAdapter()
    catalog = adapter.discover_models(runner)
    return _catalog_to_dict(catalog)


def _fetch_claude() -> dict[str, Any]:
    """Discover Claude models via the interactive /model picker PTY probe.

    The probe manages its own child process — no runner is needed.
    """
    from .adapters.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    catalog = adapter.discover_models()
    return _catalog_to_dict(catalog)


def discover_profile_models(
    state_root: Path,
    profile: str,
    *,
    refresh: bool = False,
) -> ModelCatalog:
    """Discover models for *profile* with caching.

    Profiles without live discovery (chatgpt_pro) return an honest
    ``ModelCatalog`` with ``error`` set.
    """
    if profile not in _PROFILES_WITH_DISCOVERY:
        return ModelCatalog(
            models=(),
            default_model=None,
            native_efforts=(),
            source=f"{profile} model discovery unavailable",
            error=f"Profile '{profile}' does not support live model discovery.",
        )

    runner = _BoundedSubprocessRunner()
    version = _get_cli_version(profile)

    if profile == "codex":

        def fetcher():
            return _fetch_codex(runner)
    elif profile == "claude":
        fetcher = _fetch_claude  # probe manages its own process, no runner
    elif profile == "reasonix":

        def fetcher():
            return _fetch_reasonix(runner)
    else:

        def fetcher():
            return _fetch_opencode(runner)

    data, from_cache = cache_get_or_fetch(
        state_root,
        profile,
        version,
        fetcher,
        refresh=refresh,
    )

    source = data.get("source", "unknown")
    catalog = _dict_to_catalog(
        data,
        source,
        version,
        data.get("fetched_at", time.time()),
        cache_hit=from_cache,
    )

    return catalog


def build_profile_health_entry(
    state_root: Path,
    profile: str,
    catalog: ModelCatalog | None = None,
    *,
    discovery_available: bool = True,
) -> dict[str, Any]:
    """Build a single profile health entry from a (possibly cached) catalog."""
    if profile in _PROFILES_WITH_DISCOVERY and discovery_available:
        if catalog is None:
            catalog = discover_profile_models(state_root, profile)
        return {
            "name": profile,
            "status": "registered",
            "runtime_checked": True,
            "discovery_available": True,
            "native_efforts": list(catalog.native_efforts),
            "model_info": _group_model_info_by_provider(
                _compact_model_info_with_fallback(catalog.model_info, catalog.models)
            ),
            "source": catalog.source,
            "cli_version": catalog.cli_version,
            "fetched_at": catalog.fetched_at,
            "stale": catalog.stale,
            "cache_hit": catalog.cache_hit,
            "error": catalog.error,
        }
    else:
        return {
            "name": profile,
            "status": "registered",
            "runtime_checked": False,
            "discovery_available": False,
            "models": [],
            "error": f"Profile '{profile}' does not support live model discovery.",
        }


# ── profiles_list / profile_health model surfaces ────────────────────────────


def cached_models_for_listing(
    state_root: Path,
    profile: str,
    fallback_models: list[str],
    fallback_default: str | None,
) -> tuple[list[str], str | None]:
    """Return live-discovered models for *profile*, or the static fallback.

    Attempts bounded live discovery via ``discover_profile_models`` — which
    itself reads a fresh on-disk cache first, so a subsequent call after a
    successful discovery stays cache-only (no subprocess spawn) as long as
    the cache remains fresh and version-matched. Falls back to
    (*fallback_models*, *fallback_default*) only when *profile* has no live
    discovery at all (e.g. ``chatgpt_pro``). For a discovery-capable profile
    (codex, claude, opencode, reasonix) a crashed probe, or a discovered
    catalog carrying a discovery error or empty model list, yields an empty
    list instead — a static allowlist is never surfaced as if it were a
    live-verified catalog. The actual failure reason is not discarded: it
    is preserved (sanitized, bounded) in ``discovery_diagnostics`` for
    ``profile_health`` to surface, and cleared on the next success.
    """
    if profile not in _PROFILES_WITH_DISCOVERY:
        return fallback_models, fallback_default

    try:
        catalog = discover_profile_models(state_root, profile)
    except Exception as exc:
        discovery_diagnostics.record(profile, f"{profile} live discovery crashed: {exc}")
        return [], None

    if not catalog.models or catalog.error:
        discovery_diagnostics.record(
            profile, catalog.error or f"{profile} discovery returned no models"
        )
        return [], None

    discovery_diagnostics.clear_profile(profile)
    return list(catalog.models), catalog.default_model or fallback_default


def live_profile_registry(state_root: Path) -> dict[str, Any]:
    """``profile_registry()`` with codex/claude/opencode/reasonix models
    overlaid by a bounded live-discovery attempt (see
    ``cached_models_for_listing``).

    The four discovery-capable profiles are probed in parallel so
    aggregate latency is bounded by the slowest single probe rather than
    their sum — each probe already enforces its own subprocess timeout.
    A failed probe yields an empty model list for that profile — never the
    static registry entry, which would otherwise advertise hardcoded
    models no live probe actually proved; a fresh, version-matched cache
    short-circuits back to a cache read with no subprocess spawn. Entry
    schema stays identical to ``profile_registry()`` (no keys added or
    removed), only the ``models`` value may change.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .profiles import profile_registry

    registry = profile_registry()
    disabled = {
        value.strip()
        for value in os.environ.get("AGENT_CROSSBAR_SKIP_PROFILES", "").split(",")
        if value.strip()
    }
    for profile in disabled:
        if profile in registry:
            registry[profile]["models"] = []

    with ThreadPoolExecutor(max_workers=max(len(registry), 1)) as pool:
        futures = {
            profile: pool.submit(
                cached_models_for_listing,
                state_root,
                profile,
                entry["models"],
                entry.get("default_model"),
            )
            for profile, entry in registry.items()
            if profile not in disabled
        }
        for profile, future in futures.items():
            models, _default_model = future.result()
            registry[profile]["models"] = models

    return registry


def cached_profile_health_entry(state_root: Path, profile: str) -> dict[str, Any]:
    """Cache-only model-discovery entry for ``profile_health``.

    Unlike ``build_profile_health_entry``, this never spawns a live
    discovery subprocess — it surfaces whatever is already cached (even
    stale or version-mismatched), or an honest explanation when nothing
    has been discovered yet.  Keeps ``profile_health`` fast; the cache
    is populated by real usage (agent_start validation, `doctor`, or an
    explicit refresh).
    """
    if profile not in _PROFILES_WITH_DISCOVERY:
        return {
            "name": profile,
            "status": "registered",
            "runtime_checked": False,
            "discovery_available": False,
            "models": [],
            "error": f"Profile '{profile}' does not support live model discovery.",
        }

    version = _get_cli_version(profile)
    envelope = read_cache_any(state_root, profile)
    if envelope is None:
        last_failure = discovery_diagnostics.get(profile)
        error = (
            f"No cached model discovery yet — last discovery attempt failed: {last_failure}"
            if last_failure is not None
            else (
                "No cached model discovery yet — it is populated by starting a job "
                "or running `agent-crossbar doctor`."
            )
        )
        return {
            "name": profile,
            "status": "registered",
            "runtime_checked": False,
            "discovery_available": True,
            "native_efforts": [],
            "model_info": {},
            "source": None,
            "cli_version": version,
            "fetched_at": None,
            "stale": None,
            "cache_hit": False,
            "error": error,
        }

    data = envelope.get("data", {})
    fetched_at = envelope.get("fetched_at")
    stale = bool(fetched_at is None or time.time() - fetched_at > CACHE_TTL_SECONDS)
    last_failure = discovery_diagnostics.get(profile)
    if last_failure is not None:
        stale = True
    return {
        "name": profile,
        "status": "registered",
        "runtime_checked": True,
        "discovery_available": True,
        "native_efforts": list(data.get("native_efforts", [])),
        "model_info": _group_model_info_by_provider(
            _compact_model_info_with_fallback(data.get("model_info", []), data.get("models", []))
        ),
        "source": envelope.get("source", data.get("source")),
        "cli_version": version,
        "fetched_at": fetched_at,
        "stale": stale,
        "cache_hit": True,
        "error": last_failure if last_failure is not None else data.get("error"),
    }
