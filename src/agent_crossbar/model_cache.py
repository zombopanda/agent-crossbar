"""On-disk model-discovery cache with atomic writes and last-good-on-failure.

Each cache entry is stored as a JSON file under
``<state_root>/model_cache/<profile>.json``.  The file is written
atomically (temp + rename) and has restricted permissions (0o600).

When a fresh fetch fails and any previous cache exists (even stale),
``cache_get_or_fetch`` returns the last-good cached data marked
``stale`` with the fresh error.  If no cache exists the failure
propagates so callers can surface the error.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_dir(state_root: Path) -> Path:
    d = state_root / "model_cache"
    os.makedirs(d, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def _cache_path(state_root: Path, profile: str) -> Path:
    return _cache_dir(state_root) / f"{profile}.json"


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(data, indent=2, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def read_cache(
    state_root: Path,
    profile: str,
    version: str,
    *,
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> dict[str, Any] | None:
    """Return the full cache envelope if fresh, or None.

    Freshness requires:
    - The cache file exists and is valid JSON.
    - ``version`` matches the stored CLI version.
    - ``fetched_at`` is within *ttl_seconds*.
    """
    path = _cache_path(state_root, profile)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if data.get("version") != version:
        return None

    fetched_at = data.get("fetched_at", 0)
    if time.time() - fetched_at > ttl_seconds:
        return None

    return data


def _read_cache_any(state_root: Path, profile: str) -> dict[str, Any] | None:
    """Return the raw cache envelope regardless of TTL or version.

    Returns None only when the file is missing or corrupt.
    """
    path = _cache_path(state_root, profile)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_cache_any(state_root: Path, profile: str) -> dict[str, Any] | None:
    """Public wrapper for :func:`_read_cache_any`.

    Used by diagnostic surfaces (``profile_health``) that want to display
    whatever was last discovered — even a stale or version-mismatched
    entry — rather than hiding it entirely.
    """
    return _read_cache_any(state_root, profile)


def write_cache(
    state_root: Path,
    profile: str,
    version: str,
    data: dict[str, Any],
) -> None:
    """Atomically write a cache entry from the fetcher's output dict.

    Cache failures never corrupt or replace the last good file.
    """
    entry: dict[str, Any] = {
        "version": version,
        "fetched_at": time.time(),
        "source": data.get("source", "unknown"),
        "data": data,
    }
    path = _cache_path(state_root, profile)

    # Read existing entry so we can fall back if the write fails
    backup: dict[str, Any] | None = None
    if path.exists():
        try:
            backup = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    try:
        _atomic_write(path, entry)
    except OSError:
        # If the write failed and we have a backup, restore it
        if backup is not None:
            try:
                _atomic_write(path, backup)
            except OSError:
                pass
        raise


def _build_last_good_result(
    envelope: dict[str, Any],
    fetch_error: Exception,
) -> dict[str, Any]:
    """Build a last-good result dict from a cached envelope + fresh error."""
    inner = dict(envelope["data"])
    inner["fetched_at"] = envelope.get("fetched_at")
    inner["source"] = envelope.get("source", "cache")
    inner["stale"] = True
    inner["error"] = str(fetch_error)
    return inner


def cache_get_or_fetch(
    state_root: Path,
    profile: str,
    version: str,
    fetcher: Any,
    *,
    refresh: bool = False,
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> tuple[dict[str, Any], bool]:
    """Get cached data or fetch fresh.

    Returns ``(data, from_cache)`` where *from_cache* is True when the
    response came from the on-disk cache.  The returned dict includes
    ``fetched_at`` (Unix timestamp) so callers can surface staleness.

    When *refresh* is True the cache is bypassed and fresh data is
    always fetched.

    **Last-good guarantee**: if a fresh fetch raises an exception and
    *any* previous cache file exists (even stale by TTL or
    version), the cached data is returned marked ``stale`` alongside
    the fresh ``error``.  The exception is only propagated when there
    is no cache at all.
    """
    if not refresh:
        cached = read_cache(state_root, profile, version, ttl_seconds=ttl_seconds)
        if cached is not None:
            # Merge envelope metadata into the inner data so callers can
            # access fetched_at without knowing about the envelope.
            inner = dict(cached["data"])
            inner["fetched_at"] = cached.get("fetched_at")
            inner["source"] = cached.get("source", "cache")
            return inner, True

    try:
        fresh = fetcher()
    except Exception as exc:
        # Try to return last-good cached data, even if stale
        last_good = _read_cache_any(state_root, profile)
        if last_good is not None:
            return _build_last_good_result(last_good, exc), True
        # No cache at all — propagate the failure
        raise

    try:
        write_cache(state_root, profile, version, fresh)
    except OSError:
        pass
    # Ensure fetched_at is set even for fresh (non-cached) results
    if isinstance(fresh, dict) and "fetched_at" not in fresh:
        fresh["fetched_at"] = time.time()
    return fresh, False
