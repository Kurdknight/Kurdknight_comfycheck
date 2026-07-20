"""What torch-family versions have ACTUALLY shipped — live from PyPI when
possible, cached on disk, with a baked snapshot as the offline fallback.

Why this module exists: every version-pairing bug this tool has had came from
asserting a version out of a formula ("torch 2.N needs torchaudio 2.N") that
reality had stopped honouring. PyPI's JSON API is the ground truth for "does
torchaudio 2.13 exist at all?" — one cached request answers it with certainty.

Reality check that motivated this (2026-07): torch stable releases reached
2.13, torchvision 0.28, but torchaudio froze at 2.11 (maintenance mode). The
lockstep formula is simply dead for torchaudio — no table bump can fix that,
only knowing what shipped.

Resolution order per package:
  1. fresh in-memory / on-disk cache (< CACHE_TTL old)
  2. live PyPI (timeout TIMEOUT_S, silent failure), result written to cache
  3. stale on-disk cache (better than the snapshot: it was live once)
  4. BAKED snapshot (below)

Network use is a single HTTPS GET to pypi.org per package per day, only for
torch/torchvision/torchaudio, and only when a rule asks a pairing question.
Set COMFYDOCTOR_NO_NETWORK=1 to forbid the fetch entirely (tests do).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import urllib.request

PYPI_URL = "https://pypi.org/pypi/{pkg}/json"
TIMEOUT_S = 4.0
CACHE_TTL = 24 * 3600
CACHE_FILE = os.path.join(tempfile.gettempdir(), "comfydoctor_shipped_versions.json")

# Only final releases count as "shipped" — an rc/dev/a/b upload must never
# make us assert its version pair onto a stable install.
_FINAL_RELEASE = re.compile(r"(\d+)\.(\d+)\.(\d+)(\.post\d+)?")

# Snapshot of PyPI taken 2026-07-20. Used only when both the network and the
# disk cache are unavailable. Being stale here is SAFE in the direction that
# matters: a too-old snapshot makes us say "can't verify" (INFO), never
# "you need version X" for an X we can't see.
BAKED: dict[str, list[list[int]]] = {
    "torch": [[1, m] for m in range(0, 14)] + [[2, m] for m in range(0, 14)],
    "torchvision": [[0, m] for m in range(1, 29)],
    "torchaudio": [[0, m] for m in range(3, 14)] + [[2, m] for m in range(0, 12)],
}

_memo: dict[str, tuple[frozenset[tuple[int, int]], str]] = {}


def _network_allowed() -> bool:
    return os.environ.get("COMFYDOCTOR_NO_NETWORK", "") != "1"


def _load_cache() -> dict:
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CACHE_FILE), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, CACHE_FILE)
    except Exception:
        pass  # a cache that can't be written is just a cache miss next time


def _fetch_pypi_minors(pkg: str) -> frozenset[tuple[int, int]] | None:
    """One GET to PyPI; the set of (major, minor) with at least one final
    release actually uploaded. None on any failure."""
    try:
        req = urllib.request.Request(
            PYPI_URL.format(pkg=pkg),
            headers={"User-Agent": "comfydoctor (version compatibility check)"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.load(resp)
        out = set()
        for ver, files in data.get("releases", {}).items():
            if not files:
                continue  # version registered but nothing ever uploaded
            m = _FINAL_RELEASE.fullmatch(ver)
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
        return frozenset(out) if out else None
    except Exception:
        return None


def shipped_minors(pkg: str) -> tuple[frozenset[tuple[int, int]], str]:
    """The (major, minor) pairs of PKG that have shipped a final release,
    plus where the answer came from: 'live', 'cache', 'stale-cache', 'baked'."""
    if pkg in _memo:
        return _memo[pkg]

    cache = _load_cache()
    entry = cache.get(pkg)
    now = time.time()

    result: tuple[frozenset[tuple[int, int]], str] | None = None

    if entry and now - entry.get("fetched_at", 0) < CACHE_TTL:
        result = (frozenset(tuple(x) for x in entry["minors"]), "cache")

    if result is None and _network_allowed():
        live = _fetch_pypi_minors(pkg)
        if live:
            cache[pkg] = {"fetched_at": now, "minors": sorted(list(x) for x in live)}
            _save_cache(cache)
            result = (live, "live")

    if result is None and entry:  # network down: stale beats baked
        result = (frozenset(tuple(x) for x in entry["minors"]), "stale-cache")

    if result is None:
        result = (frozenset(tuple(x) for x in BAKED.get(pkg, [])), "baked")

    _memo[pkg] = result
    return result


def minor_shipped(pkg: str, mm: tuple[int, int] | None) -> bool:
    """Has PKG ever shipped a final release in the (major, minor) series?"""
    if not mm:
        return False
    minors, _source = shipped_minors(pkg)
    return tuple(mm) in minors


def clear_caches() -> None:
    """Testing hook: forget the in-memory memo (the disk cache is left alone)."""
    _memo.clear()
