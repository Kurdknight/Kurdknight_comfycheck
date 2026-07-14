"""The installed-package truth, read without importing a single package.

This is the hard rule that the old implementation broke: never `import` a
package to find out about it. In a broken environment, importing xformers or
flash_attn against a mismatched torch does not raise a tidy ImportError - it
can abort the process. A diagnostic tool that crashes the thing it is
diagnosing is worse than useless.

Everything here comes from importlib.metadata, i.e. from the .dist-info
directories on disk. It is fast, it is safe, and unlike the old code it gets
opencv/pillow/scikit-learn right, because it works in *distribution* names and
maps them to import names rather than guessing that they are the same string.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import metadata as md
from pathlib import Path
from typing import Iterable

try:  # packaging ships with pip and with torch; near-certain to be present.
    from packaging.markers import UndefinedEnvironmentName
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    HAVE_PACKAGING = True
except Exception:  # pragma: no cover - degraded mode
    HAVE_PACKAGING = False

    def canonicalize_name(n: str) -> str:  # type: ignore[misc]
        return re.sub(r"[-_.]+", "-", n).lower()


@dataclass
class Dist:
    name: str                    # canonical, e.g. "opencv-python-headless"
    raw_name: str                # as declared
    version: str                 # e.g. "2.6.0+cu124"
    location: str | None         # the site dir it lives in
    requires: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)  # top-level import names it provides

    @property
    def base_version(self) -> str:
        """Version with the local tag stripped: 2.6.0+cu124 -> 2.6.0"""
        return self.version.split("+", 1)[0]

    @property
    def local_tag(self) -> str | None:
        """The +local part: 2.6.0+cu124 -> cu124. This is how you tell a CUDA
        torch build from a CPU one without importing torch."""
        return self.version.split("+", 1)[1] if "+" in self.version else None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "location": self.location,
            "local_tag": self.local_tag,
            "modules": self.modules,
        }


@dataclass
class Inventory:
    dists: dict[str, Dist]                       # canonical name -> Dist (first wins = the one that imports)
    duplicates: dict[str, list[Dist]]            # canonical name -> every copy found, when >1
    module_owners: dict[str, list[str]]          # import name -> dists claiming it
    # The pip-check equivalent. Each entry carries the *parsed* target name, so
    # consumers never have to re-derive it from the requirement string - doing
    # that by hand turns "numpy>=2.0" into a package called "numpy>=2-0".
    unsatisfied: list[dict]                      # {dist, requirement, target, installed, reason}

    def get(self, name: str) -> Dist | None:
        return self.dists.get(canonicalize_name(name))

    def version(self, name: str) -> str | None:
        d = self.get(name)
        return d.version if d else None

    def has(self, name: str) -> bool:
        return canonicalize_name(name) in self.dists

    def to_dict(self) -> dict:
        return {
            "packages": {k: v.to_dict() for k, v in sorted(self.dists.items())},
            "duplicates": {k: [d.to_dict() for d in v] for k, v in self.duplicates.items()},
            "shared_modules": {m: o for m, o in self.module_owners.items() if len(o) > 1},
            "unsatisfied": self.unsatisfied,
            "count": len(self.dists),
        }


# pip and setuptools bundle private copies of their own dependencies under
# `_vendor/`. Those copies have real .dist-info directories, so they show up in
# importlib.metadata - but they are NOT on sys.path and can never shadow
# anything. Counting them produces a confident, completely wrong finding
# ("packaging is installed twice!") whose suggested fix would break setuptools.
# They are not installs; they are cargo.
_VENDORED = ("_vendor", "_internal", "_bundled")


def _is_vendored(location: str | None) -> bool:
    if not location:
        return False
    parts = location.replace("\\", "/").lower().split("/")
    return any(v in parts for v in _VENDORED)


def build() -> Inventory:
    dists: dict[str, Dist] = {}
    duplicates: dict[str, list[Dist]] = defaultdict(list)
    module_owners: dict[str, list[str]] = defaultdict(list)

    for dist in md.distributions():
        try:
            raw = dist.metadata["Name"]
            if not raw:
                continue
        except Exception:
            continue

        if _is_vendored(_location_of(dist)):
            continue
        name = canonicalize_name(raw)
        try:
            version = dist.version or "unknown"
        except Exception:
            version = "unknown"

        d = Dist(
            name=name,
            raw_name=raw,
            version=version,
            location=_location_of(dist),
            requires=list(dist.requires or []),
            modules=_top_level_modules(dist),
        )

        duplicates[name].append(d)
        # importlib.metadata yields in sys.path order, so the first copy of a
        # name is the one that actually wins an import. Keep that one as truth.
        if name not in dists:
            dists[name] = d
            for m in d.modules:
                if name not in module_owners[m]:
                    module_owners[m].append(name)
        else:
            for m in d.modules:
                if name not in module_owners[m]:
                    module_owners[m].append(name)

    real_dupes = {k: v for k, v in duplicates.items() if len(v) > 1}
    unsat = _check_requirements(dists)
    return Inventory(
        dists=dists,
        duplicates=real_dupes,
        module_owners=dict(module_owners),
        unsatisfied=unsat,
    )


def _location_of(dist: md.Distribution) -> str | None:
    try:
        p = getattr(dist, "_path", None)
        if p:
            return str(Path(p).parent)
        loc = dist.locate_file("")
        return str(loc) if loc else None
    except Exception:
        return None


def _top_level_modules(dist: md.Distribution) -> list[str]:
    """Which import names does this distribution actually provide?

    This is what lets us say "cv2 is claimed by both opencv-python and
    opencv-python-headless" - a conflict no version check would ever catch,
    and one that silently breaks insightface / GPU decoding for thousands of
    people.
    """
    mods: list[str] = []
    try:
        tl = dist.read_text("top_level.txt")
        if tl:
            mods = [ln.strip() for ln in tl.splitlines() if ln.strip() and not ln.startswith("_")]
    except Exception:
        pass

    if not mods:
        # No top_level.txt (modern wheels often omit it). Derive from RECORD.
        try:
            for f in dist.files or []:
                s = str(f)
                if s.startswith(("..", "__pycache__")) or "/" not in s.replace("\\", "/"):
                    # A bare top-level .py module counts.
                    if s.endswith(".py") and "/" not in s and "\\" not in s:
                        stem = s[:-3]
                        if stem not in ("setup", "conftest") and not stem.startswith("_"):
                            mods.append(stem)
                    continue
                head = s.replace("\\", "/").split("/", 1)[0]
                if head.endswith((".dist-info", ".data", ".egg-info")) or head.startswith("_"):
                    continue
                if head not in mods:
                    mods.append(head)
        except Exception:
            pass
    return sorted(set(mods))


def _check_requirements(dists: dict[str, Dist]) -> list[dict]:
    """In-process `pip check`, plus the reason in words.

    We evaluate every installed distribution's own Requires-Dist against what is
    actually installed. Extras are skipped (an extra you did not ask for is not
    a broken install) and environment markers are honoured.

    The parsed `target` name is carried in the result. Callers must never try to
    recover it by string-slicing the requirement - that is how you end up
    reporting a missing package called "numpy>=2-0".
    """
    if not HAVE_PACKAGING:
        return []

    problems: list[dict] = []
    for name, d in dists.items():
        for req_str in d.requires:
            try:
                req = Requirement(req_str)
            except InvalidRequirement:
                continue

            if req.marker is not None:
                try:
                    # Requirements gated behind an extra are optional by definition.
                    if "extra" in str(req.marker):
                        continue
                    if not req.marker.evaluate():
                        continue
                except UndefinedEnvironmentName:
                    continue
                except Exception:
                    continue

            target_name = canonicalize_name(req.name)
            target = dists.get(target_name)

            if target is None:
                problems.append({
                    "dist": name,
                    "requirement": req_str,
                    "target": target_name,
                    "specifier": str(req.specifier),
                    "installed": None,
                    "reason": f"{req.name} is not installed, but {name} requires it",
                })
                continue

            if not req.specifier:
                continue
            try:
                installed = Version(target.base_version)
            except InvalidVersion:
                continue
            if not req.specifier.contains(installed, prereleases=True):
                problems.append({
                    "dist": name,
                    "requirement": req_str,
                    "target": target_name,
                    "specifier": str(req.specifier),
                    "installed": target.version,
                    "reason": (
                        f"{req.name} {target.version} is installed, but {name} "
                        f"requires {req.specifier}"
                    ),
                })
    return problems


def parse_version(v: str):
    """Version object from a possibly-local version string, or None."""
    if not HAVE_PACKAGING or not v:
        return None
    try:
        return Version(v.split("+", 1)[0])
    except Exception:
        return None


def satisfies(installed: str | None, specifier: str) -> bool | None:
    """True/False, or None when we genuinely cannot tell (don't guess)."""
    if not HAVE_PACKAGING or not installed or not specifier:
        return None
    try:
        from packaging.specifiers import SpecifierSet

        v = Version(installed.split("+", 1)[0])
        return SpecifierSet(specifier).contains(v, prereleases=True)
    except Exception:
        return None


def requirement_pins(dist: Dist, target: str) -> list[str]:
    """Every specifier `dist` places on `target`.

    Used to catch the classic: xformers 0.0.28.post1 declares `torch==2.5.1`,
    you have torch 2.6.0, and the import aborts. The pin is right there in the
    metadata on disk - we never have to import xformers to find it.
    """
    if not HAVE_PACKAGING:
        return []
    want = canonicalize_name(target)
    out: list[str] = []
    for r in dist.requires:
        try:
            req = Requirement(r)
        except InvalidRequirement:
            continue
        if canonicalize_name(req.name) != want:
            continue
        if req.marker is not None and "extra" in str(req.marker):
            continue
        if req.specifier:
            out.append(str(req.specifier))
    return out


def iter_import_names(inv: Inventory, names: Iterable[str]) -> dict[str, str | None]:
    """dist name -> version, for a list of dists. Missing ones map to None."""
    return {n: inv.version(n) for n in names}


def module_is_importable(module: str) -> bool:
    """Can this module be found on the path? Uses find_spec - does NOT import.

    Note find_spec still executes parent __init__ for submodules, so we only
    ever call this on top-level names.
    """
    if module in sys.modules:
        return True
    try:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except Exception:
        return False
