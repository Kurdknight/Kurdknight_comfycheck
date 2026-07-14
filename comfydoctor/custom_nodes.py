"""What every installed custom node *demands*, and whether it got it.

Nobody surfaces this, and it is where the real pain lives. A ComfyUI install is
one shared site-packages being fought over by 40 independently-authored
requirements.txt files. Node A pins `numpy<2`, node B needs `numpy>=2`, and one
of them is quietly broken - but neither tells you, because pip installed them at
different times and the loser just gets a weird runtime error six clicks into a
workflow.

We read the requirements files (we never run them) and we look at which node
packages failed to import, then join the two: "IPAdapter_plus failed to load,
and it requires insightface, which is not installed."
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .inventory import HAVE_PACKAGING, Inventory, canonicalize_name

if HAVE_PACKAGING:
    from packaging.requirements import InvalidRequirement, Requirement

# Lines in a requirements.txt we cannot meaningfully evaluate.
_SKIP_PREFIX = ("-", "--", "#", "git+", "http://", "https://", ".", "/")


@dataclass
class CustomNode:
    name: str
    path: Path
    requirements: list[str] = field(default_factory=list)   # raw requirement lines
    loaded: bool | None = None                              # None = unknown (CLI mode)
    disabled: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "requirements": self.requirements,
            "loaded": self.loaded,
            "disabled": self.disabled,
        }


@dataclass
class NodeSurvey:
    nodes: list[CustomNode] = field(default_factory=list)
    # canonical package -> [(node name, specifier string)] - who wants what
    demands: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    runtime_known: bool = False   # did we get to observe which nodes loaded?

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "demands": {k: [{"node": n, "specifier": s} for n, s in v] for k, v in self.demands.items()},
            "runtime_known": self.runtime_known,
            "count": len(self.nodes),
        }


def survey(custom_nodes_dir: Path | None) -> NodeSurvey:
    s = NodeSurvey()
    if not custom_nodes_dir or not custom_nodes_dir.is_dir():
        return s

    loaded = _loaded_node_dirs()
    s.runtime_known = loaded is not None

    for entry in sorted(custom_nodes_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "__")):
            continue
        # ComfyUI-Manager disables a node by suffixing the directory.
        disabled = entry.name.endswith((".disabled", ".disabled.bak"))
        node = CustomNode(name=entry.name, path=entry, disabled=disabled)

        req_file = entry / "requirements.txt"
        if req_file.is_file():
            node.requirements = _read_requirements(req_file)

        if disabled:
            node.loaded = False
        elif loaded is not None:
            node.loaded = entry.name in loaded

        s.nodes.append(node)

    s.demands = _build_demands(s.nodes)
    return s


def _read_requirements(path: Path) -> list[str]:
    out: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out
    for line in text.splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith(_SKIP_PREFIX):
            continue
        out.append(line)
    return out


def _build_demands(nodes: list[CustomNode]) -> dict[str, list[tuple[str, str]]]:
    if not HAVE_PACKAGING:
        return {}
    demands: dict[str, list[tuple[str, str]]] = {}
    for node in nodes:
        if node.disabled:
            continue
        for raw in node.requirements:
            try:
                req = Requirement(raw)
            except InvalidRequirement:
                continue
            if req.marker is not None:
                try:
                    if not req.marker.evaluate():
                        continue
                except Exception:
                    pass
            key = canonicalize_name(req.name)
            demands.setdefault(key, []).append((node.name, str(req.specifier)))
    return demands


def _loaded_node_dirs() -> set[str] | None:
    """Which custom_nodes directories did ComfyUI actually import?

    Returns None when we are not running inside ComfyUI, so that rules which
    depend on this can skip cleanly instead of reporting phantom failures.

    ComfyUI imports each custom node as a top-level module named after its
    directory, and records the directory it came from in __file__. Matching on
    __file__ is far more robust than matching module names, which get mangled.
    """
    if "nodes" not in sys.modules or "folder_paths" not in sys.modules:
        return None

    loaded: set[str] = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            parts = Path(f).resolve().parts
        except Exception:
            continue
        try:
            i = len(parts) - 1 - parts[::-1].index("custom_nodes")
        except ValueError:
            continue
        if i + 1 < len(parts):
            loaded.add(parts[i + 1])
    return loaded


def unsatisfied_demands(sv: NodeSurvey, inv: Inventory) -> list[dict]:
    """Node requirements the environment does not currently meet."""
    if not HAVE_PACKAGING:
        return []
    out: list[dict] = []
    for node in sv.nodes:
        if node.disabled:
            continue
        for raw in node.requirements:
            try:
                req = Requirement(raw)
            except InvalidRequirement:
                continue
            if req.marker is not None:
                try:
                    if not req.marker.evaluate():
                        continue
                except Exception:
                    pass
            dist = inv.get(req.name)
            if dist is None:
                out.append({
                    "node": node.name, "package": canonicalize_name(req.name),
                    "requirement": raw, "installed": None,
                    "reason": f"{req.name} is not installed at all",
                })
                continue
            if not req.specifier:
                continue
            from .inventory import satisfies

            ok = satisfies(dist.version, str(req.specifier))
            if ok is False:
                out.append({
                    "node": node.name, "package": dist.name,
                    "requirement": raw, "installed": dist.version,
                    "reason": f"{dist.name} {dist.version} does not satisfy {req.specifier}",
                })
    return out


def conflicting_demands(sv: NodeSurvey, inv: Inventory | None = None) -> list[dict]:
    """Requirements that NO single version could ever satisfy.

    Two nodes whose specifiers have an empty intersection. There is no install
    that makes both happy - the user has to make a call, and we should say so
    rather than pretending an upgrade will fix it.

    The bar for reporting this is deliberately very high, because a false
    positive here is expensive: we'd be telling someone to uninstall a node that
    is working perfectly. So we only claim "irreconcilable" when we can find no
    version at all that satisfies the combined constraints - and we look hard,
    using the versions the specifiers themselves name rather than a synthetic
    ladder (an earlier ladder capped at major version 4, which cheerfully
    declared pillow>=10.2 and pillow>=11.1 unsatisfiable).
    """
    if not HAVE_PACKAGING:
        return []
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    out: list[dict] = []
    for pkg, claims in sv.demands.items():
        pinned = [(n, s) for n, s in claims if s]
        if len(pinned) < 2:
            continue

        combined = SpecifierSet()
        ok = True
        for _, spec in pinned:
            try:
                combined &= SpecifierSet(spec)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        # Shortcut, and the one that matters most: if what is ALREADY installed
        # satisfies every claim, there is by definition no conflict. ftfy==6.1.1
        # and ftfy>=6.1.1 are both happy with 6.1.1.
        installed = inv.version(pkg) if inv else None
        if installed:
            try:
                if combined.contains(Version(installed.split("+")[0]), prereleases=True):
                    continue
            except InvalidVersion:
                pass

        if _satisfiable(combined, pinned):
            continue

        out.append({
            "package": pkg,
            "claims": [{"node": n, "specifier": s} for n, s in pinned],
            "combined": str(combined),
            "installed": installed,
        })
    return out


def _satisfiable(combined, pinned: list[tuple[str, str]]) -> bool:
    """Is there ANY version satisfying all these specifiers?

    Candidates are drawn from the version literals the specifiers actually
    mention, plus small bumps of each. If a constraint set is satisfiable, a
    solution almost always sits on or just above one of its own boundaries -
    which is a far sounder basis than guessing at a range of version numbers.
    """
    import re as _re

    from packaging.version import InvalidVersion, Version

    literals: set[str] = set()
    for _, spec in pinned:
        for m in _re.finditer(r"(\d+(?:\.\d+)*)", spec):
            literals.add(m.group(1))

    candidates: set[str] = set(literals)
    for lit in literals:
        parts = [int(p) for p in lit.split(".")]
        while len(parts) < 3:
            parts.append(0)
        maj, min_, patch = parts[0], parts[1], parts[2]
        # Just above each boundary, and the next minor/major - enough to land
        # inside any open interval the specifiers leave available.
        candidates.update({
            f"{maj}.{min_}.{patch + 1}",
            f"{maj}.{min_ + 1}.0",
            f"{maj + 1}.0.0",
            f"{max(0, maj - 1)}.0.0",
        })
    candidates.add("0.0.0")

    for c in candidates:
        try:
            if combined.contains(Version(c), prereleases=True):
                return True
        except InvalidVersion:
            continue
    return False


def nodes_pinning_torch(sv: NodeSurvey) -> list[dict]:
    """Custom nodes whose requirements.txt mentions torch.

    This deserves its own warning. When ComfyUI-Manager pip-installs such a
    node's requirements, pip resolves `torch` against PyPI - whose default torch
    wheel is CPU-only - and cheerfully *replaces* the user's cu124 build. The
    user's GPU then silently stops being used and everything gets 30x slower,
    with no error anywhere. It is the nastiest failure mode in the ecosystem.
    """
    out: list[dict] = []
    for node in sv.nodes:
        if node.disabled:
            continue
        for raw in node.requirements:
            name = re.split(r"[<>=!~\[; ]", raw.strip(), 1)[0].strip()
            if canonicalize_name(name) in ("torch", "torchvision", "torchaudio", "xformers"):
                out.append({"node": node.name, "requirement": raw, "package": canonicalize_name(name)})
    return out
