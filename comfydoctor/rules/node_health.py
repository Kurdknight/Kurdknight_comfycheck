"""Custom nodes: who failed to load, who's fighting whom, and who's about to
wreck your PyTorch install.

This is the section that did not exist before, and it is the one people actually
need. A ComfyUI install is dozens of independent requirements.txt files sharing
one site-packages. Nobody arbitrates between them. We can at least name the
fight.
"""

from __future__ import annotations

from typing import Iterator

from .. import remedy
from ..custom_nodes import conflicting_demands, nodes_pinning_torch, unsatisfied_demands
from ..models import Finding, Severity
from . import Context, rule

CAT = "Custom nodes"


@rule
def failed_imports(ctx: Context) -> Iterator[Finding]:
    """Nodes ComfyUI tried to load and couldn't - joined to *why*.

    ComfyUI prints IMPORT FAILED in a console the user has usually already
    scrolled past, and never says what was missing. We cross-reference each
    failed node against its own requirements.txt and tell them.
    """
    if not ctx.comfy_runtime:
        return  # CLI mode: we can't observe what loaded, so don't invent failures.

    unsat = unsatisfied_demands(ctx.nodes, ctx.inv)
    by_node: dict[str, list[dict]] = {}
    for u in unsat:
        by_node.setdefault(u["node"], []).append(u)

    for node in ctx.nodes.nodes:
        if node.loaded is not False or node.disabled:
            continue

        missing = by_node.get(node.name, [])
        if missing:
            detail = (
                f"'{node.name}' did not load, and its requirements are not met:\n"
                + "\n".join(f"  - {m['reason']}" for m in missing[:6])
            )
            pkgs = sorted({m["package"] for m in missing if m["installed"] is None})
            rem = remedy.Remedy(
                title=f"Install what {node.name} is missing",
                commands=[ctx.env.pip_argv(
                    "install", "-r", str(node.path / "requirements.txt"),
                )],
                explain=(
                    f"Installs {node.name}'s requirements.txt into the Python that ComfyUI is "
                    f"actually running on - which is the part people usually get wrong.\n\n"
                    + (f"Missing packages: {', '.join(pkgs)}." if pkgs else "")
                ),
                danger=(
                    "This node's requirements mention torch. Installing them can replace your CUDA "
                    "PyTorch with a CPU build. Read the requirements file first."
                    if any(p in ("torch", "torchvision", "torchaudio") for p in pkgs) else None
                ),
            ) if (node.path / "requirements.txt").is_file() else None
        else:
            detail = (
                f"'{node.name}' failed to import, but its declared requirements are all satisfied. "
                f"The cause is inside the node itself, not a missing package - a syntax error, an "
                f"undeclared dependency, or an incompatibility with your ComfyUI version."
            )
            rem = remedy.manual(
                title="Read the actual error",
                explain=(
                    f"Look in the ComfyUI console for the traceback next to `IMPORT FAILED: "
                    f"{node.name}` - it names the real cause. If it mentions a module that isn't in "
                    f"this node's requirements.txt, the node author forgot to declare it, and "
                    f"installing that module by hand will usually fix it."
                ),
            )

        yield Finding(
            id=f"nodes.import_failed.{node.name}",
            severity=Severity.ERROR,
            category=CAT,
            title=f"Custom node '{node.name}' failed to load",
            detail=detail,
            impact="Every node it provides is missing from your node menu. Workflows using them won't open.",
            evidence={"node": node.name, "path": str(node.path), "unsatisfied": missing},
            remedy=rem,
        )


@rule
def node_requirements_unmet(ctx: Context) -> Iterator[Finding]:
    """Nodes that *did* load but whose requirements aren't actually met.

    These are the worst kind, because everything looks fine until you run the
    workflow and hit the one code path that touches the missing package.
    """
    unsat = unsatisfied_demands(ctx.nodes, ctx.inv)
    loaded_failures = {n.name for n in ctx.nodes.nodes if n.loaded is False}

    by_node: dict[str, list[dict]] = {}
    for u in unsat:
        if u["node"] in loaded_failures:
            continue  # already reported above, with a better story
        by_node.setdefault(u["node"], []).append(u)

    for node_name, items in sorted(by_node.items()):
        req_path = next((n.path / "requirements.txt" for n in ctx.nodes.nodes
                         if n.name == node_name), None)
        # Same guard as failed_imports: if this node's unmet requirements include
        # torch/vision/audio, `pip install -r requirements.txt` can pull the
        # CPU wheel over a working CUDA build. Warn loudly instead of handing over
        # a silent GPU-killer.
        touches_torch = any(
            (i.get("package") or "") in ("torch", "torchvision", "torchaudio") for i in items
        )
        yield Finding(
            id=f"nodes.requirements_unmet.{node_name}",
            severity=Severity.WARNING,
            category=CAT,
            title=f"'{node_name}' is loaded, but {len(items)} of its requirements aren't met",
            detail="\n".join(f"  - {i['reason']}" for i in items[:6]),
            impact=(
                "The node appears in your menu and looks fine, so this fails late - typically "
                "mid-run, with an ImportError, after you've already waited on a model load."
            ),
            evidence={"node": node_name, "unsatisfied": items},
            remedy=remedy.Remedy(
                title=f"Install {node_name}'s requirements",
                commands=[ctx.env.pip_argv("install", "-r", str(req_path))],
                explain=(
                    "Installs this node's requirements into the interpreter ComfyUI is really using."
                ),
                danger=(
                    "This node's requirements mention torch. Running this can replace your CUDA "
                    "PyTorch with a CPU-only build from PyPI and silently kill GPU acceleration. "
                    "Open the requirements.txt first; if it lists torch/torchvision/torchaudio, "
                    "install the OTHER packages by hand instead."
                    if touches_torch else None
                ),
            ) if req_path and req_path.is_file() else None,
        )


@rule
def irreconcilable_pins(ctx: Context) -> Iterator[Finding]:
    """Two nodes whose version demands have no overlap. No install satisfies both.

    Worth its own severity, because the honest answer here is "you cannot have
    both of these nodes" - and no amount of pip-installing will change that.
    Telling someone to upgrade when the conflict is unwinnable just wastes their
    afternoon.
    """
    for c in conflicting_demands(ctx.nodes, ctx.inv):
        pkg = c["package"]
        claims = c["claims"]
        installed = c.get("installed") or ctx.inv.version(pkg)
        yield Finding(
            id=f"nodes.irreconcilable.{pkg}",
            severity=Severity.WARNING,
            category=CAT,
            title=f"Custom nodes disagree about {pkg}, and no version can satisfy them all",
            detail=(
                "\n".join(f"  - {c_['node']} needs {pkg}{c_['specifier']}" for c_ in claims)
                + f"\n\nCurrently installed: {installed or 'nothing'}."
            ),
            impact=(
                "There is no version of "
                f"{pkg} that makes all of these nodes happy - their requirements genuinely "
                "contradict each other. Whichever node loses will misbehave. You have to pick: "
                "disable one of them, or check whether the stricter one has since relaxed its pin "
                "(node authors often pin defensively and never revisit it)."
            ),
            evidence=c,
            remedy=remedy.manual(
                title="Choose which node to keep",
                explain=(
                    "This one needs a human decision, so ComfyDoctor won't touch it.\n\n"
                    "  1. Check each node's repo - the pin above may be stale and already fixed "
                    "upstream. Updating the node is the best outcome.\n"
                    "  2. If it's real, disable the node you need least (ComfyUI-Manager can "
                    "disable without deleting).\n"
                    "  3. Whatever you do, don't just pip-install one of the pins - you'll break "
                    "the other node and this will look like a new bug."
                ),
            ),
        )


@rule
def nodes_that_can_break_torch(ctx: Context) -> Iterator[Finding]:
    """Custom nodes whose requirements.txt lists torch.

    This is the booby-trap nobody warns you about. `pip install -r
    requirements.txt` on such a node resolves `torch` against PyPI - where the
    default wheel is CPU-only - and happily *replaces* your CUDA build. Your GPU
    silently stops being used. No error is printed anywhere. It is the single
    most common way a working ComfyUI install becomes a slow one, and the user
    has no way to connect the cause to the effect days later.
    """
    offenders = nodes_pinning_torch(ctx.nodes)
    if not offenders:
        return

    # Unpinned `torch` is the dangerous case; `torch==2.6.0` is merely rude.
    risky = [o for o in offenders if o["package"] in ("torch", "torchvision", "torchaudio")]
    if not risky:
        return

    yield Finding(
        id="nodes.requirements_pin_torch",
        severity=Severity.WARNING,
        category=CAT,
        title=f"{len(risky)} custom node(s) list PyTorch in their requirements",
        detail="\n".join(f"  - {o['node']}: {o['requirement']}" for o in risky[:10]),
        impact=(
            "If you ever run `pip install -r requirements.txt` for these nodes - which is exactly "
            "what ComfyUI-Manager's 'Try fix' button does - pip will resolve `torch` against PyPI, "
            "whose default wheel is CPU-only, and quietly overwrite your CUDA build. Everything "
            "keeps working; it just becomes 30x slower, with nothing in any log to explain it.\n\n"
            "Nothing is broken right now. This is a landmine, not a wound."
        ),
        evidence={"nodes": risky},
        remedy=remedy.manual(
            title="How to install these nodes' requirements safely",
            explain=(
                "When you do need to install one of these, protect your torch first:\n\n"
                "  1. Note your current version from the PyTorch section above.\n"
                "  2. Install the node's requirements, then immediately re-run this scan.\n"
                "  3. If 'Your GPU is not being used' has appeared, use its Fix button to put your "
                "CUDA torch back.\n\n"
                "Better: edit the node's requirements.txt and delete the torch lines before "
                "installing. Nodes should never pin torch, and every one of these is a bug you "
                "could report upstream."
            ),
        ),
    )


@rule
def node_inventory(ctx: Context) -> Iterator[Finding]:
    total = len([n for n in ctx.nodes.nodes if not n.disabled])
    if not total:
        return
    failed = len([n for n in ctx.nodes.nodes if n.loaded is False and not n.disabled])
    disabled = len([n for n in ctx.nodes.nodes if n.disabled])

    if ctx.comfy_runtime and failed == 0:
        yield Finding(
            id="nodes.all_loaded",
            severity=Severity.OK,
            category=CAT,
            title=f"All {total} custom nodes loaded successfully",
            detail=f"{disabled} disabled." if disabled else "",
        )
    elif not ctx.comfy_runtime:
        yield Finding(
            id="nodes.inventory",
            severity=Severity.INFO,
            category=CAT,
            title=f"{total} custom nodes installed",
            detail=(
                "Running outside ComfyUI, so we can't see which ones actually loaded. Run the scan "
                "from the ComfyDoctor panel inside ComfyUI to get import-failure detection."
            ),
        )
