"""Conflicts between the packages themselves.

The old checker listed versions. Listing versions tells you nothing - `numpy
2.1.0` looks perfectly healthy right up until you notice that six of your nodes
were compiled against numpy 1.x and every one of them will throw
`_ARRAY_API not found` on import.

These rules are about *relationships*: who shadows whom, who is fighting whom,
and which two packages are both claiming the same import name.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

from .. import remedy
from ..inventory import parse_version
from ..models import Finding, Severity
from . import Context, rule

CAT = "Package conflicts"

# Distributions that all install into the same `cv2` directory. Having more than
# one is not "extra features" - it is a coin flip decided by install order.
OPENCV_VARIANTS = [
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
]

# Same disease, different package: both write an `onnxruntime` module.
ONNX_VARIANTS = ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"]

# Import names too generic to be worth reporting as a "shared module" conflict -
# lots of packages legitimately ship a `tests` or `docs` folder.
_NOISY_MODULES = {
    "tests", "test", "docs", "examples", "bin", "data", "scripts", "utils",
    "src", "lib", "include", "share", "etc", "conftest", "typing_extensions",
}


@rule
def broken_dependencies(ctx: Context) -> Iterator[Finding]:
    """The in-process equivalent of `pip check`, but it explains itself.

    Every installed package declares what it needs. This finds the ones whose
    needs are not met - the definitive, no-judgement-required list of things pip
    would complain about, computed from the .dist-info on disk.
    """
    if not ctx.inv.unsatisfied:
        yield Finding(
            id="packages.dependencies_ok",
            severity=Severity.OK,
            category=CAT,
            title=f"All {ctx.inv.to_dict()['count']} installed packages have their dependencies satisfied",
            detail="Equivalent to a clean `pip check`.",
        )
        return

    # Group by the package that is *causing* the problem, because that is what
    # you install to fix a whole cluster of complaints at once. The target name
    # is already parsed for us - never re-derive it from the requirement string.
    by_target: dict[str, list[dict]] = defaultdict(list)
    for u in ctx.inv.unsatisfied:
        by_target[u["target"]].append(u)

    for target, items in sorted(by_target.items(), key=lambda kv: -len(kv[1])):
        complainers = sorted({u["dist"] for u in items})
        installed = ctx.inv.version(target)
        missing = installed is None

        lines = [f"  - {u['dist']} requires {u['requirement']}" for u in items[:8]]
        if len(items) > 8:
            lines.append(f"  ...and {len(items) - 8} more")
        if not missing:
            lines.insert(0, f"  installed: {target} {installed}")

        # Combine every specifier so the install command asks for a version that
        # satisfies ALL the complainers at once, not just the last one we saw.
        # `pip install "numpy>=2,<3"` beats a bare `--upgrade numpy` that might
        # sail straight past someone else's upper bound.
        spec = _combined_specifier([u["specifier"] for u in items if u["specifier"]])
        install_arg = f"{target}{spec}" if spec else target

        yield Finding(
            id=f"packages.unsatisfied.{target}",
            severity=Severity.ERROR if missing else Severity.WARNING,
            category=CAT,
            title=(
                f"{target} is required by {len(complainers)} package(s) but isn't installed"
                if missing else
                f"{target} {installed} doesn't satisfy what {len(complainers)} package(s) require"
            ),
            detail="\n".join(lines),
            impact=(
                f"Any node that reaches code in {', '.join(complainers[:3])} can fail at import or "
                f"at runtime. These are the same conflicts `pip check` reports, and they are "
                f"usually left over from installing custom nodes in a different order than their "
                f"authors expected."
            ),
            evidence={"target": target, "installed": installed, "combined_specifier": spec,
                      "requirements": items},
            remedy=remedy.install(
                ctx.env, [install_arg],
                why=(
                    (f"Installs {target}{spec}. " if missing else
                     f"Moves {target} to a version that satisfies all {len(complainers)} of the "
                     f"packages above at once.\n\n")
                    + "Re-run the scan afterwards - resolving one conflict sometimes reveals the "
                      "next one underneath it, and if a new conflict appears in the opposite "
                      "direction, those two dependants genuinely cannot coexist."
                ),
            ),
        )


def _combined_specifier(specs: list[str]) -> str:
    """Intersect every specifier into one pip-installable string."""
    if not specs:
        return ""
    try:
        from packaging.specifiers import SpecifierSet

        combined = SpecifierSet()
        for s in specs:
            combined &= SpecifierSet(s)
        return str(combined)
    except Exception:
        return specs[0]


@rule
def opencv_pileup(ctx: Context) -> Iterator[Finding]:
    present = [v for v in OPENCV_VARIANTS if ctx.inv.has(v)]
    if len(present) < 2:
        return

    # Prefer the plain package on desktop; headless is for servers with no GUI.
    keep = "opencv-python" if "opencv-python" in present else present[0]
    drop = [p for p in present if p != keep]

    yield Finding(
        id="packages.opencv_variants",
        severity=Severity.WARNING,
        category=CAT,
        title=f"{len(present)} different OpenCV packages are installed at once",
        detail=(
            "Installed: " + ", ".join(f"{p} {ctx.inv.version(p)}" for p in present) + ".\n"
            "All of these install into the same `cv2` folder and overwrite each other's files."
        ),
        impact=(
            "Which cv2 you actually get depends on the order they happened to be installed in, so "
            "the same workflow can behave differently on two machines. A later `pip uninstall` of "
            "any one of them deletes files the others still need, which is how people end up with "
            "a cv2 that imports but crashes. Symptoms: preview windows failing to open, or "
            "`cv2.error: function not implemented`."
        ),
        evidence={"variants": {p: ctx.inv.version(p) for p in present}},
        remedy=remedy.resolve_opencv(ctx.env, keep, drop),
    )


@rule
def onnxruntime_pileup(ctx: Context) -> Iterator[Finding]:
    present = [v for v in ONNX_VARIANTS if ctx.inv.has(v)]
    if len(present) < 2:
        return
    keep = "onnxruntime-gpu" if "onnxruntime-gpu" in present and ctx.gpu.has_nvidia_hardware else present[0]
    drop = [p for p in present if p != keep]

    yield Finding(
        id="packages.onnxruntime_variants",
        severity=Severity.ERROR,
        category=CAT,
        title="Both CPU and GPU builds of onnxruntime are installed",
        detail="Installed: " + ", ".join(f"{p} {ctx.inv.version(p)}" for p in present) + ".",
        impact=(
            "They both provide the `onnxruntime` module, so the CPU build usually wins and the GPU "
            "one is dead weight. This is the standard reason face-swap and face-detection nodes "
            "(InsightFace, ReActor, IPAdapter FaceID) run on the CPU at a crawl, or report "
            "'no CUDAExecutionProvider available' while onnxruntime-gpu is clearly installed."
        ),
        evidence={"variants": {p: ctx.inv.version(p) for p in present}, "keeping": keep},
        remedy=remedy.Remedy(
            title=f"Keep only {keep}",
            commands=[
                ctx.env.pip_argv("uninstall", "-y", *present),
                ctx.env.pip_argv("install", "--no-cache-dir", keep),
            ],
            explain=(
                f"Removes every onnxruntime build and reinstalls only {keep}. They share a module "
                f"directory, so uninstalling just the loser would leave a broken remainder - both "
                f"have to go before one comes back."
            ),
            danger="Between the two commands, onnxruntime will not exist. Let it finish.",
        ),
    )


@rule
def numpy_abi_break(ctx: Context) -> Iterator[Finding]:
    """numpy 2.x vs packages compiled against numpy 1.x.

    Computed from real metadata, not from a hand-maintained blocklist: we look
    for anything that actually pins numpy<2 and check it against what's on disk.
    """
    np = ctx.inv.get("numpy")
    if not np:
        return
    v = parse_version(np.version)
    if not v or v.major < 2:
        return

    from ..inventory import requirement_pins

    blocked: list[tuple[str, str]] = []
    for name, dist in ctx.inv.dists.items():
        for spec in requirement_pins(dist, "numpy"):
            if "<2" in spec.replace(" ", ""):
                blocked.append((name, spec))
                break

    if not blocked:
        return

    yield Finding(
        id="packages.numpy2_abi",
        severity=Severity.ERROR,
        category=CAT,
        title=f"numpy {np.version} is installed, but {len(blocked)} package(s) require numpy 1.x",
        detail="\n".join(f"  - {n} requires numpy{s}" for n, s in blocked[:10]),
        impact=(
            "numpy 2 changed its binary interface. Packages compiled against numpy 1 fail on "
            "import with `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` "
            "or `_ARRAY_API not found`. The nodes that depend on them will show up as IMPORT "
            "FAILED in the ComfyUI console."
        ),
        evidence={"numpy": np.version, "requires_numpy1": dict(blocked)},
        remedy=remedy.pin(
            ctx.env, "numpy<2",
            why=(
                "Pins numpy back to the 1.x series, which everything above was built for. This is "
                "the standard fix and costs you nothing - almost nothing in the ComfyUI ecosystem "
                "needs numpy 2 features yet.\n\n"
                "If a node later demands numpy>=2, check whether the packages listed above have "
                "shipped numpy-2 builds before you upgrade."
            ),
        ),
    )


@rule
def shadowed_installs(ctx: Context) -> Iterator[Finding]:
    """The same package installed twice, in two different site directories.

    Classic on Windows: pip installs into the user site-packages while ComfyUI
    imports from the embedded one (or vice versa). The user sees pip say
    "Successfully installed X-1.2.3", ComfyUI keeps loading X-0.9, and everyone
    concludes ComfyUI is haunted.
    """
    for name, copies in sorted(ctx.inv.duplicates.items()):
        locs = [c.location for c in copies]
        versions = {c.version for c in copies}
        if len(set(locs)) < 2:
            continue

        winner = copies[0]
        losers = copies[1:]
        differing = len(versions) > 1

        yield Finding(
            id=f"packages.shadowed.{name}",
            severity=Severity.ERROR if differing else Severity.WARNING,
            category=CAT,
            title=(
                f"{name} is installed twice, at different versions"
                if differing else f"{name} is installed twice"
            ),
            detail="\n".join(
                f"  - {c.version}  in  {c.location}" + ("   <- this is the one that loads" if i == 0 else "")
                for i, c in enumerate(copies)
            ),
            impact=(
                f"Python imports whichever comes first on sys.path - here that's {winner.version}. "
                f"The other copy is invisible, which means `pip install --upgrade {name}` can "
                f"appear to succeed while ComfyUI keeps using the old one. If you have ever "
                f"upgraded a package and seen no change, this is why."
                if differing else
                f"Harmless today, but the moment the two copies drift apart, upgrades will appear "
                f"to do nothing."
            ),
            evidence={"copies": [c.to_dict() for c in copies]},
            remedy=remedy.Remedy(
                title=f"Remove the shadowed copies of {name}",
                commands=[ctx.env.pip_argv("uninstall", "-y", name)] * min(len(copies), 3),
                explain=(
                    f"pip uninstall only removes one copy per run, so this runs it once per copy "
                    f"({len(copies)} found), then you should reinstall {name} once. "
                    f"Run the scan again afterwards to confirm only one is left."
                ),
                danger=f"This removes ALL copies of {name}. You will need to reinstall it afterwards.",
            ),
            # Locations are the interesting part; keep the raw list for the report.
        )


@rule
def contested_module_names(ctx: Context) -> Iterator[Finding]:
    """Two unrelated distributions both claiming the same import name.

    Beyond the curated opencv/onnxruntime cases, this catches the ones we've
    never heard of - which, in an ecosystem where anyone can publish a node with
    any requirements.txt, is most of them.
    """
    known = set(OPENCV_VARIANTS) | set(ONNX_VARIANTS)

    for module, owners in sorted(ctx.inv.module_owners.items()):
        if len(owners) < 2 or module in _NOISY_MODULES or module.startswith("_"):
            continue
        if set(owners) & known:
            continue  # already reported with a better, specific remedy

        yield Finding(
            id=f"packages.contested_module.{module}",
            severity=Severity.WARNING,
            category=CAT,
            title=f"`import {module}` is claimed by {len(owners)} different packages",
            detail=(
                "Provided by: "
                + ", ".join(f"{o} {ctx.inv.version(o)}" for o in owners)
                + ".\nThey write into the same folder, so the files on disk are a mixture of "
                  "whichever was installed last."
            ),
            impact=(
                f"`import {module}` gives you an unpredictable blend of these packages. Uninstalling "
                f"any one of them will delete files the others still need."
            ),
            evidence={"module": module, "owners": {o: ctx.inv.version(o) for o in owners}},
            remedy=remedy.manual(
                title="Decide which one you actually need",
                explain=(
                    f"ComfyDoctor won't guess this one for you - these packages aren't in our known "
                    f"list, so we can't be sure which is the right one to keep. Work out which of "
                    f"{', '.join(owners)} your nodes actually need, then:\n\n"
                    f"  1. uninstall ALL of them\n"
                    f"  2. install only the one you need\n\n"
                    f"Uninstalling only the unwanted one will break the one you keep, because they "
                    f"share files."
                ),
            ),
        )
