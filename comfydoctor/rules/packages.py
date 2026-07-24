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

# These ship as CUDA wheels from PyTorch's own index. A bare `pip install torch`
# resolves against PyPI, whose default wheel is CPU-ONLY, and silently replaces
# a working CUDA build — the single worst thing this tool could ever tell a user
# to do. Any remedy touching these must pin the PyTorch index, so they are
# routed to reinstall_torch_stack instead of a bare install command.
TORCH_FAMILY = {"torch", "torchvision", "torchaudio"}

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

    from ..inventory import requirement_pins

    for target, items in sorted(by_target.items(), key=lambda kv: -len(kv[1])):
        complainers = sorted({u["dist"] for u in items})
        installed = ctx.inv.version(target)
        missing = installed is None

        lines = [f"  - {u['dist']} requires {u['requirement']}" for u in items[:8]]
        if len(items) > 8:
            lines.append(f"  ...and {len(items) - 8} more")
        if not missing:
            lines.insert(0, f"  installed: {target} {installed}")

        # THE "will this fix break something else?" guard: every OTHER installed
        # package's declared pin on this target joins the install specifier too
        # — not just the complainers'. That makes pip itself the verifier: the
        # command can never satisfy one package by violating another's declared
        # requirement, and if the pins genuinely contradict, pip refuses up
        # front and installs nothing. (Undeclared needs can still exist — which
        # is why the remedy also says that doing nothing is a valid choice on a
        # working machine.)
        bystander_pins: list[tuple[str, str]] = []
        for name, dist in ctx.inv.dists.items():
            if name == target or name in complainers:
                continue
            for pin in requirement_pins(dist, target):
                bystander_pins.append((name, pin))

        if bystander_pins and not missing:
            shown = sorted({n for n, _ in bystander_pins})
            lines.append(
                f"  also pinned by {len(shown)} other package(s) "
                f"({', '.join(shown[:5])}{'...' if len(shown) > 5 else ''}) - "
                f"the fix respects their pins too"
            )

        # Combine every specifier so the install command asks for a version that
        # satisfies ALL the complainers at once, not just the last one we saw.
        # `pip install "numpy>=2,<3"` beats a bare `--upgrade numpy` that might
        # sail straight past someone else's upper bound.
        spec = _combined_specifier(
            [u["specifier"] for u in items if u["specifier"]]
            + [p for _, p in bystander_pins]
        )
        install_arg = f"{target}{spec}" if spec else target

        # torch/vision/audio must NEVER get a bare `pip install` remedy — that
        # pulls the CPU wheel from PyPI over a working CUDA build. Route them to
        # the matched-stack reinstall, which pins the correct PyTorch index.
        if target in TORCH_FAMILY:
            fix = remedy.reinstall_torch_stack(
                ctx.env, ctx.gpu,
                torch_version=ctx.gpu.torch_version if target == "torch" else None,
                reason=(
                    f"{target} is what other packages are unsatisfied with. It must be "
                    f"reinstalled from the PyTorch index (never plain PyPI, which is the "
                    f"CPU-only wheel), together with its matched torch/vision/audio siblings."
                ),
            )
        elif missing:
            fix = remedy.install(
                ctx.env, [install_arg],
                why=(
                    f"Installs {target}{spec}. Re-run the scan afterwards - resolving one "
                    f"conflict sometimes reveals the next one underneath it."
                ),
            )
        else:
            fix = remedy.Remedy(
                title=f"Install {install_arg}",
                commands=[ctx.env.pip_argv("install", install_arg)],
                explain=(
                    f"Moves {target} to a version that satisfies all {len(complainers)} "
                    f"complaining package(s) at once - AND every other installed package's "
                    f"declared pin on {target}, which is baked into the command. If no such "
                    f"version exists, pip refuses and changes nothing; that would mean these "
                    f"packages genuinely cannot coexist, and no version change can help.\n\n"
                    f"Re-run the scan afterwards - resolving one conflict sometimes reveals "
                    f"the next one underneath it."
                ),
                danger=(
                    f"This moves a shared library that other packages also use. Declared "
                    f"requirements are respected automatically, but a package can rely on "
                    f"behaviour it never declared. If the nodes that use "
                    f"{', '.join(complainers[:3])} currently work, the pin above may simply "
                    f"be stale caution - doing nothing is also a valid choice."
                ),
            )

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
            remedy=fix,
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


# Order of preference when collapsing an OpenCV pile-up. contrib is a strict
# SUPERSET of plain (all main modules + the contrib extras like ximgproc that
# controlnet_aux and friends import), so keeping plain when contrib is present
# would break every node that touches a contrib module. Headless last: it's
# for servers, and desktop nodes may want the GUI bits.
_OPENCV_KEEP_ORDER = [
    "opencv-contrib-python",
    "opencv-python",
    "opencv-contrib-python-headless",
    "opencv-python-headless",
]


@rule
def opencv_pileup(ctx: Context) -> Iterator[Finding]:
    present = [v for v in OPENCV_VARIANTS if ctx.inv.has(v)]
    if len(present) < 2:
        return

    keep = next((p for p in _OPENCV_KEEP_ORDER if p in present), present[0])
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
    # Keep the build that matches the HARDWARE, not just onnxruntime-gpu. On a
    # non-NVIDIA machine (AMD/Intel on Windows), onnxruntime-directml / -openvino
    # IS the GPU path — defaulting to present[0] (often plain CPU onnxruntime)
    # would uninstall the user's only accelerator and keep the slow one.
    if "onnxruntime-gpu" in present and ctx.gpu.has_nvidia_hardware:
        keep = "onnxruntime-gpu"
    elif not ctx.gpu.has_nvidia_hardware and "onnxruntime-directml" in present:
        keep = "onnxruntime-directml"
    elif not ctx.gpu.has_nvidia_hardware and "onnxruntime-openvino" in present:
        keep = "onnxruntime-openvino"
    else:
        keep = present[0]
    drop = [p for p in present if p != keep]

    # A dev/rc build of the keeper was almost certainly installed on purpose
    # from a special index (e.g. the ORT nightly feed for a new CUDA major).
    # Plain `pip install` would replace it with the latest STABLE from PyPI,
    # which may not support that CUDA yet — trading slow-but-working for
    # broken. Say so instead of silently downgrading.
    keep_ver = parse_version(ctx.inv.version(keep))
    danger = "Between the two commands, onnxruntime will not exist. Let it finish."
    if keep_ver is not None and keep_ver.is_prerelease:
        danger += (
            f" Also: your current {keep} {ctx.inv.version(keep)} is a PRE-RELEASE build, "
            f"probably installed from a nightly index on purpose (new CUDA support). This "
            f"command reinstalls the latest stable from PyPI instead. If a node afterwards "
            f"reports 'no CUDAExecutionProvider', reinstall the nightly from its original index."
        )

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
            danger=danger,
        ),
    )


@rule
def numpy_abi_break(ctx: Context) -> Iterator[Finding]:
    """numpy 2.x vs packages compiled against numpy 1.x.

    Computed from real metadata, not from a hand-maintained blocklist — and the
    specifiers are EVALUATED, never string-matched. The old `"<2" in spec` test
    read `numpy>=1.24,<2.3` (a normal, *satisfied* numpy-2 pin) as "needs numpy
    1.x" and pushed a downgrade remedy on healthy machines: the torchaudio
    formula disease in a different coat.
    """
    np = ctx.inv.get("numpy")
    if not np:
        return
    v = parse_version(np.version)
    if not v or v.major < 2:
        return

    from ..inventory import requirement_pins, satisfies

    # 1.26.4 is the final numpy 1.x ever released — a fixed historical fact, so
    # baking it cannot rot. It stands in for "does this spec accept ANY numpy 1?".
    LAST_NUMPY1 = "1.26.4"

    blocked: list[tuple[str, str]] = []       # rejects installed numpy, accepts 1.x
    need_np2: list[tuple[str, str]] = []      # rejects every numpy 1.x
    for name, dist in ctx.inv.dists.items():
        for spec in requirement_pins(dist, "numpy"):
            ok_installed = satisfies(np.version, spec)
            ok_np1 = satisfies(LAST_NUMPY1, spec)
            if ok_installed is False and ok_np1 is not False:
                blocked.append((name, spec))
                break
            if ok_np1 is False:
                need_np2.append((name, spec))
                break

    if not blocked:
        return

    # If OTHER packages require numpy>=2, downgrading to 1.x just breaks THEM
    # instead — an unwinnable conflict, not a one-click pin. (In 2026 much of the
    # stack has moved to numpy 2, so this is increasingly the real situation.)
    # Report it honestly and let the human choose, rather than firing a
    # destructive ERROR on one side of a genuine standoff.
    if need_np2:
        yield Finding(
            id="packages.numpy2_conflict",
            severity=Severity.WARNING,
            category=CAT,
            title=f"Packages disagree about numpy: {len(blocked)} need 1.x, {len(need_np2)} need 2.x",
            detail=(
                "Need numpy 1.x:\n" + "\n".join(f"  - {n} requires numpy{s}" for n, s in blocked[:6])
                + "\n\nNeed numpy 2.x:\n" + "\n".join(f"  - {n} requires numpy{s}" for n, s in need_np2[:6])
                + f"\n\nCurrently installed: numpy {np.version}."
            ),
            impact=(
                "No single numpy version satisfies both groups. Pinning numpy<2 would fix the first "
                "group and break the second; upgrading breaks the first. This is a genuine conflict "
                "between two nodes, not something a version pin can resolve."
            ),
            evidence={"numpy": np.version, "requires_numpy1": dict(blocked),
                      "requires_numpy2": dict(need_np2)},
            remedy=remedy.manual(
                title="Pick which set of nodes you need",
                explain=(
                    "Check whether the numpy<2 pins above are stale — many were added defensively "
                    "in 2024 and the projects have since shipped numpy-2 builds; updating those "
                    "nodes is the clean fix. If the conflict is real, disable the node you need "
                    "least. Do NOT blindly pin numpy<2 — it will break the numpy-2 packages listed."
                ),
            ),
        )
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
                "Pins numpy back to the 1.x series, which every package listed above was built "
                "for. Nothing else you have installed requires numpy 2, so this is safe here.\n\n"
                "If you later add a node that demands numpy>=2, check whether the packages above "
                "have shipped numpy-2 builds before you upgrade."
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
            remedy=_shadowed_remedy(ctx, name, copies, winner),
            # Locations are the interesting part; keep the raw list for the report.
        )


def _shadowed_remedy(ctx: Context, name: str, copies, winner) -> remedy.Remedy:
    """Uninstall every copy, then PUT THE WINNER BACK in the same click.

    The old advice ran only the uninstalls and told the user to reinstall
    afterwards — anyone who stopped there was left without the package at all.
    A fix must leave the machine working, not half-repaired.
    """
    cmds = [ctx.env.pip_argv("uninstall", "-y", name)] * min(len(copies), 3)

    # torch/vision/audio must NEVER be reinstalled from bare PyPI (CPU wheel).
    # Pin the winning version and point at the correct PyTorch index instead.
    if name in TORCH_FAMILY:
        tag = remedy.preferred_cuda_tag(ctx.env, ctx.gpu)
        cmds.append(ctx.env.pip_argv(
            "install", f"{name}=={winner.base_version}",
            "--index-url", remedy.TORCH_INDEX.format(tag=tag),
        ))
    else:
        cmds.append(ctx.env.pip_argv("install", f"{name}=={winner.base_version}"))

    return remedy.Remedy(
        title=f"Collapse {name} down to one copy ({winner.base_version})",
        commands=cmds,
        explain=(
            f"pip uninstall only removes one copy per run, so this runs it once per copy "
            f"({len(copies)} found), then reinstalls {name} {winner.base_version} — the version "
            f"that was winning the import — exactly once. Run the scan again afterwards to "
            f"confirm only one is left."
        ),
        danger=(
            f"Between the uninstalls and the reinstall, {name} will not exist. Let it finish."
        ),
    )


@rule
def contested_module_names(ctx: Context) -> Iterator[Finding]:
    """Two distributions that each OWN the same import name.

    `ctx.inv.module_owners` only counts a dist as an owner when it ships
    `<module>/__init__.py`, and that distinction is doing all the work here.
    google-auth, google-cloud-storage, opentelemetry-sdk, the 14 nvidia-* wheels,
    jaraco-*, ruamel-* and pyannote-* all write into a shared folder - but none of
    them owns it. They are PEP 420 namespace packages and the sharing is the
    entire design. Flagging those means screaming about `import google` on every
    healthy machine alive.

    opencv-python and opencv-python-headless, by contrast, BOTH ship
    cv2/__init__.py. That is a real fight over the same files.
    """
    known = set(OPENCV_VARIANTS) | set(ONNX_VARIANTS)

    for module, owners in sorted(ctx.inv.module_owners.items()):
        if len(owners) < 2 or module in _NOISY_MODULES or module.startswith("_"):
            continue
        if set(owners) & known:
            continue  # already reported with a better, specific remedy

        # On a machine where everything currently loads, a shared module is a
        # LATENT hazard, not a failure — and the destructive act here IS
        # uninstalling. So the advice must never be "uninstall and pick one"
        # for packages we admit we don't understand. Do-nothing is the correct
        # default; the non-destructive repair (force-reinstall the one you want
        # to win) is reserved for the day something actually breaks.
        versions = {ctx.inv.version(o) for o in owners}

        if len(versions) == 1:
            # Same version on both sides: one is almost certainly a
            # re-published wheel of the other (filterpy 1.4.5 vs filterpywhl
            # 1.4.5 — common when an abandoned project never shipped a wheel).
            # The files on disk are the same code either way. Nothing to fix.
            yield Finding(
                id=f"packages.contested_module.{module}",
                severity=Severity.INFO,
                category=CAT,
                title=f"`import {module}` is provided by {len(owners)} packages at the same version",
                detail=(
                    "Provided by: "
                    + ", ".join(f"{o} {ctx.inv.version(o)}" for o in owners)
                    + ".\nSame version on both sides usually means one is simply a re-published "
                      "wheel of the other, so the files on disk are the same code either way."
                ),
                impact=(
                    "Almost certainly harmless — leave it exactly as it is. The only way this "
                    f"goes wrong is uninstalling one of them, which deletes files from the shared "
                    f"`{module}` folder and breaks the other."
                ),
                evidence={"module": module, "owners": {o: ctx.inv.version(o) for o in owners}},
            )
            continue

        yield Finding(
            id=f"packages.contested_module.{module}",
            severity=Severity.WARNING,
            category=CAT,
            title=f"`import {module}` is claimed by {len(owners)} different packages",
            detail=(
                "Provided by: "
                + ", ".join(f"{o} {ctx.inv.version(o)}" for o in owners)
                + ".\nThey write into the same folder, so the files on disk are a mixture of "
                  "whichever was installed last. (Some packages do this deliberately — "
                  "urllib3-future, for example, is designed to shadow urllib3.)"
            ),
            impact=(
                f"Right now nothing is observably broken — your nodes are loading with the "
                f"current blend. The one action guaranteed to make this worse is uninstalling "
                f"either package: they share the `{module}` folder, so removing one deletes "
                f"files the other still needs."
            ),
            evidence={"module": module, "owners": {o: ctx.inv.version(o) for o in owners}},
            remedy=remedy.manual(
                title="Leave it alone unless something actually breaks",
                explain=(
                    f"Do nothing now — this is a note, not a wound.\n\n"
                    f"If `import {module}` ever starts failing (an ImportError or AttributeError "
                    f"naming {module}), repair it WITHOUT uninstalling anything: decide which "
                    f"package the failing node needs, and force-reinstall that one so its files "
                    f"win the shared folder:\n\n"
                    f"  pip install --force-reinstall --no-deps <one of: {', '.join(owners)}>\n\n"
                    f"(using the exact pip command from the System section below). That rewrites "
                    f"`{module}` as one coherent copy and deletes nothing. Never uninstall one of "
                    f"these on its own."
                ),
            ),
        )
