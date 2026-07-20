"""The machine underneath: Python, RAM, VRAM, disk, and where pip will land."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Iterator

from ..models import Finding, Severity
from .. import remedy
from . import Context, rule

CAT = "System"

# ComfyUI itself runs on 3.13, but a large share of the custom-node ecosystem
# still ships wheels only up to 3.12. 3.10-3.12 is the sweet spot.
# ComfyUI officially runs on 3.13 (mid-2026), so it belongs INSIDE the range —
# warning on the interpreter the project itself recommends was a false alarm.
# 3.14+ is still genuinely ahead of most compiled wheels, so it still warns.
PY_SWEET_SPOT = ((3, 10), (3, 13))


@rule
def python_version(ctx: Context) -> Iterator[Finding]:
    v = sys.version_info[:2]
    lo, hi = PY_SWEET_SPOT
    if lo <= v <= hi:
        yield Finding(
            id="system.python_ok",
            severity=Severity.OK,
            category=CAT,
            title=f"Python {ctx.env.python_version} ({ctx.env.kind})",
            detail=ctx.env.kind_detail,
        )
        return

    newer = v > hi
    yield Finding(
        id="system.python_version",
        severity=Severity.WARNING,
        category=CAT,
        title=f"Python {ctx.env.python_version} is outside the range most custom nodes support",
        detail=(
            f"Most of the ComfyUI ecosystem targets Python {lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}. "
            f"You're on {ctx.env.python_version}."
        ),
        impact=(
            "Packages with compiled extensions (insightface, dlib, onnxruntime, some Triton "
            "builds) often have no wheel for this version. pip then tries to build them from "
            "source, which fails on most machines with a wall of compiler errors that look "
            "unrelated to the real cause."
            if newer else
            "Newer nodes increasingly require 3.10+ syntax and will fail to import with "
            "SyntaxError on this interpreter."
        ),
        evidence={"python": ctx.env.python_version, "supported": f"{lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}"},
        remedy=remedy.manual(
            title="Consider a Python in the supported range",
            explain=(
                "Nothing is necessarily broken - but when a package refuses to install and the "
                "error mentions compilers, wheels, or 'building from source', this is why. "
                "The lowest-effort route is a fresh ComfyUI portable, which ships a Python that "
                "the ecosystem is tested against."
            ),
        ),
    )


@rule
def interpreter_kind(ctx: Context) -> Iterator[Finding]:
    """Tell people which pip is the right pip. Half of all failed installs are
    'I installed it into a different Python'."""
    yield Finding(
        id="system.interpreter",
        severity=Severity.INFO,
        category=CAT,
        title="Install packages with this exact command",
        detail=(
            f"{' '.join(ctx.env.pip_argv('install', '<package>'))}\n\n"
            f"{ctx.env.kind_detail}"
        ),
        impact=(
            "Typing plain `pip install` in a terminal usually targets a completely different "
            "Python than the one ComfyUI is running on. The install succeeds, ComfyUI still can't "
            "find the package, and everyone loses an hour. Use the command above and that can't "
            "happen. Every Fix button in this panel already uses it."
        ),
        evidence={"python_exe": ctx.env.python_exe, "kind": ctx.env.kind,
                  "site_dirs": ctx.env.site_dirs},
    )


@rule
def system_python_warning(ctx: Context) -> Iterator[Finding]:
    if ctx.env.kind != "system":
        return
    yield Finding(
        id="system.no_isolation",
        severity=Severity.WARNING,
        category=CAT,
        title="ComfyUI is running on your system-wide Python",
        detail=f"No virtualenv, conda env, or embedded Python - just {ctx.env.python_exe}.",
        impact=(
            "Every custom node you install writes into the same Python your OS and other projects "
            "use. One node pinning `numpy<2` can break unrelated software on this machine, and "
            "there's no way to roll back a bad install short of repairing everything by hand."
        ),
        remedy=remedy.manual(
            title="Move ComfyUI into its own environment",
            explain=(
                "Not urgent, but worth doing before you accumulate more nodes. Create a venv, "
                "reinstall ComfyUI's requirements into it, and launch from there. After that, a "
                "wrecked environment is a folder you delete rather than an afternoon you lose."
            ),
        ),
    )


@rule
def disk_space(ctx: Context) -> Iterator[Finding]:
    """Check the drive ComfyUI is actually on.

    The old code checked psutil.disk_usage('/'), which on Windows reports the C:
    drive - while most people keep ComfyUI and its 200 GB of models on D:. It
    was measuring the wrong disk.
    """
    root = ctx.env.comfy_root or os.getcwd()
    try:
        usage = shutil.disk_usage(str(root))
    except Exception:
        return
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)

    if free_gb >= 20:
        return

    critical = free_gb < 5
    yield Finding(
        id="system.disk_space",
        severity=Severity.CRITICAL if critical else Severity.WARNING,
        category=CAT,
        title=f"Only {free_gb:.1f} GB free on the drive ComfyUI lives on",
        detail=f"{root} - {free_gb:.1f} GB free of {total_gb:.0f} GB.",
        impact=(
            "PyTorch reinstalls need ~5 GB of temporary space, a single model download can be "
            "20 GB, and pip fails messily when it runs out mid-extract, sometimes leaving a "
            "half-written package that then won't import. Clear space before running any fix on "
            "this page."
        ),
        evidence={"path": str(root), "free_gb": round(free_gb, 1), "total_gb": round(total_gb)},
    )


@rule
def memory(ctx: Context) -> Iterator[Finding]:
    try:
        import psutil
    except ImportError:
        return

    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)

    if total_gb < 16:
        yield Finding(
            id="system.low_ram",
            severity=Severity.WARNING,
            category=CAT,
            title=f"{total_gb:.0f} GB of system RAM",
            detail=f"{mem.available / (1024 ** 3):.1f} GB currently available.",
            impact=(
                "ComfyUI loads models through system RAM on their way to VRAM. Under 16 GB, "
                "SDXL and Flux workflows commonly die with an out-of-memory kill that gets blamed "
                "on the GPU. On Windows, make sure the pagefile is system-managed and on a fast "
                "drive - that alone rescues a lot of these."
            ),
            evidence={"total_gb": round(total_gb), "percent_used": mem.percent},
        )


@rule
def vram(ctx: Context) -> Iterator[Finding]:
    if not ctx.gpu.torch_devices:
        return
    dev = ctx.gpu.torch_devices[0]
    vram_gb = (dev.get("vram_total_mb") or 0) / 1024
    if vram_gb == 0 or vram_gb >= 8:
        return

    yield Finding(
        id="system.low_vram",
        severity=Severity.INFO,
        category=CAT,
        title=f"{vram_gb:.0f} GB of VRAM on {dev.get('name')}",
        detail="Below 8 GB, the bigger model families need help to fit.",
        impact=(
            "SDXL and Flux will out-of-memory at default settings. Launch ComfyUI with --lowvram "
            "(or --novram if that still fails), and prefer GGUF or fp8 model variants. This is a "
            "hardware limit, not a misconfiguration - nothing here is broken."
        ),
        evidence={"vram_gb": round(vram_gb, 1), "device": dev.get("name")},
    )
