"""PyTorch: the load-bearing wall. If this is wrong, nothing above it is right."""

from __future__ import annotations

from typing import Iterator

from .. import remedy
from ..gpu import min_driver_for
from ..models import Finding, Remedy, Severity
from ..remedy import expected_torchaudio, expected_torchvision, is_prerelease_torch
from . import Context, rule

CAT = "PyTorch"


@rule
def torch_present(ctx: Context) -> Iterator[Finding]:
    if ctx.gpu.torch_ok:
        return
    yield Finding(
        id="torch.missing_or_broken",
        severity=Severity.CRITICAL,
        category=CAT,
        title="PyTorch is missing or cannot be imported",
        detail=ctx.gpu.torch_error or "torch could not be imported.",
        impact="ComfyUI cannot run at all. Nothing else in this report matters until this is fixed.",
        evidence={"error": ctx.gpu.torch_error},
        remedy=remedy.reinstall_torch_stack(
            ctx.env, ctx.gpu,
            reason="torch is not importable, so we install a fresh matched stack.",
        ),
    )


@rule
def cpu_torch_on_gpu_machine(ctx: Context) -> Iterator[Finding]:
    """The silent killer: a working ComfyUI that is 30x too slow.

    Nothing errors. Nothing warns. Renders just take four minutes instead of
    eight seconds, and the user assumes their GPU is bad. Usually caused by a
    custom node's requirements.txt pulling `torch` from PyPI (CPU-only) over the
    top of a perfectly good CUDA build.
    """
    if not ctx.gpu.torch_ok or not ctx.gpu.has_nvidia_hardware:
        return
    if ctx.gpu.cuda_available:
        return

    tag = ctx.gpu.torch_local_tag or ""
    # torch.version.cuda is the source of truth for "is this a CUDA build?".
    # A conda-installed CUDA torch reports NO local tag (tag == "") yet has a
    # real torch_cuda_build like "12.1" — calling that "CPU-only" was wrong and
    # sent conda users to reinstall over pip. Only an explicit "cpu" tag, or the
    # genuine absence of a CUDA build, means CPU-only.
    cpu_build = tag == "cpu" or not ctx.gpu.torch_cuda_build
    gpu_names = ", ".join(d["name"] for d in ctx.gpu.devices)

    if cpu_build:
        detail = (
            f"nvidia-smi sees {gpu_names} (driver {ctx.gpu.driver_version}), but the installed "
            f"PyTorch is a CPU-only build (torch {ctx.gpu.torch_version}). It has no CUDA support "
            f"compiled in at all."
        )
        impact = (
            "Every render is running on your CPU. Expect roughly 20-50x slower generation, "
            "with no error message anywhere - this is almost certainly why things feel broken. "
            "The usual cause is a custom node's requirements.txt installing plain `torch` from "
            "PyPI, which is the CPU wheel, on top of your CUDA build."
        )
    else:
        detail = (
            f"PyTorch is a CUDA build ({ctx.gpu.torch_version}, CUDA {ctx.gpu.torch_cuda_build}) "
            f"and nvidia-smi sees {gpu_names}, but torch.cuda.is_available() is False. "
            f"The build is right; something is stopping it from initialising the driver."
        )
        impact = (
            "ComfyUI will fall back to CPU and be unusably slow. Common causes: the driver is "
            "older than this CUDA build needs, or CUDA_VISIBLE_DEVICES is set to an empty value."
        )

    yield Finding(
        id="torch.cpu_build_on_gpu_machine" if cpu_build else "torch.cuda_unavailable",
        severity=Severity.CRITICAL,
        category=CAT,
        title="Your GPU is not being used" + (" (CPU-only PyTorch installed)" if cpu_build else ""),
        detail=detail,
        impact=impact,
        evidence={
            "nvidia_smi_devices": ctx.gpu.devices,
            "driver": ctx.gpu.driver_version,
            "torch_version": ctx.gpu.torch_version,
            "torch_cuda_build": ctx.gpu.torch_cuda_build,
            "cuda_available": ctx.gpu.cuda_available,
        },
        remedy=remedy.reinstall_torch_stack(
            ctx.env, ctx.gpu,
            torch_version=None if cpu_build else ctx.gpu.torch_version,
            reason="Installs the CUDA build of PyTorch matched to your driver.",
        ),
    )


@rule
def triplet_mismatch(ctx: Context) -> Iterator[Finding]:
    """torch / torchvision / torchaudio must be one matched release.

    They ship compiled extensions linked against each other's ABI. A mismatched
    torchvision does not warn - it raises something inscrutable about
    `operator torchvision::nms does not exist` the first time a node touches it.
    """
    torch_d = ctx.inv.get("torch")
    if not torch_d:
        return

    tv = ctx.inv.get("torchvision")
    ta = ctx.inv.get("torchaudio")

    want_tv = expected_torchvision(torch_d.version)
    want_ta = expected_torchaudio(torch_d.version)

    # Nightly / dev / newer-than-we-know torch: we CANNOT compute a trustworthy
    # matched version (extrapolating once demanded a torchaudio that doesn't
    # exist and cost a user their working stack). Don't assert a version here —
    # the build-tag check below still catches genuinely mixed CUDA builds, which
    # is version-independent and reliable. Offer a calm, non-destructive note.
    if is_prerelease_torch(torch_d.version) and (tv or ta):
        yield Finding(
            id="torch.triplet_unverified_prerelease",
            severity=Severity.INFO,
            category=CAT,
            title="Can't verify the torch/torchvision/torchaudio pairing (nightly or newer build)",
            detail=(
                f"torch {torch_d.version} looks like a nightly/pre-release or is newer than this "
                f"checker knows about, so there is no published torchvision/torchaudio version to "
                f"check it against. This is NOT necessarily a problem."
            ),
            impact=(
                "If all three came from the same PyTorch index (the same nightly build), you are "
                "fine. Only mismatched builds cause ABI errors like `operator torchvision::nms "
                "does not exist`. Do NOT uninstall anything on the strength of a version number "
                "alone."
            ),
            evidence={
                "torch": torch_d.version,
                "torchvision": tv.version if tv else None,
                "torchaudio": ta.version if ta else None,
            },
        )
        # Skip the version-based mismatch below; fall through to the build-tag check.
        want_tv = want_ta = None

    problems: list[str] = []
    if tv and want_tv and not tv.base_version.startswith(want_tv + "."):
        problems.append(
            f"torchvision {tv.version} is installed, but torch {torch_d.version} needs "
            f"torchvision {want_tv}.x"
        )
    if ta and want_ta and not ta.base_version.startswith(want_ta + "."):
        problems.append(
            f"torchaudio {ta.version} is installed, but torch {torch_d.version} needs "
            f"torchaudio {want_ta}.x"
        )

    if problems:
        yield Finding(
            id="torch.triplet_version_mismatch",
            severity=Severity.CRITICAL,
            category=CAT,
            title="PyTorch and torchvision/torchaudio are from different releases",
            detail="\n".join(problems),
            impact=(
                "These packages contain compiled code linked against each other. Mismatched, they "
                "throw errors like `operator torchvision::nms does not exist` - typically the "
                "moment a node tries to load an image model, long after startup, so it looks like "
                "the node is broken rather than the environment."
            ),
            evidence={
                "torch": torch_d.version,
                "torchvision": tv.version if tv else None,
                "torchaudio": ta.version if ta else None,
                "expected_torchvision": want_tv,
                "expected_torchaudio": want_ta,
            },
            remedy=remedy.reinstall_torch_stack(
                ctx.env, ctx.gpu, torch_version=torch_d.version,
                reason="Reinstalls all three at versions that belong to the same release.",
            ),
        )
        return

    # Local build tags must agree too: a cu124 torch beside a cpu torchvision is
    # a mismatch even when the version numbers look perfectly compatible.
    tags = {
        "torch": torch_d.local_tag,
        "torchvision": tv.local_tag if tv else None,
        "torchaudio": ta.local_tag if ta else None,
    }
    present = {k: v for k, v in tags.items() if v is not None}
    if len(set(present.values())) > 1:
        yield Finding(
            id="torch.build_tag_mismatch",
            severity=Severity.ERROR,
            category=CAT,
            title="PyTorch packages were built for different CUDA versions",
            detail=(
                "The build tags do not agree: "
                + ", ".join(f"{k} is {v}" for k, v in present.items())
                + ". They must all come from the same PyTorch index."
            ),
            impact=(
                "Whichever package has the odd tag will either refuse to import or fall back to "
                "CPU silently, depending on which one it is."
            ),
            evidence=tags,
            remedy=remedy.reinstall_torch_stack(
                ctx.env, ctx.gpu, torch_version=torch_d.version,
                reason="Reinstalls all three from one index so the build tags match.",
            ),
        )


@rule
def driver_too_old(ctx: Context) -> Iterator[Finding]:
    if not ctx.gpu.has_nvidia_hardware or not ctx.gpu.torch_local_tag:
        return
    # Reality beats arithmetic: if CUDA actually initialised, the driver is by
    # definition new enough for this build — never tell someone with a working
    # GPU to downgrade over a version-number comparison.
    if ctx.gpu.cuda_available:
        return
    tag = ctx.gpu.torch_local_tag
    if not tag.startswith("cu"):
        return
    need = min_driver_for(tag, ctx.env.is_windows)
    if need is None or not ctx.gpu.driver_version:
        return
    try:
        have = float(".".join(ctx.gpu.driver_version.split(".")[:2]))
    except ValueError:
        return
    if have >= need:
        return

    yield Finding(
        id="torch.driver_too_old",
        severity=Severity.CRITICAL,
        category=CAT,
        title="Your NVIDIA driver is too old for this PyTorch build",
        detail=(
            f"PyTorch is built for {tag}, which needs driver {need:.2f} or newer. "
            f"You have {ctx.gpu.driver_version}."
        ),
        impact=(
            "CUDA will fail to initialise. Either the GPU is ignored entirely (silent, slow) or "
            "you get a hard 'CUDA driver version is insufficient' crash on the first render."
        ),
        evidence={"driver": ctx.gpu.driver_version, "required": need, "torch_tag": tag},
        remedy=Remedy(
            title="Update your NVIDIA driver (recommended), or downgrade PyTorch",
            commands=[ctx.env.pip_argv(
                "install", "--force-reinstall", "torch", "torchvision", "torchaudio",
                "--index-url", f"https://download.pytorch.org/whl/{_older_tag(tag)}",
            )],
            explain=(
                f"The better fix is to update your NVIDIA driver to {need:.0f}+ from nvidia.com - "
                f"it takes five minutes and keeps you on the faster PyTorch build.\n\n"
                f"If you cannot update the driver (locked-down machine, laptop OEM driver), the "
                f"command below instead downgrades PyTorch to {_older_tag(tag)}, which your current "
                f"driver can run."
            ),
            danger="Downgrading CUDA can slow you down and may break packages built for the newer tag. Update the driver first if you possibly can.",
            doc_url="https://www.nvidia.com/download/index.aspx",
        ),
    )


def _older_tag(tag: str) -> str:
    # Oldest-first ladder derived from the single source of truth in gpu.py,
    # so adding a new CUDA tag there automatically updates this rule too.
    from ..gpu import CUDA_TAG_LADDER
    ladder = list(reversed(CUDA_TAG_LADDER))
    try:
        i = ladder.index(tag)
    except ValueError:
        return "cu121"
    return ladder[max(0, i - 1)]


@rule
def torch_healthy(ctx: Context) -> Iterator[Finding]:
    """Say what's *right*, too. A report that only lists problems gives the user
    no way to tell 'checked and fine' from 'never checked'."""
    if not ctx.gpu.torch_ok or not ctx.gpu.cuda_available:
        return
    dev = ctx.gpu.torch_devices[0] if ctx.gpu.torch_devices else {}
    yield Finding(
        id="torch.ok",
        severity=Severity.OK,
        category=CAT,
        title=f"PyTorch {ctx.gpu.torch_version} is using your GPU",
        detail=(
            f"{dev.get('name', 'GPU')} - {dev.get('vram_total_mb', '?')} MB VRAM, "
            f"compute {dev.get('compute_capability', '?')}, CUDA {ctx.gpu.torch_cuda_build}, "
            f"driver {ctx.gpu.driver_version}."
        ),
        evidence={"devices": ctx.gpu.torch_devices, "backends": ctx.gpu.backends},
    )
