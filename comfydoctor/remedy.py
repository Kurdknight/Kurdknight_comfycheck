"""Turning a diagnosis into the exact command for *this* machine.

The old node told you "flash_attn: Not installed" and stopped. That is not help;
that is a fact. Help is knowing that on your portable install, with your driver,
the command is:

    python_embeded\\python.exe -s -m pip install torch==2.6.0 torchvision==0.21.0 \\
        torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

Every remedy here is built from the live Environment + GPUInfo, so the
interpreter, the index URL and the CUDA tag are correct by construction rather
than by the user guessing.
"""

from __future__ import annotations

from .env import Environment
from .gpu import GPUInfo, cuda_tag_for_driver
from .models import Remedy
from .shipped import minor_shipped

TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"

# torchvision and torchaudio track torch on a fixed offset:
#   torch 2.N  <->  torchvision 0.(N+15)  <->  torchaudio 2.N
# The formula gives the CANDIDATE pairing; comfydoctor.shipped (live PyPI with
# a baked fallback) decides whether that candidate ever actually shipped. We
# only assert a pairing both halves of which exist — this is what killed the
# "torch 2.13 needs torchaudio 2.13.x" false positive: torch 2.13 is real, but
# torchaudio froze at 2.11, so the formula's answer must be discarded there.
TV_OFFSET = 15


def is_prerelease_torch(torch_version: str | None) -> bool:
    """A nightly / dev / rc build (e.g. '2.13.0.dev20260710', '2.9.0rc1', or a
    minor no stable release has ever shipped for). The stable-release pairing
    rules do not apply to these — their matched torchvision/torchaudio come
    from the nightly index and can carry different version numbers."""
    if not torch_version:
        return False
    v = torch_version.lower()
    # Local-tag markers (after '+') and pre-release markers (in the base).
    if any(m in v for m in ("+git", "nightly", ".dev", "rc", "a0", "b0")):
        return True
    return not minor_shipped("torch", _major_minor(torch_version))


def expected_torchvision(torch_version: str) -> str | None:
    """The torchvision series matching this torch — or None when we cannot
    stand behind an answer (unshipped torch, or the paired torchvision was
    never released). None means 'can't verify', never 'wrong'."""
    mm = _major_minor(torch_version)
    if not mm or mm[0] != 2 or not minor_shipped("torch", mm):
        return None
    want = (0, mm[1] + TV_OFFSET)
    return f"0.{want[1]}" if minor_shipped("torchvision", want) else None


def expected_torchaudio(torch_version: str) -> str | None:
    """The torchaudio series matching this torch — or None when no such
    torchaudio ever shipped (it froze at 2.11 while torch kept releasing, so
    for newer torch there IS no matched version and nothing to demand)."""
    mm = _major_minor(torch_version)
    if not mm or mm[0] != 2 or not minor_shipped("torch", mm):
        return None
    want = (2, mm[1])
    return f"2.{want[1]}" if minor_shipped("torchaudio", want) else None


def _major_minor(v: str | None) -> tuple[int, int] | None:
    if not v:
        return None
    base = v.split("+", 1)[0]
    parts = base.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def preferred_cuda_tag(env: Environment, gpu: GPUInfo) -> str:
    """The cu-tag we should be installing for, given the driver actually present."""
    if gpu.has_nvidia_hardware:
        return cuda_tag_for_driver(gpu.driver_cuda_version, env.is_windows)
    return "cpu"


def reinstall_torch_stack(
    env: Environment,
    gpu: GPUInfo,
    torch_version: str | None = None,
    reason: str = "",
) -> Remedy:
    """Reinstall torch + torchvision + torchaudio as one matched set.

    Always all three together and always with an explicit --index-url. Installing
    them one at a time is how people end up with a cu124 torch next to a CPU
    torchvision, which fails at import with an error message that mentions
    neither CUDA nor versions.
    """
    tag = preferred_cuda_tag(env, gpu)
    v = (torch_version or gpu.torch_version or "").split("+", 1)[0]
    tv = expected_torchvision(v) if v else None
    ta = expected_torchaudio(v) if v else None

    if v and tv:
        pkgs = [f"torch=={v}", f"torchvision=={tv}.*"]
        # torchaudio stopped releasing at 2.11; for newer torch there is no
        # matched version, so install it unpinned and let the index pick the
        # newest one — never pin to a version we can't confirm exists.
        pkgs.append(f"torchaudio=={ta}.*" if ta else "torchaudio")
        pin_note = f"pinned to your current torch {v}, so nothing else in your environment shifts"
    else:
        pkgs = ["torch", "torchvision", "torchaudio"]
        pin_note = "latest matched set"

    index = TORCH_INDEX.format(tag=tag)
    cmd = env.pip_argv("install", "--force-reinstall", *pkgs, "--index-url", index)

    explain = (
        f"Reinstalls all three PyTorch packages together from the {tag} index ({pin_note}). "
        "They must come from the same index and the same release, or torchvision's compiled "
        "extensions won't match torch's ABI and imports fail."
    )
    if reason:
        explain = f"{reason}\n\n{explain}"

    return Remedy(
        title=f"Reinstall the PyTorch stack for {tag}",
        commands=[cmd],
        explain=explain,
        danger=(
            "This re-downloads ~2.5 GB and replaces your current torch. If you are on a "
            "metered connection or mid-render, do it later."
        ),
        doc_url="https://pytorch.org/get-started/locally/",
    )


def reinstall_matching(env: Environment, package: str, pin: str, why: str) -> Remedy:
    """Reinstall one package at a version compatible with the installed torch."""
    return Remedy(
        title=f"Reinstall {package} to match your PyTorch",
        commands=[env.pip_argv("install", "--force-reinstall", f"{package}=={pin}")],
        explain=why,
        doc_url=None,
    )


def uninstall(env: Environment, packages: list[str], why: str, danger: str | None = None) -> Remedy:
    return Remedy(
        title=f"Uninstall {', '.join(packages)}",
        commands=[env.pip_argv("uninstall", "-y", *packages)],
        explain=why,
        danger=danger,
    )


def install(env: Environment, packages: list[str], why: str, index_url: str | None = None) -> Remedy:
    args = ["install", *packages]
    if index_url:
        args += ["--index-url", index_url]
    return Remedy(
        title=f"Install {', '.join(packages)}",
        commands=[env.pip_argv(*args)],
        explain=why,
    )


def pin(env: Environment, spec: str, why: str) -> Remedy:
    return Remedy(
        title=f"Install {spec}",
        commands=[env.pip_argv("install", spec)],
        explain=why,
    )


def resolve_opencv(env: Environment, keep: str, drop: list[str]) -> Remedy:
    """The cv2 three-way fight, resolved by removing all of them and reinstalling one.

    Uninstalling only the loser leaves a half-deleted cv2 package directory
    behind - the files of the two distributions overlap on disk, so pip's
    uninstall of one removes files the other still needs. The only reliable cure
    is to remove every opencv variant and then install exactly one.
    """
    return Remedy(
        title=f"Collapse the OpenCV installs down to {keep}",
        commands=[
            env.pip_argv("uninstall", "-y", *([keep] + drop)),
            env.pip_argv("install", "--no-cache-dir", keep),
        ],
        explain=(
            f"You have {len(drop) + 1} OpenCV distributions installed and they all write into the "
            f"same `cv2` folder, so which one you actually get is decided by install order. "
            f"This removes all of them and reinstalls only {keep}, giving you one predictable cv2."
        ),
        danger="Between the uninstall and the install, cv2 will not exist. Don't interrupt it.",
    )


def manual(title: str, explain: str, doc_url: str | None = None) -> Remedy:
    """A remedy we can describe but must not run (driver updates, file edits)."""
    return Remedy(
        title=title, commands=[], explain=explain, doc_url=doc_url,
        runnable=False, restart_required=False,
    )
