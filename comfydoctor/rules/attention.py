"""Attention backends: xformers, flash-attn, triton, sageattention.

These are all compiled against a specific torch build. When they don't match,
importing them doesn't raise a friendly error - it throws an undefined-symbol
abort deep in a .so/.pyd, which ComfyUI reports as "IMPORT FAILED" for whatever
unlucky node happened to import it first. The user then goes and reinstalls that
innocent node, repeatedly, for an hour.

We catch it from metadata alone: these packages declare the torch they were
built against in their Requires-Dist. We compare that to the torch on disk. No
imports, no crashes.
"""

from __future__ import annotations

import re
from typing import Iterator

from .. import remedy
from ..inventory import requirement_pins, satisfies
from ..models import Finding, Severity
from . import Context, rule

CAT = "Attention backends"

# Packages that carry compiled kernels linked against torch's ABI.
ABI_BOUND = ["xformers", "flash-attn", "sageattention", "natten", "causal-conv1d", "mamba-ssm"]


@rule
def abi_pin_mismatch(ctx: Context) -> Iterator[Finding]:
    torch_d = ctx.inv.get("torch")
    if not torch_d:
        return

    for pkg in ABI_BOUND:
        dist = ctx.inv.get(pkg)
        if not dist:
            continue

        pins = requirement_pins(dist, "torch")
        for spec in pins:
            ok = satisfies(torch_d.version, spec)
            if ok is not False:
                continue
            yield Finding(
                id=f"attention.{pkg}.torch_pin_mismatch",
                severity=Severity.ERROR,
                category=CAT,
                title=f"{pkg} was built for a different PyTorch",
                detail=(
                    f"{pkg} {dist.version} declares it needs torch{spec}, but torch "
                    f"{torch_d.version} is installed."
                ),
                impact=(
                    f"Importing {pkg} will abort with an undefined-symbol error. ComfyUI will "
                    f"report whichever node imports it first as failed to load, which sends people "
                    f"off reinstalling the wrong thing. If you don't need {pkg}, uninstalling it "
                    f"is a perfectly good fix."
                ),
                evidence={"package": pkg, "installed": dist.version, "requires": spec,
                          "torch": torch_d.version},
                remedy=remedy.uninstall(
                    ctx.env, [pkg],
                    why=(
                        f"Removing {pkg} is the safe, fast fix: PyTorch's built-in SDPA attention "
                        f"is close to xFormers speed on modern GPUs, so most users lose almost "
                        f"nothing.\n\n"
                        f"If you specifically need {pkg}, install a build made for torch "
                        f"{torch_d.base_version} instead - check the project's release notes for "
                        f"which version pairs with your torch."
                    ),
                ),
            )
            break  # one finding per package is enough

    # A CUDA build tag that disagrees with torch's is the same disease, and it
    # shows up even when the package forgot to pin torch at all.
    torch_cuda = _cuda_of(torch_d.local_tag)
    for pkg in ABI_BOUND:
        dist = ctx.inv.get(pkg)
        if not dist or not dist.local_tag or not torch_cuda:
            continue

        pkg_cuda = _cuda_of(dist.local_tag)
        pkg_torch = _torch_of(dist.local_tag)

        # Compare the parts, never the whole tag. These wheels encode BOTH the
        # CUDA version and the torch they were built against, in one string:
        #   sageattention 2.2.0+cu130torch2.9.1.post5
        # Naive string equality against torch's own "cu130" calls that a
        # mismatch when it is in fact a perfect match.
        cuda_bad = pkg_cuda is not None and pkg_cuda != torch_cuda
        torch_bad = (
            pkg_torch is not None
            and not torch_d.base_version.startswith(pkg_torch)
        )
        if not cuda_bad and not torch_bad:
            continue

        if cuda_bad:
            what = f"a different CUDA version than PyTorch (cu{pkg_cuda} vs cu{torch_cuda})"
        else:
            what = f"a different PyTorch (built for torch {pkg_torch}, you have {torch_d.base_version})"

        yield Finding(
            id=f"attention.{pkg}.build_tag_mismatch",
            severity=Severity.ERROR,
            category=CAT,
            title=f"{pkg} was built for {what.split('(')[0].strip()}",
            detail=f"{pkg} {dist.version} was built for {what}.",
            impact=(
                "Loading its compiled kernels will fail at import time with an undefined-symbol "
                "error, and ComfyUI will blame whichever node imported it first."
            ),
            evidence={"package": pkg, "package_tag": dist.local_tag,
                      "torch_tag": torch_d.local_tag, "torch_version": torch_d.version},
            remedy=remedy.uninstall(
                ctx.env, [pkg],
                why=(
                    f"Remove the mismatched build. Reinstall only if you can find a wheel built for "
                    f"torch {torch_d.base_version} / cu{torch_cuda}."
                ),
            ),
        )


def _cuda_of(tag: str | None) -> str | None:
    """'cu130torch2.9.1.post5' -> '130'.  'cu124' -> '124'.  'cpu' -> None."""
    if not tag:
        return None
    m = re.search(r"cu(\d+)", tag)
    return m.group(1) if m else None


def _torch_of(tag: str | None) -> str | None:
    """'cu130torch2.9.1.post5' -> '2.9.1'. Absent in most tags, and that's fine."""
    if not tag:
        return None
    m = re.search(r"torch(\d+\.\d+(?:\.\d+)?)", tag)
    return m.group(1) if m else None


@rule
def triton_on_windows(ctx: Context) -> Iterator[Finding]:
    """`triton` proper has no Windows wheels. Windows users need triton-windows.

    People install `triton` on Windows because a node's requirements.txt asks for
    it, pip finds *something*, and then every torch.compile path explodes.
    """
    if not ctx.env.is_windows:
        return
    t = ctx.inv.get("triton")
    tw = ctx.inv.get("triton-windows")
    if not t or tw:
        return
    yield Finding(
        id="attention.triton_linux_wheel_on_windows",
        severity=Severity.WARNING,
        category=CAT,
        title="The Linux `triton` package is installed on Windows",
        detail=(
            f"triton {t.version} is installed, but upstream Triton does not publish working "
            f"Windows wheels. The Windows port is a separate package, `triton-windows`."
        ),
        impact=(
            "Anything using torch.compile or a Triton kernel (SageAttention, some samplers, "
            "several speed-up nodes) will fail. Nodes that merely check `import triton` may also "
            "take a broken fast path."
        ),
        evidence={"triton": t.version},
        remedy=remedy.Remedy(
            title="Replace triton with triton-windows",
            commands=[
                ctx.env.pip_argv("uninstall", "-y", "triton"),
                ctx.env.pip_argv("install", "triton-windows"),
            ],
            explain=(
                "Removes the Linux-only package and installs the maintained Windows port, which "
                "provides the same `triton` import name."
            ),
        ),
    )


@rule
def attention_summary(ctx: Context) -> Iterator[Finding]:
    """What attention can this machine actually do? Informational, but it's the
    question people are really asking when they run a checker at all."""
    if not ctx.gpu.torch_ok:
        return

    have = []
    if ctx.gpu.backends.get("flash_sdp"):
        have.append("PyTorch SDPA (flash)")
    if ctx.gpu.backends.get("mem_efficient_sdp"):
        have.append("PyTorch SDPA (mem-efficient)")
    for pkg, label in (("xformers", "xFormers"), ("flash-attn", "FlashAttention"),
                       ("sageattention", "SageAttention")):
        d = ctx.inv.get(pkg)
        if d:
            have.append(f"{label} {d.version}")

    yield Finding(
        id="attention.available",
        severity=Severity.INFO,
        category=CAT,
        title="Attention backends available: " + (", ".join(have) if have else "none"),
        detail=(
            "PyTorch's built-in SDPA is enough for almost everyone on a modern GPU. xFormers and "
            "FlashAttention are optional speed-ups, and both are common sources of the version "
            "conflicts above - if you're not sure you need them, you probably don't."
        ),
        evidence={"backends": ctx.gpu.backends, "packages": have},
    )
