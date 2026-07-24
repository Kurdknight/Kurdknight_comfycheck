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
import sys
from typing import Iterator

from .. import remedy
from ..inventory import requirement_pins, satisfies
from ..models import Finding, Severity
from . import Context, rule

CAT = "Attention backends"

# Packages that carry compiled kernels linked against torch's ABI.
ABI_BOUND = ["xformers", "flash-attn", "sageattention", "natten", "causal-conv1d", "mamba-ssm"]


def _is_lower_bound_only(spec: str) -> bool:
    """True when a torch pin is ONLY a minimum (>= / >), with no upper bound or
    exact/exclusion operator. An unmet lower bound means torch is too OLD — that
    is an UPGRADE, not an ABI break, and uninstalling the package is the wrong
    fix. A real ABI ceiling uses ==, ~=, <, <=, or != and is handled normally.
    """
    ops = re.findall(r"(===|==|~=|!=|<=|>=|<|>)", spec or "")
    return bool(ops) and all(o in (">", ">=") for o in ops)


def _mm(v: str | None) -> str | None:
    """'2.9.1' / '2.9.1.post5' -> '2.9'. Patch/post differences do NOT break the
    torch ABI, so the build-tag check must compare only major.minor."""
    if not v:
        return None
    parts = v.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else v


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
            # A purely lower-bound pin that isn't met means torch is too OLD, not
            # that the package is ABI-broken. Don't tell people to uninstall a
            # working accelerator over an upgrade suggestion.
            if _is_lower_bound_only(spec):
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
                        f"Removing {pkg} is the safe fix: nothing breaks, ComfyUI falls back to "
                        f"PyTorch's built-in attention automatically. You do lose {pkg}'s "
                        f"speed-up until you reinstall a matching build - for SageAttention on "
                        f"video workloads that can be a noticeable slowdown; for xFormers on "
                        f"modern cards the difference is usually small.\n\n"
                        f"If you want to keep {pkg}, install a build made for torch "
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
        # Compare only major.minor: a wheel built for torch2.9.1 runs fine on
        # torch 2.9.0 (patch releases keep the ABI). Comparing the full version
        # false-flagged every patch-level difference and told people to uninstall.
        torch_bad = (
            pkg_torch is not None
            and _mm(pkg_torch) != _mm(torch_d.base_version)
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


def _ships_windows_binaries(dist_name: str) -> bool:
    """Does the installed dist actually contain Windows binaries (.pyd)?
    Metadata-only — never imports. This is the reality check that keeps rules
    from asserting ecosystem facts ("no Windows wheels exist") that can rot."""
    try:
        from importlib import metadata as md

        return any(str(f).endswith(".pyd") for f in (md.distribution(dist_name).files or []))
    except Exception:
        return False


@rule
def triton_on_windows(ctx: Context) -> Iterator[Finding]:
    """`triton` proper has no Windows wheels (as of 2026). Windows users need
    triton-windows.

    People install `triton` on Windows because a node's requirements.txt asks for
    it, pip finds *something*, and then every torch.compile path explodes.
    """
    if not ctx.env.is_windows:
        return
    t = ctx.inv.get("triton")
    tw = ctx.inv.get("triton-windows")
    if not t or tw:
        return
    # Believe the files on disk over our own claim: if this triton actually
    # ships Windows binaries, upstream has started publishing Windows wheels
    # and the premise of this warning is gone.
    if _ships_windows_binaries("triton"):
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


# Launch flags that decide which attention path ComfyUI uses. Reading them from
# sys.argv (we run inside ComfyUI's process) is a FACT about this session —
# unlike guessing what ComfyUI "probably" picked.
_ATTENTION_FLAGS = (
    "--use-sage-attention", "--use-flash-attention", "--use-pytorch-cross-attention",
    "--use-split-cross-attention", "--use-quad-cross-attention", "--disable-xformers",
)


@rule
def attention_summary(ctx: Context) -> Iterator[Finding]:
    """Which attention speed-ups exist on this machine, and — the part everyone
    misses — whether ComfyUI is actually set to use them. Installing one changes
    nothing by itself. We report what we can verify (installed packages, launch
    flags) and label reported speed-ups as exactly that: reported, and
    workload-dependent."""
    if not ctx.gpu.torch_ok:
        return

    installed = []
    for pkg, label in (("sageattention", "SageAttention"), ("flash-attn", "FlashAttention"),
                       ("xformers", "xFormers")):
        d = ctx.inv.get(pkg)
        if d:
            installed.append(f"{label} {d.version}")

    lines = []
    if installed:
        lines.append("Installed speed-ups: " + ", ".join(installed) + ".")
    lines.append(
        "Always available: PyTorch's built-in attention. It is the fallback whenever no "
        "speed-up is switched on."
    )
    if ctx.comfy_runtime:
        flags = [a for a in sys.argv if a in _ATTENTION_FLAGS]
        if flags:
            lines.append("Launch flags in effect: " + " ".join(flags) + ".")
        elif installed:
            lines.append(
                "Launch flags: none of the attention flags are set - ComfyUI is picking the "
                "attention path by its own defaults, and installed speed-ups that need a flag "
                "(like SageAttention) are NOT being used globally."
            )

    yield Finding(
        id="attention.available",
        severity=Severity.INFO,
        category=CAT,
        title=(
            "Attention speed-ups installed: " + ", ".join(i.split(" ")[0] for i in installed)
            if installed else
            "No attention speed-up packages installed (using PyTorch's built-in)"
        ),
        detail="\n".join(lines),
        impact=(
            "Attention is the part of the maths that takes most of your render time; these "
            "packages replace it with faster versions. What users commonly report (it varies by "
            "workload and card): SageAttention about 1.3-2x faster sampling, largest on video "
            "models like Wan or Hunyuan; FlashAttention and xFormers smaller gains over the "
            "built-in on modern cards. Installing one does nothing by itself - it must be "
            "switched on at launch (for example --use-sage-attention) or by a workflow node."
        ),
        evidence={"backends": ctx.gpu.backends, "packages": installed},
    )
