"""Speed & quality: nothing is broken, but you're leaving performance on the table.

This is the half of the job the old checker never did. It would faithfully print
`Triton Available: False` and stop — a fact, with no consequence attached and no
action implied. The user is left to work out for themselves that this is why
torch.compile silently no-ops, and to go and find the right wheel.

Every rule here is the same shape: *your hardware can do X, you haven't got X,
here is the command, here is roughly what it buys you.* They are TIPs, worth zero
health penalty, because declining an optional speed-up is not a defect.

Two hard rules for this file, because "recommendations" is where a diagnostic
tool most easily turns into a nag or, worse, a liar:

  1. **Never recommend into a broken environment.** If torch is a mess, telling
     someone to add SageAttention on top is actively harmful. Every rule bails
     unless CUDA is genuinely working.
  2. **Never promise a number we can't stand behind.** Speed-ups are hardware- and
     workflow-dependent. Give honest ranges, name the condition, and say when
     something is a maybe.
"""

from __future__ import annotations

from typing import Iterator

from .. import remedy
from ..facts import _cudnn
from ..models import Finding, Remedy, Severity
from . import Context, rule

CAT = "Speed & quality"


def _gpu_healthy(ctx: Context) -> bool:
    """The gate on every recommendation in this file. Optimising a broken
    environment is how you turn one problem into two."""
    return bool(ctx.gpu.torch_ok and ctx.gpu.cuda_available and ctx.gpu.torch_devices)


def _compute_capability(ctx: Context) -> float:
    try:
        return float(ctx.gpu.torch_devices[0].get("compute_capability", "0"))
    except (ValueError, IndexError, TypeError):
        return 0.0


@rule
def sage_attention(ctx: Context) -> Iterator[Finding]:
    """SageAttention is the single biggest free win for most ComfyUI users."""
    if not _gpu_healthy(ctx):
        return
    if ctx.inv.has("sageattention"):
        return

    cc = _compute_capability(ctx)
    if cc < 8.0:  # needs Ampere or newer
        return

    name = ctx.gpu.torch_devices[0].get("name", "your GPU")
    yield Finding(
        id="tip.sageattention",
        severity=Severity.TIP,
        category=CAT,
        title="SageAttention could speed up your sampling by roughly 20-30%",
        detail=(
            f"{name} (compute {cc}) supports SageAttention's INT8 kernels, and it isn't installed."
        ),
        impact=(
            "SageAttention replaces the attention maths with quantised kernels. On Ampere and "
            "newer cards it typically cuts 20-30% off sampling time, and more on video models "
            "where attention dominates. Quality loss is generally imperceptible.\n\n"
            "It needs Triton to compile its kernels, so install that too if you don't have it. "
            "You then launch ComfyUI with --use-sage-attention."
        ),
        evidence={"compute_capability": cc, "gpu": name},
        remedy=Remedy(
            title="Install SageAttention",
            commands=[ctx.env.pip_argv("install", "sageattention")],
            explain=(
                "Installs SageAttention, then launch ComfyUI with --use-sage-attention to turn it "
                "on.\n\n"
                "It is a compiled package built against a specific torch — if a future torch "
                "upgrade breaks it, ComfyDoctor will tell you, and uninstalling it is always a "
                "safe fallback (you lose speed, nothing else)."
            ),
            danger=None,
            doc_url="https://github.com/thu-ml/SageAttention",
        ),
    )


@rule
def triton_for_compile(ctx: Context) -> Iterator[Finding]:
    if not _gpu_healthy(ctx):
        return
    if ctx.inv.has("triton") or ctx.inv.has("triton-windows"):
        return

    pkg = "triton-windows" if ctx.env.is_windows else "triton"
    yield Finding(
        id="tip.triton",
        severity=Severity.TIP,
        category=CAT,
        title="Triton isn't installed, so torch.compile and several speed-up nodes can't work",
        detail=(
            "Triton compiles the custom GPU kernels behind torch.compile, SageAttention, and a "
            "number of optimisation nodes."
        ),
        impact=(
            "Without it, torch.compile silently falls back to eager mode — no error, just no "
            "speed-up — and anything Triton-based either fails to load or quietly takes a slow "
            "path. This is the most common reason a 'speed boost' node does nothing at all."
            + (
                "\n\nOn Windows you need `triton-windows`; the upstream `triton` package has no "
                "working Windows wheels."
                if ctx.env.is_windows else ""
            )
        ),
        remedy=remedy.install(
            ctx.env, [pkg],
            why=f"Installs {pkg}, unlocking torch.compile and the Triton-based optimisation nodes.",
        ),
    )


@rule
def onnx_gpu_for_face_nodes(ctx: Context) -> Iterator[Finding]:
    """insightface on CPU onnxruntime is the classic 'why is face-swap so slow'."""
    if not _gpu_healthy(ctx):
        return

    face_pkgs = [p for p in ("insightface", "facexlib", "facenet-pytorch") if ctx.inv.has(p)]
    if not face_pkgs:
        return
    if ctx.inv.has("onnxruntime-gpu"):
        return
    if not ctx.inv.has("onnxruntime"):
        return  # they aren't using onnx at all

    yield Finding(
        id="tip.onnxruntime_gpu",
        severity=Severity.TIP,
        category=CAT,
        title="Your face nodes are running on the CPU",
        detail=(
            f"You have {', '.join(face_pkgs)} installed, but only the CPU build of onnxruntime. "
            f"onnxruntime-gpu is not installed."
        ),
        impact=(
            "Face detection and face-swap models run through onnxruntime. On the CPU build they "
            "are roughly 5-10x slower, and nothing warns you — the node works, it's just slow. "
            "This is the single most common complaint about ReActor, IPAdapter FaceID and "
            "InsightFace workflows."
        ),
        evidence={"face_packages": face_pkgs},
        remedy=Remedy(
            title="Switch onnxruntime to the GPU build",
            commands=[
                ctx.env.pip_argv("uninstall", "-y", "onnxruntime"),
                ctx.env.pip_argv("install", "onnxruntime-gpu"),
            ],
            explain=(
                "The CPU and GPU builds provide the same `onnxruntime` module, so they cannot "
                "coexist — the CPU one wins and the GPU one sits unused. This removes the CPU "
                "build first, then installs the GPU one."
            ),
            danger="Between the two commands onnxruntime will not exist. Let it finish.",
        ),
    )


@rule
def cuda_malloc_fragmentation(ctx: Context) -> Iterator[Finding]:
    """A free fix for a whole class of 'out of memory' that isn't really OOM."""
    import os

    if not _gpu_healthy(ctx):
        return
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return

    vram_gb = (ctx.gpu.torch_devices[0].get("vram_total_mb") or 0) / 1024
    if vram_gb >= 20:
        return  # plenty of headroom; fragmentation rarely bites

    yield Finding(
        id="tip.expandable_segments",
        severity=Severity.TIP,
        category=CAT,
        title="One environment variable can prevent a lot of false out-of-memory errors",
        detail="PYTORCH_CUDA_ALLOC_CONF is not set.",
        impact=(
            "PyTorch's allocator fragments VRAM over a long session. You end up with plenty of "
            "free memory in total but no single block big enough, and you get an out-of-memory "
            "error on a workflow that ran fine an hour ago. Setting "
            "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lets the allocator grow blocks "
            "instead of fragmenting, and it costs nothing.\n\n"
            f"With {vram_gb:.0f} GB of VRAM you are well within the range where this matters."
        ),
        evidence={"vram_gb": round(vram_gb, 1)},
        remedy=remedy.manual(
            title="Set it before ComfyUI starts",
            explain=(
                "This has to be set in the environment before PyTorch initialises, so it goes in "
                "your launch script, not in a node.\n\n"
                "Windows (.bat, before the python line):\n"
                "    set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n\n"
                "Linux/macOS (.sh):\n"
                "    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n\n"
                "Restart ComfyUI afterwards. Re-run this scan and the tip will disappear."
            ),
        ),
    )


@rule
def bf16_capable(ctx: Context) -> Iterator[Finding]:
    if not _gpu_healthy(ctx):
        return
    cc = _compute_capability(ctx)
    if cc < 8.9:  # fp8 needs Ada (8.9) or Hopper+
        return

    name = ctx.gpu.torch_devices[0].get("name", "your GPU")
    vram_gb = (ctx.gpu.torch_devices[0].get("vram_total_mb") or 0) / 1024
    if vram_gb >= 24:
        return  # they can already fit the big models comfortably

    yield Finding(
        id="tip.fp8",
        severity=Severity.TIP,
        category=CAT,
        title=f"{name} supports fp8 — you can run bigger models than you think",
        detail=f"Compute capability {cc} includes native fp8 tensor cores. You have {vram_gb:.0f} GB of VRAM.",
        impact=(
            "Flux, Qwen-Image and the larger video models ship fp8 variants that use roughly half "
            "the VRAM of fp16 with very little quality loss, and run faster on your card because "
            "the fp8 path is hardware-accelerated. If you've been avoiding a model because it "
            "doesn't fit, look for its fp8 (or GGUF) variant before you conclude you need more "
            "VRAM.\n\n"
            "Launching with --fast also enables fp8 matrix-multiply acceleration on this hardware."
        ),
        evidence={"compute_capability": cc, "vram_gb": round(vram_gb, 1)},
        remedy=remedy.manual(
            title="Nothing to install",
            explain=(
                "This is a hardware capability you already have — it's about which model files you "
                "download. Prefer the fp8 checkpoint variants, and add --fast to your ComfyUI "
                "launch arguments."
            ),
        ),
    )


@rule
def cudnn_benchmark(ctx: Context) -> Iterator[Finding]:
    if not _gpu_healthy(ctx):
        return
    if ctx.gpu.backends.get("cudnn_version") is None:
        return
    # ComfyUI does not enable this by default, and for the fixed-resolution
    # batches typical of image generation it is close to free speed.
    yield Finding(
        id="tip.cudnn_benchmark",
        severity=Severity.TIP,
        category=CAT,
        title="Try --cuda-malloc and cuDNN autotuning for a few percent more throughput",
        detail=f"cuDNN {_cudnn(ctx.gpu.backends['cudnn_version'])} is available.",
        impact=(
            "cuDNN can benchmark convolution algorithms on first use and then reuse the fastest "
            "one. It costs a few seconds on the first run of each new resolution and gives a small "
            "but free speed-up on every run after — which suits image generation, where you tend "
            "to hammer the same resolution repeatedly.\n\n"
            "This is a modest win (a few percent), not a transformative one. Mentioned for "
            "completeness; ignore it if you switch resolutions constantly."
        ),
        remedy=remedy.manual(
            title="Add to your launch arguments",
            explain=(
                "Add `--cuda-malloc` to your ComfyUI launch command. Most builds enable cuDNN "
                "benchmarking automatically once a workload settles; there is nothing to install."
            ),
        ),
    )
