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
        title="SageAttention could make sampling noticeably faster - users report up to ~2x on video models",
        detail=(
            f"{name} (compute {cc}) supports SageAttention's quantised kernels, and it isn't installed."
        ),
        impact=(
            "SageAttention replaces the attention maths with faster quantised kernels. Reported "
            "gains vary by workload: commonly 20-40% faster sampling on image models, and "
            "1.5-2x on video models (Wan, Hunyuan), where attention dominates the time. Quality "
            "loss is generally imperceptible.\n\n"
            "It needs Triton to compile its kernels, so install that too if you don't have it. "
            "IMPORTANT: installing it changes nothing by itself - you must also launch ComfyUI "
            "with --use-sage-attention."
        ),
        evidence={"compute_capability": cc, "gpu": name},
        remedy=Remedy(
            title="Install SageAttention",
            commands=[ctx.env.pip_argv("install", "sageattention")],
            explain=(
                "Installs SageAttention, then add --use-sage-attention to your ComfyUI launch "
                "command to turn it on.\n\n"
                "It is a compiled package built against a specific torch — if a future torch "
                "upgrade breaks it, ComfyDoctor will tell you, and uninstalling it is always a "
                "safe fallback (you lose speed, nothing else)."
            ),
            danger=None,
            doc_url="https://github.com/thu-ml/SageAttention",
        ),
    )


@rule
def sage_installed_but_off(ctx: Context) -> Iterator[Finding]:
    """The saddest configuration: the speed-up installed, sitting idle.

    We run inside ComfyUI's process, so sys.argv IS the launch command — whether
    the flag is set is a verifiable fact about this session, not a guess."""
    import sys

    if not _gpu_healthy(ctx):
        return
    if not ctx.inv.has("sageattention"):
        return
    if not ctx.comfy_runtime:
        return  # from the CLI we can't see the launch flags - say nothing rather than guess
    if "--use-sage-attention" in sys.argv:
        return

    yield Finding(
        id="tip.sageattention_not_enabled",
        severity=Severity.TIP,
        category=CAT,
        title="SageAttention is installed but not switched on",
        detail=(
            f"sageattention {ctx.inv.version('sageattention')} is installed, but ComfyUI was not "
            f"started with --use-sage-attention."
        ),
        impact=(
            "Installing SageAttention changes nothing by itself. Unless a workflow node (for "
            "example KJNodes' 'Patch Sage Attention') switches it on for a specific run, your "
            "sampling is still using the slower built-in path and the speed-up you installed is "
            "sitting idle. Users commonly report 1.3-2x faster sampling with it on, most on "
            "video models."
        ),
        evidence={"launch_flags": [a for a in sys.argv if a.startswith("--")]},
        remedy=remedy.manual(
            title="Add --use-sage-attention to your launch command",
            explain=(
                "Open the .bat (or .sh) file you start ComfyUI with and add "
                "--use-sage-attention to the line that runs main.py, then restart ComfyUI.\n\n"
                "If sampling afterwards errors mentioning sageattention or triton, just remove "
                "the flag again - nothing else is affected."
            ),
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


# NOTE: there used to be a "--cuda-malloc / cuDNN autotuning" tip here. It was
# removed deliberately: it conflated two unrelated mechanisms, promised "a few
# percent" nobody had measured, and claimed builds enable benchmarking
# automatically - an assertion we cannot verify. A tip that confuses people to
# maybe save 2% fails this file's own second rule. Do not resurrect it without
# a verifiable claim and a verifiable action.
