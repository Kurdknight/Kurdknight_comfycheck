"""The inventory view: what you have, what you don't, grouped so it reads.

This restores what v1 was actually *for* — being able to glance at one screen and
see your whole AI stack — but it fixes the bug that made v1's list untrustworthy.

The old code did `importlib.import_module("opencv-python")` and reported "Not
installed" when that failed. It always failed: the *distribution* is called
opencv-python but the *module* is called cv2. Same for scikit-learn (sklearn),
pillow (PIL), face-recognition (face_recognition), flash-attn (flash_attn). Half
the list was permanently, confidently wrong.

Here, everything is looked up by distribution name through importlib.metadata,
which is what pip itself uses. Nothing is imported. Nothing is guessed.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from .env import Environment
from .gpu import GPUInfo
from .inventory import Inventory

# Grouped roughly the way people actually think about their stack. The `note`
# is what the package is FOR - because "einops: 0.8.0" tells a newcomer nothing.
LIBRARY_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("PyTorch", [
        ("torch", "the engine everything runs on"),
        ("torchvision", "image ops; must match torch's release"),
        ("torchaudio", "audio ops; must match torch's release"),
    ]),
    ("Attention & speed", [
        ("xformers", "memory-efficient attention (alternative to the built-in)"),
        ("flash-attn", "FlashAttention kernels"),
        ("sageattention", "quantised attention — reported 1.3-2x faster sampling, biggest on video; needs --use-sage-attention"),
        ("triton", "compiles GPU kernels; needed by torch.compile"),
        ("triton-windows", "the Windows port of Triton"),
        ("deepspeed", "large-model training/inference optimisation"),
    ]),
    ("Diffusion & models", [
        ("transformers", "text encoders, CLIP, T5"),
        ("diffusers", "diffusion pipelines"),
        ("accelerate", "device placement and offloading"),
        ("safetensors", "the model file format"),
        ("huggingface-hub", "model downloading"),
        ("tokenizers", "fast tokenisation"),
        ("sentencepiece", "tokeniser for T5/LLaMA-family text encoders"),
        ("peft", "LoRA loading"),
        ("timm", "vision backbones"),
    ]),
    ("Quantisation", [
        ("bitsandbytes", "8/4-bit quantisation"),
        ("gguf", "GGUF model support — run big models on small cards"),
        ("optimum", "hardware-optimised inference"),
        ("nunchaku", "4-bit SVDQuant inference"),
    ]),
    ("Core scientific", [
        ("numpy", "arrays. The 1.x/2.x split breaks things — see conflicts"),
        ("scipy", "scientific routines"),
        ("pillow", "image loading/saving (imports as PIL)"),
        ("opencv-python", "computer vision (imports as cv2)"),
        ("opencv-python-headless", "the server build of OpenCV — pick ONE"),
        ("opencv-contrib-python", "OpenCV plus contrib modules — pick ONE"),
        ("einops", "tensor reshaping used across the ecosystem"),
        ("scikit-image", "image processing"),
        ("scikit-learn", "classical ML (imports as sklearn)"),
        ("pandas", "dataframes"),
        ("matplotlib", "plotting"),
    ]),
    ("Face & detection", [
        ("insightface", "face analysis — powers ReActor, IPAdapter FaceID"),
        ("onnxruntime", "ONNX inference (CPU build)"),
        ("onnxruntime-gpu", "ONNX inference (GPU build) — much faster for face nodes"),
        ("facexlib", "face restoration helpers"),
        ("dlib", "classical face landmarks"),
        ("mediapipe", "Google's face/pose detection"),
        ("ultralytics", "YOLO detection/segmentation"),
        ("segment-anything", "SAM segmentation"),
    ]),
    ("Video & audio", [
        ("av", "PyAV — video decoding"),
        ("imageio", "image/video IO"),
        ("imageio-ffmpeg", "the ffmpeg bridge"),
        ("moviepy", "video editing"),
        ("decord", "fast video frame loading"),
        ("librosa", "audio analysis"),
        ("soundfile", "audio IO"),
    ]),
]

# The env vars that actually change ComfyUI's behaviour. Anything holding a token
# is listed as "set"/"not set" and never printed - a report is meant to be
# pasteable into a public issue.
ENV_VARS: list[tuple[str, str, bool]] = [
    ("PYTORCH_CUDA_ALLOC_CONF", "allocator tuning; expandable_segments:True avoids fragmentation OOM", False),
    ("CUDA_VISIBLE_DEVICES", "which GPUs ComfyUI can see", False),
    ("CUDA_HOME", "CUDA toolkit location", False),
    ("CUDA_PATH", "CUDA toolkit location (Windows)", False),
    ("TORCH_CUDA_ARCH_LIST", "which GPU architectures to compile kernels for", False),
    ("PYTORCH_ENABLE_MPS_FALLBACK", "Apple Silicon fallback", False),
    ("HF_HOME", "where HuggingFace caches models", False),
    ("HF_TOKEN", "HuggingFace auth", True),          # secret
    ("TRANSFORMERS_CACHE", "legacy HF cache location", False),
    ("XFORMERS_ENABLE_VERSION_CHECK", "xFormers version guard", False),
    ("TRITON_CACHE_DIR", "where Triton caches compiled kernels", False),
    ("PYTHONPATH", "extra import paths — a common source of shadowed packages", False),
    ("VIRTUAL_ENV", "active virtualenv", False),
    ("CONDA_PREFIX", "active conda env", False),
]


def build(env: Environment, gpu: GPUInfo, inv: Inventory) -> dict[str, Any]:
    """A grouped, human-readable snapshot of the whole environment."""
    return {
        "system": _system(env),
        "python": _python(env, inv),
        "gpu": _gpu(gpu),
        "pytorch": _pytorch(gpu),
        "libraries": _libraries(inv),
        "environment_variables": _env_vars(),
    }


def _system(env: Environment) -> list[dict]:
    rows = [
        _row("OS", f"{platform.system()} {platform.release()}"),
        _row("Platform", env.platform_tag),
        _row("Machine", platform.machine()),
        _row("CPU", platform.processor() or "unknown"),
    ]
    try:
        import psutil

        rows += [
            _row("CPU cores", f"{psutil.cpu_count(logical=False)} physical / "
                              f"{psutil.cpu_count(logical=True)} logical"),
            _row("RAM", f"{psutil.virtual_memory().total / 1024**3:.1f} GB "
                        f"({psutil.virtual_memory().percent}% in use)"),
        ]
    except Exception:
        pass

    if env.comfy_root:
        import shutil

        try:
            u = shutil.disk_usage(str(env.comfy_root))
            rows.append(_row("Disk (ComfyUI's drive)",
                             f"{u.free / 1024**3:.1f} GB free of {u.total / 1024**3:.0f} GB"))
        except Exception:
            pass
        rows.append(_row("ComfyUI root", str(env.comfy_root)))
    return rows


def _python(env: Environment, inv: Inventory) -> list[dict]:
    return [
        _row("Version", env.python_version),
        _row("Implementation", platform.python_implementation()),
        _row("Install type", f"{env.kind} — {env.kind_detail}"),
        _row("Executable", env.python_exe),
        _row("pip", inv.version("pip") or "not found"),
        _row("setuptools", inv.version("setuptools") or "not found"),
        _row("Install command", " ".join(env.pip_argv("install", "<package>")),
             note="use exactly this — a bare `pip` usually targets a different Python"),
        _row("Site-packages", "\n".join(env.site_dirs) or "unknown"),
        _row("Packages installed", str(len(inv.dists))),
    ]


def _gpu(gpu: GPUInfo) -> list[dict]:
    if not gpu.has_nvidia_hardware:
        return [_row("GPU", "no NVIDIA GPU detected",
                     note=gpu.smi_error or "nvidia-smi found nothing")]

    rows = [_row("Driver", gpu.driver_version or "unknown")]
    if gpu.driver_cuda_version:
        rows.append(_row("Max CUDA the driver supports", gpu.driver_cuda_version,
                         note="the ceiling on which PyTorch build you can run"))

    for i, d in enumerate(gpu.devices):
        prefix = f"GPU {i}" if len(gpu.devices) > 1 else "GPU"
        rows.append(_row(prefix, d.get("name", "?")))
        vram = d.get("vram_total_mb")
        if vram:
            used = d.get("vram_used_mb") or 0
            rows.append(_row(f"{prefix} VRAM",
                             f"{vram / 1024:.1f} GB total, {used / 1024:.1f} GB in use"))
        if d.get("compute_capability"):
            cc = d["compute_capability"]
            rows.append(_row(f"{prefix} compute capability", cc, note=_cc_note(cc)))
    return rows


def _cc_note(cc: str) -> str:
    try:
        v = float(cc)
    except ValueError:
        return ""
    if v >= 9.0:
        return "Hopper/Blackwell — fp8 and every modern kernel"
    if v >= 8.9:
        return "Ada — native fp8; use fp8 model variants and --fast"
    if v >= 8.0:
        return "Ampere — bf16 and SageAttention supported"
    if v >= 7.5:
        return "Turing — fp16; no bf16"
    return "older than Turing — expect limited kernel support"


def _cudnn(v) -> str:
    """torch reports cuDNN as a packed int: 91200 -> 9.12.0. Printing the raw
    number is how you get support threads full of people saying "I have cuDNN
    ninety-one thousand"."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{n // 10000}.{(n % 10000) // 100}.{n % 100}"


def _pytorch(gpu: GPUInfo) -> list[dict]:
    if not gpu.torch_ok:
        return [_row("PyTorch", "not installed or not importable", note=gpu.torch_error or "")]

    b = gpu.backends
    rows = [
        _row("Version", gpu.torch_version or "?"),
        _row("Built for CUDA", gpu.torch_cuda_build or "CPU-only build",
             note=("this is a CPU build — your GPU will not be used"
                   if not gpu.torch_cuda_build else "")),
        _row("CUDA usable now", "yes" if gpu.cuda_available else "NO"),
    ]
    if b.get("cudnn_version"):
        rows.append(_row("cuDNN", _cudnn(b["cudnn_version"])))
    if b.get("bf16_supported") is not None:
        rows.append(_row("bf16 supported", "yes" if b["bf16_supported"] else "no"))

    sdpa = [name for key, name in (
        ("flash_sdp", "Flash"),
        ("mem_efficient_sdp", "Mem-efficient"),
        ("math_sdp", "Math (fallback)"),
    ) if b.get(key)]
    rows.append(_row("SDPA backends", ", ".join(sdpa) or "none",
                     note="PyTorch's built-in attention — the path used when no speed-up is switched on"))
    for key, label in (("mkl", "Intel MKL"), ("openmp", "OpenMP")):
        if key in b:
            rows.append(_row(label, "enabled" if b[key] else "disabled"))
    return rows


def _libraries(inv: Inventory) -> list[dict]:
    """Every group, with what's installed and what isn't — the v1 feature, correct."""
    out = []
    for group, entries in LIBRARY_GROUPS:
        items = []
        for name, note in entries:
            v = inv.version(name)
            items.append({
                "name": name,
                "version": v,
                "installed": v is not None,
                "note": note,
            })
        out.append({
            "group": group,
            "installed": sum(1 for i in items if i["installed"]),
            "total": len(items),
            "items": items,
        })
    return out


def _env_vars() -> list[dict]:
    rows = []
    for name, note, secret in ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            value = None
        elif secret:
            # Never print a token, even into a report the user thinks is private.
            value = "set (value hidden)"
        else:
            value = raw
        rows.append({"name": name, "value": value, "set": raw is not None, "note": note})
    return rows


def _row(label: str, value: str, note: str = "") -> dict:
    return {"label": label, "value": value, "note": note}
