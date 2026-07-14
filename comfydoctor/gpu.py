"""Hardware and driver truth, from outside PyTorch.

Two independent sources, deliberately:

  * nvidia-smi tells us what the *machine* has - GPU, driver, the maximum CUDA
    runtime that driver supports. It works even when torch is completely broken.
  * torch tells us what PyTorch *thinks* it has.

The interesting findings all live in the gap between those two. "nvidia-smi sees
a 4090 but torch.cuda.is_available() is False" is the single most common silent
failure in ComfyUI, and you cannot detect it by asking torch alone.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

# Minimum NVIDIA driver for each CUDA runtime, from NVIDIA's compatibility table.
# Only the majors matter in practice; ComfyUI users are on cu118..cu130.
CUDA_MIN_DRIVER = {
    "cu118": (452.39, 450.80),   # (windows, linux)
    "cu121": (527.41, 525.60),
    "cu124": (528.33, 525.60),   # 12.x has minor-version compatibility from 525+
    "cu126": (528.33, 525.60),
    "cu128": (570.00, 570.00),   # CUDA 12.8 needs a 570-series driver
    "cu129": (570.00, 570.00),
    "cu130": (580.00, 580.00),
}


@dataclass
class GPUInfo:
    nvidia_smi_ok: bool = False
    driver_version: str | None = None
    driver_cuda_version: str | None = None   # highest CUDA the driver supports
    devices: list[dict] = field(default_factory=list)   # from nvidia-smi
    smi_error: str | None = None

    torch_ok: bool = False
    torch_version: str | None = None
    torch_cuda_build: str | None = None      # "12.4" - what torch was compiled for
    torch_local_tag: str | None = None       # "cu124"
    cuda_available: bool = False
    torch_devices: list[dict] = field(default_factory=list)
    torch_error: str | None = None
    backends: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def has_nvidia_hardware(self) -> bool:
        return self.nvidia_smi_ok and bool(self.devices)


def probe() -> GPUInfo:
    info = GPUInfo()
    _probe_nvidia_smi(info)
    _probe_torch(info)
    return info


def _probe_nvidia_smi(info: GPUInfo) -> None:
    exe = _find_nvidia_smi()
    if not exe:
        info.smi_error = "nvidia-smi not found on PATH"
        return
    try:
        q = "name,driver_version,memory.total,memory.used,compute_cap"
        r = subprocess.run(
            [exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            info.smi_error = (r.stderr or r.stdout or "nvidia-smi failed").strip()[:300]
            return
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            info.devices.append({
                "name": parts[0],
                "driver": parts[1],
                "vram_total_mb": _int(parts[2]),
                "vram_used_mb": _int(parts[3]),
                "compute_capability": parts[4],
            })
        if info.devices:
            info.driver_version = info.devices[0]["driver"]
        info.nvidia_smi_ok = True

        # The "CUDA Version: 12.8" in the smi header is the max runtime the
        # driver can load - not what is installed. It is exactly the ceiling we
        # need to check a torch cu-tag against.
        r2 = subprocess.run([exe], capture_output=True, text=True, timeout=15)
        m = re.search(r"CUDA Version:\s*([0-9.]+)", r2.stdout or "")
        if m:
            info.driver_cuda_version = m.group(1)
    except subprocess.TimeoutExpired:
        info.smi_error = "nvidia-smi timed out (driver may be hung)"
    except Exception as e:
        info.smi_error = f"{type(e).__name__}: {e}"


def _find_nvidia_smi() -> str | None:
    import shutil

    p = shutil.which("nvidia-smi")
    if p:
        return p
    if os.name == "nt":
        for c in (
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.exists(c):
                return c
    return None


_TORCH_PROBE = r"""
import json, sys
out = {"ok": False}
try:
    import torch
    out["ok"] = True
    out["version"] = torch.__version__
    out["cuda_build"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["devices"] = []
    if out["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out["devices"].append({
                "index": i,
                "name": p.name,
                "vram_total_mb": int(p.total_memory / (1024**2)),
                "compute_capability": "%d.%d" % (p.major, p.minor),
                "multi_processor_count": p.multi_processor_count,
            })
        try:
            out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            out["bf16_supported"] = None
        try:
            out["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            out["cudnn_version"] = None
    out["backends"] = {
        "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
        "mkl": bool(torch.backends.mkl.is_available()),
        "openmp": bool(torch.backends.openmp.is_available()),
    }
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
print("<<<COMFYDOCTOR>>>" + json.dumps(out))
"""


def _probe_torch(info: GPUInfo) -> None:
    """Read torch's view of the world - in-process if it is already loaded,
    in a throwaway subprocess if it is not.

    Inside ComfyUI, torch is already imported: using it directly is free and
    safe. From the CLI on a broken environment, `import torch` can segfault or
    hang on a bad CUDA/driver pairing. Isolating that in a subprocess means the
    doctor survives to *report* the crash instead of dying of it - which is the
    whole point of a tool you reach for when things are broken.
    """
    if "torch" in sys.modules:
        data = _run_probe_inline()
    else:
        data = _run_probe_subprocess()

    if data is None:
        info.torch_error = "torch probe crashed (the interpreter died importing torch)"
        return
    if not data.get("ok"):
        info.torch_error = data.get("error", "torch is not installed")
        return

    info.torch_ok = True
    info.torch_version = data.get("version")
    info.torch_cuda_build = data.get("cuda_build")
    info.cuda_available = bool(data.get("cuda_available"))
    info.torch_devices = data.get("devices", [])
    info.backends = data.get("backends", {})
    if info.torch_version and "+" in info.torch_version:
        info.torch_local_tag = info.torch_version.split("+", 1)[1]
    if data.get("cudnn_version"):
        info.backends["cudnn_version"] = data["cudnn_version"]
    if data.get("bf16_supported") is not None:
        info.backends["bf16_supported"] = data["bf16_supported"]


def _run_probe_inline() -> dict | None:
    scope: dict = {}
    try:
        exec(compile(_TORCH_PROBE.replace('print("<<<COMFYDOCTOR>>>" + json.dumps(out))', "RESULT = out"),
                     "<torch_probe>", "exec"), scope)
        return scope.get("RESULT")
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _run_probe_subprocess() -> dict | None:
    try:
        r = subprocess.run(
            [sys.executable, "-c", _TORCH_PROBE],
            capture_output=True, text=True, timeout=120,
        )
        for line in (r.stdout or "").splitlines():
            if line.startswith("<<<COMFYDOCTOR>>>"):
                return json.loads(line[len("<<<COMFYDOCTOR>>>"):])
        # No marker: the process died before printing. That IS the finding.
        return None
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "importing torch timed out after 120s"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def min_driver_for(cu_tag: str, windows: bool) -> float | None:
    entry = CUDA_MIN_DRIVER.get(cu_tag)
    if not entry:
        return None
    return entry[0] if windows else entry[1]


def cuda_tag_for_driver(driver_cuda: str | None, windows: bool) -> str:
    """Best torch cu-tag this driver can actually run. Used to build install
    commands that will not immediately fail."""
    if not driver_cuda:
        return "cu124"
    try:
        major, minor = (driver_cuda.split(".") + ["0"])[:2]
        v = int(major) * 10 + int(minor)
    except Exception:
        return "cu124"
    for tag in ("cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118"):
        need = int(tag[2:4]) * 10 + int(tag[4:])
        if v >= need:
            return tag
    return "cu118"


def _int(s: str) -> int | None:
    try:
        return int(float(s))
    except Exception:
        return None
