"""THE invariant that prevents the next Reddit thread:

    A healthy machine must scan clean — zero CRITICAL, zero ERROR.

Every rule runs against synthetic snapshots of real, WORKING setups (portable
Windows + CUDA, conda, nightly torch, CPU-only, driver-550 + cu128). If any
rule fires CRITICAL/ERROR on one of these, that is a false positive of exactly
the class that made a user uninstall a working stack — and this test fails
before it ships. Add a new fixture here whenever a false positive is reported
in the wild.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfydoctor.env import Environment                    # noqa: E402
from comfydoctor.gpu import GPUInfo                        # noqa: E402
from comfydoctor.inventory import Dist, Inventory          # noqa: E402
from comfydoctor.custom_nodes import NodeSurvey            # noqa: E402
from comfydoctor.models import Severity                    # noqa: E402
from comfydoctor.rules import Context, run_all             # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _env(windows=True, kind="embedded") -> Environment:
    return Environment(
        python_exe="C:/ComfyUI/python_embeded/python.exe" if windows else "/opt/comfy/venv/bin/python",
        python_version="3.12.7",
        kind=kind,
        kind_detail="test fixture",
        comfy_root=None,           # disk rule falls back to cwd — patched below
        custom_nodes_dir=None,
        site_dirs=[],
        is_windows=windows,
        platform_tag="win_amd64" if windows else "linux_x86_64",
    )


def _dist(name, version, requires=None, modules=None) -> Dist:
    return Dist(
        name=name, raw_name=name, version=version, location="/site",
        requires=requires or [], modules=modules or [name],
        owned_modules=modules or [name],
    )


def _inv(dists: list) -> Inventory:
    return Inventory(
        dists={d.name: d for d in dists},
        duplicates={},
        module_owners={},
        unsatisfied=[],
    )


def _gpu_cuda(tag="cu126", torch_v="2.6.0", cuda_build="12.6", driver="551.86") -> GPUInfo:
    g = GPUInfo()
    g.nvidia_smi_ok = True
    g.driver_version = driver
    g.driver_cuda_version = "12.6"
    g.devices = [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24564,
                  "vram_used_mb": 800, "compute_capability": "8.9"}]
    g.torch_ok = True
    g.torch_version = f"{torch_v}+{tag}" if tag else torch_v
    g.torch_cuda_build = cuda_build
    g.torch_local_tag = tag
    g.cuda_available = True
    g.torch_devices = [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24564,
                        "compute_capability": "8.9"}]
    g.backends = {"cudnn": True, "flash_sdp": True}
    return g


def _matched_stack(torch_v="2.6.0", tag="cu126", tv="0.21.0", ta="2.6.0") -> list:
    suffix = f"+{tag}" if tag else ""
    return [
        _dist("torch", f"{torch_v}{suffix}"),
        _dist("torchvision", f"{tv}{suffix}"),
        _dist("torchaudio", f"{ta}{suffix}"),
        _dist("numpy", "1.26.4"),
        _dist("pillow", "10.4.0"),
    ]


def _ctx(env, gpu, dists) -> Context:
    return Context(env=env, gpu=gpu, inv=_inv(dists), nodes=NodeSurvey())


def _bad_findings(ctx):
    """Run EVERY rule; return the CRITICAL/ERROR findings (should be empty),
    with system probes pinned to healthy values so the test machine's own
    disk/RAM can't pollute the result."""
    import collections
    du = collections.namedtuple("usage", "total used free")(
        total=2_000_000_000_000, used=800_000_000_000, free=1_200_000_000_000)
    with patch("shutil.disk_usage", return_value=du):
        findings = run_all(ctx)
    return [f for f in findings if f.severity in (Severity.CRITICAL, Severity.ERROR)]


def _assert_clean(ctx, label):
    bad = _bad_findings(ctx)
    assert not bad, (
        f"FALSE POSITIVE on a healthy {label}: "
        + "; ".join(f"[{f.severity.value}] {f.id}: {f.title}" for f in bad)
    )


# --------------------------------------------------------------------------- #
# Healthy machines
# --------------------------------------------------------------------------- #

class TestHealthyMachinesScanClean:
    def test_windows_portable_cuda(self):
        _assert_clean(_ctx(_env(windows=True), _gpu_cuda(), _matched_stack()),
                      "Windows portable + 4090 + matched cu126 stack")

    def test_linux_venv_cu128_on_driver_550(self):
        # The audited case: cu128 wheels on a 550 driver is a WORKING config
        # (CUDA 12.x minor-version compatibility). Used to fire a CRITICAL
        # 'driver too old' + downgrade remedy.
        gpu = _gpu_cuda(tag="cu128", torch_v="2.7.1", cuda_build="12.8", driver="550.54")
        dists = _matched_stack(torch_v="2.7.1", tag="cu128", tv="0.22.1", ta="2.7.1")
        _assert_clean(_ctx(_env(windows=False, kind="venv"), gpu, dists),
                      "Linux venv + cu128 torch on driver 550")

    def test_nightly_torch_same_index(self):
        # The reported case: torch nightly newer than any released torchaudio.
        # Used to fire CRITICAL 'needs torchaudio 2.13.x' (nonexistent).
        gpu = _gpu_cuda(tag="cu130", torch_v="2.13.0", cuda_build="13.0", driver="581.00")
        dists = _matched_stack(torch_v="2.13.0", tag="cu130", tv="0.28.0", ta="2.13.0")
        _assert_clean(_ctx(_env(windows=True), gpu, dists),
                      "nightly torch stack from one index")

    def test_conda_cuda_torch_no_local_tag(self):
        # conda CUDA builds carry no +cuXXX tag but a real torch.version.cuda.
        # Used to be mislabelled 'CPU-only PyTorch installed'.
        gpu = _gpu_cuda(tag=None, torch_v="2.6.0", cuda_build="12.4")
        gpu.torch_local_tag = None
        dists = _matched_stack(torch_v="2.6.0", tag=None, tv="0.21.0", ta="2.6.0")
        _assert_clean(_ctx(_env(windows=False, kind="conda"), gpu, dists),
                      "conda CUDA torch (no local tag)")

    def test_cpu_only_machine(self):
        # No NVIDIA hardware at all (Mac/other): CPU torch is CORRECT here.
        gpu = GPUInfo()
        gpu.torch_ok = True
        gpu.torch_version = "2.6.0"
        gpu.cuda_available = False
        dists = _matched_stack(torch_v="2.6.0", tag=None)
        _assert_clean(_ctx(_env(windows=False, kind="venv"), gpu, dists),
                      "CPU-only machine (no NVIDIA hardware)")

    def test_patch_level_xformers_tag(self):
        # xformers wheel tagged torch2.9.1 on torch 2.9.0: ABI-compatible.
        # Used to fire ERROR 'built for a different PyTorch' + uninstall.
        gpu = _gpu_cuda(tag="cu128", torch_v="2.9.0", cuda_build="12.8", driver="576.00")
        dists = _matched_stack(torch_v="2.9.0", tag="cu128", tv="0.24.0", ta="2.9.0")
        dists.append(_dist("xformers", "0.0.30+cu128torch2.9.1"))
        _assert_clean(_ctx(_env(windows=True), gpu, dists),
                      "xformers patch-level build tag")

    def test_numpy2_with_only_numpy2_pins(self):
        gpu = _gpu_cuda()
        dists = _matched_stack()
        dists = [d for d in dists if d.name != "numpy"]
        dists.append(_dist("numpy", "2.1.0"))
        dists.append(_dist("scipy", "1.14.0", requires=["numpy>=2.0"]))
        _assert_clean(_ctx(_env(windows=True), gpu, dists),
                      "numpy 2 environment with numpy>=2 consumers")


class TestKnownConflictsStillCaught:
    """The guard must not have lobotomized the tool: REAL problems still fire."""

    def test_mixed_build_tags_still_error(self):
        gpu = _gpu_cuda()
        dists = [
            _dist("torch", "2.6.0+cu126"),
            _dist("torchvision", "0.21.0+cpu"),   # genuinely mixed index
            _dist("torchaudio", "2.6.0+cu126"),
        ]
        bad = _bad_findings(_ctx(_env(), gpu, dists))
        assert any(f.id == "torch.build_tag_mismatch" for f in bad), \
            "a cpu torchvision beside a cu126 torch must still be flagged"

    def test_real_triplet_mismatch_still_critical(self):
        gpu = _gpu_cuda()
        dists = [
            _dist("torch", "2.6.0+cu126"),
            _dist("torchvision", "0.18.0+cu126"),  # belongs to torch 2.3
            _dist("torchaudio", "2.6.0+cu126"),
        ]
        bad = _bad_findings(_ctx(_env(), gpu, dists))
        assert any(f.id == "torch.triplet_version_mismatch" for f in bad), \
            "a genuinely mismatched torchvision must still be flagged"

    def test_genuinely_old_driver_still_critical(self):
        # Driver 450 truly cannot run cu126 — and CUDA is NOT available.
        gpu = _gpu_cuda(tag="cu126", driver="450.80")
        gpu.cuda_available = False
        dists = _matched_stack()
        bad = _bad_findings(_ctx(_env(windows=False), gpu, dists))
        assert any(f.id in ("torch.driver_too_old", "torch.cuda_unavailable") for f in bad)
