"""Regression tests for the 2026-07 false-positive audit.

Same class as the torchaudio bug: a hardcoded/extrapolated number + a
CRITICAL/ERROR + a destructive remedy. These pin the fixes so they don't
regress and re-earn a Reddit thread.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestDriverTable:
    """cu12x must share the CUDA-12.0 driver floor (minor-version compat), not
    the driver that shipped with each minor."""

    def test_cuda12_wheels_share_the_major_floor(self):
        from comfydoctor.gpu import min_driver_for
        # 4090 on driver 550 running cu128 torch is a working config — the floor
        # must be ~525 (linux), never 570.
        for tag in ("cu121", "cu124", "cu126", "cu128", "cu129"):
            assert min_driver_for(tag, windows=False) == 525.60, tag
            assert min_driver_for(tag, windows=True) == 528.33, tag

    def test_real_major_bump_still_raises_floor(self):
        from comfydoctor.gpu import min_driver_for
        assert min_driver_for("cu130", windows=False) == 580.0

    def test_550_driver_is_enough_for_cu128(self):
        from comfydoctor.gpu import min_driver_for
        assert 550.0 >= min_driver_for("cu128", windows=False)


class TestAbiHelpers:
    def test_lower_bound_only(self):
        from comfydoctor.rules.attention import _is_lower_bound_only
        assert _is_lower_bound_only(">=2.7") is True
        assert _is_lower_bound_only(">2.0") is True
        assert _is_lower_bound_only("==2.4.*") is False   # exact = real ABI pin
        assert _is_lower_bound_only(">=2.0,<2.5") is False  # has an upper bound
        assert _is_lower_bound_only("!=2.8.0") is False    # exclusion
        assert _is_lower_bound_only("") is False

    def test_patch_releases_are_abi_equal(self):
        from comfydoctor.rules.attention import _mm
        # a wheel built for torch2.9.1 runs on torch 2.9.0
        assert _mm("2.9.1") == _mm("2.9.0") == "2.9"
        assert _mm("2.9.1.post5") == "2.9"
        # a real minor difference is still caught
        assert _mm("2.8.0") != _mm("2.9.0")


class TestTorchFamilyGuard:
    def test_torch_family_constant(self):
        from comfydoctor.rules.packages import TORCH_FAMILY
        assert {"torch", "torchvision", "torchaudio"} <= TORCH_FAMILY


class TestPythonSweetSpot:
    def test_313_is_supported(self):
        from comfydoctor.rules.system import PY_SWEET_SPOT
        lo, hi = PY_SWEET_SPOT
        assert lo <= (3, 13) <= hi, "ComfyUI runs on 3.13 — it must not warn"

    def test_314_still_ahead(self):
        from comfydoctor.rules.system import PY_SWEET_SPOT
        _, hi = PY_SWEET_SPOT
        assert (3, 14) > hi
