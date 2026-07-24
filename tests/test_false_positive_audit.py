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


class TestCudaTagFormulas:
    """The other formula-shaped code paths: no fragile int encodings, no
    hand-maintained ladders that can drift from the driver table."""

    def test_tag_key_is_a_real_tuple(self):
        from comfydoctor.gpu import cu_tag_key
        assert cu_tag_key("cu124") == (12, 4)
        assert cu_tag_key("cu130") == (13, 0)
        assert cu_tag_key("cu118") == (11, 8)
        assert cu_tag_key("cpu") is None
        assert cu_tag_key("") is None
        # The old major*10+minor int encoding collapsed 12.10 and 13.0 into
        # the same number. Tuples never do.
        assert (12, 10) < (13, 0)

    def test_ladder_derived_from_driver_table(self):
        from comfydoctor.gpu import CUDA_MIN_DRIVER, CUDA_TAG_LADDER, cu_tag_key
        assert set(CUDA_TAG_LADDER) == set(CUDA_MIN_DRIVER), \
            "ladder and driver table must never drift apart"
        keys = [cu_tag_key(t) for t in CUDA_TAG_LADDER]
        assert keys == sorted(keys, reverse=True), "ladder must be newest-first"

    def test_older_tag_ladder_in_sync(self):
        from comfydoctor.gpu import CUDA_MIN_DRIVER
        from comfydoctor.rules.torch_stack import _older_tag
        for tag in CUDA_MIN_DRIVER:
            assert _older_tag(tag) in CUDA_MIN_DRIVER, tag
        assert _older_tag("cu118") == "cu118"      # nothing older exists
        assert _older_tag("cu130") == "cu129"

    def test_tag_for_driver_uses_tuple_compare(self):
        from comfydoctor.gpu import cuda_tag_for_driver
        assert cuda_tag_for_driver("12.4", windows=False) == "cu124"
        assert cuda_tag_for_driver("12.8", windows=False) == "cu128"
        assert cuda_tag_for_driver("13.0", windows=False) == "cu130"
        # A hypothetical CUDA 12.10 driver is still CUDA 12 — the old int
        # encoding (12*10+10 == 130) would have claimed it runs cu130 wheels.
        assert cuda_tag_for_driver("12.10", windows=False) == "cu129"
        assert cuda_tag_for_driver("11.8", windows=False) == "cu118"
        assert cuda_tag_for_driver("11.0", windows=False) == "cu118"

    def test_unknown_driver_falls_back_to_default(self):
        from comfydoctor.gpu import DEFAULT_CU_TAG, cuda_tag_for_driver
        assert cuda_tag_for_driver(None, windows=False) == DEFAULT_CU_TAG
        assert cuda_tag_for_driver("garbage", windows=True) == DEFAULT_CU_TAG


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


class TestTritonRealityCheck:
    """The 'no Windows wheels' claim is checked against the files on disk, not
    just asserted. If the installed triton ships .pyd binaries, upstream has
    started publishing Windows wheels and the warning must stay silent."""

    def _ctx(self):
        from comfydoctor.custom_nodes import NodeSurvey
        from comfydoctor.env import Environment
        from comfydoctor.gpu import GPUInfo
        from comfydoctor.inventory import Dist, Inventory
        from comfydoctor.rules import Context

        env = Environment.__new__(Environment)
        env.is_windows = True
        env.python_exe = "C:/x/python.exe"
        env.kind = "venv"
        triton = Dist(name="triton", raw_name="triton", version="3.1.0",
                      location="/site", modules=["triton"], owned_modules=["triton"])
        inv = Inventory(dists={"triton": triton}, duplicates={},
                        module_owners={}, unsatisfied=[])
        return Context(env=env, gpu=GPUInfo(), inv=inv, nodes=NodeSurvey())

    def test_real_windows_build_is_believed(self):
        from unittest.mock import patch

        from comfydoctor.rules import attention

        with patch.object(attention, "_ships_windows_binaries", return_value=True):
            assert list(attention.triton_on_windows(self._ctx())) == []

    def test_linux_wheel_on_windows_still_warns(self):
        from unittest.mock import patch

        from comfydoctor.rules import attention

        with patch.object(attention, "_ships_windows_binaries", return_value=False):
            findings = list(attention.triton_on_windows(self._ctx()))
        assert [f.id for f in findings] == ["attention.triton_linux_wheel_on_windows"]


class TestPythonSweetSpot:
    def test_313_is_supported(self):
        from comfydoctor.rules.system import PY_SWEET_SPOT
        lo, hi = PY_SWEET_SPOT
        assert lo <= (3, 13) <= hi, "ComfyUI runs on 3.13 — it must not warn"

    def test_314_still_ahead(self):
        from comfydoctor.rules.system import PY_SWEET_SPOT
        _, hi = PY_SWEET_SPOT
        assert (3, 14) > hi
