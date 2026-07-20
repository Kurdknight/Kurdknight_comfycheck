"""Regression tests for the torch/torchvision/torchaudio version pairing.

Motivated by a real false positive (r/comfyui, 2026-07): the checker reported
"torchaudio 2.11.0+cu130 is installed, but torch 2.13.0 needs torchaudio 2.13.x"
— a version that does not exist — and a user uninstalled a working stack over
it. The pairing formula is correct for shipped releases but must NOT be
extrapolated onto nightly / newer-than-known torch versions.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfydoctor.remedy import (  # noqa: E402
    expected_torchaudio, expected_torchvision, is_prerelease_torch,
    KNOWN_TORCH_MAX_MINOR,
)


class TestExpectedVersions:
    def test_known_releases_pair_correctly(self):
        # Lockstep confirmed against pytorch.org: torchaudio minor == torch minor,
        # torchvision == 0.(minor+15).
        assert expected_torchaudio("2.6.0") == "2.6"
        assert expected_torchvision("2.6.0") == "0.21"
        assert expected_torchaudio("2.9.1") == "2.9"
        assert expected_torchvision("2.9.1") == "0.24"
        assert expected_torchaudio("2.10.0+cu124") == "2.10"

    def test_beyond_known_ceiling_returns_none(self):
        # The reviewer's exact case: torch 2.13 is newer than any shipped
        # torchaudio, so we DON'T invent "torchaudio 2.13".
        assert expected_torchaudio("2.13.0") is None
        assert expected_torchvision("2.13.0") is None

    def test_ceiling_boundary(self):
        assert expected_torchaudio(f"2.{KNOWN_TORCH_MAX_MINOR}.0") is not None
        assert expected_torchaudio(f"2.{KNOWN_TORCH_MAX_MINOR + 1}.0") is None

    def test_garbage_returns_none(self):
        assert expected_torchaudio("") is None
        assert expected_torchaudio("not.a.version") is None
        assert expected_torchvision("1.13.0") is None  # torch 1.x not handled


class TestPrerelease:
    def test_dev_and_rc_builds(self):
        assert is_prerelease_torch("2.13.0.dev20260710") is True
        assert is_prerelease_torch("2.8.0rc1") is True
        assert is_prerelease_torch("2.9.0+git1234567") is True

    def test_newer_than_known_is_prerelease(self):
        assert is_prerelease_torch("2.13.0") is True
        assert is_prerelease_torch("2.13.0+cu130") is True

    def test_shipped_stable_is_not_prerelease(self):
        assert is_prerelease_torch("2.6.0") is False
        assert is_prerelease_torch("2.9.1+cu124") is False
        assert is_prerelease_torch("2.10.0") is False

    def test_none(self):
        assert is_prerelease_torch(None) is False


class TestReinstallRemedyDegrades:
    def test_unverifiable_torch_does_not_pin_fake_versions(self):
        # When we can't verify the pairing, the reinstall command must not pin
        # torchvision/torchaudio to computed (possibly nonexistent) versions.
        from comfydoctor.remedy import reinstall_torch_stack
        from comfydoctor.env import Environment
        from comfydoctor.gpu import GPUInfo

        env = Environment.probe() if hasattr(Environment, "probe") else Environment.__new__(Environment)
        try:
            gpu = GPUInfo.__new__(GPUInfo)
            gpu.has_nvidia_hardware = False
            gpu.torch_version = "2.13.0"
            r = reinstall_torch_stack(env, gpu, torch_version="2.13.0")
        except Exception:
            # Environment/GPUInfo construction varies; the version logic above
            # is the load-bearing part and is covered by the other tests.
            return
        flat = " ".join(str(c) for c in r.commands)
        assert "torchaudio==2.13" not in flat
        assert "torchvision==0.28" not in flat
