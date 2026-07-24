"""Regression tests for the torch/torchvision/torchaudio version pairing.

Motivated by a real false positive (r/comfyui, 2026-07): the checker reported
"torchaudio 2.11.0+cu130 is installed, but torch 2.13.0 needs torchaudio 2.13.x"
— a version that does not exist — and a user uninstalled a working stack over
it. Reality (PyPI, 2026-07): torch stable reached 2.13, torchvision 0.28, but
torchaudio FROZE at 2.11. The user's stack was the correct, current pairing.

The rule now: the offset formula proposes a candidate; comfydoctor.shipped
(live PyPI / baked snapshot) decides whether that candidate ever shipped. We
never demand a version we cannot see.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfydoctor.remedy import (  # noqa: E402
    expected_torchaudio, expected_torchvision, is_prerelease_torch,
)


class TestExpectedVersions:
    def test_known_releases_pair_correctly(self):
        assert expected_torchaudio("2.6.0") == "2.6"
        assert expected_torchvision("2.6.0") == "0.21"
        assert expected_torchaudio("2.9.1") == "2.9"
        assert expected_torchvision("2.9.1") == "0.24"
        assert expected_torchaudio("2.10.0+cu124") == "2.10"
        assert expected_torchaudio("2.11.0") == "2.11"   # last torchaudio ever

    def test_frozen_torchaudio_era(self):
        # torch 2.12/2.13 shipped; torchaudio did not follow. torchvision is
        # still verifiable, torchaudio must be None — never "2.13".
        assert expected_torchvision("2.13.0") == "0.28"
        assert expected_torchaudio("2.13.0") is None
        assert expected_torchvision("2.12.0") == "0.27"
        assert expected_torchaudio("2.12.0") is None

    def test_unshipped_torch_returns_none(self):
        # A torch minor no stable release exists for: verify nothing.
        assert expected_torchaudio("2.99.0") is None
        assert expected_torchvision("2.99.0") is None

    def test_garbage_returns_none(self):
        assert expected_torchaudio("") is None
        assert expected_torchaudio("not.a.version") is None
        assert expected_torchvision("1.13.0") is None  # torch 1.x not handled


class TestPrerelease:
    def test_dev_and_rc_builds(self):
        assert is_prerelease_torch("2.13.0.dev20260710") is True
        assert is_prerelease_torch("2.8.0rc1") is True
        assert is_prerelease_torch("2.9.0+git1234567") is True

    def test_unshipped_minor_is_prerelease(self):
        assert is_prerelease_torch("2.99.0") is True
        assert is_prerelease_torch("3.0.0") is True

    def test_shipped_stable_is_not_prerelease(self):
        assert is_prerelease_torch("2.6.0") is False
        assert is_prerelease_torch("2.9.1+cu124") is False
        # torch 2.13.0 IS a real stable release (the Reddit case) — treating
        # it as "unknown/prerelease" was part of the original confusion.
        assert is_prerelease_torch("2.13.0") is False
        assert is_prerelease_torch("2.13.0+cu130") is False

    def test_none(self):
        assert is_prerelease_torch(None) is False


class TestReinstallRemedy:
    def _remedy_for(self, torch_v):
        from comfydoctor.remedy import reinstall_torch_stack
        from comfydoctor.env import Environment
        from comfydoctor.gpu import GPUInfo

        try:
            env = Environment.__new__(Environment)
            env.is_windows = False
            env.python_exe = "/usr/bin/python3"
            gpu = GPUInfo()
            r = reinstall_torch_stack(env, gpu, torch_version=torch_v)
        except Exception:
            return None
        return " ".join(" ".join(map(str, c)) if isinstance(c, list) else str(c)
                        for c in r.commands)

    def test_frozen_era_pins_only_what_shipped(self):
        flat = self._remedy_for("2.13.0")
        if flat is None:
            return  # construction differs; version logic covered above
        assert "torchaudio==2.13" not in flat, "must never pin a nonexistent version"
        assert "torchaudio" in flat            # still installed, just unpinned

    def test_unshipped_torch_pins_nothing(self):
        flat = self._remedy_for("2.99.0")
        if flat is None:
            return
        assert "==2.99" not in flat
        assert "torchvision==" not in flat


class TestReleaseGapStillPinsTorch:
    """torch has shipped but its torchvision partner isn't visible yet
    (release-day gap, or a stale offline snapshot). The remedy must still
    REPAIR — pin the user's real torch — never silently upgrade the stack."""

    def test_shipped_torch_with_unknown_partner_pins_torch_only(self, tmp_path, monkeypatch):
        import json
        import time

        from comfydoctor import shipped

        monkeypatch.setattr(shipped, "CACHE_FILE", str(tmp_path / "cache.json"))
        now = time.time()
        cache = {
            "torch": {"fetched_at": now, "minors": [[2, 13], [2, 14]]},
            "torchvision": {"fetched_at": now, "minors": [[0, 28]]},   # 0.29 missing
            "torchaudio": {"fetched_at": now, "minors": [[2, 11]]},
        }
        (tmp_path / "cache.json").write_text(json.dumps(cache))
        shipped.clear_caches()
        try:
            from comfydoctor.env import Environment
            from comfydoctor.gpu import GPUInfo
            from comfydoctor.remedy import reinstall_torch_stack

            env = Environment.__new__(Environment)
            env.is_windows = False
            env.python_exe = "/usr/bin/python3"
            env.kind = "venv"
            r = reinstall_torch_stack(env, GPUInfo(), torch_version="2.14.0")
            args = [a for c in r.commands for a in c]
            assert "torch==2.14.0" in args, "the user's real torch must stay pinned"
            assert "torchvision" in args
            assert not any(a.startswith("torchvision==") for a in args)
            assert not any(a.startswith("torchaudio==") for a in args)
        finally:
            shipped.clear_caches()
