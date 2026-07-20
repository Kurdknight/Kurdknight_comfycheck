"""Tests for comfydoctor.shipped — the live-PyPI / cached / baked resolver of
"which versions actually exist". The whole false-positive saga came from
asserting versions no one had verified; this module is the verification."""

import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfydoctor import shipped  # noqa: E402


def _isolate(tmp_path, offline=True, monkeypatch=None):
    """Point the module at a fresh cache file and clear the memo."""
    monkeypatch.setattr(shipped, "CACHE_FILE", str(tmp_path / "cache.json"))
    if offline:
        monkeypatch.setenv("COMFYDOCTOR_NO_NETWORK", "1")
    else:
        monkeypatch.delenv("COMFYDOCTOR_NO_NETWORK", raising=False)
    shipped.clear_caches()


class TestBakedFallback:
    def test_offline_uses_baked(self, tmp_path, monkeypatch):
        _isolate(tmp_path, offline=True, monkeypatch=monkeypatch)
        minors, source = shipped.shipped_minors("torch")
        assert source == "baked"
        assert (2, 13) in minors        # newest stable torch in the snapshot
        assert (2, 14) not in minors

    def test_baked_reflects_torchaudio_freeze(self, tmp_path, monkeypatch):
        _isolate(tmp_path, offline=True, monkeypatch=monkeypatch)
        assert shipped.minor_shipped("torchaudio", (2, 11)) is True
        assert shipped.minor_shipped("torchaudio", (2, 12)) is False
        assert shipped.minor_shipped("torchaudio", (2, 13)) is False
        assert shipped.minor_shipped("torchvision", (0, 28)) is True

    def test_unknown_package_is_empty_not_crash(self, tmp_path, monkeypatch):
        _isolate(tmp_path, offline=True, monkeypatch=monkeypatch)
        minors, source = shipped.shipped_minors("no-such-package")
        assert minors == frozenset()
        assert shipped.minor_shipped("no-such-package", (1, 0)) is False

    def test_none_mm_is_false(self, tmp_path, monkeypatch):
        _isolate(tmp_path, offline=True, monkeypatch=monkeypatch)
        assert shipped.minor_shipped("torch", None) is False


class TestLiveFetch:
    def _fake_pypi(self, releases):
        body = json.dumps({"releases": releases}).encode()

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return FakeResp(body)

    def test_live_fetch_parses_and_caches(self, tmp_path, monkeypatch):
        _isolate(tmp_path, offline=False, monkeypatch=monkeypatch)
        releases = {
            "2.13.0": [{"f": 1}],
            "2.14.0": [{"f": 1}],          # newer than the baked snapshot
            "2.15.0rc1": [{"f": 1}],       # rc only — must NOT count
            "2.16.0": [],                  # registered, never uploaded — must NOT count
        }
        with patch.object(shipped.urllib.request, "urlopen",
                          return_value=self._fake_pypi(releases)):
            minors, source = shipped.shipped_minors("torch")
        assert source == "live"
        assert (2, 14) in minors
        assert (2, 15) not in minors
        assert (2, 16) not in minors
        # Second call hits the memo/cache, no network needed:
        shipped.clear_caches()
        minors2, source2 = shipped.shipped_minors("torch")
        assert source2 == "cache"
        assert minors2 == minors

    def test_network_failure_falls_back_to_baked(self, tmp_path, monkeypatch):
        _isolate(tmp_path, offline=False, monkeypatch=monkeypatch)
        with patch.object(shipped.urllib.request, "urlopen",
                          side_effect=OSError("no route")):
            minors, source = shipped.shipped_minors("torch")
        assert source == "baked"
        assert (2, 13) in minors

    def test_stale_cache_beats_baked_when_offline(self, tmp_path, monkeypatch):
        _isolate(tmp_path, offline=True, monkeypatch=monkeypatch)
        stale = {"torch": {"fetched_at": time.time() - 90 * 86400,
                           "minors": [[2, 13], [2, 14]]}}
        (tmp_path / "cache.json").write_text(json.dumps(stale))
        minors, source = shipped.shipped_minors("torch")
        assert source == "stale-cache"
        assert (2, 14) in minors     # data a live run once saw, kept


class TestSafetyDirection:
    def test_staleness_can_only_underclaim(self, tmp_path, monkeypatch):
        """The failure mode of old data must be 'can't verify' (INFO), never
        'demand a version'. expected_* returning None on anything the data
        source can't see guarantees that."""
        _isolate(tmp_path, offline=True, monkeypatch=monkeypatch)
        from comfydoctor.remedy import expected_torchaudio, expected_torchvision
        # 2.14 doesn't exist in the baked snapshot: both must decline to answer.
        assert expected_torchvision("2.14.0") is None
        assert expected_torchaudio("2.14.0") is None
