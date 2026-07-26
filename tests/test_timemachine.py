"""The environment time machine: journal, regression point, diff, restore.

The load-bearing test in here is TestPersistentErrorMachine: an install with
dozens of nodes usually carries one long-standing ERROR the owner has decided
to live with. A reference point keyed on "last scan with zero errors" is
something such a machine never produces - the whole feature would be silently
dead exactly where it is needed. The reference point is per-problem for that
reason, and this suite pins it.

Contract also under test: the restore never uninstalls, never touches packages
added since the reference point, and never sends torch-family through bare
PyPI.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfydoctor import timemachine as tm                  # noqa: E402
from comfydoctor.env import Environment                    # noqa: E402
from comfydoctor.inventory import Dist, Inventory          # noqa: E402
from comfydoctor.models import Finding, Severity           # noqa: E402


def _env(root) -> Environment:
    env = Environment.__new__(Environment)
    env.comfy_root = root
    env.python_exe = "C:/x/python.exe"
    env.kind = "venv"
    env.is_windows = True
    return env


def _inv(pkgs: dict) -> Inventory:
    dists = {
        name: Dist(name=name, raw_name=name, version=v, location="/site")
        for name, v in pkgs.items()
    }
    return Inventory(dists=dists, duplicates={}, module_owners={}, unsatisfied=[])


def _problem(fid: str, sev=Severity.ERROR) -> Finding:
    return Finding(id=fid, severity=sev, category="c", title=f"title of {fid}")


CLEAN = [Finding(id="ok.x", severity=Severity.OK, category="c", title="fine"),
         Finding(id="warn.x", severity=Severity.WARNING, category="c", title="meh")]


def _journal(root) -> dict:
    return json.loads((root / "user" / "comfydoctor" / "env_journal.json").read_text())


class TestJournal:
    def test_records_packages_and_problems(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"torch": "2.9.1+cu130"}), [_problem("onnx.dup")])
        entry = _journal(tmp_path)["recent"][-1]
        assert entry["packages"] == {"torch": "2.9.1+cu130"}
        assert entry["problems"] == ["onnx.dup"]

    def test_warnings_are_not_problems(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"numpy": "1.26.4"}), CLEAN)
        assert _journal(tmp_path)["recent"][-1]["problems"] == []
        assert tm.last_clean(env) is not None

    def test_identical_scans_do_not_flush_history(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"numpy": "1.26.4"}), CLEAN)
        for _ in range(5):
            tm.record(env, _inv({"numpy": "1.26.4"}), CLEAN)
        assert len(_journal(tmp_path)["recent"]) == 1

    def test_recent_window_is_capped(self, tmp_path):
        env = _env(tmp_path)
        for i in range(tm.MAX_RECENT + 10):
            tm.record(env, _inv({"numpy": f"1.{i}.0"}), CLEAN)
        assert len(_journal(tmp_path)["recent"]) == tm.MAX_RECENT

    def test_clean_snapshot_survives_a_long_run_of_broken_scans(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"torch": "2.9.1+cu130"}), CLEAN)
        for i in range(tm.MAX_RECENT + 5):
            tm.record(env, _inv({"torch": f"2.9.{i}"}), [_problem("torch.cpu")])
        assert tm.last_clean(env)["packages"]["torch"] == "2.9.1+cu130"

    def test_corrupt_journal_is_a_fresh_start_not_a_crash(self, tmp_path):
        env = _env(tmp_path)
        p = tmp_path / "user" / "comfydoctor" / "env_journal.json"
        p.parent.mkdir(parents=True)
        p.write_text("{not json")
        tm.record(env, _inv({"numpy": "1.26.4"}), CLEAN)
        assert tm.last_clean(env)["packages"] == {"numpy": "1.26.4"}

    def test_no_comfy_root_is_a_silent_noop(self):
        env = _env(None)
        tm.record(env, _inv({"numpy": "1.26.4"}), CLEAN)
        assert tm.last_clean(env) is None
        assert tm.what_changed_finding(env, _inv({}), [_problem("x")]) is None


class TestPersistentErrorMachine:
    """THE real-world case: a machine that has carried an onnxruntime ERROR
    for months and the owner is fine with it. A new problem must still be
    traceable."""

    def test_new_problem_is_found_despite_a_standing_error(self, tmp_path):
        env = _env(tmp_path)
        # Three months of scans, all carrying the same accepted error.
        tm.record(env, _inv({"torch": "2.9.1+cu130", "numpy": "1.26.4"}),
                  [_problem("packages.onnxruntime_variants")])
        # Today: torch got replaced by a CPU wheel. New problem, same old error.
        f = tm.what_changed_finding(
            env,
            _inv({"torch": "2.9.1", "numpy": "1.26.4"}),
            [_problem("packages.onnxruntime_variants"), _problem("torch.cpu_build_on_gpu_machine")],
        )
        assert f is not None, "a standing error must not blind the time machine"
        assert "torch.cpu_build_on_gpu_machine" in f.evidence["new_problems"]
        assert "packages.onnxruntime_variants" not in f.evidence["new_problems"]
        assert "torch: 2.9.1+cu130 -> 2.9.1" in f.detail

    def test_unchanged_standing_error_alone_says_nothing(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"numpy": "1.26.4"}), [_problem("packages.onnxruntime_variants")])
        f = tm.what_changed_finding(
            env, _inv({"numpy": "2.1.0"}), [_problem("packages.onnxruntime_variants")])
        assert f is None, "an old, unchanged problem is not a regression"


class TestReferencePoint:
    def test_picks_the_tightest_window(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"a": "1.0"}), CLEAN)                 # old
        tm.record(env, _inv({"a": "2.0"}), CLEAN)                 # newest without the problem
        f = tm.what_changed_finding(env, _inv({"a": "3.0"}), [_problem("new.problem")])
        assert "a: 2.0 -> 3.0" in f.detail
        assert "a: 1.0" not in f.detail

    def test_saturated_window_stays_quiet_rather_than_guessing(self, tmp_path):
        # Every retained snapshot already had this problem: we no longer hold
        # the evidence for when it appeared, so we must not answer with a
        # months-wide diff off the pinned clean snapshot.
        env = _env(tmp_path)
        tm.record(env, _inv({"a": "1.0"}), CLEAN)                 # clean, then pruned out
        for i in range(tm.MAX_RECENT + 3):
            tm.record(env, _inv({"a": f"2.{i}"}), [_problem("p.always")])
        assert tm.what_changed_finding(env, _inv({"a": "9.0"}), [_problem("p.always")]) is None
        # ...but the clean state itself is still on record for a manual restore.
        assert tm.last_clean(env)["packages"] == {"a": "1.0"}

    def test_a_genuinely_new_problem_still_reports_inside_a_busy_window(self, tmp_path):
        env = _env(tmp_path)
        for i in range(tm.MAX_RECENT + 3):
            tm.record(env, _inv({"a": f"2.{i}"}), [_problem("p.always")])
        f = tm.what_changed_finding(env, _inv({"a": "9.0"}),
                                    [_problem("p.always"), _problem("p.new")])
        assert f is not None
        assert f.evidence["new_problems"] == ["p.new"]

    def test_silent_with_no_history(self, tmp_path):
        env = _env(tmp_path)
        assert tm.what_changed_finding(env, _inv({"a": "1.0"}), [_problem("p")]) is None

    def test_silent_while_healthy(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"a": "1.0"}), CLEAN)
        assert tm.what_changed_finding(env, _inv({"a": "2.0"}), CLEAN) is None


class TestHumanReadableTime:
    """The journal stores ISO-8601 UTC; a user must never be shown it."""

    def test_relative_phrasing(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        assert tm.when(now.isoformat()).startswith("today at")
        assert tm.when((now - timedelta(days=1)).isoformat()).startswith("yesterday at")
        assert "3 days ago" in tm.when((now - timedelta(days=3)).isoformat())
        old = tm.when((now - timedelta(days=400)).isoformat())
        assert "T" not in old and "+00:00" not in old

    def test_garbage_and_missing_degrade_gracefully(self):
        assert tm.when(None) == "an earlier scan"
        assert tm.when("not a date") == "not a date"

    def test_no_iso_timestamp_reaches_the_user(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"a": "1.0"}), CLEAN)
        f = tm.what_changed_finding(env, _inv({"a": "2.0"}), [_problem("p")])
        shown = " ".join([f.title, f.detail, f.impact, f.remedy.title,
                          f.remedy.explain, f.remedy.danger])
        assert "+00:00" not in shown and "T0" not in shown and "T1" not in shown


class TestNonPackageBreakage:
    def test_identical_packages_blames_the_environment_not_pip(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"torch": "2.9.1+cu130"}), CLEAN)
        f = tm.what_changed_finding(
            env, _inv({"torch": "2.9.1+cu130"}), [_problem("torch.cuda_unavailable")])
        assert f is not None
        assert f.id == "env.broke_without_package_changes"
        assert f.remedy is None, "there is nothing safe to reinstall here"
        assert "driver" in f.impact.lower()

    def test_finding_survives_repeated_scans_of_the_broken_state(self, tmp_path):
        # Regression guard: recording the broken scan must NOT overwrite the
        # clean same-packages entry, or this finding would show exactly once
        # and then go silent while the problem persists.
        env = _env(tmp_path)
        inv = _inv({"torch": "2.9.1+cu130"})
        problem = [_problem("torch.cuda_unavailable")]
        tm.record(env, inv, CLEAN)
        for _ in range(3):
            f = tm.what_changed_finding(env, inv, problem)
            assert f is not None
            assert f.id == "env.broke_without_package_changes"
            tm.record(env, inv, problem)

    def test_empty_inventory_is_a_failed_probe_not_a_mass_removal(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"torch": "2.9.1+cu130", "numpy": "1.26.4"}), CLEAN)
        f = tm.what_changed_finding(env, _inv({}), [_problem("p")])
        assert f is None, "diffing against a failed probe would offer a full reinstall"


class TestRestore:
    def _broken(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"torch": "2.9.1+cu130", "einops": "0.8.0"}), CLEAN)
        return env, tm.what_changed_finding(
            env, _inv({"torch": "2.9.1", "insightface": "0.7.3"}), [_problem("torch.cpu")])

    def test_torch_returns_via_its_own_index(self, tmp_path):
        _env_, f = self._broken(tmp_path)
        torch_cmd = " ".join(f.remedy.commands[0])
        assert "torch==2.9.1" in torch_cmd
        assert "download.pytorch.org/whl/cu130" in torch_cmd

    def test_removed_package_is_reinstalled_pinned(self, tmp_path):
        _env_, f = self._broken(tmp_path)
        assert any("einops==0.8.0" in " ".join(c) for c in f.remedy.commands)

    def test_never_uninstalls_and_never_touches_additions(self, tmp_path):
        _env_, f = self._broken(tmp_path)
        flat = " ".join(" ".join(c) for c in f.remedy.commands)
        assert "uninstall" not in flat
        assert "insightface" not in flat
        assert "left in place" in f.remedy.explain

    def test_additions_only_means_nothing_to_run(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"numpy": "1.26.4"}), CLEAN)
        f = tm.what_changed_finding(
            env, _inv({"numpy": "1.26.4", "newpkg": "1.0.0"}), [_problem("p")])
        assert f is not None            # the diff is still worth showing
        assert f.remedy is None         # but there is nothing safe to run

    def test_untagged_torch_says_so_instead_of_pretending(self, tmp_path):
        env = _env(tmp_path)
        tm.record(env, _inv({"torch": "2.9.1"}), CLEAN)      # conda build, no local tag
        f = tm.what_changed_finding(env, _inv({"torch": "2.8.0"}), [_problem("p")])
        assert "No CUDA build tag was recorded" in f.remedy.explain
