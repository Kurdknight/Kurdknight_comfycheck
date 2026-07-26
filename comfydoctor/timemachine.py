"""The environment time machine: answers "it worked yesterday - what changed?"

Every scan already computes the full package inventory. This module journals
that state to disk together with the problems found, so that when a NEW problem
appears we can point at the exact package changes that came with it, and offer
a pinned way back.

The reference point is per-problem, not a global "healthy" flag. That
distinction is the whole design: a working install with dozens of custom nodes
very often carries one long-standing ERROR the user has consciously decided to
live with. A global "last scan with zero errors" reference would never be
recorded there, and the feature would be dead on precisely the machines that
need it. So instead we look for the most recent snapshot that did NOT have one
of today's problems - that snapshot is, by definition, from before this
regression.

Contracts this feature lives under (same as everything else here):

  * Facts only. A snapshot is the dist-info state at scan time; a diff is
    arithmetic between two snapshots. Nothing is guessed.
  * Non-destructive. The restore reinstalls changed/removed packages at their
    recorded versions. Packages ADDED since then are left alone - removing
    things is how working machines get broken, and the new package may belong
    to a node the user installed on purpose.
  * torch/torchvision/torchaudio are never reinstalled from bare PyPI (the
    CPU-wheel trap). Their restore carries the --index-url derived from the
    recorded build tag.

Storage: <comfy_root>/user/comfydoctor/env_journal.json - ComfyUI's user-data
convention, so it survives extension updates. With no comfy_root (bare CLI in
an unusual layout) the feature quietly does nothing.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .env import Environment
from .inventory import Inventory
from .models import Finding, Remedy, Severity

MAX_RECENT = 20          # rolling window of snapshots kept
MAX_SHOWN = 12           # diff lines shown before "...and N more"
TORCH_FAMILY = ("torch", "torchvision", "torchaudio")
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
CAT = "What changed"


# --------------------------------------------------------------------------- #
# Journal I/O
# --------------------------------------------------------------------------- #

def journal_path(env: Environment) -> Path | None:
    if not env.comfy_root:
        return None
    return Path(env.comfy_root) / "user" / "comfydoctor" / "env_journal.json"


def _load(path: Path | None) -> dict:
    """A journal that can't be read is an empty journal, never a crash."""
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(path: Path | None, journal: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(journal, f)
        os.replace(tmp, str(path))
    except Exception:
        pass  # a journal that can't be written is just a missing snapshot


def _packages_of(inv: Inventory) -> dict[str, str]:
    return {name: d.version for name, d in inv.dists.items()}


def when(ts: str | None) -> str:
    """An ISO timestamp as something a human reads without decoding it.

    The journal stores ISO-8601 UTC because that is unambiguous to compare and
    survives timezone changes. Nobody should ever SEE that: the whole point of
    this feature is a person asking "what changed since yesterday", so the
    answer says "yesterday at 14:03" in their own local time.
    """
    if not ts:
        return "an earlier scan"
    try:
        from datetime import datetime as _dt

        moment = _dt.fromisoformat(ts).astimezone()
        now = _dt.now().astimezone()
        days = (now.date() - moment.date()).days
        clock = moment.strftime("%H:%M")
        if days <= 0:
            return f"today at {clock}"
        if days == 1:
            return f"yesterday at {clock}"
        if days < 7:
            return f"{days} days ago ({moment.strftime('%a %d %b')} at {clock})"
        if moment.year == now.year:
            return moment.strftime("%d %b at %H:%M")
        return moment.strftime("%d %b %Y")
    except Exception:
        return ts


def problem_ids(findings: list[Finding]) -> list[str]:
    """The things that are actually broken right now. Warnings are excluded on
    purpose: a real install always carries a handful, and treating them as
    regressions would make every scan look like a disaster."""
    return sorted(
        f.id for f in findings if f.severity in (Severity.CRITICAL, Severity.ERROR)
    )


def _entries(journal: dict) -> list[dict]:
    return [e for e in journal.get("recent", []) if isinstance(e, dict)
            and isinstance(e.get("packages"), dict)]


def record(env: Environment, inv: Inventory, findings: list[Finding]) -> None:
    """Journal this scan. Called at the END of a scan, after the what-changed
    finding has been computed against the PREVIOUS journal state."""
    path = journal_path(env)
    if path is None or not inv.dists:
        return

    journal = _load(path)
    packages = _packages_of(inv)
    problems = problem_ids(findings)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packages": packages,
        "problems": problems,
    }

    recent = _entries(journal)
    # Back-to-back scans of an unchanged environment just refresh the newest
    # entry, so idle re-scans can't flush real history out of the window.
    # Both packages AND problems must match: when a problem appears with no
    # package change (driver update, edited launch script), the previous entry
    # is the only proof the machine once ran this exact package set cleanly -
    # overwriting it would silence broke_without_package_changes after one scan.
    if recent and recent[-1].get("packages") == packages \
            and recent[-1].get("problems") == problems:
        recent[-1] = entry
    else:
        recent.append(entry)
    journal["recent"] = recent[-MAX_RECENT:]

    # The last fully clean state is pinned separately from the rolling window.
    # Nothing consumes it automatically (see reference_point on why the diff
    # must stay inside the retained window); it is the durable "this machine
    # was perfect on <date>, here is exactly what it looked like" record, which
    # is what the user is really asking for when they restore by hand.
    if not problems:
        journal["last_clean"] = entry

    _save(path, journal)


def last_clean(env: Environment) -> dict | None:
    entry = _load(journal_path(env)).get("last_clean")
    if isinstance(entry, dict) and isinstance(entry.get("packages"), dict):
        return entry
    return None


# --------------------------------------------------------------------------- #
# Finding the moment a problem appeared
# --------------------------------------------------------------------------- #

def reference_point(env: Environment, current: list[str]) -> tuple[dict | None, list[str]]:
    """The newest snapshot from BEFORE one of today's problems existed.

    Returns (entry, new_problem_ids). `new_problem_ids` are the problems that
    exist now and did not exist in that snapshot - i.e. what regressed.

    Searching newest-first means we land on the tightest window around the
    regression, so the package diff stays small and readable rather than
    dragging in months of unrelated churn.

    If EVERY retained snapshot already had all of today's problems, we return
    nothing and the feature stays quiet. We deliberately do not fall back to
    the pinned clean snapshot: a problem present in every scan we kept is not
    news, and answering it with a months-wide package diff would be both
    unhelpful and a claim we cannot support - we no longer hold the evidence
    for when it actually appeared.
    """
    if not current:
        return None, []
    for entry in reversed(_entries(_load(journal_path(env)))):
        had = set(entry.get("problems") or [])
        new = [p for p in current if p not in had]
        if new:
            return entry, new
    return None, []


def diff(old: dict[str, str], new: dict[str, str]) -> dict:
    """Arithmetic between two snapshots. Sorted, so output is stable."""
    changed = [(n, old[n], new[n]) for n in sorted(old) if n in new and old[n] != new[n]]
    removed = [(n, old[n]) for n in sorted(old) if n not in new]
    added = [(n, new[n]) for n in sorted(new) if n not in old]
    return {"changed": changed, "removed": removed, "added": added}


def what_changed_finding(
    env: Environment, inv: Inventory, findings: list[Finding],
) -> Finding | None:
    """When a problem is NEW since a recorded snapshot, show what changed with
    it - and offer the way back."""
    current = problem_ids(findings)
    if not current:
        return None  # nothing broken: the journal stays silent, per contract
    if not inv.dists:
        # A probe that saw zero packages is a failed probe, not a machine with
        # everything uninstalled. record() refuses to journal such a state for
        # the same reason; diffing against it would offer a full reinstall.
        return None

    entry, new_problems = reference_point(env, current)
    if entry is None or not new_problems:
        return None  # no history, or nothing has regressed since we've been watching

    d = diff(entry["packages"], _packages_of(inv))
    n_changes = len(d["changed"]) + len(d["removed"]) + len(d["added"])
    if not n_changes:
        # Something broke without any package moving - a driver update, an
        # edited launch script, a disk problem. Say that plainly instead of
        # implying packages are to blame.
        return Finding(
            id="env.broke_without_package_changes",
            severity=Severity.INFO,
            category=CAT,
            title="This problem is new, but no package has changed",
            detail=(
                f"Every installed package is exactly as it was {when(entry.get('ts'))}, when this "
                f"problem did not exist. Nothing was installed, removed or upgraded in between."
            ),
            impact=(
                "So the cause is outside pip: a GPU driver update, a change to your launch "
                "script or environment variables, a moved or deleted folder, or a disk issue. "
                "Reinstalling packages will not help here - do not start uninstalling things."
            ),
            evidence={"reference": entry.get("ts"), "new_problems": new_problems},
        )

    titles = {f.id: f.title for f in findings}
    ago = when(entry.get("ts"))

    lines = [f"Your last scan without this problem was {ago}. New since then:"]
    for pid in new_problems[:5]:
        lines.append(f"  ! {titles.get(pid, pid)}")
    lines.append("")
    lines.append("What changed on your machine in between:")
    shown = 0
    for n, old_v, new_v in d["changed"]:
        if shown >= MAX_SHOWN:
            break
        lines.append(f"  ~ {n}: {old_v} -> {new_v}")
        shown += 1
    for n, old_v in d["removed"]:
        if shown >= MAX_SHOWN:
            break
        lines.append(f"  - {n} {old_v} (removed)")
        shown += 1
    for n, new_v in d["added"]:
        if shown >= MAX_SHOWN:
            break
        lines.append(f"  + {n} {new_v} (newly installed)")
        shown += 1
    if n_changes > shown:
        lines.append(f"  ...and {n_changes - shown} more")

    thing = "package" if n_changes == 1 else "packages"
    return Finding(
        id="env.changed_since_working",
        severity=Severity.INFO,
        category=CAT,
        title=f"It worked {ago} — {n_changes} {thing} changed since then",
        detail="\n".join(lines),
        impact=(
            "This is everything that changed on your machine between the last scan where "
            "this problem did not exist and now, so the cause is very likely in this list. "
            "Fixing the specific findings above is usually the better route. If you would "
            "rather just go back to what worked, the button below reinstalls these packages "
            "at the versions they had then - it does not uninstall anything."
        ),
        evidence={"reference": entry.get("ts"), "new_problems": new_problems, "diff": d},
        remedy=_restore_remedy(env, d, ago),
    )


def _restore_remedy(env: Environment, d: dict, ago: str) -> Remedy | None:
    """Pinned reinstall of everything that changed or disappeared. Packages
    added since the reference point are deliberately left installed."""
    torch_pins: list[str] = []
    other_pins: list[str] = []
    tag: str | None = None

    reverts = [(n, old_v) for n, old_v, _new in d["changed"]] + list(d["removed"])
    for name, old_v in reverts:
        base = old_v.split("+", 1)[0]
        if name in TORCH_FAMILY:
            torch_pins.append(f"{name}=={base}")
            if "+" in old_v:
                tag = old_v.split("+", 1)[1]
        else:
            other_pins.append(f"{name}=={base}")

    if not torch_pins and not other_pins:
        return None  # only additions - nothing to restore, and we never uninstall

    commands: list[list[str]] = []
    if torch_pins:
        argv = env.pip_argv("install", *torch_pins)
        if tag:
            argv += ["--index-url", TORCH_INDEX.format(tag=tag)]
        commands.append(argv)
    if other_pins:
        commands.append(env.pip_argv("install", *other_pins))

    added_note = ""
    if d["added"]:
        names = ", ".join(n for n, _ in d["added"][:6])
        more = f" (+{len(d['added']) - 6} more)" if len(d["added"]) > 6 else ""
        added_note = (
            f"\n\nPackages installed since then ({names}{more}) are left in place - they "
            f"probably belong to a node you added on purpose. If the problem survives the "
            f"restore, one of them is the next thing to look at."
        )

    torch_note = ""
    if torch_pins:
        torch_note = (
            f" torch/torchvision/torchaudio go back through the PyTorch index for their "
            f"recorded build ({tag}), never bare PyPI, so a CUDA build cannot be replaced "
            f"by a CPU one."
            if tag else
            " No CUDA build tag was recorded for torch, so it is reinstalled from the "
            "default index - check the PyTorch findings above afterwards."
        )

    n = len(reverts)
    return Remedy(
        title=f"Put these {n} package{'' if n == 1 else 's'} back to how they were {ago}",
        commands=commands,
        explain=(
            f"Reinstalls every package that changed or was removed since {ago}, at the exact "
            f"version it had then.{torch_note}{added_note}"
        ),
        danger=(
            f"This changes {n} package{'' if n == 1 else 's'} back to the version"
            f"{'' if n == 1 else 's'} they had {ago}. Nothing is uninstalled. If you installed a "
            f"new custom node since then, it might want the newer versions - if it complains "
            f"afterwards, run the scan again and read what it says before changing anything else."
        ),
    )
