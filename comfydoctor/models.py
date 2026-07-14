"""Core data types for ComfyDoctor.

Everything a rule produces is a Finding. Everything a Finding can do about
itself is a Remedy. Both are plain dataclasses so they serialise straight to
JSON for the panel, the CLI and the HTML report without a translation layer.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Ordered worst-to-best. `.rank` drives sorting and the health score."""

    CRITICAL = "critical"  # ComfyUI is broken or will crash. Fix before anything else.
    ERROR = "error"        # A feature is definitively broken (node won't load, import dies).
    WARNING = "warning"    # Works today, will bite you. Or: silently degraded (CPU torch).
    TIP = "tip"            # Nothing is wrong - but you could be faster, or fit bigger models.
    INFO = "info"          # Worth knowing. Not a problem.
    OK = "ok"              # Explicitly verified healthy. Shown so users can see what passed.

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    @property
    def weight(self) -> int:
        """Health-score penalty."""
        return _SEVERITY_WEIGHT[self]


_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.ERROR: 1,
    Severity.WARNING: 2,
    Severity.TIP: 3,
    Severity.INFO: 4,
    Severity.OK: 5,
}

# A TIP costs nothing. An opportunity you haven't taken is not a defect, and a
# tool that docks your score for declining an optional speed-up is nagging, not
# diagnosing. Keeping this line sharp is what stops a health score from decaying
# into noise people learn to ignore.
_SEVERITY_WEIGHT = {
    Severity.CRITICAL: 35,
    Severity.ERROR: 8,
    Severity.WARNING: 2,
    Severity.TIP: 0,
    Severity.INFO: 0,
    Severity.OK: 0,
}


@dataclass
class Remedy:
    """A concrete, runnable cure for one Finding.

    `commands` are argv lists, already resolved against the *actual* interpreter
    running ComfyUI - never a bare "pip install x" string, because on a portable
    install that would hit the wrong Python entirely. They are argv lists rather
    than shell strings so the runner never touches a shell.
    """

    title: str
    commands: list[list[str]] = field(default_factory=list)
    explain: str = ""
    restart_required: bool = True
    # Shown in the confirm dialog. Set it whenever the fix can make things worse.
    danger: str | None = None
    doc_url: str | None = None
    # False for remedies we can describe but must not execute (e.g. "edit this file
    # by hand", "reinstall your driver"). The panel shows the text, hides the button.
    runnable: bool = True

    def as_shell(self) -> list[str]:
        """Copy-pasteable form for the clipboard / CLI / issue reports."""
        return [_quote_argv(c) for c in self.commands]


@dataclass
class Finding:
    id: str                 # stable, e.g. "torch.triplet_mismatch". Used to look up remedies server-side.
    severity: Severity
    category: str           # groups the panel: "PyTorch", "Attention", "Custom nodes", ...
    title: str              # one line, plain language
    detail: str = ""        # what exactly is wrong, with the numbers
    impact: str = ""        # what the user will actually experience. The bit everyone omits.
    evidence: dict[str, Any] = field(default_factory=dict)
    remedy: Remedy | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class ScanResult:
    findings: list[Finding]
    snapshot: dict[str, Any]
    health: int             # 0-100
    scanned_at: str
    duration_ms: int
    comfy_runtime: bool     # False when run from the CLI outside ComfyUI
    # Grouped inventory for the Environment view (see facts.py)
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "snapshot": self.snapshot,
            "health": self.health,
            # Computed once, server-side, so the panel, the CLI and the HTML
            # report can never disagree about what the number means.
            "health_label": health_label(self.findings),
            "scanned_at": self.scanned_at,
            "duration_ms": self.duration_ms,
            "comfy_runtime": self.comfy_runtime,
            "counts": self.counts(),
            "facts": self.facts,
        }

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for f in self.findings:
            out[f.severity.value] += 1
        return out

    def worst(self) -> Severity:
        if not self.findings:
            return Severity.OK
        return min((f.severity for f in self.findings), key=lambda s: s.rank)


def health_score(findings: list[Finding]) -> int:
    """100 = clean. Each finding subtracts its weight, with diminishing returns.

    A flat per-finding penalty does not survive contact with a real machine. A
    working ComfyUI with 80 custom nodes routinely carries ~25 warnings (stale
    pins, unmet optional deps) while running perfectly - and a linear score sends
    that straight to 0/100 "Broken". Once every real install reads as broken, the
    number carries no information and people stop trusting the whole report.

    So warnings taper: the first few cost the full weight, the twentieth costs
    almost nothing. It's the presence of a critical that should sink the score,
    not the accumulation of nits.
    """
    penalty = 0.0
    counts: dict[Severity, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    for sev, n in counts.items():
        w = sev.weight
        if not w:
            continue
        # Harmonic taper: 1st costs w, 2nd w/2, 3rd w/3... Sums slowly, so ten
        # warnings hurt roughly three times as much as one, not ten times.
        penalty += sum(w / (i + 1) for i in range(n))

    return max(0, min(100, round(100 - penalty)))


def health_label(findings: list[Finding]) -> str:
    """The word next to the number - driven by the WORST thing found, not the
    arithmetic. "Broken" must mean "something is actually broken", or it is a
    lie the user will catch us in immediately."""
    sevs = {f.severity for f in findings}
    if Severity.CRITICAL in sevs:
        return "Broken"
    if Severity.ERROR in sevs:
        return "Needs attention"
    if Severity.WARNING in sevs:
        return "Minor issues"
    return "Healthy"


# Characters that make a shell do something other than pass the argument along.
# `<` and `>` matter enormously here: a version specifier like `numpy<3,>=2.0`
# is a perfectly ordinary pip argument, but pasted unquoted into a terminal it
# becomes a redirect and either truncates a file called `3,` or errors out. We
# run commands via argv (no shell at all), so this quoting exists purely to make
# the *copied* command safe - which is the path most users actually take.
_SHELL_META = set(' \t"\'<>|&;$()`*?[]{}#!~^%')


def _quote_argv(argv: list[str]) -> str:
    out = []
    for a in argv:
        if not a or any(ch in _SHELL_META for ch in a):
            out.append('"' + a.replace('"', '\\"') + '"')
        else:
            out.append(a)
    return " ".join(out)
