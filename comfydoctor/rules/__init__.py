"""The rule engine.

A rule is a function that takes the Context (everything we probed) and yields
Findings. That's the whole contract. Rules never print, never fix, never import
the package they are judging - they look at data that was already gathered
safely and make a call.

Keeping them as small independent functions means each one can be tested against
a captured snapshot of a broken machine, without needing a broken machine.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, Iterable

from ..custom_nodes import NodeSurvey
from ..env import Environment
from ..gpu import GPUInfo
from ..inventory import Inventory
from ..models import Finding, Severity


@dataclass
class Context:
    env: Environment
    gpu: GPUInfo
    inv: Inventory
    nodes: NodeSurvey

    @property
    def comfy_runtime(self) -> bool:
        """True when we're running inside a live ComfyUI, not the CLI."""
        return self.nodes.runtime_known


Rule = Callable[[Context], Iterable[Finding]]

_RULES: list[tuple[str, Rule]] = []


def rule(fn: Rule) -> Rule:
    _RULES.append((fn.__name__, fn))
    return fn


def run_all(ctx: Context) -> list[Finding]:
    # Import for side effect: each module registers its rules on import.
    from . import attention, node_health, opportunities, packages, system, torch_stack  # noqa: F401

    findings: list[Finding] = []
    for name, fn in _RULES:
        try:
            findings.extend(fn(ctx) or [])
        except Exception:
            # A rule that crashes is a bug in ComfyDoctor, not in the user's
            # environment. Say so plainly rather than silently dropping a check
            # and letting them believe that area is healthy.
            findings.append(Finding(
                id=f"internal.rule_failed.{name}",
                severity=Severity.INFO,
                category="ComfyDoctor",
                title=f"Internal check '{name}' failed to run",
                detail=traceback.format_exc(limit=3),
                impact="That one check was skipped. Everything else in this report is still valid.",
            ))
    findings.sort(key=lambda f: (f.severity.rank, f.category, f.id))
    return findings


def rule_count() -> int:
    from . import attention, node_health, opportunities, packages, system, torch_stack  # noqa: F401

    return len(_RULES)
