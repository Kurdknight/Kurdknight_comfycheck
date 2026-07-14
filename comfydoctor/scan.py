"""Run every probe, then every rule. This is the only entry point anyone needs."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from . import custom_nodes, env, gpu, inventory
from .models import ScanResult, health_score
from .rules import Context, run_all

# The last scan is kept so that /fix can look up a remedy *by finding id*.
# The browser never sends us a command to run - it sends an id, and we execute
# only the command we ourselves generated for it. That's the whole security
# model, and it's why there is no way to turn this panel into a remote shell.
_LAST: ScanResult | None = None
_LAST_CTX: Context | None = None


def scan() -> ScanResult:
    global _LAST, _LAST_CTX
    t0 = time.perf_counter()

    e = env.detect()
    g = gpu.probe()
    inv = inventory.build()
    nodes = custom_nodes.survey(e.custom_nodes_dir)

    ctx = Context(env=e, gpu=g, inv=inv, nodes=nodes)
    findings = run_all(ctx)

    snapshot = {
        "environment": e.to_dict(),
        "gpu": g.to_dict(),
        "packages": inv.to_dict(),
        "custom_nodes": nodes.to_dict(),
    }

    result = ScanResult(
        findings=findings,
        snapshot=snapshot,
        health=health_score(findings),
        scanned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_ms=int((time.perf_counter() - t0) * 1000),
        comfy_runtime=ctx.comfy_runtime,
    )
    _LAST, _LAST_CTX = result, ctx
    return result


def last() -> ScanResult | None:
    return _LAST


def remedy_for(finding_id: str):
    """The only way a remedy ever gets executed: looked up from our own last scan."""
    if _LAST is None:
        return None
    for f in _LAST.findings:
        if f.id == finding_id and f.remedy and f.remedy.runnable and f.remedy.commands:
            return f.remedy
    return None
