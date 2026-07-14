"""The one surviving node.

The panel and the CLI are the product now. This exists for the minority who
genuinely want the report inside a graph - piping it into a text overlay, saving
it alongside a batch render, feeding it to an LLM node that explains it.

The two old nodes (SystemCheck, SystemViz) are aliased onto this one so that
workflows saved with the old version still open. Their data was wrong anyway:
the old code called importlib.import_module("opencv-python"), which can never
succeed, so it reported "Not installed" for packages that were installed. Anyone
relying on that output was relying on a bug.
"""

from __future__ import annotations

from . import comfydoctor
from .comfydoctor import report as report_mod


class ComfyDoctorReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (["summary", "markdown", "problems_only"], {"default": "summary"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report", "health")
    FUNCTION = "run"
    CATEGORY = "utils/ComfyDoctor"
    DESCRIPTION = (
        "Scans this ComfyUI environment for broken packages, version conflicts and failed "
        "custom nodes. For the interactive version with one-click fixes, open the Doctor tab "
        "in the sidebar."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The environment can change under us (someone pip-installs in another
        # window), so never let ComfyUI serve a cached result for this node.
        return float("nan")

    def run(self, format: str):
        result = comfydoctor.scan()

        if format == "markdown":
            text = report_mod.to_markdown(result, include_snapshot=False)
        elif format == "problems_only":
            text = _plain(result, problems_only=True)
        else:
            text = _plain(result, problems_only=False)

        return (text, result.health)


def _plain(result, problems_only: bool) -> str:
    from .comfydoctor.models import Severity

    lines = [
        f"ComfyDoctor - {result.health}/100 ({report_mod.health_label(result)})",
        "",
    ]
    for f in result.findings:
        if problems_only and f.severity in (Severity.OK, Severity.INFO):
            continue
        lines.append(f"[{f.severity.value.upper()}] {f.title}")
        if f.detail:
            lines.extend("    " + ln for ln in f.detail.splitlines())
        if f.remedy and f.remedy.commands:
            lines.append(f"    FIX: {f.remedy.as_shell()[0]}")
        lines.append("")
    return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "ComfyDoctorReport": ComfyDoctorReport,
    # Back-compat: old workflows referencing these keep loading. Both old nodes
    # collapse onto the new one, which is a superset of what either did.
    "SystemCheck": ComfyDoctorReport,
    "SystemViz": ComfyDoctorReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyDoctorReport": "ComfyDoctor Report",
    "SystemCheck": "ComfyDoctor Report (was: System Check)",
    "SystemViz": "ComfyDoctor Report (was: System Visualization)",
}
