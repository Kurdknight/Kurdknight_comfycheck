"""The doctor that still works when the patient can't stand up.

    python -m comfydoctor

This is the point of the whole rewrite. The old version was a node - it could
only run inside a healthy ComfyUI. But a broken torch means ComfyUI never
finishes booting, which means the node never registers, which means the
diagnostic is unavailable at exactly the moment you need a diagnostic. It could
only ever tell you about environments that were already fine.

This runs standalone, off the same core and the same rules.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import report, runner
from .models import Severity
# Import the functions, not the module: the package __init__ re-exports `scan`
# as a function, which shadows the submodule of the same name.
from .scan import remedy_for
from .scan import scan as run_scan

_COLOR = {
    Severity.CRITICAL: "\033[1;31m",
    Severity.ERROR: "\033[0;31m",
    Severity.WARNING: "\033[0;33m",
    Severity.INFO: "\033[0;36m",
    Severity.OK: "\033[0;32m",
}
_GLYPH = {
    Severity.CRITICAL: "STOP",
    Severity.ERROR: "FAIL",
    Severity.WARNING: "WARN",
    Severity.INFO: "INFO",
    Severity.OK: " OK ",
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"

# Box-drawing and middle-dot are not in cp1252, which is still the default
# console encoding on a stock Windows Python - including the one ComfyUI
# portable ships. Printing them raises UnicodeEncodeError and takes the whole
# tool down. We ask for UTF-8, and if the console refuses, we degrade to ASCII
# rather than crash. A diagnostic tool that dies on its own output is not a
# diagnostic tool.
_RULE = "-"
_SEP = " | "


def _setup_encoding() -> None:
    global _RULE, _SEP
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    enc = (getattr(sys.stdout, "encoding", None) or "ascii").lower()
    if enc.replace("-", "") in ("utf8", "utf16", "utf32"):
        _RULE = "─"   # ─
        _SEP = " · "  # ·


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # Modern Windows Terminal / PowerShell handle ANSI; enable it explicitly
        # so we also work in the old conhost that ComfyUI portable launches with.
        try:
            import ctypes

            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="comfydoctor",
        description="Diagnose a ComfyUI Python environment. Works even when ComfyUI won't start.",
    )
    p.add_argument("--json", action="store_true", help="emit the full ScanResult as JSON")
    p.add_argument("--markdown", "-m", action="store_true",
                   help="emit an anonymized markdown report, ready to paste into an issue")
    p.add_argument("--html", metavar="PATH", help="write a self-contained HTML report to PATH")
    p.add_argument("--quiet", "-q", action="store_true", help="show only problems, hide passing checks")
    p.add_argument("--fix", metavar="FINDING_ID",
                   help="run the fix for one finding (use the id shown in brackets)")
    p.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt for --fix")
    args = p.parse_args(argv)

    _setup_encoding()
    color = _supports_color()

    if not args.json and not args.markdown:
        print("Examining your environment...", file=sys.stderr)

    result = run_scan()

    if args.json:
        import json

        print(json.dumps(result.to_dict(), indent=1, default=str))
        return _exit_code(result)

    if args.markdown:
        print(report.to_markdown(result))
        return _exit_code(result)

    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(report.to_html(result))
        print(f"Wrote {args.html}")
        return _exit_code(result)

    if args.fix:
        return _do_fix(args.fix, args.yes)

    _print_human(result, color, quiet=args.quiet)
    return _exit_code(result)


def _print_human(result, color: bool, quiet: bool) -> None:
    def c(s: str, code: str) -> str:
        return f"{code}{s}{_RESET}" if color else s

    label = report.health_label(result)
    tone = {"Healthy": Severity.OK, "Minor issues": Severity.WARNING,
            "Needs attention": Severity.ERROR, "Broken": Severity.CRITICAL}[label]

    print()
    print("  " + c(f"{result.health}/100  {label}", _BOLD + _COLOR[tone] if color else ""))

    counts = result.counts()
    bits = [f"{counts[s.value]} {s.value}" for s in
            (Severity.CRITICAL, Severity.ERROR, Severity.WARNING) if counts[s.value]]
    print("  " + c(_SEP.join(bits) or "no problems found", _DIM))
    if not result.comfy_runtime:
        print("  " + c("run from the panel inside ComfyUI to also catch node import failures", _DIM))
    print()

    cat = None
    for f in result.findings:
        if quiet and f.severity in (Severity.OK, Severity.INFO):
            continue
        if f.category != cat:
            cat = f.category
            print(c(f"  {_RULE * 2} {cat} " + _RULE * max(0, 60 - len(cat)), _DIM))

        print(f"  {c('[' + _GLYPH[f.severity] + ']', _COLOR[f.severity])} {f.title}")
        if f.detail:
            for line in f.detail.splitlines():
                print(c(f"        {line}", _DIM))
        if f.impact:
            print()
            for line in _wrap(f.impact, 72):
                print(f"        {line}")
        if f.remedy:
            print()
            print(f"        {c('FIX:', _COLOR[Severity.INFO])} {f.remedy.title}")
            for cmd in f.remedy.as_shell():
                print(c(f"          $ {cmd}", _BOLD if color else ""))
            if f.remedy.danger:
                print(c(f"          ! {f.remedy.danger}", _COLOR[Severity.WARNING]))
            if f.remedy.runnable:
                print(c(f"          run it: python -m comfydoctor --fix {f.id}", _DIM))
        print()

    if not quiet:
        print(c("  Full report:  python -m comfydoctor --markdown > report.md", _DIM))
        print()


def _do_fix(finding_id: str, assume_yes: bool) -> int:
    remedy = remedy_for(finding_id)
    if remedy is None:
        print(f"No runnable fix for '{finding_id}'.", file=sys.stderr)
        print("Run `python -m comfydoctor` and use an id from the [brackets].", file=sys.stderr)
        return 2

    print()
    print(f"  {remedy.title}")
    print()
    for cmd in remedy.as_shell():
        print(f"    $ {cmd}")
    print()
    if remedy.danger:
        print(f"  ! {remedy.danger}")
        print()

    if not assume_yes:
        try:
            if input("  Run this? [y/N] ").strip().lower() not in ("y", "yes"):
                print("  Cancelled.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 1

    job, err = runner.start(finding_id, remedy)
    if job is None:
        print(f"  {err}", file=sys.stderr)
        return 2

    seen = 0
    import time

    while True:
        snap = job.snapshot(seen)
        for line in snap["lines"]:
            print("  " + line)
        seen = snap["total_lines"]
        if snap["status"] not in ("pending", "running"):
            break
        time.sleep(0.2)

    return 0 if job.status == "success" else 1


def _exit_code(result) -> int:
    """0 = clean, 1 = warnings, 2 = errors/critical. Lets people gate a launch
    script on it: `python -m comfydoctor -q || echo "fix your env first"`."""
    worst = result.worst()
    if worst in (Severity.CRITICAL, Severity.ERROR):
        return 2
    if worst == Severity.WARNING:
        return 1
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    out: list[str] = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out
