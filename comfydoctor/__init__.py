"""ComfyDoctor - diagnose and repair a ComfyUI Python environment.

Three surfaces, one core:
  * the sidebar panel inside ComfyUI  (api.py + ../web/)
  * `python -m comfydoctor` from a terminal, which works even when ComfyUI won't
    start - the case the old node could never cover  (cli.py)
  * a single node that outputs the report as a STRING, for people who want it in
    a graph  (../nodes.py)

Nothing here imports torch, numpy or any package it is judging. Everything is
read from importlib.metadata and from nvidia-smi. That is deliberate: importing
a package compiled against the wrong torch does not raise, it aborts the
process, and a diagnostic that kills the thing it is diagnosing is worthless.
"""

from .models import Finding, Remedy, ScanResult, Severity
from .scan import last, scan

__version__ = "2.0.0"

__all__ = ["scan", "last", "Finding", "Remedy", "ScanResult", "Severity", "__version__"]
