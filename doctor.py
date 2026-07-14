#!/usr/bin/env python
"""Standalone launcher. Run this when ComfyUI will not start.

    python doctor.py                 # diagnose
    python doctor.py --markdown      # a report you can paste into an issue
    python doctor.py --fix <id>      # apply one fix

Why this file exists instead of just `python -m comfydoctor`:

ComfyUI portable ships an *embedded* Python, whose python3xx._pth file replaces
sys.path entirely and does not include the current directory. `-m comfydoctor`
therefore fails with "No module named comfydoctor" no matter where you cd to.
Running a script by path always puts that script's own directory on sys.path
first, which is the one invocation that works on every kind of install -
portable, venv, conda, or system.

On a ComfyUI portable install:

    python_embeded\\python.exe ComfyUI\\custom_nodes\\Kurdknight_comfycheck\\doctor.py

or just double-click comfydoctor.bat next to this file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfydoctor.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
