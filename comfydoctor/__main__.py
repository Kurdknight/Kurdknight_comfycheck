"""Entry point for `python -m comfydoctor`.

Works from inside the custom_nodes folder, or from anywhere if you add the
folder to sys.path. On a ComfyUI portable install:

    python_embeded\\python.exe -s -m comfydoctor
"""

import sys
from pathlib import Path

# Allow `python -m comfydoctor` to be run from inside the package directory
# itself, which is what people will naturally try.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfydoctor.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
