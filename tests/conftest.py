"""Test bootstrap: force the shipped-version resolver offline and onto a
throwaway cache so every test run is deterministic (baked snapshot only) and
never touches pypi.org or a cache left by a previous live run."""

import os
import sys
import tempfile
from pathlib import Path

os.environ["COMFYDOCTOR_NO_NETWORK"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfydoctor import shipped  # noqa: E402

shipped.CACHE_FILE = os.path.join(tempfile.mkdtemp(prefix="comfydoctor_test_"),
                                  "shipped_cache.json")
shipped.clear_caches()
