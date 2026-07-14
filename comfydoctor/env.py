"""Where are we, and which Python is this really?

Getting this wrong is the single most common reason a "just run pip install x"
answer fails a ComfyUI user: they paste it into a terminal whose `pip` belongs
to a completely different Python than the one ComfyUI is running on. Every
command ComfyDoctor emits is built from `Environment.pip_argv()`, so it always
targets the interpreter that is actually executing this code.
"""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Environment:
    python_exe: str
    python_version: str
    kind: str                    # "embedded" | "venv" | "conda" | "system"
    kind_detail: str
    comfy_root: Path | None
    custom_nodes_dir: Path | None
    site_dirs: list[str] = field(default_factory=list)
    is_windows: bool = False
    platform_tag: str = ""

    def pip_argv(self, *args: str) -> list[str]:
        """Build a pip command that provably targets THIS interpreter.

        The `-s` on embedded installs matters: ComfyUI's portable python_embeded
        will otherwise happily pick up packages from a user site-packages dir
        belonging to a system Python, which produces the "I installed it and it
        still says not found" class of bug.
        """
        argv = [self.python_exe]
        if self.kind == "embedded":
            argv.append("-s")
        argv += ["-m", "pip", *args]
        return argv

    def to_dict(self) -> dict:
        return {
            "python_exe": self.python_exe,
            "python_version": self.python_version,
            "kind": self.kind,
            "kind_detail": self.kind_detail,
            "comfy_root": str(self.comfy_root) if self.comfy_root else None,
            "custom_nodes_dir": str(self.custom_nodes_dir) if self.custom_nodes_dir else None,
            "site_dirs": self.site_dirs,
            "is_windows": self.is_windows,
            "platform_tag": self.platform_tag,
            "pip_prefix": " ".join(self.pip_argv()),
        }


def detect() -> Environment:
    exe = Path(sys.executable)
    kind, detail = _classify(exe)
    comfy_root = find_comfy_root()
    return Environment(
        python_exe=str(exe),
        python_version=platform.python_version(),
        kind=kind,
        kind_detail=detail,
        comfy_root=comfy_root,
        custom_nodes_dir=(comfy_root / "custom_nodes") if comfy_root else None,
        site_dirs=_site_dirs(),
        is_windows=os.name == "nt",
        platform_tag=sysconfig.get_platform(),
    )


def _classify(exe: Path) -> tuple[str, str]:
    parts = {p.lower() for p in exe.parts}

    # ComfyUI portable ships python_embeded (sic - the typo is upstream's).
    if "python_embeded" in parts or "python_embedded" in parts:
        return "embedded", f"ComfyUI portable embedded Python at {exe.parent}"

    # An embedded distribution has a python3xx._pth file next to the exe and no
    # ensurepip. That is the real signal; the folder name is just a convention.
    if list(exe.parent.glob("python*._pth")):
        return "embedded", f"Embedded Python distribution at {exe.parent}"

    if os.environ.get("CONDA_PREFIX"):
        return "conda", f"Conda env '{os.environ.get('CONDA_DEFAULT_ENV', '?')}' at {os.environ['CONDA_PREFIX']}"

    # sys.prefix != sys.base_prefix is the canonical venv/virtualenv test.
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return "venv", f"Virtual environment at {sys.prefix}"

    return "system", f"System-wide Python at {exe}"


def _site_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        import site

        for d in site.getsitepackages():
            if d not in dirs:
                dirs.append(d)
        if site.ENABLE_USER_SITE:
            u = site.getusersitepackages()
            if u and u not in dirs:
                dirs.append(u)
    except Exception:
        pass
    purelib = sysconfig.get_paths().get("purelib")
    if purelib and purelib not in dirs:
        dirs.insert(0, purelib)
    return dirs


def find_comfy_root() -> Path | None:
    """Locate the ComfyUI root, working both in-process and standalone.

    In-process we can just ask the folder_paths module. From the CLI we walk up
    from this file, since we live at <root>/custom_nodes/<us>/comfydoctor/env.py.
    """
    mod = sys.modules.get("folder_paths")
    base = getattr(mod, "base_path", None) if mod else None
    if base and Path(base).exists():
        return Path(base)

    here = Path(__file__).resolve()
    for parent in here.parents:
        if _looks_like_comfy_root(parent):
            return parent

    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if _looks_like_comfy_root(p):
            return p
    return None


def _looks_like_comfy_root(p: Path) -> bool:
    return (p / "custom_nodes").is_dir() and (
        (p / "main.py").is_file() or (p / "comfy").is_dir() or (p / "nodes.py").is_file()
    )


def anonymize(text: str) -> str:
    """Strip identity out of paths so a report can be pasted into a public issue.

    Windows users' reports are full of C:\\Users\\<realname>\\... - people either
    hand-edit it out or, more often, just paste it and leak their name.
    """
    if not text:
        return text
    out = text
    home = str(Path.home())
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    for needle, mask in ((home, "<HOME>"),):
        if needle:
            out = out.replace(needle, mask).replace(needle.replace("\\", "/"), mask)
    if user and len(user) > 2:
        out = out.replace(user, "<USER>")
    return out
