"""The advice-safety contract: on a working machine, ComfyDoctor must never
hand out advice that can break a running ComfyUI.

Rules under test:
  * contested modules the tool admits it doesn't understand -> never
    "uninstall all of them"; do-nothing default, force-reinstall escalation.
  * same-version contested pair (filterpy/filterpywhl) -> INFO, harmless.
  * OpenCV pile-up -> keep the contrib SUPERSET, never plain over contrib.
  * onnxruntime keeper on a pre-release build -> warn that the fix installs
    stable and may lose deliberate nightly CUDA support.
  * shadowed installs -> the one-click fix must put the package back, and
    torch-family goes back via the PyTorch index, never bare PyPI.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfydoctor.custom_nodes import NodeSurvey          # noqa: E402
from comfydoctor.env import Environment                  # noqa: E402
from comfydoctor.gpu import GPUInfo                      # noqa: E402
from comfydoctor.inventory import Dist, Inventory        # noqa: E402
from comfydoctor.models import Severity                  # noqa: E402
from comfydoctor.rules import Context                    # noqa: E402
from comfydoctor.rules import packages as pkg_rules      # noqa: E402


def _env() -> Environment:
    env = Environment.__new__(Environment)
    env.is_windows = True
    env.python_exe = "C:/x/python.exe"
    env.kind = "venv"
    return env


def _gpu_nvidia() -> GPUInfo:
    g = GPUInfo()
    g.nvidia_smi_ok = True
    g.devices = [{"name": "RTX 4090", "driver": "610.47"}]
    g.driver_cuda_version = "13.0"
    return g


def _dist(name, version, location="/site"):
    return Dist(name=name, raw_name=name, version=version, location=location,
                modules=[name], owned_modules=[name])


def _ctx(dists, module_owners=None, duplicates=None, gpu=None) -> Context:
    inv = Inventory(
        dists={d.name: d for d in dists},
        duplicates=duplicates or {},
        module_owners=module_owners or {},
        unsatisfied=[],
    )
    return Context(env=_env(), gpu=gpu or GPUInfo(), inv=inv, nodes=NodeSurvey())


class TestContestedModules:
    def test_same_version_pair_is_info_and_hands_off(self):
        # filterpy 1.4.5 vs filterpywhl 1.4.5: a re-published wheel of the same
        # code. Must be INFO and must not carry any uninstall advice.
        ctx = _ctx(
            [_dist("filterpy", "1.4.5"), _dist("filterpywhl", "1.4.5")],
            module_owners={"filterpy": ["filterpy", "filterpywhl"]},
        )
        found = list(pkg_rules.contested_module_names(ctx))
        assert len(found) == 1
        f = found[0]
        assert f.severity is Severity.INFO
        assert f.remedy is None
        assert "uninstall all" not in (f.impact or "").lower()

    def test_different_versions_never_say_uninstall_all(self):
        # urllib3 vs urllib3-future: unknown pair, nothing observably broken.
        # The advice must be do-nothing + a force-reinstall escalation path,
        # never "uninstall ALL of them".
        ctx = _ctx(
            [_dist("urllib3", "2.6.3"), _dist("urllib3-future", "2.17.900")],
            module_owners={"urllib3": ["urllib3", "urllib3-future"]},
        )
        found = list(pkg_rules.contested_module_names(ctx))
        assert len(found) == 1
        f = found[0]
        assert f.severity is Severity.WARNING
        text = (f.remedy.explain + f.remedy.title).lower()
        assert "uninstall all" not in text
        assert "do nothing" in text
        assert "--force-reinstall" in f.remedy.explain
        assert f.remedy.commands == []          # nothing runnable, by design


class TestOpencvKeepChoice:
    def test_contrib_superset_wins_over_plain(self):
        # contrib contains everything plain has PLUS the extras that
        # controlnet_aux etc. import - keeping plain would break them.
        ctx = _ctx([
            _dist("opencv-python", "4.13.0.92"),
            _dist("opencv-python-headless", "4.13.0.92"),
            _dist("opencv-contrib-python", "4.13.0.92"),
        ])
        found = list(pkg_rules.opencv_pileup(ctx))
        assert len(found) == 1
        assert "opencv-contrib-python" in found[0].remedy.title
        install_cmd = found[0].remedy.commands[-1]
        assert install_cmd[-1] == "opencv-contrib-python"

    def test_plain_wins_over_headless(self):
        ctx = _ctx([
            _dist("opencv-python", "4.13.0.92"),
            _dist("opencv-python-headless", "4.13.0.92"),
        ])
        found = list(pkg_rules.opencv_pileup(ctx))
        assert "opencv-python" in found[0].remedy.title
        assert "headless" not in found[0].remedy.title


class TestOnnxruntimePrereleaseKeeper:
    def test_prerelease_keeper_warns_about_stable_downgrade(self):
        # onnxruntime-gpu 1.25.0.devXXXX was installed from a nightly index on
        # purpose (new CUDA). The fix installs stable from PyPI - it must SAY so.
        ctx = _ctx(
            [_dist("onnxruntime", "1.24.2"),
             _dist("onnxruntime-gpu", "1.25.0.dev20260307001")],
            gpu=_gpu_nvidia(),
        )
        found = list(pkg_rules.onnxruntime_pileup(ctx))
        assert len(found) == 1
        assert "PRE-RELEASE" in found[0].remedy.danger
        assert "CUDAExecutionProvider" in found[0].remedy.danger

    def test_stable_keeper_keeps_the_short_danger(self):
        ctx = _ctx(
            [_dist("onnxruntime", "1.24.2"), _dist("onnxruntime-gpu", "1.25.0")],
            gpu=_gpu_nvidia(),
        )
        found = list(pkg_rules.onnxruntime_pileup(ctx))
        assert "PRE-RELEASE" not in found[0].remedy.danger


class TestVersionMoveRespectsBystanders:
    """The pillow/moviepy case: a fix that moves a shared library must carry
    EVERY installed package's declared pin on it, so pip refuses outright
    rather than satisfying one package by breaking another. And on a working
    machine the danger text must say that doing nothing is a valid choice."""

    def _pillow_ctx(self):
        dists = [
            _dist("pillow", "12.1.0"),
            _dist("moviepy", "2.2.1"),
            _dist("qrcode", "8.0"),
        ]
        # qrcode's pin is satisfied today - it must STILL end up in the command.
        dists[2].requires = ["pillow>=9.0"]
        inv = Inventory(
            dists={d.name: d for d in dists},
            duplicates={}, module_owners={},
            unsatisfied=[{
                "dist": "moviepy", "requirement": "pillow<12.0,>=9.2.0",
                "target": "pillow", "specifier": "<12.0,>=9.2.0",
                "installed": "12.1.0",
                "reason": "pillow 12.1.0 is installed, but moviepy requires <12.0,>=9.2.0",
            }],
        )
        return Context(env=_env(), gpu=GPUInfo(), inv=inv, nodes=NodeSurvey())

    def test_bystander_pins_join_the_install_spec(self):
        found = [f for f in pkg_rules.broken_dependencies(self._pillow_ctx())
                 if f.id == "packages.unsatisfied.pillow"]
        assert len(found) == 1
        arg = found[0].remedy.commands[0][-1]
        assert arg.startswith("pillow")
        assert "<12.0" in arg          # the complainer's ceiling
        assert ">=9.0" in arg          # the satisfied bystander's floor, kept

    def test_danger_says_doing_nothing_is_valid(self):
        found = [f for f in pkg_rules.broken_dependencies(self._pillow_ctx())
                 if f.id == "packages.unsatisfied.pillow"]
        d = found[0].remedy.danger or ""
        assert "doing nothing" in d.lower()
        assert "moviepy" in d


class TestNoOpinionAsFact:
    """Phrases that once presented opinion or invented numbers as fact must
    never reappear anywhere in the package. This is the systemic guard: every
    user-visible sentence lives in these source files, so a static scan covers
    advice for hardware and situations no single test machine can reproduce."""

    BANNED = [
        "is enough for almost everyone",     # SDPA opinion (false for video workloads)
        "enough for most people",
        "lose almost nothing",               # uninstall-a-2x-speedup framing
        "uninstall ALL of them",             # destructive advice for unknown pairs
        "20-50x slower",                     # invented precision
        "becomes 30x slower",
        "typically 20-30%",                  # undersold SageAttention
        "autotuning for a few percent",      # the confused cuDNN/--cuda-malloc tip
    ]

    def test_banned_phrases_absent_from_all_sources(self):
        root = Path(__file__).resolve().parent.parent / "comfydoctor"
        offenders = []
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="replace")
            for phrase in self.BANNED:
                if phrase in text:
                    offenders.append(f"{py.name}: {phrase!r}")
        js = root.parent / "web" / "comfydoctor.js"
        if js.is_file():
            text = js.read_text(encoding="utf-8", errors="replace")
            for phrase in self.BANNED:
                if phrase in text:
                    offenders.append(f"comfydoctor.js: {phrase!r}")
        assert not offenders, "opinion-as-fact phrasing crept back in:\n" + "\n".join(offenders)


class TestSageEnablementIsChecked:
    """Installing SageAttention does nothing without --use-sage-attention.
    Whether the flag is set is verifiable from inside ComfyUI (sys.argv), and
    the tip must fire exactly on installed-but-off."""

    def _ctx(self, sage=True, runtime=True):
        import comfydoctor.rules.opportunities as opp  # noqa: F401 - registers rules

        dists = [_dist("torch", "2.9.1+cu130")]
        if sage:
            dists.append(_dist("sageattention", "2.2.0"))
        gpu = _gpu_nvidia()
        gpu.torch_ok = True
        gpu.cuda_available = True
        gpu.torch_devices = [{"name": "RTX 4090", "compute_capability": "8.9",
                              "vram_total_mb": 24564}]
        ctx = _ctx_base = _ctx_build(dists, gpu)
        ctx.nodes.runtime_known = runtime
        return ctx

    def test_installed_but_flag_missing_fires(self):
        import sys
        from unittest.mock import patch

        from comfydoctor.rules import opportunities

        with patch.object(sys, "argv", ["main.py", "--listen"]):
            found = list(opportunities.sage_installed_but_off(self._ctx()))
        assert [f.id for f in found] == ["tip.sageattention_not_enabled"]
        assert found[0].remedy.commands == []   # advice only, nothing destructive

    def test_flag_present_stays_silent(self):
        import sys
        from unittest.mock import patch

        from comfydoctor.rules import opportunities

        with patch.object(sys, "argv", ["main.py", "--use-sage-attention"]):
            assert list(opportunities.sage_installed_but_off(self._ctx())) == []

    def test_cli_mode_stays_silent_rather_than_guess(self):
        import sys
        from unittest.mock import patch

        from comfydoctor.rules import opportunities

        with patch.object(sys, "argv", ["doctor.py"]):
            assert list(opportunities.sage_installed_but_off(self._ctx(runtime=False))) == []


def _ctx_build(dists, gpu):
    inv = Inventory(dists={d.name: d for d in dists}, duplicates={},
                    module_owners={}, unsatisfied=[])
    return Context(env=_env(), gpu=gpu, inv=inv, nodes=NodeSurvey())


class TestShadowedInstallFixIsComplete:
    def test_fix_puts_the_package_back(self):
        # The old fix ran only the uninstalls and left the user without the
        # package. The last command must now reinstall the winning version.
        copies = [_dist("einops", "0.8.0", "/site-a"), _dist("einops", "0.7.0", "/site-b")]
        ctx = _ctx(copies[:1], duplicates={"einops": copies})
        found = list(pkg_rules.shadowed_installs(ctx))
        assert len(found) == 1
        cmds = found[0].remedy.commands
        assert cmds[-1][-1] == "einops==0.8.0"   # winner goes back, pinned
        assert sum("uninstall" in " ".join(c) for c in cmds) == 2

    def test_torch_family_goes_back_via_pytorch_index(self):
        copies = [_dist("torch", "2.9.1+cu130", "/site-a"),
                  _dist("torch", "2.9.1", "/site-b")]
        ctx = _ctx(copies[:1], duplicates={"torch": copies}, gpu=_gpu_nvidia())
        found = list(pkg_rules.shadowed_installs(ctx))
        flat = " ".join(found[0].remedy.commands[-1])
        assert "download.pytorch.org" in flat, \
            "torch must never be reinstalled from bare PyPI (CPU wheel)"
        assert "torch==2.9.1" in flat
