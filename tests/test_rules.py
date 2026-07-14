"""Rules tested against synthetic broken environments.

The value of a rule engine is that you can hand it a fake machine. Every one of
these fixtures is a real failure mode someone has hit - a CPU torch on a 4090, a
cu126 xformers next to a cu124 torch, two opencvs - and none of them requires
actually breaking a machine to test.

Run:  python -m pytest tests/ -q     (or just: python tests/test_rules.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfydoctor.custom_nodes import CustomNode, NodeSurvey, conflicting_demands
from comfydoctor.env import Environment
from comfydoctor.gpu import GPUInfo, cuda_tag_for_driver
from comfydoctor.inventory import Dist, Inventory
from comfydoctor.models import Severity, health_score
from comfydoctor.remedy import expected_torchaudio, expected_torchvision
from comfydoctor.rules import Context, run_all


def env(windows: bool = True) -> Environment:
    return Environment(
        python_exe=r"C:\ComfyUI\python_embeded\python.exe",
        python_version="3.12.10",
        kind="embedded",
        kind_detail="portable",
        comfy_root=Path("C:/ComfyUI"),
        custom_nodes_dir=Path("C:/ComfyUI/custom_nodes"),
        is_windows=windows,
    )


def dist(name: str, version: str, requires=None, modules=None, location="site-packages") -> Dist:
    return Dist(
        name=name, raw_name=name, version=version, location=location,
        requires=requires or [], modules=modules or [name.replace("-", "_")],
    )


def inv(*dists: Dist, unsatisfied=None) -> Inventory:
    d = {x.name: x for x in dists}
    owners: dict[str, list[str]] = {}
    for x in dists:
        for m in x.modules:
            owners.setdefault(m, []).append(x.name)
    return Inventory(dists=d, duplicates={}, module_owners=owners, unsatisfied=unsatisfied or [])


def gpu_4090(torch_version="2.6.0+cu124", cuda_available=True, driver="551.86") -> GPUInfo:
    g = GPUInfo()
    g.nvidia_smi_ok = True
    g.driver_version = driver
    g.driver_cuda_version = "12.4"
    g.devices = [{"name": "NVIDIA GeForce RTX 4090", "driver": driver,
                  "vram_total_mb": 24564, "vram_used_mb": 1000, "compute_capability": "8.9"}]
    g.torch_ok = True
    g.torch_version = torch_version
    g.torch_local_tag = torch_version.split("+")[1] if "+" in torch_version else None
    g.torch_cuda_build = "12.4" if "cu" in torch_version else None
    g.cuda_available = cuda_available
    g.torch_devices = [{"index": 0, "name": "NVIDIA GeForce RTX 4090",
                        "vram_total_mb": 24564, "compute_capability": "8.9"}] if cuda_available else []
    g.backends = {"flash_sdp": True, "mem_efficient_sdp": True, "math_sdp": True}
    return g


def run(inventory: Inventory, gpu: GPUInfo, nodes: NodeSurvey | None = None,
        windows: bool = True) -> dict[str, object]:
    ctx = Context(env=env(windows), gpu=gpu, inv=inventory, nodes=nodes or NodeSurvey())
    return {f.id: f for f in run_all(ctx)}


# ---------------------------------------------------------------- torch stack

def test_cpu_torch_on_gpu_machine_is_critical():
    """The silent killer. Nothing errors; renders are just 30x slower forever."""
    f = run(inv(dist("torch", "2.6.0")), gpu_4090("2.6.0", cuda_available=False))
    hit = f.get("torch.cpu_build_on_gpu_machine")
    assert hit is not None, "must catch a CPU torch on a machine with a 4090"
    assert hit.severity is Severity.CRITICAL
    assert "not being used" in hit.title
    # And it must offer to actually fix it, with the right index for the driver.
    cmd = " ".join(hit.remedy.commands[0])
    assert "--index-url" in cmd and "cu124" in cmd
    assert "python_embeded" in cmd, "must target ComfyUI's Python, not a system one"


def test_healthy_cuda_torch_reports_ok_not_silence():
    f = run(inv(dist("torch", "2.6.0+cu124")), gpu_4090())
    assert "torch.cpu_build_on_gpu_machine" not in f
    assert f["torch.ok"].severity is Severity.OK


def test_torchvision_release_mismatch():
    f = run(
        inv(dist("torch", "2.6.0+cu124"), dist("torchvision", "0.19.0+cu124")),
        gpu_4090(),
    )
    hit = f.get("torch.triplet_version_mismatch")
    assert hit is not None
    assert hit.severity is Severity.CRITICAL
    assert "0.21" in hit.detail  # tells you the version you should have


def test_matched_triplet_is_silent():
    f = run(
        inv(dist("torch", "2.6.0+cu124"),
            dist("torchvision", "0.21.0+cu124"),
            dist("torchaudio", "2.6.0+cu124")),
        gpu_4090(),
    )
    assert "torch.triplet_version_mismatch" not in f
    assert "torch.build_tag_mismatch" not in f


def test_build_tag_mismatch_cpu_torchvision_beside_cuda_torch():
    f = run(
        inv(dist("torch", "2.6.0+cu124"), dist("torchvision", "0.21.0+cpu")),
        gpu_4090(),
    )
    assert "torch.build_tag_mismatch" in f


def test_driver_too_old_for_cuda_build():
    f = run(inv(dist("torch", "2.8.0+cu128")), gpu_4090("2.8.0+cu128", driver="522.06"))
    hit = f.get("torch.driver_too_old")
    assert hit is not None and hit.severity is Severity.CRITICAL


def test_torchvision_offset_formula():
    assert expected_torchvision("2.6.0") == "0.21"
    assert expected_torchvision("2.1.0") == "0.16"
    assert expected_torchaudio("2.6.0+cu124") == "2.6"


# ------------------------------------------------------------------ attention

def test_xformers_pinned_to_other_torch():
    f = run(
        inv(dist("torch", "2.6.0+cu124"),
            dist("xformers", "0.0.28.post1", requires=["torch==2.5.1"])),
        gpu_4090(),
    )
    hit = f.get("attention.xformers.torch_pin_mismatch")
    assert hit is not None and hit.severity is Severity.ERROR


def test_sageattention_compound_tag_is_not_a_false_positive():
    """Regression: sageattention 2.2.0+cu130torch2.9.1.post5 encodes BOTH the
    CUDA version and the torch version in its local tag. Comparing that string
    against torch's plain 'cu130' called a perfect match a mismatch."""
    f = run(
        inv(dist("torch", "2.9.1+cu130"),
            dist("sageattention", "2.2.0+cu130torch2.9.1.post5")),
        gpu_4090("2.9.1+cu130"),
    )
    assert "attention.sageattention.build_tag_mismatch" not in f


def test_sageattention_genuinely_wrong_cuda_is_caught():
    f = run(
        inv(dist("torch", "2.9.1+cu130"),
            dist("sageattention", "2.2.0+cu124torch2.9.1")),
        gpu_4090("2.9.1+cu130"),
    )
    assert "attention.sageattention.build_tag_mismatch" in f


def test_linux_triton_on_windows():
    f = run(inv(dist("torch", "2.6.0+cu124"), dist("triton", "3.2.0")), gpu_4090(), windows=True)
    assert "attention.triton_linux_wheel_on_windows" in f
    # ...and not on Linux, where it is entirely correct.
    f2 = run(inv(dist("torch", "2.6.0+cu124"), dist("triton", "3.2.0")), gpu_4090(), windows=False)
    assert "attention.triton_linux_wheel_on_windows" not in f2


# ----------------------------------------------------------------- packages

def test_onnxruntime_cpu_and_gpu_both_installed():
    f = run(
        inv(dist("torch", "2.6.0+cu124"),
            dist("onnxruntime", "1.20.0", modules=["onnxruntime"]),
            dist("onnxruntime-gpu", "1.20.0", modules=["onnxruntime"])),
        gpu_4090(),
    )
    hit = f.get("packages.onnxruntime_variants")
    assert hit is not None and hit.severity is Severity.ERROR
    # The cure must remove BOTH before reinstalling one - removing only the
    # loser leaves a half-deleted shared module directory.
    cmds = hit.remedy.commands
    assert "uninstall" in cmds[0] and "install" in cmds[1]
    assert "onnxruntime" in cmds[0] and "onnxruntime-gpu" in cmds[0]


def test_opencv_variants_collapse_to_one():
    f = run(
        inv(dist("torch", "2.6.0+cu124"),
            dist("opencv-python", "4.10.0", modules=["cv2"]),
            dist("opencv-python-headless", "4.9.0", modules=["cv2"])),
        gpu_4090(),
    )
    assert "packages.opencv_variants" in f


def test_numpy2_against_numpy1_packages():
    f = run(
        inv(dist("torch", "2.6.0+cu124"),
            dist("numpy", "2.1.0"),
            dist("insightface", "0.7.3", requires=["numpy<2"])),
        gpu_4090(),
    )
    hit = f.get("packages.numpy2_abi")
    assert hit is not None and hit.severity is Severity.ERROR
    assert "numpy<2" in " ".join(hit.remedy.commands[0])


def test_unsatisfied_dependency_keeps_the_package_name_intact():
    """Regression: the target name was being recovered by string-slicing the
    requirement, which turned `numpy>=2.0` into a package called `numpy>=2-0`
    and reported it as 'not installed'."""
    f = run(
        inv(dist("torch", "2.6.0+cu124"), dist("numpy", "1.26.4"),
            unsatisfied=[{
                "dist": "opencv-python", "requirement": "numpy>=2", "target": "numpy",
                "specifier": ">=2", "installed": "1.26.4",
                "reason": "numpy 1.26.4 is installed, but opencv-python requires >=2",
            }]),
        gpu_4090(),
    )
    hit = f.get("packages.unsatisfied.numpy")
    assert hit is not None
    assert "numpy>=2-0" not in hit.title and "numpy>=2" not in hit.title
    assert "numpy 1.26.4" in hit.title
    # And the fix must ask for a version, not a bare upgrade.
    assert '"numpy>=2"' in hit.remedy.as_shell()[0] or "numpy>=2" in hit.remedy.as_shell()[0]


def test_shell_metacharacters_are_quoted_for_copy_paste():
    """A version specifier is an ordinary pip argument, but `<` pasted into a
    terminal is a redirect. We run via argv (no shell), so this quoting exists
    purely so the *copied* command is safe."""
    from comfydoctor.models import Remedy

    r = Remedy(title="t", commands=[["pip", "install", "numpy<3,>=2.0"]])
    assert r.as_shell()[0] == 'pip install "numpy<3,>=2.0"'


# -------------------------------------------------------------- custom nodes

def _nodes(*specs: tuple[str, list[str]]) -> NodeSurvey:
    sv = NodeSurvey(runtime_known=True)
    sv.nodes = [
        CustomNode(name=n, path=Path(f"C:/ComfyUI/custom_nodes/{n}"), requirements=reqs, loaded=True)
        for n, reqs in specs
    ]
    from comfydoctor.custom_nodes import _build_demands

    sv.demands = _build_demands(sv.nodes)
    return sv


def test_equal_and_gte_pins_are_not_a_conflict():
    """Regression: ftfy==6.1.1 and ftfy>=6.1.1 are both satisfied by 6.1.1.
    The old satisfiability probe used a hardcoded version ladder that stopped at
    major version 4, so ftfy 6 and pillow 12 fell off the end and every one was
    reported as an irreconcilable conflict."""
    sv = _nodes(("node_a", ["ftfy==6.1.1"]), ("node_b", ["ftfy>=6.1.1"]))
    assert conflicting_demands(sv, inv(dist("ftfy", "6.1.1"))) == []


def test_ascending_lower_bounds_are_not_a_conflict():
    sv = _nodes(("a", ["pillow>=10.2.0"]), ("b", ["pillow>=10.3.0"]), ("c", ["pillow>=11.1.0"]))
    assert conflicting_demands(sv, inv(dist("pillow", "12.1.0"))) == []


def test_genuinely_irreconcilable_pins_are_caught():
    sv = _nodes(("old_node", ["numpy<2"]), ("new_node", ["numpy>=2"]))
    out = conflicting_demands(sv, inv(dist("numpy", "1.26.4")))
    assert len(out) == 1 and out[0]["package"] == "numpy"


def test_node_pinning_torch_is_flagged_as_a_landmine():
    sv = _nodes(("some_node", ["torch", "numpy"]))
    f = run(inv(dist("torch", "2.6.0+cu124")), gpu_4090(), nodes=sv)
    hit = f.get("nodes.requirements_pin_torch")
    assert hit is not None
    assert "CPU-only" in hit.impact


def test_cli_mode_does_not_invent_import_failures():
    """Outside ComfyUI we cannot observe which nodes loaded. We must not guess."""
    sv = NodeSurvey(runtime_known=False)
    sv.nodes = [CustomNode(name="x", path=Path("C:/x"), loaded=None)]
    f = run(inv(dist("torch", "2.6.0+cu124")), gpu_4090(), nodes=sv)
    assert not any(k.startswith("nodes.import_failed") for k in f)


def test_vendored_copies_are_not_shadowed_installs():
    """Regression: setuptools and pip bundle private copies of their deps under
    `_vendor/`, complete with .dist-info. They are not on sys.path and cannot
    shadow anything - but importlib.metadata lists them, so we were reporting
    "packaging is installed twice, at different versions" and offering a fix
    that would have broken setuptools."""
    from comfydoctor.inventory import _is_vendored

    assert _is_vendored(r"C:\ComfyUI\python_embeded\Lib\site-packages\setuptools\_vendor")
    assert _is_vendored("/usr/lib/python3/site-packages/pip/_vendor")
    assert not _is_vendored(r"C:\ComfyUI\python_embeded\Lib\site-packages")
    assert not _is_vendored("/usr/lib/python3/site-packages")


# -------------------------------------------------------------------- scoring

def test_health_score_and_driver_tag_selection():
    from comfydoctor.models import Finding, health_label

    def f(sev, i=0):
        return Finding(id=f"x{i}", severity=sev, category="c", title="t")

    assert health_score([]) == 100
    assert health_label([]) == "Healthy"

    # A real, working install with 80 nodes carries a long tail of warnings.
    # It must NOT read as "Broken" - if every healthy machine scores 0, the
    # number is noise and people stop believing the whole report.
    many_warnings = [f(Severity.WARNING, i) for i in range(24)]
    score = health_score(many_warnings)
    assert score > 80, f"24 warnings alone should not tank the score, got {score}"
    assert health_label(many_warnings) == "Minor issues"

    # But a genuine critical must sink it, and the word must follow the worst
    # finding rather than the arithmetic.
    assert health_label([f(Severity.CRITICAL)]) == "Broken"
    assert health_label([f(Severity.ERROR)]) == "Needs attention"
    assert health_score([f(Severity.CRITICAL, i) for i in range(4)]) < 30
    assert health_score([f(Severity.CRITICAL, i) for i in range(20)]) >= 0  # never negative

    assert cuda_tag_for_driver("12.4", windows=True) == "cu124"
    assert cuda_tag_for_driver("11.8", windows=True) == "cu118"


def test_no_rule_crashes_on_a_totally_empty_environment():
    """No torch, no GPU, no nodes. Must produce findings, not a traceback."""
    f = run(inv(), GPUInfo())
    assert "torch.missing_or_broken" in f
    assert not any(k.startswith("internal.rule_failed") for k in f), \
        "a rule raised on an empty environment"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
