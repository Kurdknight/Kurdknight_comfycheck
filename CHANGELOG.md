# Changelog

## 2.0.0 — 2026-07-14

Complete rewrite. The old version was two nodes that printed a list of versions. This is a
diagnostic tool that finds conflicts, explains what they'll do to you, and fixes them.

### The headline

- **It works when ComfyUI won't start.** `comfydoctor.bat` (double-click) or `doctor.py` run
  standalone off the same engine. This is the whole reason for the rewrite: a broken torch
  means ComfyUI never boots, which means a diagnostic *node* can never load. The old tool
  could only ever describe environments that were already healthy.
- **A sidebar panel**, not a node. No graph, no Queue button. It opens, it scans, it shows you
  what's wrong.
- **One-click fixes.** Every finding that can be cured carries the exact command for *your*
  interpreter — embedded, venv, conda or system — and a button that runs it, streaming pip's
  output live into the panel.
- **Conflict detection**, which didn't exist before at all. This is the part people actually
  needed.

### Added — conflict rules

- CPU-only PyTorch installed on an NVIDIA machine (silent 20–50× slowdown, no error anywhere)
- torch / torchvision / torchaudio release mismatch, and disagreeing `+cu` build tags
- NVIDIA driver too old for the installed CUDA build
- xformers / flash-attn / sageattention built against a *different torch* — detected from
  package metadata, without importing them
- Linux `triton` installed on Windows
- Full `pip check` equivalent, computed in-process, with plain-English reasons
- `onnxruntime` + `onnxruntime-gpu` co-installed (the standard cause of InsightFace/ReActor
  silently running on CPU)
- Multiple OpenCV variants fighting over `cv2`
- numpy 2.x against packages built for numpy 1.x
- The same package installed twice in different site-packages, so upgrades appear to do nothing
- Any two distributions claiming the same import name
- **Custom nodes:** which failed to import *and why*; nodes loaded but with unmet requirements;
  nodes whose pins genuinely contradict each other; nodes that list `torch` in
  `requirements.txt` and can therefore replace your CUDA build with the CPU wheel

### Added — surfaces

- `doctor.py` / `comfydoctor.bat` standalone launcher
- Sidebar panel with live-streaming repair, both light and dark themes
- Anonymized Markdown report (safe to paste into a public issue — your Windows username and
  home path are stripped) and a self-contained HTML report
- CLI exit codes: `0` clean, `1` warnings, `2` errors — gate a launch script on it
- 0–100 health score

### Fixed — bugs that had been in v1 the whole time

- **Version lookups were wrong.** `_get_module_version()` called
  `importlib.import_module("opencv-python")`, `("scikit-learn")`, `("pillow")`,
  `("face-recognition")` — using *distribution* names as *module* names. Those imports can
  never succeed. The tool reported **"Not installed"** for packages that were installed, and
  had done so since day one.
- **Inspecting packages by importing them.** Importing xformers/flash-attn/bitsandbytes on a
  mismatched torch doesn't raise — it can abort the process. The diagnostic could take ComfyUI
  down. Everything now reads `importlib.metadata` and never imports what it inspects.
- **The web UI was dead on modern ComfyUI.** `document.querySelector(".comfy-menu")` returns
  null on the Vue frontend, so the extension threw during `setup()`.
- **The API routes never existed.** `api_route()` used a Flask-style `@server.route` decorator
  that ComfyUI's aiohttp server doesn't have — and `register_extension()` was never called by
  anything regardless. The Refresh and Save Report buttons had never worked.
- **Disk space checked the wrong disk.** `psutil.disk_usage('/')` reports C: on Windows, while
  most people keep ComfyUI on D:.
- `torch.cuda.is_bf16_supported()` was reported as "FP16 Available".
- `pkg_resources` (removed in setuptools ≥81 — this would have broken the extension outright)
  replaced with `importlib.metadata`.
- Bare `except:` clauses that swallowed `KeyboardInterrupt` and `SystemExit`.

### Changed

- `SystemCheck` and `SystemViz` are aliased onto the new `ComfyDoctorReport` node, so saved
  workflows still open.
- Reports render in the panel instead of being written to a file the user then had to go and
  find.

### Security

- The browser cannot supply a command. It sends a *finding id*; the server executes only the
  argv it generated itself during the last scan.
- Commands run as argv lists with `shell=False` — no shell, no injection surface.
- One repair at a time. Two concurrent pip processes writing the same site-packages is how a
  broken environment becomes an unrecoverable one.

### Packaging

- Added `pyproject.toml` (`PublisherId = "kurdknight"`), which the project never had — this is
  why it was only ever installable as a *nightly* git-clone and had zero published versions in
  the Comfy Registry.
- Added a GitHub Action to publish to the registry on version bump.
