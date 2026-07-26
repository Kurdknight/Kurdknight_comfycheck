# Changelog

## 2026-07-26 — v2.1.1

### New: environment time machine — "it worked yesterday, what changed?"
Every scan journals the installed-package state plus the problems found, to
`<comfy_root>/user/comfydoctor/env_journal.json`. When a **new** problem
appears, a *What changed* finding names it, lists every package that changed,
vanished or arrived since the last scan that did not have it, and offers a
one-click restore pinning those packages back to their recorded versions.

- **Per-problem reference point, not a global "healthy" flag.** Installs with
  many custom nodes often carry one long-standing ERROR the owner accepts, so
  a "last scan with zero errors" reference would never be recorded and the
  feature would be dead exactly where it is needed. The reference is the
  newest snapshot that lacked *today's* problem.
- Restore never uninstalls, never touches packages added since the reference,
  and sends torch/torchvision/torchaudio back through the PyTorch index for
  their recorded build tag (never bare PyPI's CPU wheel).
- Says so plainly when a problem appeared with **no** package change at all —
  driver, launch script or disk — instead of implying pip is to blame.
- Goes quiet rather than guessing when every retained snapshot already had the
  problem; timestamps are shown as "yesterday at 14:03", never ISO-8601.
- Silent on healthy machines; a no-op when there is no ComfyUI root; a corrupt
  journal is a fresh start, never a crash.
- Re-scanning an unchanged-but-broken environment no longer overwrites the
  last clean same-packages snapshot, so the "problem is new but no package
  changed" finding persists instead of showing exactly once. An empty package
  inventory (failed probe) is never diffed against — it would have offered a
  full reinstall.

### Also in this release (2026-07-25)

- **Panel:** copied commands now quote all shell metacharacters (`<`/`>` in pip
  specifiers previously became shell redirects when pasted). Score colour now
  follows the health label instead of the raw number.
- **Advice-text audit:** removed every opinion-stated-as-fact and invented
  number ("SDPA is enough for almost everyone", "20-30%", "20-50x"…). Attention
  reporting now states what is installed AND whether ComfyUI's launch flags
  actually enable it; new tip when SageAttention is installed but switched off.
  The confused cuDNN/--cuda-malloc tip was removed. A test now bans these
  phrases from every source file.
- **Advice safety:** unknown contested-module pairs get "do nothing / non-
  destructive repair" advice instead of "uninstall all"; same-version pairs
  (repackaged wheels) drop to INFO. OpenCV pile-ups keep the contrib superset.
  onnxruntime fixes warn when they would replace a deliberate nightly build.
  Shadowed-install fixes reinstall the winner in the same click. Version-move
  fixes carry every installed package's pin so pip refuses rather than breaks.
- **Verification hardening:** numpy requirements are evaluated with `packaging`
  (the `"<2" in spec` substring bug misread satisfied pins as numpy-1
  requirements and pushed downgrades on healthy machines). Yanked PyPI releases
  no longer count as "shipped". Malformed caches read as cache misses. A
  shipped torch with an unconfirmable torchvision partner stays pinned instead
  of being silently upgraded.

## 2026-07-24 — v2.0.x

- Torch-family version pairings verified against PyPI (live, cached, baked
  fallback) instead of a lockstep formula; the formula's answer is discarded
  when the paired release never shipped (torchaudio froze at 2.11).
- Healthy-machine invariant test suite: real working configurations (portable
  CUDA, conda, nightly, cu128-on-550-driver, frozen torchaudio) must scan with
  zero CRITICAL/ERROR.
- One-click fixes off by default behind an "at your own risk" toggle; dangerous
  remedies additionally require an explicit acknowledgement.
