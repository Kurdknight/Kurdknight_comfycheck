# Publishing to the Comfy Registry

Everything here is one-time setup except step 4, which becomes your whole release
process afterwards.

## Where things actually stand

Checked live against `api.comfy.org`:

| Thing | State |
|---|---|
| Publisher `kurdknight` ("ComfyKurd") | **Exists, you own it** (role `owner`) |
| Node `Kurdknight_comfycheck` | **Exists, attached to your publisher**, points at the right repo |
| Published versions | **Zero.** `GET /nodes/Kurdknight_comfycheck/versions` → `[]` |
| `/install` endpoint | **404** — the registry has no artifact to serve |

The node record was created **2025-03-12**, four months *before* your publisher
account (2025-07-29). You didn't register it — Comfy Org bulk-imported it from
ComfyUI-Manager's legacy `custom-node-list.json`, into an "Unclaimed" publisher,
and it was later attached to `kurdknight`.

**That is why Manager shows "nightly".** With no published version, Manager has
nothing to install except a raw `git clone` of whatever is currently on `main`.
No version pinning, no dependency install, no download counter.

## Step 1 — Get a registry API key

1. Go to <https://registry.comfy.org> and sign in with GitHub.
2. Your publisher `kurdknight` is already there.
3. Create an **API key** for it.

## Step 2 — Add it to the repo as a secret

GitHub → the repo → **Settings → Secrets and variables → Actions → New
repository secret**

- Name: `REGISTRY_ACCESS_TOKEN`
- Value: the key from step 1

This is the only step nobody can do for you, because it requires your GitHub
login.

## Step 3 — Push

`.github/workflows/publish.yml` is already committed. It fires whenever
`pyproject.toml` changes on `main`.

## Step 4 — Every release from now on

Bump one line:

```toml
# pyproject.toml
version = "2.0.1"
```

Push it. The action publishes to the registry automatically. Within a few
minutes ComfyUI Manager stops saying "nightly" and offers a real versioned
install, `/install` starts working, and downloads start counting.

Verify:

```bash
curl -s https://api.comfy.org/nodes/Kurdknight_comfycheck/versions
```

Should no longer be `[]`.

## Two ways to publish, from now on

### A. Automated (recommended — what publishes today)

Bump `version` in `pyproject.toml`, commit, push to `main`. The GitHub Action
does the rest. No key on your machine, nothing to remember.

### B. Manual, from your terminal

For publishing without a push, or a quick local release. One-time:

```bash
pip install comfy-cli
```

Then, **from inside this node's folder** (the one with `pyproject.toml`):

```bash
comfy node publish
```

It prompts for an API key — paste the same `comfy_...` key from the registry
(or make a fresh one; you can hold several). It zips the node from
`pyproject.toml` and uploads. Same result as the automated path.

**Either way: bump `version` first.** The registry rejects a duplicate — you
cannot republish `2.0.1` over `2.0.1`. Versions are immutable and additive:
publishing `2.0.2` never touches `2.0.1`; both stay downloadable forever.

## Publishing a brand-new node (a different one) later

1. Make the folder with your code + `__init__.py`.
2. Add a `pyproject.toml` (`comfy node init` scaffolds one):

   ```toml
   [project]
   name = "your_new_node_name"   # globally unique — becomes the registry ID and URL. PERMANENT.
   version = "1.0.0"
   description = "..."
   license = { file = "LICENSE" }

   [tool.comfy]
   PublisherId = "kurdknight"     # ALWAYS this — it's your identity, same on every node you own
   DisplayName = "Your Node's Nice Name"
   ```
3. `comfy node publish` — or drop this repo's `.github/workflows/publish.yml`
   into the new repo and add its own `REGISTRY_ACCESS_TOKEN` secret.

**The rules that never change:**

- `PublisherId = "kurdknight"` on *every* node — that's how the registry knows
  it's yours.
- `name` is permanent, globally unique, and is the URL — choose once, never
  rename (renaming orphans the node, see below).
- `DisplayName` is the pretty name; change it freely.
- Bump `version` on every publish.

## A trap to avoid

**Do not rename `name` in `pyproject.toml`.** It must stay exactly:

```toml
name = "Kurdknight_comfycheck"
```

That string is the *existing registry node id*. Changing it (e.g. to
"ComfyDoctor") does not rename your node — it mints a brand-new registry entry
and orphans the current one, along with its GitHub stars and its place in the
Manager list. The display name is a separate field, and that one is free to
change:

```toml
[tool.comfy]
DisplayName = "ComfyDoctor"    # <- this is what users see
```

## Optional: the legacy Manager list

The flat `custom-node-list.json` in `Comfy-Org/ComfyUI-Manager` still carries the
old title and description:

> "KurdKnight ComfyUI System Check Node — A comprehensive system information
> node…"

The registry entry supersedes it in the modern Manager UI, so this is cosmetic.
If you want it updated, open a PR against that repo changing the `title` and
`description` for the `Kurdknight/Kurdknight_comfycheck` entry.
