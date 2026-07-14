# Publishing to the Comfy Registry

Everything here is one-time setup except step 4, which becomes your whole release
process afterwards.

## Where things actually stand

Checked live against `api.comfy.org`:

| Thing | State |
|---|---|
| Publisher `kurdknight` ("ComfyKurd") | **Exists, you own it** — `twana.info@gmail.com`, role `owner` |
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
