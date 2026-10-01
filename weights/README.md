# Trained weights

Nothing in this directory is committed by default. This file explains what the weights are, what
licence they carry, and the three ways to ship them — because the obvious one does not work.

## What they are

| model | file | size | corpus |
|---|---|---:|---|
| Sunny baseline **v75q** | `carla2real_semantic_v75_pz_tex_G_fp16.pth` | 350 MB | PandaSet · ZOD · Mapillary Vistas · Cityscapes |
| Night baseline **v79** | `carla2real_semantic_v79_clean_night_G_fp16.pth` | 350 MB | PandaSet · ZOD only |

The generator is 183.5M parameters. At fp32 that is **700 MB per model**; the files above are fp16,
which halves it. `scripts/weights/export_weights.py` does the conversion and reports the largest
relative weight change it caused — measured at **4.2e-04**, an order of magnitude below the 3.9e-03
that one step of an 8-bit pixel represents, so the rendered output is unchanged.

The discriminator (`latest_net_D.pth`, 33 MB) is only needed to resume training and is not shipped.

## Why you cannot simply commit them

GitHub rejects any single file over **100 MB**. Even at fp16 these are three and a half times that.
There are three ways round it and they are not equally good:

| | limit | cost | verdict |
|---|---|---|---|
| **Release asset** | 2 GB per file | free, unmetered downloads | **use this** |
| Git LFS | 100 MB+ per file | free tier is 1 GB storage **and 1 GB/month bandwidth** — one clone of both models is 700 MB, so the second clone in a month fails until you buy a data pack | works, but throttles |
| Commit the raw file | 100 MB | — | rejected by GitHub |

A release asset is how essentially every model repository ships weights, and it keeps the clone of
this repository small for anyone who only wants the code.

### Publishing as a release asset

```bash
bash scripts/weights/export.sh          # writes fp16 copies into weights/
```

Then either drag the two `.pth` files onto a new release at
`https://github.com/<you>/CARLAtoreal/releases/new`, or with the `gh` CLI:

```bash
gh release create v1.0 weights/*.pth \
  --title "Baselines v75q (sunny) and v79 (night)" \
  --notes "fp16 generator weights. CC BY-SA 4.0 — see weights/LICENSE."
```

`scripts/weights/fetch.sh` pulls them back down into `pix2pixHD/checkpoints/`.

### If you would rather commit them with Git LFS

```bash
git lfs install
git lfs track "weights/*.pth"
git add .gitattributes weights/*.pth
git commit -m "Add fp16 baseline weights"
```

`.gitattributes` in the repository root already has the tracking rule, so this works as soon as
`git lfs` is installed. Read the bandwidth row above first — LFS objects also cannot be removed
from history without a rewrite.

## Licence — read before publishing

**These weights are CC BY-SA 4.0, not Apache-2.0.**

The code in this repository is Apache-2.0. The weights are not code. They are trained on the
Zenseact Open Dataset, which is **CC BY-SA 4.0**, and share-alike propagates to derivative works.
Whether trained weights are a derivative of their training data is legally unsettled; publishing
them under a permissive licence assumes the answer, and assuming it in your own favour is not the
safe direction. `weights/LICENSE` states CC BY-SA 4.0 for exactly that reason.

PandaSet's Dataset Terms add conditions that survive into derived data: attribute Scale AI and
Hesai, do not use their names or logos beyond attribution, and do not use the data to identify any
person.

**No checkpoint is currently fit to publish.** An earlier revision of this file said `v73` (sunny)
and `v69` (night) train on PandaSet only. They do not: v73's corpus includes Mapillary and Cityscapes,
v69's includes Dark Zurich, and every baseline descends from them. See
[Weight provenance](../THIRD_PARTY_NOTICES.md#weight-provenance). The first weights released will
come from a model trained from random initialisation on PandaSet and ZOD alone, which is in progress.
