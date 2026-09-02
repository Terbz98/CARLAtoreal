# carla2real — CARLA to photorealistic video for perception testing

Turns CARLA's semantic output into photorealistic driving video, so a perception stack can be
exercised on scenarios that are cheap to author in simulation but expensive to capture on the road.

A conditional GAN (pix2pixHD) maps a semantic label map to an image. On top of the label map it is
also given edge, depth, surface-normal and either chroma (day) or light (night) channels, which is
what stops the generator inventing a different world every frame.

> **Read `THIRD_PARTY_NOTICES.md` before publishing or redistributing.** Licensing is not yet
> resolved: the bundled pix2pixHD fork has no upstream licence file restored. No model weights are
> included in this repository.

## Layout

```
config.sh / config.py     every path the pipeline uses, overridable by environment variable
pix2pixHD/                the modified pix2pixHD fork (source only)
docs/                     design notes, handover, and the experiment log
*.py / *.sh               pipeline stages and measurement tools (see below)
```

Bulk data is deliberately outside the repository. Set `CARLA2REAL_DATA` and `CARLA2REAL_OUT`, or
accept the defaults of `./datasets` and `./output`.

## The idea that shapes the design

A semantic label map carries no weather information: road is the same integer in bright sun as at
midnight. A single weather-conditioned model therefore drifts toward whichever domain dominates the
training set. Separate per-condition models each keep their full capacity for one domain, which is
why there is a sunny model and a night model rather than one model with a weather switch.

The same reasoning drives the extra conditioning channels. Given only "building", the generator
invents a facade — and invents a *different* one each frame, which is exactly the flicker that makes
synthetic video useless for perception testing. Depth, normals and edges pin the geometry down.

## Pipeline

**1. Record.** `record_town_auto.py --town Town05 --weather sunny --outname Town05_sunny_inst`
drives CARLA on autopilot and captures RGB plus the semantic camera to
`$CARLA2REAL_DATA/recorded_<outname>/`. NPCs are spawned near the ego rather than scattered over
the whole map — a map-wide shuffle put 150 vehicles somewhere the ego never drove and yielded one
visible car per frame. `prepare_gt_test_label.py` converts CARLA's semantic images to the
trainId label maps the generator reads.

CARLA writes BGRA; slicing `[:, :, :3]` yields BGR-as-RGB, so the channel order must be reversed
before use. Getting this wrong is silent — the image looks plausible, just wrong.

**2. Build conditioning channels.** Label map, instance-merged edges, monocular depth and normals,
plus chroma (day) or light (night).

**3. Render.** `render_model.sh <sunny|night> <model> <tag> <Town...>` runs the generator over a
town and then the delivery chain, writing `<town>_<weather>_<tag>_FINAL_1920.mp4`.

**4. Deliver.** Stages that repair what the generator gets wrong, each guarded so it cannot make
things worse:

| Stage | What it fixes |
|---|---|
| `protect_traffic_lights_carla.py` | Composites CARLA's own lights back in, so signal state is always correct rather than a plausible-looking invention. |
| `protect_lane_markings.py` | Restores lane paint contrast to the range real paint occupies. |
| `protect_vehicle_colour.py` | Stops an invented warm cast on vehicles. Guarded: when the reference surface is near-neutral, hue is never adopted from it — on a grey surface the measured hue is decided by sensor noise, and adopting it turned a red bus magenta. |
| `protect_buildings.py` | Injects CARLA's real facade structure. Applied per region and kept only where measured detail actually increases. |
| `class_deshimmer.py` | Per-class temporal smoothing. Strength and motion tolerance vary by class, because a global filter smooths the stationary road (already stable) while missing buildings that sweep past the camera. |
| `despeckle_night.py` | Removes isolated colourless bright blobs. Lamps and signals fail all three tests and survive. |
| `fuse_colour.py` | Transfers one render's colour onto another's frames. Built because the two properties are separable: stability is temporal, vibrancy is per-frame colour. The transfer statistics are smoothed over a temporal window first, so only the slow colour trend crosses over and none of the source's frame-to-frame jitter. |

**5. Measure.** `veg_report.py`, `tail_check.py`, `road_texture.py`, `true_instability.py`,
`epoch_sweep.py`, `flicker_report.py`, and `score_vp.py` for end-to-end perception scoring against
an external stack (optional; set `PERCEPTION_ROOT`).

## A warning about metrics

Every metric in this project has been wrong at least once, and the failures were not random —
they shared a cause. Flicker metrics reward a still image, and sharpness metrics reward any
high-frequency content whether or not it means anything. A model that wove a fixed crosshatch over
every flat surface scored *better* on both than the model it replaced.

So the tools here are built to be falsifiable rather than flattering:

- `road_texture.py` asks whether there is detail where the label map says there should be none,
  instead of asking whether there is detail.
- `veg_report.py` reports near and far detail separately and refuses to average them, because
  raising near detail while distant trees stay bad is not a fix.
- `true_instability.py` separates real instability from detail in motion using optical flow.
- `road_sky_ceiling.py` compares against real photographs, not against the previous model, because
  a comparison to the parent tells you a change is new, not that it is an improvement.

Look at a frame before believing a number. `docs/EXPERIMENTS.md` records the cases where the
numbers and the picture disagreed, including the ones where the numbers won.

## Documentation

- `docs/EXPERIMENTS.md` — numbered findings, including the negative results. The dead ends are
  recorded on purpose so they are not repeated.
- `docs/STATE.md` — current state, active baselines, and standing constraints.
- `docs/Handover_pix2pixHD_EN.md` — training and inference details.
- `docs/Project_Summary_EN.md` — model history.

## Requirements

`requirements.txt` pins the versions that actually produced the delivered clips — Python 3.10,
PyTorch 2.11 on CUDA 12.8, OpenCV 4.13. CARLA 0.9.16 is needed only for recording, and its python
package version must match the running server exactly.

A CUDA GPU is required. A 2048-wide render peaks around 27 GB of VRAM, so run one GPU job at a time;
`gpu_wait.sh` exists to serialise them.

## First run

```bash
git clone <this repo> && cd carla2real
pip install -r requirements.txt
cp config.sh config.local.sh          # then edit the paths, or export them in your shell
. ./config.sh

# A. render from an existing recording (needs weights in pix2pixHD/checkpoints/<model>/)
./render_model.sh sunny carla2real_semantic_v50_graft v50 Town05

# B. or record your own first (needs a CARLA server on localhost:2000)
python3 record_town_auto.py --town Town05 --weather sunny --outname Town05_sunny_inst
python3 prepare_gt_test_label.py
```

Nothing here downloads weights or data. See the table below for what you must supply.

## Licence

This project's own code is licensed under the **Apache License, Version 2.0** — see `LICENSE` and
`NOTICE`. That covers the code in this repository and nothing else.

It does **not** cover:

- the **training corpus** (real driving footage, licensed separately — see `THIRD_PARTY_NOTICES.md`),
- any **model weights** derived from that corpus, which inherit the corpus's terms,
- the vendored **pix2pixHD** source, which keeps its own BSD licence at `pix2pixHD/LICENSE.txt`.

## What is and is not in this repository

**Code only.** No datasets, no trained weights, no rendered video. The working tree this was
packaged from is ~550 GB; the repository is under 1 MB. A clone will not run until you supply:

| Missing | Why | How to get it |
|---|---|---|
| Trained weights | ~200 MB per model, and derived from licensed training footage | Train it yourself — `train_v50.sh` is the sunny baseline recipe and `train_v51_night.sh` the night one; `train_v63_veg.sh` / `train_v64_veg.sh` are shipped as worked *negative* results. Or request the weights separately. |
| Training corpus | Real driving footage, licensed separately | Not redistributable here — see `THIRD_PARTY_NOTICES.md` |
| CARLA 0.9.16 | Records the drives | carla.org |
| MoGe, DVP, Real-ESRGAN | Depth/normal channels, optional temporal and upscale stages | Upstream projects |
| Perception stack (optional) | Only for `score_vp.py` scoring | Not part of this project; set `PERCEPTION_ROOT` |

## Which version does what

Two baselines, chosen by eye on side-by-side comparison rather than by metric:

| Condition | Baseline | Delivery chain |
|---|---|---|
| Sunny | **v50l** (since 2026-09-02) | `make_v50j.sh`, then `make_v50kl.sh` for the grade |
| Night | **v59** | `render_model.sh night <model> v59 <Town...>` with `TEMPORAL=1` |

**v50l is not a trained model, and that is the point.** It is the v50j render carrying a colour
grade lifted from v63 by `fuse_colour.py`. Reproducing it needs both: the v50j chain for the
structure, and a v63 render of the same town, frame-aligned, for the colour. Over five towns
against the previous v50d baseline: flicker −10.2%, colourfulness +23.2%, sharpness −2.9%,
CIPO recall 0.879 vs 0.887, lane MAE 0.196 vs 0.204.

The model the colour came from is the *worst* one measured on detection recall (v63, 0.836).
Borrowing its colour beat adopting it, because stability is temporal and colour is per-frame —
so the two are separable and no retrain was needed. `docs/EXPERIMENTS.md` notes 38 and 39.

Earlier chains are kept because they are the lineage, and because each carries a fix the one
before it predates:

- `make_v50d.sh` — the previous sunny baseline. Predates the vehicle-colour and building fixes.
- `make_v50i.sh` / `make_v50j.sh` — the vehicle-colour achromatic guard and the per-region
  building-structure injection. `v50j` is v50l's carrier.
- `make_v51d.sh` — night, with per-class de-shimmer and despeckle.

Do not apply the grade to `v50d` (that combination is `v50k`, and it is closed): a global grade
amplifies a per-object hue error instead of fixing it, so v50d's magenta bus came out worse. The
carrier must already have the achromatic guard. `docs/EXPERIMENTS.md` records why each was accepted or rejected, including the
vegetation-loss work (notes 35–37) that improved distant foliage but overshot road grain to ~2x a
real photograph and so was never promoted.
