# Pipeline reference

The full technical reference: every stage, every conditioning channel, the measurements behind each
design decision and the failures that produced them. [`../README.md`](../README.md) is the short
introduction; this is the document to read before changing anything.

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
carla2real/               recording, preprocessing, postprocessing, evaluation and common Python code
configs/config.sh        shell environment and repository root discovery
carla2real/config.py      Python path configuration
scripts/                 preprocessing, training, inference and delivery shell entry points
experiments/             version-specific training and delivery recipes
pix2pixHD/               modified model core; checkpoint and result locations are unchanged
datasets/ / output/      local assets and delivered clips (Git-ignored)
docs/ / notes/           guides, experiment history and project progress
```

Bulk data is deliberately outside the repository. Set `CARLA2REAL_DATA` and `CARLA2REAL_OUT`, or
accept the defaults of `./datasets` and `./output`.

Run Python tools from the repository root with `python3 -m carla2real.<group>.<module>`.
Shell drivers locate `configs/config.sh` relative to themselves; it exposes the package through
`PYTHONPATH` even when a driver changes working directory. Old root-level script paths have moved;
see [the migration guide](docs/REORGANIZATION.md) for the complete mapping and known missing helpers.
Create empty asset directories with `python3 scripts/init_asset_dirs.py`; download placeholders
and extraction locations are in [the asset guide](docs/ASSET_DOWNLOADS.md).

## The idea that shapes the design

A semantic label map carries no weather information: road is the same integer in bright sun as at
midnight. A single weather-conditioned model therefore drifts toward whichever domain dominates the
training set. Separate per-condition models each keep their full capacity for one domain, which is
why there is a sunny model and a night model rather than one model with a weather switch.

The same reasoning drives the extra conditioning channels. Given only "building", the generator
invents a facade — and invents a *different* one each frame, which is exactly the flicker that makes
synthetic video useless for perception testing. Depth, normals and edges pin the geometry down.

## Pipeline

**1. Record.** `python3 -m carla2real.recording.record_town_auto --town Town05 --weather sunny --outname Town05_sunny_inst`
drives CARLA on autopilot and captures RGB plus the semantic camera to
`$CARLA2REAL_DATA/recorded_<outname>/`. NPCs are spawned near the ego rather than scattered over
the whole map — a map-wide shuffle put 150 vehicles somewhere the ego never drove and yielded one
visible car per frame. `carla2real/preprocessing/legacy/prepare_gt_test_label.py` is the older
Town03/19-class conversion, not the current 65-class preprocessing chain. That chain remains
incomplete in this checkout; see [the inference inventory](docs/INFERENCE_FLOW.md).

CARLA writes BGRA; slicing `[:, :, :3]` yields BGR-as-RGB, so the channel order must be reversed
before use. Getting this wrong is silent — the image looks plausible, just wrong.

**2. Build conditioning channels.** Label map, instance-merged edges, monocular depth and normals,
plus chroma (day) or light (night).

**3. Render.** `scripts/inference/render_model.sh <sunny|night> <model> <tag> <Town...>` runs the generator over a
town and then the delivery chain, writing `<town>_<weather>_<tag>_FINAL_1920.mp4`.

**4. Deliver.** Stages that repair what the generator gets wrong, each guarded so it cannot make
things worse:

| Stage | What it fixes |
|---|---|
| `carla2real/postprocessing/protect_traffic_lights_carla.py` | Composites CARLA's own lights back in, so signal state is always correct rather than a plausible-looking invention. |
| `carla2real/postprocessing/protect_lane_markings.py` | Restores lane paint contrast to the range real paint occupies. |
| `carla2real/postprocessing/protect_vehicle_colour.py` | Stops an invented warm cast on vehicles. Guarded: when the reference surface is near-neutral, hue is never adopted from it — on a grey surface the measured hue is decided by sensor noise, and adopting it turned a red bus magenta. |
| `carla2real/postprocessing/protect_buildings.py` | Injects CARLA's real facade structure. Applied per region and kept only where measured detail actually increases. |
| `carla2real/postprocessing/class_deshimmer.py` | Per-class temporal smoothing. Strength and motion tolerance vary by class, because a global filter smooths the stationary road (already stable) while missing buildings that sweep past the camera. |
| `carla2real/postprocessing/despeckle_night.py` | Removes isolated colourless bright blobs. Lamps and signals fail all three tests and survive. |
| `carla2real/postprocessing/fuse_colour.py` | Transfers one render's colour onto another's frames. Built because the two properties are separable: stability is temporal, vibrancy is per-frame colour. The transfer statistics are smoothed over a temporal window first, so only the slow colour trend crosses over and none of the source's frame-to-frame jitter. |

**5. Measure.** `carla2real/evaluation/veg_report.py`, `carla2real/evaluation/tail_check.py`, `carla2real/evaluation/road_texture.py`, `carla2real/evaluation/true_instability.py`,
`carla2real/evaluation/epoch_sweep.py`, `carla2real/evaluation/flicker_report.py`, and `carla2real/evaluation/score_vp.py` for end-to-end perception scoring against
an external stack (optional; set `PERCEPTION_ROOT`).

## A warning about metrics

Every metric in this project has been wrong at least once, and the failures were not random —
they shared a cause. Flicker metrics reward a still image, and sharpness metrics reward any
high-frequency content whether or not it means anything. A model that wove a fixed crosshatch over
every flat surface scored *better* on both than the model it replaced.

So the tools here are built to be falsifiable rather than flattering:

- `carla2real/evaluation/road_texture.py` asks whether there is detail where the label map says there should be none,
  instead of asking whether there is detail.
- `carla2real/evaluation/veg_report.py` reports near and far detail separately and refuses to average them, because
  raising near detail while distant trees stay bad is not a fix.
- `carla2real/evaluation/true_instability.py` separates real instability from detail in motion using optical flow.
- `carla2real/evaluation/road_sky_ceiling.py` compares against real photographs, not against the previous model, because
  a comparison to the parent tells you a change is new, not that it is an improvement.
- `carla2real/evaluation/true_instability.py` is the one to use for any stability claim across versions of differing
  sharpness. The plain alternation metric counts a stationary detailed surface sweeping past a
  moving camera as flicker, and has already sent this project chasing a regression that was 1%.

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
`scripts/common/gpu_wait.sh` exists to serialise them.

## First run

```bash
git clone <this repo> && cd carla2real
pip install -r requirements.txt
source configs/config.sh
python3 scripts/init_asset_dirs.py

# A. render from an existing recording (needs weights in pix2pixHD/checkpoints/<model>/)
bash scripts/inference/render_model.sh sunny carla2real_semantic_v50_graft v50 Town05

# B. or record your own first (needs a CARLA server on localhost:2000)
python3 -m carla2real.recording.record_town_auto --town Town05 --weather sunny --outname Town05_sunny_inst
# Prepare matching 65-class inference channels separately (see docs/INFERENCE_FLOW.md).
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
| Trained weights | **700 MB** per model (`latest_net_G.pth`, 183.5M fp32 params; the discriminator adds 33 MB and is only needed to resume training), and derived from licensed training footage | Train it yourself — `experiments/training/train_v50.sh` is the sunny baseline recipe and `experiments/training/train_v51_night.sh` the night one; `experiments/training/train_v63_veg.sh` / `experiments/training/train_v64_veg.sh` are shipped as worked *negative* results. Or request the weights separately. |
| Training corpus | Real driving footage, licensed separately | Not redistributable here — see `THIRD_PARTY_NOTICES.md` |
| CARLA 0.9.16 | Records the drives | carla.org |
| MoGe, DVP, Real-ESRGAN | Depth/normal channels, optional temporal and upscale stages | Upstream projects |
| Perception stack (optional) | Only for `carla2real/evaluation/score_vp.py` scoring | Not part of this project; set `PERCEPTION_ROOT` |

## Which version does what

These are the documented baseline recipes, not verified runnable commands for this checkout.
The v75 texture input and `make_v50r.sh` are missing; see [known gaps](docs/INFERENCE_FLOW.md).

Two baselines, chosen by eye on side-by-side comparison rather than by metric:

| Condition | Baseline | How to produce |
|---|---|---|
| Sunny | **v85d** (since 2026-09-22) | `TEXTURE=1 bash scripts/inference/render_model.sh sunny carla2real_semantic_v85_zod_fixed v85 <Town...>`, **then** `TAG=v85q BASE_TAG=v85 COLOUR_SRC=v50m make_v50r.sh`, **then** `deepen_road_shadows.py` |
| Night | **v79** (since 2026-09-15) | `bash scripts/inference/render_model.sh night carla2real_semantic_v79_clean_night v79 <Town...>` |

**Sunny is now fully licensed too.** v85 trains on PandaSet and ZOD alone — no Mapillary Vistas, no
Cityscapes. Both baselines are free of research-only data as of 2026-09-22.

Two things got it there, and neither was more data:

**ZOD was fixed rather than avoided.** Measured per source at a common width, ZOD carries detail
2.71 and saturation 37.1 against PandaSet's 5.57 / 42.7 and Mapillary's 5.43 / 61.8. A corpus
without Mapillary is three quarters ZOD, and the first attempt at one (v81) lost 41% of its detail
and was rejected on sight. `fix_zod_appearance.py` sharpens ZOD with a **clamped** unsharp — every
pixel clamped to the local min/max of the input, so overshoot is impossible by construction —
reaching detail 4.92 at **0.00% halo**, where a plain unsharp mask that strong rings on 26% of edge
pixels. A generator trained on ringing learns to paint ringing.

**Road shadows are a post-pass, not a training problem.** Measured on road pixels as the darkest
twentieth over the median (lower = deeper):

| | road shadow depth |
|---|---|
| CARLA source | 0.542 |
| v50m, the old unlicensed baseline | **0.269** |
| v85 raw render | 0.659 |
| v85q, full delivery chain | 0.630 |
| **v85d** = v85q + `deepen_road_shadows.py` | **0.306** |

The raw render and the full chain differ by 0.03, so no corpus change could have fixed this: the
sun's position is not in the label map. It *is* in CARLA, which has the geometry and the light.
`deepen_road_shadows.py` transfers CARLA's ground-shadow shape onto road and sidewalk as a
luminance ratio only — the render keeps its own colour, grain and exposure — at 2.5x CARLA's own
depth, because CARLA's rendered shadows are themselves half as deep as a real one.

**Sunny needs the second step.** `scripts/inference/render_model.sh` is the evaluation chain — it renders, stabilises
and applies the protection passes, which is enough to score a model but is *not* the full sunny
delivery. Three stages only the delivery chain runs, and the artefact each one removes:

| stage | without it |
|---|---|
| `carla2real/postprocessing/protect_vehicle_colour.py` | the generator repaints vehicles per frame, so a car cycles through colours as it drives. The fix takes hue and saturation from CARLA — identical every frame — and keeps the render's own luminance. |
| `carla2real/postprocessing/protect_buildings.py` | facades are invented from a label that says only "building", differently each frame. That reinvention *is* the building shimmer; injecting CARLA's real window grids stops it and raises facade detail. |
| `carla2real/postprocessing/class_deshimmer.py` | vehicle shadows and contact areas break up frame to frame. Cars want strength 0.55 at flow tolerance 4.5 — looser trails, tighter flickers. |

Night does not need it: `scripts/inference/render_model.sh` composites CARLA's lamp pools back in, and the night
corpus does not show the vehicle-repaint behaviour to the same degree.

**Both baselines are trained only on openly licensed data** — PandaSet (CC BY 4.0) and the Zenseact
Open Dataset (CC BY-SA 4.0), alongside Mapillary Vistas and Cityscapes. The 21 videos of
unestablished provenance that earlier models were trained on are gone from both corpora.

**Night (v79) is the stricter of the two.** Its corpus is PandaSet and ZOD and nothing else: Dark
Zurich, which is licensed for academic research only and was 21% of the night data in v76, is gone,
verified at zero frames. Removing it cost nothing — ZOD night more than replaced it, and the clean
corpus is larger (14,674 pairs against 12,546). Sunny (v75q) still draws 56% of its corpus from
Mapillary Vistas and Cityscapes, both of which are free for research but not for commercial use, so
only the night model is clean for a commercial reader. See
`datasets/README.md` for what to download and `THIRD_PARTY_NOTICES.md` for what each licence
requires.

**ZOD is share-alike, and that reaches the weights.** ZOD's CC BY-SA 4.0 requires derivative works
to carry the same licence. Whether trained weights are a derivative of their training data is
unsettled, but anyone publishing v75q or v79 weights should assume they are and licence
them CC BY-SA 4.0. The code in this repository is unaffected and remains Apache-2.0. If you need
weights without that question, `v73` (sunny) and `v69` (night) are trained on PandaSet only — CC BY
4.0, no share-alike — and score within half a CIPO point of the baselines.

### Why these two

Measured against the previous baselines on Vision Pilot, same towns, like for like:

| | previous | new | CIPO, five towns |
|---|---|---|---|
| Night | v59 | **v79** | **+2.3 pts**, with the best lane MAE and jitter in the project |
| Sunny | v50m | **v85d** | **+1.5 pts** — the best in the project, on fully licensed data |

Night is a straight improvement. Sunny trades a point of recall for a clean licence and a lower
false-alarm rate, and was chosen on the side-by-side rather than on the number.

**Sunny no longer costs anything for its licence.** Earlier revisions of this file said the deficit
was intrinsic to dropping the unlicensed footage. That was wrong twice over: the loss came from what
*replaced* it (a corpus three quarters ZOD), and the remaining visual gap was a post-pass away, not
a training problem. Both were found by measuring per source and per class rather than by trusting
the perception score — see the note below.

**CIPO recall is not a proxy for visual fidelity.** v81q scored +0.5 against v50m — better than any
sunny model before it — while carrying 41% less detail, and was rejected at a glance. The
perception stack wants a vehicle that separates cleanly from the road, and a soft flat render gives
it one. Report `experiments/report_experiments.py` and per-class sharpness first; read CIPO second.

The image metrics are where the ZOD run paid off. Fine-tuning on a corpus dominated by one source
drifts tone badly — v73, PandaSet only, comes out at −34.0 on cars and +38.3 on road. Adding a
second, differently exposed licensed source collapses that to −8.4/−4.6 on v75 and +1.6/−2.0 on
v76, the best tone readings this project has recorded. Neither a lower learning rate nor reweighting
the corpus had moved it; a second source did.

### The previous sunny baseline, for reference

**v50m was not a trained model, and that was the point.** `experiments/delivery/make_v50m.sh` renders with `v50_graft`,
repairs it through the delivery chain, then grades it with colour lifted from a `v63` render of the
same town by `carla2real/postprocessing/fuse_colour.py`. Reproducing it needs both renders, frame-aligned. It is kept because
the delivery chain it established is the one v75 and v76 still run through.

v50m is v50l plus three fixes, each aimed at a defect found by watching the clips:

| defect | fix |
|---|---|
| ghost trails on moving vehicles | car de-shimmer strength 0.85 → 0.55, flow tolerance 6.0 → 4.5 |
| colour switching between shades | grade driven by median/MAD instead of mean/std, plus a per-frame rate cap; window 61 → 91 |
| not sharp enough | unsharp 0.55 → 0.75, CARLA facade injection 1.0 → 1.2 |

Measured against v50l: **real instability +1.2%** (warped residual 10.45 → 10.58) for **+15%
detail**, less trailing, colour held, detection recall unchanged. v50l in turn beat the older v50d
baseline by −10.2% flicker and +23.2% colourfulness over five towns.

The model the colour came from is the *worst* one measured on detection recall (v63, 0.836).
Borrowing its colour beat adopting it, because stability is temporal and colour is per-frame —
so the two are separable and no retrain was needed. `docs/EXPERIMENTS.md` notes 38 and 39.

Earlier chains are kept because they are the lineage, and because each carries a fix the one
before it predates:

- `experiments/delivery/make_v50d.sh` — the previous sunny baseline. Predates the vehicle-colour and building fixes.
- `experiments/delivery/make_v50i.sh` / `experiments/delivery/make_v50j.sh` — the vehicle-colour achromatic guard and the per-region
  building-structure injection. `v50j` is the carrier both `v50l` and `v50m` are built on.
- `experiments/delivery/make_v50kl.sh` — the first fusion, producing `v50l` (and `v50k`, which is closed).
- `experiments/delivery/make_v51d.sh` — night, with per-class de-shimmer and despeckle.

Do not apply the grade to `v50d` (that combination is `v50k`, and it is closed): a global grade
amplifies a per-object hue error instead of fixing it, so v50d's magenta bus came out worse. The
carrier must already have the achromatic guard. `docs/EXPERIMENTS.md` records why each was accepted or rejected, including the
vegetation-loss work (notes 35–37) that improved distant foliage but overshot road grain to ~2x a
real photograph and so was never promoted.
