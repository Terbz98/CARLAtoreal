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

**1. Record.** Drive CARLA and capture RGB plus the semantic camera.
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

Python 3 with PyTorch, OpenCV, NumPy and Pillow; CARLA 0.9.16 for recording; a CUDA GPU. A 2048-wide
render peaks around 27 GB of VRAM, so run one GPU job at a time.

## What is and is not in this repository

**Code only.** No datasets, no trained weights, no rendered video. The working tree this was
packaged from is ~550 GB; the repository is under 1 MB. A clone will not run until you supply:

| Missing | Why | How to get it |
|---|---|---|
| Trained weights | ~200 MB per model, and derived from licensed training footage | Train from scratch (`train_v63_veg.sh` is a worked example), or request them separately |
| Training corpus | Real driving footage, licensed separately | Not redistributable here — see `THIRD_PARTY_NOTICES.md` |
| CARLA 0.9.16 | Records the drives | carla.org |
| MoGe, DVP, Real-ESRGAN | Depth/normal channels, optional temporal and upscale stages | Upstream projects |
| Perception stack (optional) | Only for `score_vp.py` scoring | Not part of this project; set `PERCEPTION_ROOT` |

## Which version does what

Two baselines, chosen by eye on side-by-side comparison rather than by metric:

| Condition | Baseline | Delivery chain |
|---|---|---|
| Sunny | **v50d** | `make_v50d.sh` |
| Night | **v59** | `render_model.sh night <model> v59 <Town...>` with `TEMPORAL=1` |

Later candidate chains are included because they carry fixes the baselines predate:

- `make_v50i.sh` / `make_v50j.sh` — sunny, with the vehicle-colour achromatic guard and the
  per-region building-structure injection. `v50d` predates both.
- `make_v51d.sh` — night, with per-class de-shimmer and despeckle.

Neither candidate replaced its baseline: a candidate ships alongside, and the choice is made on a
side-by-side clip. `docs/EXPERIMENTS.md` records why each was accepted or rejected, including the
vegetation-loss work (notes 35–37) that improved distant foliage but overshot road grain to ~2x a
real photograph and so was never promoted.
