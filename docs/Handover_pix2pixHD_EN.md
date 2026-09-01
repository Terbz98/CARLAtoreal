# Handover Document — CARLA sim-to-real pix2pixHD Pipeline

> **Date written:** 2026-06-22 (last updated: 2026-06-25)
> **Project path:** `$CARLA2REAL_ROOT/`
> **New maintainers should read this document first, then refer to `README.md` (repository root) for additional details**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Technical Background](#2-core-technical-background)
3. [Hardware and Environment](#3-hardware-and-environment)
4. [Overall Pipeline Architecture](#4-overall-pipeline-architecture)
5. [Model Version History](#5-model-version-history)
0. [Current Progress and Future Plans](#0-current-progress-and-future-plans)
6. [Dataset Overview](#6-dataset-overview)
7. [NuRec Test Video Weather Labels](#7-nurec-test-video-weather-labels)
8. [Detailed Directory Structure](#8-detailed-directory-structure)
9. [Key Script Descriptions](#9-key-script-descriptions)
10. [Training Commands](#10-training-commands)
11. [Inference Commands](#11-inference-commands)
12. [Current Status](#12-current-status)
13. [Known Issues and Notes](#13-known-issues-and-notes)
14. [Recommended Next Steps for the New Maintainer](#14-recommended-next-steps-for-the-new-maintainer)

---

## 1. Project Overview

**Goal:** Use a conditional GAN (pix2pixHD) to convert semantic label maps (Cityscapes-19 format — note: a more detailed PS-format version exists and could be swapped in to support more label classes) into photorealistic road driving images, with support for weather-conditioned control (sunny / night / rain / snow).

**Use cases:**
- Data augmentation for autonomous-vehicle perception modules
- Generating training images under different weather conditions
- Converting CARLA simulated scene labels into near-realistic images

**Key design decision:** Training data is **entirely real NuRec footage** — no CARLA simulated images are used for training. (Otherwise the model would learn the visual style of simulated images.)

**Two subsystems (only pix2pixHD is actively developed):**
- `pix2pixHD/` — the main conditional GAN synthesis model in active use
- `cosmos-transfer1/` — NVIDIA's diffusion model (experimental, development discontinued) — discontinued because its Docker image does not support the RTX 5090

---

## 2. Core Technical Background

### 2.1 pix2pixHD

A high-resolution image-to-image translation GAN proposed by NVIDIA (CVPR 2018).

- **Generator (Global):** Encoder → 9 ResBlocks → Decoder, `ngf=64`, 4 downsampling steps
- **Discriminator:** Multi-scale (2 different scales), LSGAN loss
- **Loss:** GAN loss + Feature Matching loss (lambda_feat) + VGG Perceptual loss

**Input format:**
- `label_nc=19`: input is a Cityscapes-19 trainId grayscale label map (0~18, 255=unknown)
- `n_weather_classes=4`: an additional 4 one-hot spatial channels appended to the label (used only in v8/v9)

**Loss health indicators:**
```
G_GAN ≈ 1.0       → normal balance (each of the 2 discriminators contributes 0.5)
G_GAN > 1.5       → discriminator too strong (normal in early epochs)
G_VGG decreasing  → perceptual quality improving
D_real ≈ D_fake   → adversarial balance
```

### 2.2 Cityscapes-19 trainId Categories

```
0=road  1=sidewalk  2=building  3=wall  4=fence  5=pole
6=traffic light  7=traffic sign  8=vegetation  9=terrain  10=sky
11=person  12=rider  13=car  14=truck  15=bus  16=train
17=motorcycle  18=bicycle
```

**Note:** pixel value 255 = unknown, which causes a CUDA scatter index-out-of-bounds crash and must be mapped to 0 (road).

### 2.3 Mask2Former

A Transformer-based segmentation model used to generate Cityscapes-19 labels from RGB images.

- **Model:** `facebook/mask2former-swin-large-cityscapes-semantic`
- **Purpose:** Segments real NuRec images to produce training labels for pix2pixHD
- **Speed:** ~15-17 FPS (1024px, RTX 5090)
- The model is cached locally after the first download from Hugging Face and does not require network access afterward

### 2.4 Key Design Insight: Why We Switched from a Weather-Conditioned Model to Separate Training

**Fundamental limitation of weather conditioning:**

At the semantic level, a Cityscapes-19 label map carries no weather information — the same road looks nearly identical whether it's sunny, night, or snowy (road=0, sky=10...).
This means the only weather signal available to pix2pixHD is the appended one-hot channel (4 channels), which is too weak a signal. The model tends to drift toward the domain with the strongest visual contrast (snow's white, high-contrast appearance), causing night output to look like a snowy scene.

**Conclusion:** For this architecture (label→image), **training a dedicated per-weather model** is more effective than weather conditioning.

### 2.5 Weather Conditioning Mechanism (used in v8/v9, for reference)

In the `encode_input` function in `pix2pixHD/models/pix2pixHD_model.py`:
- The weather id (0~3) is converted into 4 spatial one-hot channels (size H×W)
- These are appended to the label tensor before being fed into the Generator
- The dataset requires a `weather_map.json` or `{phase}_weather.json`

### 2.6 CARLA Semantic Decoding

The CARLA semantic sensor outputs raw BGRA data. When saved, `[:,:,:3]` effectively stores BGR as RGB. Decoding requires two lookup tables:

```python
CARLA_BGR_FILE_TO_ID  # (B,G,R as stored) → CARLA category id
CARLA_TO_CITY19       # CARLA category id → Cityscapes trainId
```

Both tables are defined in `make_carla_videos.py` and `prepare_gt_test_label.py`.

---

## 3. Hardware and Environment

### Hardware
- **GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM)
- **CUDA:** 13.0

### Conda Environments

| Environment Name | Purpose |
| `carla_env` | **Main environment**: pix2pixHD training/inference, Mask2Former, CARLA Python API, dataset preparation |
| `cosmos-transfer1` | Cosmos diffusion inference (experimental, rarely used) |

**Activation:**
```bash
conda activate carla_env
# or without activating
conda run -n carla_env python3 script.py
```

### CARLA Server
```bash
# CARLA installation path
$CARLA2REAL_ROOT/simulator/CARLA_0.9.16/CarlaUE4.sh

# Headless launch
$CARLA2REAL_ROOT/simulator/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen -world-port=2000 &

# Check if it's running
ss -tlnp | grep 2000
```

---

## 4. Overall Pipeline Architecture

### 4.1 Training Pipeline (real NuRec data, per-weather models)

```
NuRec MP4 video (test_mp4/10~13.mp4, night)
         │
         ▼ ffmpeg extracts frames (all 900 frames per video)
    {n}_work/frames/*.jpg   (2560×1440 JPG)
         │
         ▼ Mask2Former (run_video_inference.py or manual)
    {n}_work/labels/*.png   (Cityscapes-19 grayscale PNG, 1024×512)
         │
         ▼ Build dataset (symlinks)
    datasets/training_semantic_night/
      train_img/   → frames/*.jpg
      train_label/ → labels/*.png
         │
         ▼ train_semantic_night.sh
    pix2pixHD/checkpoints/carla2real_semantic_night/
```

### 4.2 Inference Pipeline (video)

```
Input MP4 (any resolution)
    │
    ▼ ffmpeg (extract frames, skip if already exist)
    {stem}_work/frames/
    │
    ▼ Mask2Former (skip if labels already exist)
    {stem}_work/labels/
    │
    ▼ pix2pixHD test.py (label → synthesized image)
    {stem}_work/results/.../images/  (*_synthesized_image.jpg)
    │
    ▼ ffmpeg (combine → mp4, upscale back to original resolution)
    output.mp4
```

**Core script:** `run_video_inference.py` (every step is cached and not re-run unnecessarily)

---

## 5. Model Version History

| Version | Training Data | Weather | Epochs | Pretrain | lambda_feat | Status |
|------|----------|---------|--------|----------|------------|------|
| v4 | 7722 NuRec + 2975 Cityscapes | None | 200 | scratch | 10 | Done |
| **v6** | 30775 NuRec + 2975 Cityscapes | None | 200 | scratch | 10 | **Stable baseline** |
| v7 | 33750 NuRec (full 1024×512) | None | 50 | v6 | 20 | Done |
| v8_weather | 12600 NuRec | Sunny/Rain/Night/Fog¹ | 100 | v6 | 10 | Done (has issues) |
| **v9** | 9900 NuRec | Sunny/Night/Rain/Snow | 100 | v6 | 20 | **Done** |
| **night** | 3600 NuRec (10~13.mp4) | None (night-dedicated) | 150 | v7 | 20 | **Done** |
| **rain** | 2700 NuRec (14~16.mp4) | None (rain-dedicated) | 150 | v7 | 20 | **Done** |

¹ v8_weather mislabeled the Snow videos (17~19.mp4) as Fog (id=3), causing night output to resemble a snowy scene.

### v9 vs. v8_weather Differences

| Issue | v8_weather | v9 |
|------|---------|-----|
| Snow label | Mislabeled as Fog (id=3) | Snow is its own correct id (id=3) |
| Weather ID | Sunny=0, Rain=1, Night=2, Fog=3 | Sunny=0, Night=1, Rain=2, Snow=3 |
| lambda_feat | 10 (blurry output) | **20** (improved) |
| Data distribution | Imbalanced | Sunny/Night 3000 each, Rain 2100, Snow 1800 |

**v9 results:** Snow improved, but night output still looks like a snowy scene.
The root cause is that the label map has no weather information and the one-hot channel signal is too weak. → Switched to per-weather dedicated models.

### Per-Weather Dedicated Model Plan

| Model | Source Videos | Data Volume | Status |
|------|---------|--------|------|
| `v6` (sunny) | 00~09, 20 | 30k+ | Existing, used directly |
| `carla2real_semantic_night` | 10~13 | 3600 | **Done** |
| `carla2real_semantic_rain` | 14~16 | 2700 | **Done** |
| `carla2real_semantic_snow` | 17~19 | ~1800 | Not started (future) |

---

## 6. Dataset Overview

### 6.1 NuRec Test Videos (main training/testing source)

Path: `$CARLA2REAL_ROOT/datasets/test_mp4/`
Resolution: 2560×1440, 30fps, 900 frames per video, 21 videos total (00~20.mp4)

```
datasets/test_mp4/{n}_work/
  frames/    # JPG (2560×1440, 900 images per video)
  labels/    # Mask2Former Cityscapes-19 labels (PNG, 1024×512)
  dataroot/  # symlink directory used for pix2pixHD inference
  results/   # pix2pixHD inference output
```

### 6.2 Training Datasets

| Path | Purpose | Size |
|------|------|------|
| `datasets/training_semantic_v6/` | v6 training set (NuRec + Cityscapes, no weather) | — |
| `datasets/training_semantic_v9/` | v9 training set (9900 pairs, includes weather_map.json) | symlinks |
| **`datasets/training_semantic_night/`** | **Night model training set (3600 pairs, symlinks)** | symlinks |
| **`datasets/training_semantic_rain/`** | **Rain model training set (2700 pairs, symlinks)** | symlinks |

### 6.3 CARLA Static Inference Test Sets

| Path | Purpose |
|------|------|
| `datasets/test_weather_v8w/` | v8_weather inference test (Sunny/Rain/Night/Fog × gt/m2f) |
| `datasets/test_weather_v9/` | v9 inference test (Sunny/Rain/Night/Fog × gt/m2f) |
| `datasets/weather_test/{W}/` | CARLA raw data, 200 frames per weather (rgb/semantic/gt_label/m2f_label) |

### 6.4 CARLA Recorded Data

| Path | Contents |
|------|------|
| `datasets/recorded_Town03/` | Town03, 1000 frames (rgb/ + semantic/) |
| `datasets/recorded_night/` | Town03 at night, 600 frames (rgb/ + semantic/ + night.mp4) |

---

## 7. NuRec Test Video Weather Labels

| Videos | Weather | Training Use |
|------|------|---------|
| 00~09, 20 | Sunny | v6 training |
| 10~13 | **Night** | `training_semantic_night/` (3600 frames) |
| 14~16 | **Rain** | `training_semantic_rain/` (2700 frames) |
| 17~19 | **Snow** | Future snow model (~1800 frames) |

---

## 8. Detailed Directory Structure

```
$CARLA2REAL_ROOT/
│
├── pix2pixHD/                        # Modified pix2pixHD (main working directory)
│   ├── train.py / test.py            # Training/inference entry points
│   ├── models/pix2pixHD_model.py     # ★ Modified: weather conditioning (encode_input)
│   ├── data/aligned_dataset.py       # ★ Modified: reads weather_map.json
│   ├── options/base_options.py       # ★ Modified: --n_weather_classes parameter
│   ├── checkpoints/
│   │   ├── carla2real_semantic_v6/         # baseline (200ep, no weather)
│   │   ├── carla2real_semantic_v7/         # ft from v6 (50ep)
│   │   ├── carla2real_semantic_v8_weather/ # weather version (100ep, Snow mislabeled)
│   │   ├── carla2real_semantic_v9/         # fixed weather version (100ep, done)
│   │   ├── carla2real_semantic_night/      # ★ night-dedicated (150ep, training)
│   │   └── carla2real_semantic_rain/       # ★ rain-dedicated (150ep, training)
│   └── results/
│       ├── carla2real_semantic_v8_weather/ # v8 per-weather gt/m2f inference results
│       ├── carla2real_semantic_v9/         # v9 per-weather gt/m2f inference results
│       └── mp4/                      # Synthesized video output
│           ├── 00~11_synthesized.mp4        # NuRec Sunny (v6)
│           ├── 12_v9_night.mp4              # NuRec Night (v9)
│           ├── 16_v9_rain.mp4               # NuRec Rain (v9)
│           ├── 17_v9_snow.mp4               # NuRec Snow (v9)
│           ├── carla_night_v9.mp4           # CARLA Night (v9)
│           └── carla_rain_v9.mp4            # CARLA Rain (v9)
│
├── datasets/
│   ├── test_mp4/                     # 21 NuRec videos (main data)
│   │   └── {n}_work/frames/ labels/  # Preprocessing results per video
│   ├── training_semantic_night/      # ★ Night training set (3600 pairs, symlinks)
│   ├── training_semantic_rain/       # ★ Rain training set (2700 pairs, symlinks)
│   ├── training_semantic_v9/         # v9 training set (9900 pairs + weather_map.json)
│   ├── training_semantic_v6/         # v6 training set (30775 NuRec + 2975 Cityscapes)
│   ├── test_weather_v9/              # v9 static inference test (symlinks + weather JSONs)
│   ├── test_weather_v8w/             # v8 static inference test
│   ├── weather_test/{W}/             # CARLA 200 frames per weather
│   ├── recorded_night/               # CARLA night 600 frames + night.mp4
│   ├── recorded_Town03/              # CARLA Town03 1000 frames
│   └── cityscapes_parquet/           # Cityscapes cache (do not delete)
│
├── simulator/CARLA_0.9.16/CarlaUE4.sh
│
├── README.md                         # Technical details (recommended reading)
│
├── [Training scripts]
│   ├── train_semantic_v6.sh          # v6
│   ├── train_semantic_v7.sh          # v7
│   ├── train_semantic_v8_weather.sh  # v8_weather
│   ├── train_semantic_v9.sh          # v9 (done)
│   ├── train_semantic_night.sh       # ★ Night-dedicated (training)
│   └── train_semantic_rain.sh        # ★ Rain-dedicated (training)
│
├── [Inference scripts]
│   ├── run_video_inference.py        # ★ Main inference (MP4→synthesized MP4)
│   ├── test_weather_v9.sh            # v9 CARLA batch inference (done)
│   ├── test_weather_v8_weather.sh    # v8 CARLA batch inference
│   └── test_Town03_v6_gt.sh          # v6 Town03 GT label inference
│
├── [Dataset preparation]
│   ├── prepare_v9_dataset.py         # v9 training set preparation
│   └── prepare_gt_test_label.py      # CARLA GT semantic → Cityscapes-19
│
├── [CARLA recording]
│   ├── record_night_auto.py          # Headless night recording
│   └── record_weather_batch.py       # Batch weather recording
│
└── [Monitoring/Tools]
    ├── monitor_training.py            # Real-time loss monitoring (updates every 10s)
    ├── make_carla_videos.py           # CARLA synthesized videos
    └── compare_v6_v8_weather.py       # Comparison chart generation
```

---

## 9. Key Script Descriptions

### 9.1 `run_video_inference.py` (most frequently used)

MP4 → Mask2Former → pix2pixHD → synthesized MP4, with each step cached (not re-run unnecessarily).

```python
# Currently supported models (MODEL_CFG)
"v6"         → carla2real_semantic_v6 (no weather)
"v7"         → carla2real_semantic_v7 (no weather, full-resolution ft)
"v8_weather" → carla2real_semantic_v8_weather (n_weather_classes=4)
"v9"         → carla2real_semantic_v9 (n_weather_classes=4)
"night"      → carla2real_semantic_night (no weather, night-dedicated)
"rain"       → carla2real_semantic_rain (no weather, rain-dedicated)

# v9's WEATHER_ID
WEATHER_ID["v9"] = {"sunny": 0, "night": 1, "rain": 2, "snow": 3}
```

**Special use case — using the Night model to turn a sunny video into night:**
```bash
# The label map contains no weather information, so the night model will render any label in a night style
conda run -n carla_env python3 run_video_inference.py \
  --input  datasets/test_mp4/01.mp4 \
  --output pix2pixHD/results/mp4/01_night_model.mp4 \
  --model  night
```

### 9.2 `test_carla_night_model.sh` / `test_carla_rain_model.sh`

CARLA static batch inference (GT labels + M2F labels), with no weather conditioning:
```bash
conda run -n carla_env bash test_carla_night_model.sh
# Results → pix2pixHD/results/carla2real_semantic_night/test_Night_{gt|m2f}_latest/images/

conda run -n carla_env bash test_carla_rain_model.sh
# Results → pix2pixHD/results/carla2real_semantic_rain/test_Rain_{gt|m2f}_latest/images/
```

### 9.3 `compare_night_rain_vs_v9.py`

Generates side-by-side comparison images of the night/rain dedicated models vs. v9 (10 frames × 4 conditions):
```bash
conda run -n carla_env python3 compare_night_rain_vs_v9.py
# Results → pix2pixHD/results/comparison_night_rain_vs_v9/
#   compare_Night_gt.jpg / compare_Night_m2f.jpg
#   compare_Rain_gt.jpg  / compare_Rain_m2f.jpg
# Columns: [Label | v9 | Dedicated model | CARLA GT RGB]
```

### 9.4 `train_semantic_night.sh` / `train_semantic_rain.sh`

Training scripts for the per-weather dedicated models:

```bash
python train.py \
  --name carla2real_semantic_night \
  --dataroot $CARLA2REAL_ROOT/datasets/training_semantic_night \
  --label_nc 19 --no_instance \
  --loadSize 1024 --fineSize 512 \
  --resize_or_crop scale_width_and_crop \
  --batchSize 4 \
  --load_pretrain .../carla2real_semantic_v7 \
  --niter 100 --niter_decay 50 \   # 150 epochs total
  --lr 0.0001 --lambda_feat 20 \
  --save_epoch_freq 10 --gpu_ids 0
```

### 9.5 `test_weather_v9.sh`

Batch static inference on CARLA's 4 weather conditions (GT labels + M2F labels):
```bash
conda run -n carla_env bash test_weather_v9.sh          # all weathers
conda run -n carla_env bash test_weather_v9.sh Night    # single weather
# Results → pix2pixHD/results/carla2real_semantic_v9/test_{W}_{gt|m2f}_latest/images/
```

### 9.6 `monitor_training.py`

Real-time monitoring, run directly in the terminal (no conda run needed):
```bash
conda activate carla_env && python3 $CARLA2REAL_ROOT/monitor_training.py carla2real_semantic_night
```

Or use `watch` to view both at once:
```bash
watch -n 10 "tail -3 ~/carla/pix2pixHD/checkpoints/carla2real_semantic_night/loss_log.txt && echo '---' && tail -3 ~/carla/pix2pixHD/checkpoints/carla2real_semantic_rain/loss_log.txt"
```

---

## 10. Training Commands

### Currently in Progress (Night + Rain training simultaneously)

```bash
cd $CARLA2REAL_ROOT

# If a restart is needed (already running in the background, usually not necessary)
nohup conda run -n carla_env bash train_semantic_night.sh \
  > pix2pixHD/checkpoints/carla2real_semantic_night_log.txt 2>&1 &

nohup conda run -n carla_env bash train_semantic_rain.sh \
  > pix2pixHD/checkpoints/carla2real_semantic_rain_log.txt 2>&1 &
```

### Resuming Interrupted Training

```bash
# Add --continue_train at the end of the python command in train_semantic_night.sh
```

### Future Snow Model (yet to be done)

```bash
# First confirm labels for 17~19.mp4 are complete (900 images each)
# After building the dataset, run:
conda run -n carla_env bash train_semantic_snow.sh
```

---

## 11. Inference Commands

### 11.1 Video Inference (most common use case)

```bash
cd $CARLA2REAL_ROOT

# v6 (sunny, no weather conditioning)
conda run -n carla_env python3 run_video_inference.py \
  --input  datasets/test_mp4/07.mp4 \
  --output pix2pixHD/results/mp4/07_v6.mp4 \
  --model  v6

# v7 (sunny, full-resolution fine-tune)
conda run -n carla_env python3 run_video_inference.py \
  --input  datasets/test_mp4/01.mp4 \
  --output pix2pixHD/results/mp4/01_v7.mp4 \
  --model  v7

# v9 (weather-conditioned version)
conda run -n carla_env python3 run_video_inference.py \
  --input   datasets/test_mp4/17.mp4 \
  --output  pix2pixHD/results/mp4/17_v9_snow.mp4 \
  --model   v9 --weather snow

# Night model (night-dedicated, or to turn a sunny video into night)
conda run -n carla_env python3 run_video_inference.py \
  --input  datasets/test_mp4/10.mp4 \
  --output pix2pixHD/results/mp4/10_night_model.mp4 \
  --model  night

# Rain model (rain-dedicated)
conda run -n carla_env python3 run_video_inference.py \
  --input  datasets/test_mp4/15.mp4 \
  --output pix2pixHD/results/mp4/15_rain_model.mp4 \
  --model  rain
```

### 11.2 CARLA Static Image Batch Inference

```bash
conda run -n carla_env bash test_weather_v9.sh          # v9, all weathers
conda run -n carla_env bash test_weather_v8_weather.sh  # v8, all weathers
```

### 11.3 CARLA Night Scene Recording + Inference

```bash
# Start the CARLA server
~/carla/simulator/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen &

# Record 600 frames of a night scene
conda run -n carla_env python3 record_night_auto.py --town Town03 --frames 600
# → datasets/recorded_night/rgb/ + semantic/ + night.mp4

# Run inference on the recorded video
conda run -n carla_env python3 run_video_inference.py \
  --input  datasets/recorded_night/night.mp4 \
  --output pix2pixHD/results/mp4/carla_night_v9.mp4 \
  --model  v9 --weather night
```

---

## 12. Current Status

**As of 2026-06-25**

### Completed

| Item | Description |
|------|------|
| v9 training | 100 epochs, G_VGG dropped from ~6 to ~4 |
| v9 video inference | 17_v9_snow / 16_v9_rain / 12_v9_night / carla_night_v9 / carla_rain_v9 |
| v9 CARLA static inference | All 4 weathers × gt/m2f completed |
| **Night model training** | **150 epochs complete (ft from v7)** |
| **Rain model training** | **150 epochs complete (ft from v7)** |
| **CARLA static inference (night/rain)** | test_Night/Rain_gt/m2f, 200 frames each |
| **v9 comparison charts** | compare_night_rain_vs_v9.py generated 4 side-by-side comparisons |
| **run_video_inference.py** | v7 / night / rain added to MODEL_CFG |
| **01.mp4 inference** | 01_v7.mp4 (v7), 01_night_model.mp4 (night model, Day→Night conversion) |

### In Progress

None (everything completed)

### Next Steps

- [ ] Snow model (17~19.mp4, ~1800 frames, ft from v7)
- [ ] Run inference on 15.mp4 (rain) with the rain model, compare against v9
- [ ] Run inference on 10.mp4 (night) with the night model, compare against v9

---

## 13. Known Issues and Notes

### Network SSL
The corporate network intercepts SSL:
- Hugging Face models: use `wget --no-check-certificate`, do not use `hf_hub_download`
- `cityscapesscripts/download/downloader.py` has been patched: added `session.verify = False`
- Cityscapes parquet cache is at `datasets/cityscapes_parquet/` (**do not delete**)
- Mask2Former is cached after download, and can be used offline afterward

### pix2pixHD Directory Naming Convention
The `--phase` parameter determines which subdirectory is read (when `label_nc>0`):
- `--phase test_Sunny_gt` → reads `{dataroot}/test_Sunny_gt_label/`

### weather_map.json Collision Issue
If multiple phases in the same dataroot share the same filenames, **a per-phase JSON must be used** (`{phase}_weather.json`), otherwise keys will overwrite each other.

### NuRec Video Frame Counts
Each video has 900 frames (30fps × 30 seconds). Night (4 videos) gives a maximum of 3600, Rain (3 videos) gives a maximum of 2700 — these videos cannot be further augmented.

### CARLA Traffic Manager Synchronous Mode
```python
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
vehicle.set_autopilot(True, 8000)  # port must match
```

### Automatic Resolution Calculation for Video Inference
```python
infer_h = int(round(TARGET_W * out_h / out_w / 2) * 2)
# 2560×1440 → infer at 1024×576 → ffmpeg lanczos upscale back to 2560×1440
# 1024×512  → infer at 1024×512 (no upscale needed)
```

---

## 14. Recommended Next Steps for the New Maintainer

### Confirm Immediately

```bash
# Check Night + Rain training status
watch -n 10 "tail -3 ~/carla/pix2pixHD/checkpoints/carla2real_semantic_night/loss_log.txt && echo '---' && tail -3 ~/carla/pix2pixHD/checkpoints/carla2real_semantic_rain/loss_log.txt"
```

### Things You Can Do Right Away

1. **View comparison results** — side-by-side images of the Night/Rain dedicated models vs. v9:
   ```
   pix2pixHD/results/comparison_night_rain_vs_v9/
   ```

2. **Run inference on NuRec night/rain videos**
   ```bash
   conda run -n carla_env python3 run_video_inference.py \
     --input datasets/test_mp4/10.mp4 --output pix2pixHD/results/mp4/10_night_model.mp4 --model night
   conda run -n carla_env python3 run_video_inference.py \
     --input datasets/test_mp4/15.mp4 --output pix2pixHD/results/mp4/15_rain_model.mp4 --model rain
   ```

3. **Snow model** — if results are satisfactory, continue building a snow model (refer to `train_semantic_night.sh`, data source 17~19.mp4)

### Choosing a Future Direction

| Direction | Description |
|------|------|
| **Snow model** | Build `carla2real_semantic_snow`, using 17~19.mp4 (~1800 frames, ft from v7) |
| **Improve inference quality** | Use ControlNet + Stable Diffusion (requires resolving the SSL issue) |
| **More night data** | Record more CARLA night scenes to supplement training data |
| **Comparison chart generation** | Modify `compare_v6_v8_weather.py` to create v6 vs. night/rain model comparisons |

---

## 0. Current Progress and Future Plans

> This is the most important quick summary in the whole document — new maintainers should read this section first.

---

### Overall Progress Timeline

```
[Done] v4 → v6 (baseline) → v7 → v8_weather (has issues) → v9 (fixed weather labels)
                                                                    │
                                                          Discovered weather conditioning
                                                          has limited effectiveness for
                                                          this architecture
                                                                    │
[In progress]                                          night model ──┤
                                                        rain model  ──┘  trained separately
                                                                    │
[Future]                                               snow model ────┘
```

---

### Current Progress (2026-06-25)

#### Completed Models

| Model | Description | Usage |
|------|------|---------|
| `carla2real_semantic_v6` | Sunny baseline, 200 epochs | `--model v6` |
| `carla2real_semantic_v7` | v6 fine-tune, full resolution | `--model v7` |
| `carla2real_semantic_v8_weather` | 4-weather-conditioned version, Snow label is wrong | `--model v8_weather --weather {sunny/rain/night/fog}` |
| `carla2real_semantic_v9` | Fixed Snow label, night still gets confused | `--model v9 --weather {sunny/night/rain/snow}` |
| **`carla2real_semantic_night`** | **Night-dedicated, 150 epochs ft from v7** | **`--model night`** |
| **`carla2real_semantic_rain`** | **Rain-dedicated, 150 epochs ft from v7** | **`--model rain`** |

#### Completed Inference Results

| File | Model | Description |
|------|------|------|
| `results/mp4/00~11_synthesized.mp4` | v6 | NuRec sunny |
| `results/mp4/12_v9_night.mp4` | v9 | NuRec night (has confusion issue) |
| `results/mp4/16_v9_rain.mp4` | v9 | NuRec rain |
| `results/mp4/17_v9_snow.mp4` | v9 | NuRec snow |
| `results/mp4/carla_night_v9.mp4` | v9 | CARLA night |
| `results/mp4/carla_rain_v9.mp4` | v9 | CARLA rain |
| **`results/mp4/01_v7.mp4`** | v7 | NuRec sunny (v7 version) |
| **`results/mp4/01_night_model.mp4`** | night | Sunny video → night style (Day→Night conversion) |
| `results/carla2real_semantic_v9/` | v9 | CARLA static inference for 4 weathers (gt + m2f) |
| **`results/carla2real_semantic_night/`** | night | CARLA Night static inference (gt + m2f, 200 frames each) |
| **`results/carla2real_semantic_rain/`** | rain | CARLA Rain static inference (gt + m2f, 200 frames each) |
| **`results/comparison_night_rain_vs_v9/`** | — | Side-by-side comparisons of night/rain vs. v9 (4 conditions × 10 frames) |

---

### Near-term To-Do

- [ ] **NuRec night video inference** → run `10.mp4` with the night model, compare against v9's `12_v9_night.mp4`
- [ ] **NuRec rain video inference** → run `15.mp4` with the rain model, compare against v9's `16_v9_rain.mp4`
- [ ] **Snow model** → build `carla2real_semantic_snow` (data: 17~19.mp4, ~1800 frames, ft from v7)

---

### Mid-term Plans (1~2 weeks)

- [ ] **Snow model** — fine-tune v7 using 17~19.mp4 (~1800 frames), 150 epochs
  ```bash
  # Build the dataset (refer to how training_semantic_night/ was built)
  # Run train_semantic_snow.sh (needs to be created)
  ```
- [ ] **All-weather comparison chart** — four-column comparison of Sunny (v6) / Night (night model) / Rain (rain model) / Snow (snow model)
- [ ] **Handover video** — record a video demonstrating the synthesized results under different weather conditions

---

### Long-term Directions (optional)

| Direction | Description | Difficulty |
|------|------|------|
| **ControlNet + SD** | Generate using label + text prompt, higher quality | High (requires resolving SSL + VRAM issues) |
| **CUT / CycleGAN** | Real image → directly converted to a weather, no label needed | Medium (unpaired translation) |
| **More NuRec data** | Record more night/rain/snow videos covering different road segments | Low (mainly recording time) |
| **mIoU evaluation** | Run segmentation on synthesized images to quantify quality | Low (existing tools can be used directly) |

---

## Related Papers and Alternative Approaches

> The following summarizes the main research directions for sim-to-real image translation, ordered from closest to the current approach to most cutting-edge.

---

### Direction One: Better Semantic-guided GANs (closest to the current approach)

A direct upgrade of the current pix2pixHD, still taking a label map as input, but with higher synthesis quality.

**SPADE / GauGAN** — Park et al., CVPR 2019
> Replaces batch norm with spatially-adaptive normalization, so semantic information isn't diluted in deeper layers of the network. Semantic consistency in regions like roads, sky, and vegetation is significantly better than pix2pixHD.
> - Paper: arXiv 1903.07291 | Code: github.com/NVlabs/SPADE

**SEAN** — Zhu et al., CVPR 2020 (Oral)
> Builds on SPADE by learning an independent style code per semantic class, allowing a "rainy reflective road surface" style to be injected into specific regions (such as the road) while leaving other regions unchanged.
> - Code: github.com/ZPdesu/SEAN

**Difference from the current approach:** Weather control can be achieved without modifying the architecture — simply swap the style reference image (rainy-road style + sunny label = rainy synthesized image).

---

### Direction Two: Unpaired Translation (no paired training data needed)

No label map needed — directly converts CARLA RGB into a realistic style.

**CycleGAN** — Zhu et al., ICCV 2017
> Learns a bidirectional mapping (CARLA↔real) using cycle-consistency loss, without needing paired data.
> - Code: junyanz.github.io/CycleGAN

**CUT (Contrastive Unpaired Translation)** — Park et al., ECCV 2020
> Replaces CycleGAN's second generator with a patch-level contrastive loss, training twice as fast with typically sharper results.
> - Code: github.com/taesungp/contrastive-unpaired-translation

**DCLGAN** — Han et al., CVPRW 2021
> Bidirectional contrastive learning, more stable than CUT on small datasets, and has been used experimentally for CARLA→real lane detection.
> - arXiv 2104.07689

**Best fit:** If you want to convert the visual style of an entire CARLA video clip to a realistic one (without needing labels), CUT is the fastest option.

---

### Direction Three: Single Model for Multiple Weathers (replacing the current per-weather separate training)

**MUNIT** — Huang et al., ECCV 2018
> Decomposes an image into a content code (scene structure) and a style code (weather style), allowing the same label to generate multiple weather styles.
> - Code: github.com/NVlabs/MUNIT

**StarGAN v2** — Choi et al., CVPR 2020
> A single generator paired with a learned domain style code can freely switch between multiple weather domains without needing to train multiple models.
> - arXiv 1912.01865

**Difference from the current approach:** These two methods could theoretically replace the current four-model architecture ("v6 (sunny) + night + rain + snow") with a single model, while also producing more diverse output.

---

### Direction Four: Diffusion Models (highest quality, most cutting-edge direction)

**ControlNet** — Zhang et al., ICCV 2023
> Attaches a trainable encoder to a frozen Stable Diffusion model, supporting spatial conditioning inputs such as segmentation maps and depth maps.
> Directly accepts Cityscapes-format label maps, with text prompts controlling weather (e.g. `"rainy night street"`).
> - Paper: arXiv 2302.05543

**SynDiff-AD** — Goel et al., arXiv 2024, UT Austin
> Uses ControlNet with weather-specific text prompts to generate autonomous-driving data, improving Mask2Former mIoU by 2.3% on the Waymo dataset.
> Directly addresses this project's problem: insufficient data for rare weather conditions (night, rain).
> - arXiv 2411.16776

**Exploring Generative AI for Sim2Real** — Zhao et al., IEEE IV 2024, University of Warwick
> **Most directly relevant**: also uses CARLA semantic labels as input, comparing pix2pixHD, OASIS (GAN), and ControlNet (diffusion), concluding that ControlNet outperforms both GAN methods in structural fidelity and visual quality.
> - arXiv 2404.09111

**InstructPix2Pix** — Brooks et al., CVPR 2023
> Directly edits images with text instructions (e.g. `"make it rainy"`), without retraining, and can be used to apply weather post-processing to existing v6 synthesized results.
> - arXiv 2211.09800

**Implementation barrier:** Requires Stable Diffusion weights (~4GB), and resolving the corporate network's SSL issue. Inference is slower than a GAN (~2-5 seconds per image vs. real-time for a GAN).

---

### Direction Five: Tools Specifically for CARLA

**CARLA2Real** — Pasios & Nikolaidis, arXiv 2024
> Runs as a CARLA plugin, converting CARLA output to a near-Cityscapes/KITTI/Mapillary visual style in real time (~13 FPS).
> **The least-effort integration approach**: no need to modify the existing pix2pixHD pipeline — simply make CARLA's output more realistic directly.
> - arXiv 2410.18238

---

### Direction Six: Neural Rendering / NeRF (most cutting-edge)

**NeRF** — Mildenhall et al., ECCV 2020
> Reconstructs a continuous 3D scene from multi-view photos, able to synthesize realistic images from arbitrary viewpoints, without needing a label map.

**S-NeRF++** — Chen et al., arXiv 2024
> Designed for large-scale outdoor autonomous-driving scenes, incorporating LiDAR priors to generate novel-view driving data, validated on nuScenes/Waymo.
> - arXiv 2402.02112

**UniSim** — Yang et al., CVPR 2023, Waabi
> Reconstructs scenes from real dashcam/recording data, supporting dynamic object insertion and removal, generating closed-loop AV test data.
> - arXiv 2308.01898

**Limitation:** Requires multi-view calibrated images + LiDAR, with limited applicability to NuRec's single-camera data.

---

### Comparison Across Directions

| Method | Needs Labels | Needs Paired Data | Weather Control | Quality | Implementation Difficulty |
|------|-----------|------------|---------|------|---------|
| **pix2pixHD (current)** | ✅ | ✅ | Manual one-hot | Medium | Done |
| **SPADE / SEAN** | ✅ | ✅ | Style swap | Medium-high | Low (architecture swap) |
| **CUT / CycleGAN** | ❌ | ❌ | ❌ | Medium | Low |
| **MUNIT / StarGAN v2** | ❌ | ❌ | ✅ style code | Medium-high | Medium |
| **ControlNet + SD** | ✅ | Light fine-tune | ✅ text prompt | Very high | High (SSL + VRAM) |
| **CARLA2Real plugin** | ❌ | ❌ | ❌ | Medium | Very low |
| **NeRF / UniSim** | ❌ | ❌ (needs multi-view) | Limited | Highest | Very high |

**Recommended priority order:**
1. **SPADE** — the most direct architecture upgrade, clear quality improvement, fully compatible data format
2. **CUT** — quick validation of the unpaired approach, no labels needed
3. **ControlNet** — long-term goal, worth trying once the SSL issue is resolved
4. **CARLA2Real plugin** — least effort, if you need to improve the CARLA output side
