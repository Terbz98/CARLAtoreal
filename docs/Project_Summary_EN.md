# Project Summary: CARLA Sim-to-Real Image Translation

> Last updated: 2026-06-23

---

## 1. What This Project Does

Convert **semantic segmentation label maps (Cityscapes-19 format) → photorealistic real-world road images**.

Trained with **pix2pixHD** (a conditional GAN), with the goal of teaching the model:
"Give me a road label map, and I will generate a photo that looks like it was taken by a real camera."

**All training data uses real NuRec driving footage** — no CARLA simulated data is used for training.

**Key insight:** The label map itself contains no weather information (the road label looks almost identical in sunny/night/snowy conditions), so using a one-hot weather channel for conditioning has limited effectiveness. The current approach has shifted to **training a separate dedicated model for each weather condition**.

---

## 2. Overall Pipeline

```
Real road footage (NuRec MP4, 00~20.mp4)
        │
        ▼ ffmpeg extracts frames (900 frames per video)
  {n}_work/frames/*.jpg (2560×1440)
        │
        ▼ Mask2Former inference
  {n}_work/labels/*.png (Cityscapes-19, 1024×512)
        │
        ▼ Build training dataset (symlinks)
  datasets/training_semantic_{weather}/
        │
        ▼ pix2pixHD training (fine-tune from v6/v7)
  checkpoints/carla2real_semantic_{model}/
        │
        ▼ Inference / video generation
  results/mp4/
```

---

## 3. Model Versions and Current Status

### Completed Models

| Version | Training Data | Weather | Epochs | Status | Notes |
|------|---------|------|-------|------|------|
| v4 | NuRec 7722 + Cityscapes 2975 | None | 200 | Done | SegFormer labels |
| **v6** | NuRec 30775 + Cityscapes 2975 | None | 200 | **Done, sunny baseline** | Mask2Former labels |
| v7 | NuRec 33750 | None | 50 ft v6 | Done | Full-resolution version |
| v8_weather | NuRec 12600 | Sunny/Rain/Night/Fog | 100 ft v6 | Done (has bug) | Snow mislabeled as Fog |
| v9 | NuRec 9900 | Sunny/Night/Rain/Snow | 100 ft v6 | **Done** | Labels fixed, but night output is still confused |

### Models Currently in Training (Per-Weather Dedicated)

| Model | Data Source | Data Volume | Epoch Progress | Pretrain |
|------|---------|--------|-----------|---------|
| `carla2real_semantic_night` | 10~13.mp4 (night) | 3600 pairs | **6 / 150** | v7 |
| `carla2real_semantic_rain` | 14~16.mp4 (rain) | 2700 pairs | **8 / 150** | v7 |

```bash
# Monitoring
watch -n 10 "tail -3 ~/carla/pix2pixHD/checkpoints/carla2real_semantic_night/loss_log.txt && echo '---' && tail -3 ~/carla/pix2pixHD/checkpoints/carla2real_semantic_rain/loss_log.txt"
```

### Planned Models

| Model | Data Source | Status |
|------|---------|------|
| `carla2real_semantic_snow` | 17~19.mp4 (~1800 pairs) | Not started |

---

## 4. Important Paths

```
$CARLA2REAL_ROOT/
├── pix2pixHD/
│   ├── checkpoints/
│   │   ├── carla2real_semantic_v6/         ← Sunny baseline
│   │   ├── carla2real_semantic_v7/         ← v6 fine-tune
│   │   ├── carla2real_semantic_v8_weather/ ← Weather-conditioned (Snow bug)
│   │   ├── carla2real_semantic_v9/         ← Weather-conditioned, fixed (done)
│   │   ├── carla2real_semantic_night/      ← ★ Night-dedicated (training)
│   │   └── carla2real_semantic_rain/       ← ★ Rain-dedicated (training)
│   └── results/mp4/                        ← Synthesized video output
├── datasets/
│   ├── test_mp4/                           ← 21 NuRec videos (00~20.mp4)
│   │   └── {n}_work/frames & labels/
│   ├── training_semantic_night/            ← Night training set (3600 pairs)
│   ├── training_semantic_rain/             ← Rain training set (2700 pairs)
│   ├── training_semantic_v9/               ← v9 training set (9900 pairs)
│   ├── test_weather_v9/                    ← v9 CARLA inference test set
│   └── weather_test/{W}/                   ← CARLA 200 frames per weather
├── run_video_inference.py                  ← Main video inference program
├── train_semantic_night.sh                 ← Night training script
├── train_semantic_rain.sh                  ← Rain training script
├── test_weather_v9.sh                      ← v9 CARLA batch inference
├── monitor_training.py                     ← Real-time training loss monitor
└── README.md                               ← Technical details
```

---

## 5. NuRec Video-to-Weather Mapping

| Videos | Weather | Training Use |
|------|------|---------|
| 00~09, 20 | Sunny | v6 (existing) |
| 10~13 | Night | `training_semantic_night/` (3600 frames) |
| 14~16 | Rain | `training_semantic_rain/` (2700 frames) |
| 17~19 | Snow | Future snow model |

---

## 6. Existing Inference Results

### Synthesized Videos (`pix2pixHD/results/mp4/`)

| File | Model | Description |
|------|------|------|
| `00~11_synthesized.mp4` | v6 | NuRec sunny |
| `12_v9_night.mp4` | v9 | NuRec night (has confusion issue) |
| `16_v9_rain.mp4` | v9 | NuRec rain |
| `17_v9_snow.mp4` | v9 | NuRec snow |
| `carla_night_v9.mp4` | v9 | CARLA night |
| `carla_rain_v9.mp4` | v9 | CARLA rain |

### CARLA Static Inference (`pix2pixHD/results/carla2real_semantic_v9/`)
- `test_Sunny/Rain/Night/Fog_{gt|m2f}_latest/images/` — 4 weathers × GT/M2F labels

---

## 7. What's Currently Being Worked On

**Night and Rain dedicated models are training simultaneously** (started 2026-06-23)

```
Night: Epoch 6 / 150 (~3 hours remaining)
Rain : Epoch 8 / 150 (~2.5 hours remaining)
GPU  : RTX 5090 (32GB), running both trainings at once
```

**Why train separately (instead of using v9's weather conditioning):**
- The label map contains no weather information; the one-hot weather channel signal is too weak
- After v9 finished training, night output still looked like a snowy scene → the model can't distinguish weather from just 4 small channels
- Training separately: each model uses 100% of its capacity to learn a single weather condition, with no ambiguity

---

## 8. To-Do After Completion

### Once Night + Rain Models Are Done
1. Add `MODEL_CFG["night"]` and `MODEL_CFG["rain"]` to `run_video_inference.py`
2. Run inference on 10.mp4 (night) with the night model, and compare against v9's `12_v9_night.mp4`
3. Run inference on 15.mp4 (rain) with the rain model, and compare against v9's `16_v9_rain.mp4`

### Future Plans
- **Snow model**: use 17~19.mp4 (~1800 frames), fine-tune from v7, 150 epochs
- **All-weather comparison chart**: side-by-side comparison of Sunny (v6) / Night (night model) / Rain (rain model) / Snow (snow model)

---

## 9. Technical Architecture

### Training Pipeline
```
Label Map (19ch) → pix2pixHD Generator → Synthesized image (photorealistic)
```

Three original pix2pixHD source files were modified (only needed for the v8/v9 weather-conditioned versions):
- `options/base_options.py` — `--n_weather_classes`
- `models/pix2pixHD_model.py` — `encode_input()` appends a one-hot weather channel
- `data/aligned_dataset.py` — reads `weather_map.json`

### Mask2Former
- Model: `facebook/mask2former-swin-large-cityscapes-semantic`
- Output: Cityscapes-19 trainId (0~18), grayscale PNG
- **Note**: pixel=255 (unknown) must be remapped to 0, otherwise it causes a CUDA crash

### GAN Loss Health Indicators
- `G_GAN ≈ 1.0` — normal (>1.5 in early epochs is also normal)
- `G_VGG decreasing` — perceptual quality improving
- `D_real ≈ D_fake` — adversarial balance

---

## 10. Alternative Approaches and Related Papers

### Where the Current Approach Stands
```
Semantic label map → pix2pixHD → Synthesized image
```
Pros: scene structure is controllable; Cons: weather control requires training multiple separate models

### Main Alternative Directions

| Method | Representative Paper | Difference | Difficulty |
|------|---------|------|------|
| **SPADE / GauGAN** | Park et al., CVPR 2019 | Also takes labels as input, but better normalization yields higher quality | Low |
| **SEAN** | Zhu et al., CVPR 2020 | Independent style per semantic class, allowing local control of weather style | Low |
| **CUT** | Park et al., ECCV 2020 | Unpaired, no labels needed, directly converts CARLA RGB to a realistic style | Low |
| **StarGAN v2** | Choi et al., CVPR 2020 | A single model handles multiple weather conditions, controlled via style codes | Medium |
| **ControlNet** | Zhang et al., ICCV 2023 | Diffusion + label + text prompt, highest quality | High |
| **SynDiff-AD** | Goel et al., arXiv 2024 | ControlNet specialized for autonomous-driving data generation, +2.3% mIoU | High |
| **CARLA2Real** | Pasios et al., arXiv 2024 | Runs directly as a CARLA plugin, real-time conversion, least effort | Lowest |

> **The Warwick IV 2024 paper (arXiv 2404.09111)** directly compares pix2pixHD vs. ControlNet on CARLA labels, concluding that ControlNet produces better quality than pix2pixHD.

**Recommended order to try:**
1. SPADE — smallest change, a direct upgrade to pix2pixHD
2. CUT — quick validation of the unpaired direction
3. ControlNet — long-term goal (requires resolving the SSL issue)

---

## 11. Detailed Handover Document

For complete technical details, all script descriptions, and a summary of related papers, see:
`docs/Handover_pix2pixHD_EN.md`
