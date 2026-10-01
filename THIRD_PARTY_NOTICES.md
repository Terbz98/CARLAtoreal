# Third-party components and licensing status

**Read this before any public release.** The pix2pixHD question is resolved. The training-data
question is **not** resolved for any published weights: the current baselines (v90 daylight, v79
night) are fine-tuned on PandaSet and ZOD alone, but they start from weights that saw research-only
data. See "Weight provenance" below. Corrected 2026-10-01; earlier revisions of this file said
otherwise.

This repository is a derivative work. The table below lists what it is built on and what still
needs checking. Nothing here is legal advice; it is a list of the things a release review has to
answer.

| Component | Where it appears | Status |
|---|---|---|
| **NVIDIA pix2pixHD** | `pix2pixHD/` — this is a modified fork | **RESOLVED.** BSD licence, restored verbatim at `pix2pixHD/LICENSE.txt` (two notices: NVIDIA 2019, and pytorch-CycleGAN-and-pix2pix 2017). Commercial use is permitted; the copyright notice, conditions and disclaimer must be retained on redistribution, which they now are. Modifications are listed in `pix2pixHD/MODIFICATIONS.md`. |
| **CARLA simulator** | Recording scripts target CARLA 0.9.16 | Not redistributed here; users install it themselves. CARLA is MIT-licensed, its assets separately. |
| **Mapillary Vistas label space** | `--label_nc 65`, class ids throughout | Only the id scheme is referenced. No Mapillary data is redistributed. Confirm the class taxonomy may be referenced. |
| **MoGe** (monocular depth/normal) | Generates the depth and normal channels | Not redistributed here. Confirm licence for the intended use. |
| **Deep Video Prior / DVP** | Optional temporal stage | Not redistributed here. Confirm licence. |
| **Real-ESRGAN** | Optional upscaling weights | Weights not redistributed. Confirm licence. |
| **Training corpus** | Described in `datasets/README.md`; not redistributed | **Final fine-tuning data is PandaSet + ZOD only** for v90 and v79. The weights' lineage is not; see "Weight provenance" below. |
| **PandaSet** (Scale AI / Hesai) | 19,158 sunny pairs (front, front-left and front-right cameras, daylight frames only) + 4,320 night pairs | CC BY 4.0 **plus Dataset Terms**, which control where they conflict: no use of the Scale AI or Hesai name or logo beyond attribution, no use of the data to identify any person, and *derived data carries the same terms*. Commercial use permitted, no share-alike. |
| **Zenseact Open Dataset (ZOD)** | 10,000 sunny + 10,354 night pairs | Data **CC BY-SA 4.0**; devkit MIT. Commercial use permitted. **Share-alike** — see the weights note below. |
| **Dark Zurich** (ETH Zurich) | Not in v79's final corpus; in the corpora of its ancestors v76, v69 and v51 | Released for academic research. Reaches v79 through its starting weights. |
| **External perception stack** | `score_vp.py`, `PERCEPTION_ROOT` | Not part of this project and **not redistributed**. Only invoked as an optional external scorer. |

## Before the repository is made public

1. ~~Restore the upstream pix2pixHD `LICENSE`.~~ Done — BSD, permits this, attribution retained.
2. Choose and add a licence for the original work in this repository (everything outside
   `pix2pixHD/`), compatible with the above.
3. Confirm the training-data references in `docs/` are cleared for publication.
4. Note that no model weights are included. Publishing trained weights is a separate decision with
   its own licensing consequences — see "Publishing weights from these models" below.


## Training data provenance

**Changed 2026-09-11.** The 21 videos of unestablished provenance were removed from both corpora and
replaced with PandaSet and ZOD. The current baselines — **v75** sunny and **v76** night — contain
none of that footage. The full source tables are in `datasets/README.md`.

| Corpus | Pairs | Sources |
|---|---:|---|
| Sunny `training_pz` | 41,646 | Mapillary Vistas 19,293 · ZOD 10,000 · PandaSet 8,240 · Cityscapes 4,113 |
| Night `training_pz_night` | 12,546 | ZOD 5,000 · PandaSet 4,320 · Dark Zurich 2,670 · other 556 |

### Weight provenance

Every model in this project was fine-tuned from the one before it. Verified from each checkpoint's
own training options, 2026-10-01:

| Model | Its own training data | Starts from |
|---|---|---|
| **v90** (daylight baseline) | PandaSet 19,158 · ZOD 10,000 | v75 |
| v75 | Mapillary 19,293 · Cityscapes 4,113 · PandaSet 8,240 · ZOD 10,000 | v73 |
| v73 | Mapillary 19,293 · Cityscapes 4,113 · PandaSet 8,240 | v50 |
| v50 | Mapillary 19,293 · Cityscapes 4,113 · the 21 videos (9,069 frames) | earlier |
| **v79** (night baseline) | PandaSet 4,320 · ZOD 10,354 | v76 |
| v76 | Dark Zurich 2,670 · ZOD 5,000 · PandaSet 4,320 · Mapillary 22 · other 534 | v69 |
| v69, v51, v47 | Dark Zurich, Mapillary, and night video of unestablished provenance | earlier |

An earlier revision of this file said v73 and v69 trained on PandaSet only. They did not.

### Publishing weights

ZOD is **CC BY-SA 4.0**, and share-alike propagates to derivative works. Whether trained weights are
a derivative of their training data is legally unsettled; this project assumes they are. Under that
same assumption, every checkpoint above also inherits Mapillary's non-commercial terms, Cityscapes'
and Dark Zurich's research-only terms, and the unknown terms of the 21 videos. **None of them should
be published.** The code in this repository is unaffected and remains Apache-2.0.

**The fix in progress:** a daylight model trained from random initialisation on PandaSet and ZOD
alone. Its weights would be CC BY-SA 4.0 with no other restriction, and will be the first released.
A night model needs the same treatment.

### The 21 videos, for the record

### About the 21 videos

The internal name "NuRec" does not identify them. NVIDIA's NuRec
(https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) is a different thing
entirely: 3D-reconstructed driving scenes as USDZ files from six-camera driving logs, not
single-camera video. These files are not that dataset.

What inspection of the files themselves shows:

- 2560x1440, 30 fps, uniformly ~900 frames — i.e. 30-second clips.
- **They carry visible third-party creator watermarks, and different ones per video** — "ProArtInc"
  on 00 and 03, "NOMADIC" on 17, further distinct logos on 07, 12 and 20. Several different
  creators, so this is compiled footage rather than one release.
- They have audio tracks, which a reconstruction or a simulator render would not.
- Re-encoded with ffmpeg/x264 at crf 15; any original metadata was lost in that pass.
- No manifest, licence, or provenance note accompanies them anywhere in the project.

The most likely reading is that these are third-party driving videos collected from the web and
trimmed. If so:

1. **They cannot be redistributed**, and no download link can be provided for them.
2. **Model weights trained on them inherit that uncertainty.** Roughly 28% of the corpus is affected.
3. Publishing this repository is still fine — no video is included in it — but publishing the
   weights, or the clips, is a separate decision that needs this answered first.

**Action required:** establish where these files came from before releasing weights or footage. If
provenance cannot be established, the clean path is to retrain on Mapillary and Cityscapes alone,
both of which are properly licensed for research and give 23,406 pairs.
