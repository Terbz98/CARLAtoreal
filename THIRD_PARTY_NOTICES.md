# Third-party components and licensing status

**Read this before any public release.** The pix2pixHD question is resolved; the training-data question is not.

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
| **Training corpus** | Described in `docs/`; not redistributed | **UNRESOLVED, AND THE MOST SERIOUS ITEM.** See "Training data provenance" below. |
| **External perception stack** | `score_vp.py`, `PERCEPTION_ROOT` | Not part of this project and **not redistributed**. Only invoked as an optional external scorer. |

## Before the repository is made public

1. ~~Restore the upstream pix2pixHD `LICENSE`.~~ Done — BSD, permits this, attribution retained.
2. Choose and add a licence for the original work in this repository (everything outside
   `pix2pixHD/`), compatible with the above.
3. Confirm the training-data references in `docs/` are cleared for publication.
4. Note that no model weights are included. Publishing trained weights is a separate decision with
   its own licensing consequences, because the weights derive from the training corpus.


## Training data provenance

The corpus used for the current models is 32,475 image/label pairs:

| Share | Source | Status |
|---|---|---|
| 19,293 | **Mapillary Vistas** (training + validation) | Public. Free for research after registering and accepting the terms: https://www.mapillary.com/dataset/vistas |
| 4,113 | **Cityscapes** | Public. Free for research after registering: https://www.cityscapes-dataset.com/ |
| 9,069 | 21 videos in `datasets/test_mp4/`, referred to internally as "NuRec" | **Provenance unestablished. Treat as not redistributable until resolved.** |

### About the 21 videos

The internal name "NuRec" does not identify them. NVIDIA's NuRec
(https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) is a different thing
entirely: 3D-reconstructed driving scenes as USDZ files from six-camera driving logs, not
single-camera video. These files are not that dataset.

What inspection of the files themselves shows:

- 2560x1440, 30 fps, uniformly ~900 frames — i.e. 30-second clips.
- **They carry visible third-party creator watermarks, and different ones per video** — "ProArtInc"
  on 00 and 03, "NORDIC" on 17, further distinct logos on 07, 12 and 20. Several different
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
