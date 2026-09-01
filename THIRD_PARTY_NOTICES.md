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
| **Training corpus ("NuRec")** | Described in `docs/` | **CHECK BEFORE PUBLISHING.** The real-world driving footage used for training is described in the documentation. Confirm that describing it publicly, and any onward use of models trained on it, is permitted. |
| **External perception stack** | `score_vp.py`, `PERCEPTION_ROOT` | Not part of this project and **not redistributed**. Only invoked as an optional external scorer. |

## Before the repository is made public

1. ~~Restore the upstream pix2pixHD `LICENSE`.~~ Done — BSD, permits this, attribution retained.
2. Choose and add a licence for the original work in this repository (everything outside
   `pix2pixHD/`), compatible with the above.
3. Confirm the training-data references in `docs/` are cleared for publication.
4. Note that no model weights are included. Publishing trained weights is a separate decision with
   its own licensing consequences, because the weights derive from the training corpus.
