# Datasets — where to get them

No data is stored in this repository; the working tree it was packaged from is around 550 GB. This
file tells you what to download and where to put it so the scripts find it.

Set `CARLA2REAL_DATA` to wherever you keep the bulk data (default `./datasets`).

## Training corpus

The current models were trained on 32,475 image/label pairs:

| Share | Dataset | Download | Licence |
|---:|---|---|---|
| 19,293 | **Mapillary Vistas** (training + validation) | https://www.mapillary.com/dataset/vistas | Free for research. Registration and acceptance of their terms required. |
| 4,113 | **Cityscapes** (`leftImg8bit` + `gtFine`) | https://www.cityscapes-dataset.com/downloads/ | Free for research. Registration required. |
| 9,069 | 21 driving videos, internally called "NuRec" | **No link — see below** | **Unresolved.** |

### About the 21 videos

They are **not** NVIDIA's NuRec dataset despite the internal name; that is 3D-reconstructed scenes
in USDZ format, not video. Inspection of the files shows several different third-party creator
watermarks, audio tracks, and no accompanying licence or manifest, so they are most likely driving
videos collected from the web. They cannot be redistributed and no download link can be offered.

**You can train without them.** Mapillary Vistas plus Cityscapes gives 23,406 pairs — 72% of the
corpus used here — and both are properly licensed for research. See `THIRD_PARTY_NOTICES.md`.

## Simulator

**CARLA 0.9.16** — https://github.com/carla-simulator/carla/releases — needed only to record new
drives. MIT licence for the code; assets are licensed separately.

## Models used to build the conditioning channels

Not redistributed here. Install from upstream:

| Purpose | Project |
|---|---|
| Depth and surface normals | MoGe — https://github.com/microsoft/MoGe |
| Semantic labels from video | Mask2Former — https://github.com/facebookresearch/Mask2Former |
| Optional temporal stage | Deep Video Prior — https://github.com/ChenyangLEI/deep-video-prior |
| Optional upscaling | Real-ESRGAN — https://github.com/xinntao/Real-ESRGAN |

## Expected layout

```
$CARLA2REAL_DATA/
├── mapillary_vistas/
│   ├── training/{images,v2.0/labels}/
│   └── validation/{images,v2.0/labels}/
├── training_v11_city/           Cityscapes, converted to the Mapillary-65 label space
│   ├── train_img/  train_label/
├── training_v49_chroma/         the assembled training corpus
│   ├── train_img/  train_label/  train_edge/  train_depth/  train_normal/  train_chroma/
├── recorded_<Town>_<weather>_inst/
│   ├── rgb/  semantic/
└── training_v12_mapillary/      per-town inference channels
    └── test_<Town>_<weather>_inst_gt_{label,edge,depth,normal,chroma,light}/
```

## Label space

Mapillary Vistas 65 classes (`--label_nc 65`). Ids referenced throughout the code: 6 wall, 13 road,
15 sidewalk, 17 building, 24 lane marking, 27 sky, 30 vegetation, 45/46 pole, 50 fence, 55 car,
61 truck/bus. Cityscapes and CARLA labels are both remapped into this space.

## A trap worth knowing

CARLA writes BGRA. Slicing `[:, :, :3]` gives you BGR-as-RGB, so the channels must be reversed
before use. Getting it wrong is silent — the image still looks plausible, just wrong — and it will
poison every downstream channel.
