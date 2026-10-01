# Images, weights and output locations

Updated 2026-09-14. Every project asset download link is currently **pending upload**. The fields
below are release preparation; they do not mean an asset has been provided or verified. Once
uploaded, replace "pending" with the download link.

> English translation of `docs/zh/ASSET_DOWNLOADS.zh.md`, which remains the original.
>
> **Some of this is now out of date.** The baselines are v90dv (daylight) and v79 (night) as of
> 2026-10-01, not v75/v76. Their weights are **not published**: both descend from checkpoints trained
> on research-only data. See [Weight provenance](../THIRD_PARTY_NOTICES.md#weight-provenance). The
> structure and the caveats below still hold.

## Creating the directories

Run from the repository root (`python` on Windows, `python3` on Linux):

```bash
python scripts/init_asset_dirs.py
```

This only creates directories that do not exist. It downloads nothing, creates no placeholder
weights, and overwrites no files. Git does not preserve empty directories, so it has to be run once
after a fresh clone. Only the shared roots are created here; scene, model and tag directories are
not pre-populated while those are still undecided.

```text
datasets/
└── training_v12_mapillary/      inference conditioning maps
pix2pixHD/
├── checkpoints/                 model weights
└── results/mp4/
    ├── vision_pilot/            video output from render
    └── NEW/                     delivery index assembled by scripts/delivery/refresh_new.sh
output/                          final delivered video
├── vp_input_1024/               downscaled video for the external perception stack
├── calibrated/                  perception stack output video
└── gt/                          ground-truth JSON for scoring (not a model input)
```

Once the scene is decided, the matching subdirectories can be created, for example:

```bash
python scripts/init_asset_dirs.py --town Town05 --weather sunny
```

This is a naming example; it does not mean Town05 is the chosen validation data. It creates
`datasets/recorded_Town05_sunny_inst/{rgb,semantic}/` and
`datasets/training_v12_mapillary/test_Town05_sunny_inst_gt_{label,edge,depth,normal,chroma,label_rich}/`.
For night use `--weather night`, where chroma is replaced by light. `label_rich` is used by
post-processing.

Weight directories are created with `--model MODEL_NAME`, where MODEL_NAME must be a real,
confirmed model name. The options above can be combined. The instance directory is created by the
recorder only when `--instance` is enabled. Texture support is not yet wired through — see Q3 in
the inference flow document.

This tool deliberately matches the repository paths that render currently defaults to. It does not
read a custom `CARLA2REAL_DATA` or `CARLA2REAL_OUT`. Using an external disk first requires
resolving the fact that render hard-codes `ROOT/datasets`.

## Download list

All locations are relative to the repository root. `NAME`, `PHS` and `MODEL` are format variables,
not literal directory names to create.

| Asset | Download link (replace once uploaded) | Location / contents | Version, size, SHA-256 |
|---|---|---|---|
| CARLA recording image pack | pending | `datasets/recorded_NAME/`, containing `rgb/`, `semantic/`, `frame_speed.txt`, and `instance/` only when instances were recorded | to be confirmed |
| Pre-generated inference conditioning maps | pending | `datasets/training_v12_mapillary/PHS_*`: label, edge, depth, normal, and chroma (day) or light (night), plus label_rich for post-processing | to be confirmed |
| Sunny generator weights | pending | `pix2pixHD/checkpoints/MODEL/latest_net_G.pth` | MODEL, version and compatibility flags to be confirmed |
| Night generator weights | pending | `pix2pixHD/checkpoints/MODEL/latest_net_G.pth` | MODEL, version and compatibility flags to be confirmed |
| Sample output video (for comparison) | pending | `output/`; should come with the command, model and scene used to produce it | to be confirmed |
| Perception scoring ground truth (if provided) | pending | `output/gt/<town lowercase>_<weather>_gt.json` | to be confirmed |

`NAME` is for example `Town05_sunny_inst`; `PHS` is the corresponding `test_Town05_sunny_inst_gt`.
The model reads per-frame image and conditioning-map directories; a single MP4 cannot substitute
for them. There is currently no confirmed MP4 → full conditioning-map pipeline.

The README declares sunny v75 and night v76, but the compatibility of those weights with this
checkout has not been verified, so this list does not mark them as directly usable downloads.
Training image sources are documented separately in [datasets/README.md](../datasets/README.md);
the training data layout was not created here.

## Output naming

- Per-frame model results: `pix2pixHD/results/MODEL/PHS_EPOCH/images/*_synthesized_image.jpg`
- Render video: `pix2pixHD/results/mp4/vision_pilot/<town>/<town>_<weather>_<TAG>_FINAL_1920.mp4`
- Delivered video: `output/<town>_<weather>_vp55_<TAG>_FINAL_1920_visionpilot.mp4`, 1920×960. The
  word `visionpilot` in the filename does not mean a HUD has been overlaid.
- Inference log: `pix2pixHD/checkpoints/render_<TAG>_log.txt`
- External perception log and scores: `output/logs_<TAG>/`, created by the run.

Result directories that depend on model, epoch and tag are created at run time; there is no need to
create empty placeholder result directories.

## To fill in after upload

For each download pack, fill in the actual URL along with the filename, version, size, SHA-256,
archive directory depth, source and licence terms. Weights must state the matching code version and
inference flags; images and conditioning maps must state the scene they correspond to and the frame
naming. Mark an entry "verified" only after actually downloading, extracting and running inference
from a fresh clone. Known gaps are listed in [INFERENCE_FLOW.md](INFERENCE_FLOW.md); asset terms are
collected in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
