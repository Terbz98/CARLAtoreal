# Modifications to NVIDIA pix2pixHD

This directory is a modified fork of https://github.com/NVIDIA/pix2pixHD, redistributed under its
BSD licence (`LICENSE.txt`, retained verbatim). Changes made for this project:

**Extra conditioning channels** (`models/pix2pixHD_model.py`, `data/`, `options/base_options.py`)
`--edge_input`, `--depth_input`, `--normal_input`, `--light_input`, `--chroma_input` append further
channels to the generator input alongside the label map. A label map alone carries no weather and no
geometry, so the generator invents facades and foliage — differently every frame, which is the
flicker that makes synthetic video useless for perception testing. Concatenation order is
label → weather → edge → depth → light → chroma → normal.

**Per-pixel weighted perceptual loss** (`models/networks.py`, `VGGLoss`)
`forward(x, y, weight)` takes an optional weight map. With `weight=None` it is numerically identical
to the original. Note that it computes a weighted *mean* — `(w*d).sum() / w.sum()` — so raising the
weight on one class does not add gradient, it redistributes it away from everything else.
`forward_plus` returns the plain and weighted terms from a single VGG pass so a caller can add
emphasis without diluting the rest of the frame.

**Class weighting options** (`options/base_options.py`)
`--thing_weight`, `--veg_weight`, `--far_boost`, `--veg_extra`. These upweight small or distant
classes that area-weighted L1 and VGG otherwise ignore: a distant tree covers a few hundred pixels
out of two million and contributes a proportionate share of the gradient.

**Temporal and multiframe modes** (`--temporal`, `--multiframe`, `data/temporal_dataset.py`,
`data/multiframe_dataset.py`)
`--temporal` feeds the previous frame back autoregressively; `--multiframe` stacks neighbouring
label maps without feedback. Both are implemented and both are recorded as negative results in
`docs/EXPERIMENTS.md` — `--temporal` suffers exposure bias, because training feeds the real previous
frame while inference feeds the model its own output, and the compounding error weaves a visible
crosshatch over flat surfaces.

**Weather conditioning** (`--n_weather_classes`)
One-hot weather channels. Superseded by per-condition models, kept for reference.
