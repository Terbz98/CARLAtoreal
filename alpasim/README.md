# AlpaSim: does a real-world driving policy behave on our video as it does on CARLA?

[AlpaSim](https://github.com/NVlabs/alpasim) (NVIDIA, Apache-2.0) is a closed-loop AV simulator built
from gRPC services: a renderer, a driver policy, a controller and physics. Its default driver,
**VaVAM-B** (Valeo, 357M parameters), was trained on real cameras: it sees 8 frames at 2 Hz and plans
6 waypoints at 2 Hz.

This folder runs that driver **open loop** on carla2real video. It needs no Docker and no Hugging Face
token: the driver package is called straight from Python.

## The metric: plan agreement with raw CARLA

For every 2 Hz decision point, VaVAM plans a 3 s path from raw CARLA frames and from each render of the
*same* drive. **Plan agreement** is the mean distance (m) between the two plans' waypoints. 0 means the
render drives the policy exactly as CARLA does. Lower is better.

This is the target: renders that look real **and** leave a driving policy's decisions where they were.
The command is `STRAIGHT` throughout (the recordings carry no route), which is wrong at turns but
identically wrong for every render, so the comparison is fair. Run-to-run noise on CPU is ~0.1 m per
waypoint.

## Results (whole-path plan agreement, m, lower is better)

| render | Town05 | Town10HD | Town03 | Town04 | Town06 | five-town mean |
|---|---:|---:|---:|---:|---:|---:|
| v50m (old) | 1.31 | 1.39 | 2.96 | 1.07 | 1.62 | 1.67 |
| **v90dv** (daylight baseline) | 0.88 | 1.39 | 1.78 | 0.68 | 1.85 | 1.32 |
| v90dva (+ distance fade) | 0.87 | 1.36 | 1.66 | 0.73 | 1.87 | **1.30** |
| v92cdv (trained from scratch) | 0.84 | 1.31 | 2.59 | 0.86 | 1.53 | 1.43 |

**Town10HD, sunny and night (2026-10-08).** Each is compared with raw CARLA in the *same* weather
(night renders against the CARLA night drive):

| render | Town10HD plan agreement |
|---|---:|
| v90dv (daylight baseline) | 1.39 |
| v90dvt (+ softer trees) | 1.38 |
| **v93n** (night baseline, trained from scratch) | **0.89** |
| v93nw (+ warm light tint) | 1.05 |

Softening the trees leaves the policy's plans unchanged. The warm light tint moves them by 0.16 m:
the policy reacts to the colour of lit areas at night, so the tint is a look choice with a measurable
cost, not a free one.

## Running it

```bash
# 1. AlpaSim's Python workspace (uv), and the VaVAM-B checkpoint + tokenizer into alpasim/vavam/
git clone https://github.com/NVlabs/alpasim && cd alpasim && uv sync --all-packages

# 2. evaluate: tag CARLA = the raw simulator frames
export CARLA2REAL_DATA=... CARLA2REAL_OUT=... VAVAM_DIR=.../vavam
CUDA_VISIBLE_DEVICES="" uv run python ../alpasim/vavam_eval.py town10hd CARLA v90dv
WEATHER=night CUDA_VISIBLE_DEVICES="" uv run python ../alpasim/vavam_eval.py town10hd CARLA v93n

# 3. score and draw
python3 alpasim/plan_agreement.py
python3 alpasim/vavam_video.py town10hd sunny:CARLA sunny:v90dv night:CARLA night:v93n
```

`CUDA_VISIBLE_DEVICES=""` matters: VaVAM's TorchScript tokenizer hard-codes CUDA and will otherwise
grab the GPU even with `device=cpu`. On CPU the script forces float32 (the action head is fed float16,
which only works under CUDA autocast). About 7 minutes per 1000-frame clip on 16 cores.

## Not done yet: closed loop

Making carla2real an AlpaSim **renderer** (`SensorsimService.render_rgb` backed by a live CARLA server
and the generator) would let the policy drive *inside* CARLA towns. It needs Docker with the NVIDIA
container toolkit, and a CARLA town exported to AlpaSim's scene format.
