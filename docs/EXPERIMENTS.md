# Slack-time experiments, 2026-08-20 onward

Running record of what was added while the main 72-hour queue ran, so nothing has to be
reconstructed from logs later.

## Scripts added today
| file | purpose |
|---|---|
| `score_vp.py` | grade a Vision Pilot run against the CARLA GT jsons |
| `compare_models.py` | one table of every scored clip, grouped by town |
| `make_v50_init.py` | graft a new input channel onto a checkpoint without discarding it |
| `train_v50.sh` | v50 sunny, chroma grafted onto v45 |
| `train_v51_night.sh` | v51 night, corpus-only change |
| `build_v51_corpus.py` | assemble the v51 night corpus as symlinks |
| `build_dz_night.sh` | Dark Zurich -> night training pairs |
| `dz_filter.py` / `dz_label.py` | the night test, and Mask2Former pseudo-labels |
| `render_model.sh` | render+deliver+score a trained model over towns |
| `phase2.sh` | orchestrates the v50/v51 render batches |
| `eval_v50.sh` | hue/detail table, v45 vs v49 vs v50 |
| `rescore_night.sh` / `rescore_1920.sh` | resolution comparison arms |
| `repost_night.sh` | re-post the night set through the corrected chain (tag v47r) |
| `bisect_moto.sh` | find which pipeline stage loses a small vehicle |
| `dvp_ablation.sh` | is DVP worth 1 h 56 m per clip |
| `grab_dvp.sh` | capture Town05's DVP output before cleanup, for the ablation |
| `vidcodec.py` | pick codec from output extension (.avi -> lossless FFV1) |
| `refresh_new.sh` | organise deliveries into results/mp4/NEW/<model>/town<N>/<weather>/ |
| `queue_progress.sh` | live queue meter |

## Findings so far
- v49's detail loss was a shape-mismatch fallback discarding 343,392 params, not a chroma tradeoff.
- Night post-processing cost 2.6x object detection: grain/bloom at full strength + six mp4v
  re-encodes. Fixed; town04 night CIPO 0.225 -> 0.593, range MAE 7.21 -> 1.84 m.
- `conda run` silently discards heredoc stdin, exit 0. Three scripts affected.
- VP feeds must be 1024x512: lane MAE 0.27 m vs 0.87 m at 1920, confirmed on all five towns.
- The generator alone loses 45.7 points of small-object detection before any post stage.

## Open experiments

### 1. Night re-post through the corrected chain (tag `v47r`)  — RUNNING
No re-render; the generator output survived. First result, town10hd:
```
v47  (delivered)  CIPO 1.000  rng 4.36 m  lane 0.15 m
v47r (reposted)   CIPO 0.979  rng 3.83 m  lane 0.13 m
```
Modest here because town10hd's lead vehicle is a large nearby car that was never at risk. The
fix is worth 2.6x on town04, whose lead vehicle is a motorcycle. Expect the gain to track object
size, not to be uniform.

### 2. DVP ablation — RUNNING
Arm B (no DVP, CPU) and arm A (Town05's DVP output, captured free before cleanup). DVP is 20 of
the remaining ~44 GPU-hours and has never been checked against ground truth.

### 3. Generator vs stabiliser on small objects — RUNNING
`baseline` (= stabilised generator) scores 54.3% against raw CARLA's 100%. Whether that loss is
the generator or `stabilize_frames_v2.py` has not been separated. Built an unstabilised video
straight from the 2048x1024 generator frames to find out; if the stabiliser is at fault it is
another cheap fix like the bloom one.

### 2. DVP ablation — CONCLUDED: keep it
Controlled pair on town05 sunny (identical generator frames, identical lossless post chain, DVP
the only variable; arm A reused the DVP output the delivery had already paid for, so it cost no
GPU time).

```
                    CIPO    rngMAE   laneMAE   jitter   static-pixel flicker
DVP ON  (1h56/clip) 0.738   1.34 m    0.19 m   0.0202          2.727
DVP OFF (free)      0.718   1.14 m    0.19 m   0.0204          4.054
```

**Verdict: keep.** DVP removes 33% of visible shimmer, which is exactly what it was chosen for.
It does NOT help perception -- detection is +2 pts, range is 0.20 m WORSE, CTE jitter is -1%.

Two lessons worth keeping:
1. An earlier comparison suggested DVP bought 14% jitter. That was confounded -- it compared
   DVP+lossy against noDVP+lossless. The controlled pair says 1%. Never read a two-variable
   comparison as a one-variable result.
2. CTE jitter is a planning output and is not a proxy for visual flicker. Judging DVP on it
   nearly cut a stage that works, on the strength of a metric it was never meant to serve.

Cost if ever dropped: ~1 h 56 m per clip, ~29 h across the 15 clips still queued. That is the
lever to pull if throughput ever matters more than appearance -- but not while the schedule has
a day of slack.

### 4. v50 graft — TRAINED 2026-08-21 04:22, graft VERIFIED
`grep -c 'not initialized'` = **0**, so pix2pixHD's shape-mismatch fallback never fired and the
grafted weights were loaded intact. After 4 epochs:

```
                 chroma cols |w|      v45 cols drift
model.1.weight   0 -> 0.046912        31.9%  (refined, not rebuilt)
model1_1.1.weight 0 -> 0.070305       36.8%
```

The chroma columns started at exactly zero and are now non-zero, so the model learned to use the
channel; the 70 inherited columns moved only ~a third, so v45's semantic encoding was refined
rather than relearned. v49 had to rebuild that entire layer from random init in 2 epochs at
lr 5e-5, which is the whole explanation for its 44% detail loss.

### 5. v50 evaluated — HYPOTHESIS WRONG, and worth recording as such
```
model   HUE car   HUE bldg    lane     car det
v45      61.5      30.1      +66.3      970
v49      59.5      33.7      +41.4      541
v50      54.5      30.3      +39.1      472
```
v50 wins colour uniformity outright (54.5, the best of any model tested, against v48's failed
80.1) and restores building uniformity to v45's level. **But detail fell further, 541 -> 472.**

The graft is not in doubt: zero shape-mismatch fallbacks, chroma columns learned from exactly
zero, v45's 70 columns retained with ~a third drift. It did what it was designed to do. What it
did NOT do is recover detail — so the 343,392 discarded parameters, though a genuine bug, were
not the cause of v49's detail loss. That diagnosis was wrong and is corrected here.

Detail instead tracks CHROMA TRAINING EXPOSURE, monotonically:
  v45 none -> 970,  v49 two epochs -> 541,  v50 four epochs -> 472.
New hypothesis: the chroma prior is a blurred, flat-luminance field (sigma 8, L pinned to 128)
and the generator learns to follow its smoothness. `test_chroma_blur.sh` tests the inference half
by feeding the same model a NEUTRAL grey prior on the same frames.

Consequence for the plan: v50 does NOT automatically replace v49. It is better on the patchiness
complaint and worse on sharpness, which is exactly the trade the user resolved by eye last time.
Both are being rendered in phase2 so the comparison can be made the same way.

### 6. v51 night — TRAINED 2026-08-21 09:57
Corpus assembled with zero tuples dropped: 4156 (v47 set) + 2670 (Dark Zurich) = 6826, all six
channels complete. `grep -c 'not initialized'` = 0, so the architecture stayed byte-identical to
v47 and the pretrained weights loaded intact — the only variable is the data, as intended.
Share of the corpus coming from the four NuRec dashcam drives falls from 87% to 53%.
Phase2 is now rendering all five night towns with it for scoring against v47.

### 7. HEADLINE RESULT — town04 night, the motorcycle case
```
                        CIPO     range MAE
raw CARLA (ceiling)    1.000       0.32 m
v47 as delivered       0.225       7.21 m
v51 + post fixes       0.997       0.27 m
```
The clip that lost a motorcycle for 381 consecutive frames (12.7 s) now tracks it on 99.7% of
frames, and ranges it to 0.27 m -- at or slightly better than raw CARLA's own output. The
sim-to-real pipeline has gone from destroying that vehicle to preserving it completely.

Two changes combined, and both were needed:
  1. v51's night corpus: 6826 images instead of 4156, with the four-dashcam share cut 87% -> 53%
  2. the post-processing fixes now baked into render_model.sh -- grain/bloom off (night had been
     running them at full strength) and lossless FFV1 intermediates instead of six mp4v re-encodes

Matched-town scores so far, v47 -> v51:
  town03  CIPO 0.497 -> 0.617   rng 8.68 -> 8.16 m   lane 0.39 -> 0.42 m
  town04  CIPO 0.225 -> 0.997   rng 7.21 -> 0.27 m   lane 0.33 -> 0.34 m
Lane is a touch worse on both; detection and range are dramatically better.

### 8. NIGHT COMPLETE — v51 beats v47 across all five towns
```
town       v47 CIPO v51 CIPO    v47 rng  v51 rng    v47 lane v51 lane
town03        0.497    0.617      8.68m    8.16m       0.39m    0.42m
town04        0.225    0.997      7.21m    0.27m       0.33m    0.34m
town05        0.750    0.730      3.62m    3.45m       0.21m    0.24m
town06        0.996    1.000      0.52m    0.46m       0.09m    0.08m
town10hd      1.000    1.000      4.36m    4.03m       0.15m    0.14m

  MEAN v47:  CIPO 0.694   rng 4.88 m   lane 0.23 m
  MEAN v51:  CIPO 0.869   rng 3.27 m   lane 0.24 m
```
v51 wins detection (+25%) and range (-33% error) and ties on lane (0.24 vs 0.23 m). Every town
improves or holds on range; only town05 CIPO dips slightly (0.750 -> 0.730).

The gain is concentrated exactly where the weakness was: town04's motorcycle, 0.225 -> 0.997 with
range collapsing 7.21 m -> 0.27 m. Towns whose lead vehicle was a large nearby car (town06,
town10hd) were already near ceiling and stay there. So the improvement is not uniform polish --
it is the hard cases becoming solvable, which is the result worth having.

Attribution caveat, stated plainly: this compares v47-as-delivered against v51-with-post-fixes,
so it bundles the larger night corpus WITH the grain/bloom and lossless-intermediate fixes. The
earlier town04 bisect separates them -- post fixes alone took 0.225 -> 0.593, so roughly the
first half of the gain is post-processing and the second half is the model.

RECOMMENDATION: v51 replaces v47 as the night baseline.

### 9. Sunny v50 vs v49 — first matched town, and a caveat about the detail metric
```
town03 sunny      CIPO     rng MAE    lane MAE
v49 (baseline)   0.773      1.30 m     0.27 m
v50 (grafted)    0.938      1.12 m     0.26 m
```
v50 wins detection by 21% and range by 14%, lane tied — despite measuring WORSE on the hue
table's car-detail score (472 vs 541) and lane contrast (+39.1 vs +41.4).

Worth stating plainly: Laplacian variance inside car regions is a sharpness proxy, and it does
NOT predict whether Vision Pilot can find and range the car. v50 is blurrier by that measure and
better by the one that matters for perception. Any future model choice made on the hue table
alone risks picking the wrong model; score_vp.py against GT is the arbiter.

This does not settle the baseline question by itself — the user chose v49 by eye, and v50's
visible sharpness is what they would be trading away. Both sets are delivered side by side.

### 10. PHASE 2 COMPLETE 2026-08-22 09:16 — all 10 clips delivered and scored
Matched 5-town comparison, both weathers (the four missing v49 sunny runs were scored from the
VP logs the chain had left in /tmp, so the sunny arms are now like-for-like):

```
         model   n   laneMAE   rngMAE   CIPO recall
  night  v47     5    0.23 m   4.88 m      0.694
  night  v51     5    0.24 m   3.27 m      0.869     <- night winner
  sunny  v49     5    0.19 m   1.82 m      0.769
  sunny  v50     5    0.19 m   1.49 m      0.899     <- sunny winner on metrics
```

Both new models win on object perception and tie on lane:
  v51 night: recall +25%, range -33% error
  v50 sunny: recall +17%, range -18% error, lane identical

Caveat carried forward: v50 measures BLURRIER than v49 on the hue table (car detail 472 vs 541)
while scoring better here. The user picked v49 by eye, so the sunny baseline is a visual call
they should make between two sets now sitting side by side in results/mp4/NEW/v49 and /v50.
Night has no such tension — v51 wins on metrics and inherits the ghosting and post-processing
fixes, so it should simply replace v47.

### 11. v52 thing-weighted night — INCONCLUSIVE, my test town was badly chosen
```
town04 night      CIPO     rng MAE
v47              0.225      7.21 m
v51              0.997      0.27 m
v52 thing-weight 1.000      0.28 m
```
v52 is indistinguishable from v51 here, but that proves nothing: v51 already sits at 0.997 on
town04, so there is no headroom for any model to demonstrate a gain. I picked town04 because it
is where the ORIGINAL defect was most visible under v47 -- which was the right choice when the
baseline was v47 and the wrong one now that the baseline is v51.

v51 per-town CIPO shows where headroom actually remains:
   town03 0.617   town05 0.730   town04 0.997   town06 1.000   town10hd 1.000

`eval_v52_town03.sh` re-tests v52 on town03, the only genuinely hard case left. Gated behind the
coverage job so it never competes for the GPU. Until that lands, the thing-weighted loss is
untested, not disproven.

### 12. Thing-weighted loss (v52/v53) — NEGATIVE RESULT
```
town03 sunny        CIPO     rng MAE
v49 (baseline)     0.773      1.30 m
v50                0.939      1.12 m
v53 thing-weight   0.858      1.05 m     <- 8 pts WORSE detection than v50
```
Range improves slightly (1.05 vs 1.12 m) but detection drops. On night, v52 was indistinguishable
from v51 (1.000 vs 0.997) on a town with no headroom — retest on town03 pending.

Read honestly: upweighting thing-class pixels 10x in the perceptual loss did NOT deliver the
small-object gain it was designed for, and on sunny it cost detection. The hypothesis that the
uniform L1 was starving small objects of gradient is not supported by this test. The generator's
45.7-point loss on small vehicles remains unexplained and unfixed at the model level -- though
the post-processing fixes plus v51's corpus already closed most of the practical gap
(town04 0.225 -> 0.997).

The `--thing_weight` option defaults to 1.0 and is bit-identical to the original loss at that
value, so nothing needs reverting; it simply should not be used at 10.

### 13. Town07 coverage — rejected take, cause identified
check_take.py failed Town07 sunny (14 frames, 1.4%, driven by road L-R balance). Inspecting the
flagged frames showed the real fault is NOT tumbling: at frame 398 the ego camera is clipped
INSIDE another vehicle, looking through its rear window at an ambulance; elsewhere it is nosed
into a picket fence. Town07 is CARLA's smallest map and 220 NPCs boxes the ego in until it
collides. The verdict was right, the label was misleading.

`retry_coverage.sh` re-records Town07 sunny and night at 80 / 45 / 25 NPCs, keeping the first
take that passes, then re-runs expand_coverage.sh (whose per-stage guards skip everything already
finished). Queued behind the coverage job so it never competes for the GPU.

### 14. Two wait-loop bugs that wedged the tail of the queue (fixed 2026-08-23 04:45)
`retry_coverage.sh` sat idle for two hours after COVERAGE DONE. Its GPU-free check was
`while pgrep -f '<patterns>' | grep -qv "^$$\$"; do sleep 120; done`, which can never exit: the
harness runs commands as `bash -c '<whole script text>'`, so pgrep matches the script's OWN
wrapper, and `grep -v "^$$"` removes only the current shell's pid, not the wrapper's.
`eval_v52_town03.sh` carried the identical defect.

Both now source `gpu_wait.sh`, which filters the caller's full process ancestry plus the
shell-snapshot signature. A second bug surfaced immediately: `pgrep` can return a pid that exits
before `ps -o args=` reads it, and the empty result was being read as "busy" — so an unreadable
process is now treated as gone, not running.

This is the third distinct appearance of the pgrep self-match trap in this project (queue_progress.sh,
then these two). Any new script that waits on a process must use `gpu_wait.sh`, never a bare
`pgrep -f`.

### 15. GPU collision lost two coverage clips (2026-08-23 ~05:58)
Town05dense (v50) and Town07 night (v51) each rendered exactly 1 frame of 1000, despite every
input channel being complete. Both were running while eval_v52_town03.sh had the card.

Cause, and it is a design mistake not a race I got unlucky with: eval_v52_town03.sh gates on
`grep -q 'COVERAGE DONE'`, and that marker was already written by the FIRST coverage run at
02:37. So when retry_coverage.sh re-ran coverage at 05:42, the v52 job had long since passed its
gate and was mid-render. Two 2048 renders on a 32 GB card, and the loser produced one frame.

**A completion marker is not a lock.** Gating on one that can already be set from an earlier pass
gives no mutual exclusion at all. `finish_coverage.sh` waits on the GPU itself via gpu_wait.sh
instead of on any marker, and re-runs expand_coverage2.sh, whose per-town guard skips anything
with a FINAL_1920 already on disk — so only the two lost clips are redone.

### 16. Town07 sunny cannot pass the take check — and the check is the thing at fault
Re-recorded at 80, 45 and 25 NPCs; all three rejected, so crowding is not the cause. The signals
tell the story (45-NPC take):
```
road centroid max step 0.013     excellent — the rebuild gate allows 0.10
road fraction  max step 0.087    fine
sky tilt(L-R)  max step 1.324    this is what fails it
```
The reliable road-plane signals are clean; only the SKY signals spike. Town07 is CARLA's hilly
rural map, and cresting a rise tilts the horizon exactly the way a roll would. check_take.py's
sky thresholds were calibrated on flat urban towns.

Town07 NIGHT passes at 45 NPCs (CLEAN, 0.9% flagged), which supports the terrain reading rather
than a broken drive. Town07 sunny is left undelivered rather than force it through: the honest
options are to widen the sky thresholds for hilly maps or drop the sky signals where road
signals are strong, and neither should be decided by loosening a gate to get a green light.

### 17. Thing-weighted loss — CONCLUSIVELY NO EFFECT (retest on the discriminating town)
```
town03 night        CIPO     rng MAE   lane MAE
v47                0.497      8.68 m    0.39 m
v51 (baseline)     0.617      8.16 m    0.42 m
v52 thing-weight   0.615      8.93 m    0.38 m
```
town03 is where v51 has real headroom (0.617, versus ceiling on three of the five towns), so this
is the test town04 could not be. v52 lands within 0.002 CIPO of v51 and is 0.77 m WORSE on range.

Combined with v53's 8-point detection loss on sunny, the verdict across both weathers is that
upweighting thing-class pixels 10x in the VGG loss does nothing useful and sometimes hurts.

**The hypothesis is disproven, not merely unsupported.** The reasoning was that a motorcycle at
0.3% of pixels supplies 0.3% of the perceptual gradient and so is never learned. That is true
arithmetic about the loss, and it still did not translate into better small-object synthesis --
which means the bottleneck is somewhere other than gradient share. Plausible remaining suspects:
the label/edge channels simply do not carry enough shape information at that scale, or the
generator's receptive field at 2048 wide resolves a 59x125 px object into too few features.

`--thing_weight` stays in the codebase at its default of 1.0, which is bit-identical to the
original loss. It should not be used. The 45.7-point generator gap on small objects remains open.

### 18. Chroma prior is NOT the cause of the detail loss (last open experiment, closed)
Same v50 checkpoint, same 300 frames, only the chroma map swapped for uniform grey:
```
                HUE car   car detail
real prior       54.5        472
neutral grey     51.1        493
```
Detail barely moves (472 -> 493, +4.4%), so the blurred prior is not smoothing the output at
inference. The detail damage is baked into the weights during training, not produced at runtime
by the prior's smoothness. That kills the hypothesis from note 5.

An unexpected second result: colour uniformity is BETTER with no prior at all (51.1 vs 54.5).
So v50's colour-uniformity win -- the whole point of the chroma channel -- is a property the
model LEARNED during training, not something the prior delivers at inference. On this evidence
the chroma map could be dropped at inference with a small gain, which would remove
gen_chroma_maps.py from every render. Worth confirming on more than one town before acting.

Net position on v49's detail loss after four experiments: not the discarded first conv (note 5),
not gradient share (note 17), not the prior at inference (here). Still open.

### 19. Post-processing de-shimmer has a hard ceiling: shimmer trades 1:1 against ghosting
Full strength sweep on town10hd (150 frames, 960x480), ghost measured against un-deshimmered
v50b on pixels optical flow says actually moved:
```
                          alt p99   sharp    ghost
v50b (no de-shimmer)        45.48   772.0     0.00
3-frame gated (= v50d)      26.72   634.3     4.01
3-frame no gate (= v50e)    18.82   625.4     5.87
5-frame no gate             12.76   525.0     8.79
v49 ("the stable one")      27.51   440.9       --
```
Every setting is a point on one curve; nothing moves the curve. **Calibration point: the v1
stabiliser scored ghost 6.30, and that is the build where the user could plainly see car-outline
trails.** Treat 6.30 as the visible-ghost threshold for this metric.

### 20. Motion-compensated median FAILS — the flow is corrupted by the shimmer it is meant to fix
Warping each neighbour into the current frame by optical flow before taking the median should
have broken the trade in note 19: once aligned, a moving car sits on top of itself, so the median
has nothing to smear. With forward-backward consistency checking:
```
motion-compensated 3-frame   alt 37.20  sharp 486.9  ghost 3.89
motion-compensated 5-frame   alt 36.00  sharp 413.7  ghost 4.80
plain 3-frame gated          alt 26.72  sharp 634.3  ghost 4.01
```
Worse on every axis that matters. The flow is estimated FROM the shimmering video, so the shimmer
corrupts the flow, the warp adds its own error, and the FB check then rejects so many pixels the
median runs out of votes. Dead end — do not retry without a flow source that is not the render.

### 21. v50e: lowering flicker is easy, lowering it without ghosting is the actual problem
v50e (max-strength de-shimmer) delivered on all 5 sunny towns:
```
              alt p99   sharpness  |  CIPO   rng MAE  |  ghost t03 / t10hd
v49             24.25       542    | 0.769     1.82   |
v50d            27.01       854    | 0.887     1.62   |  4.83 / 4.96
v50e            17.20       799    | 0.879     1.60   |  7.92 / 8.30
```
36% less flicker than v50d for only 0.008 CIPO — and it would have been an easy, wrong call to
ship it. Ghost 7.9-8.3 is well past the 6.30 visible-trail line from note 19. Rejected; kept in
`calibrated/v50e/` with a README. This is the experiment that justified spending GPU hours on
temporal training instead of another post-processing pass.

### 22. Six silent-failure blockers in the temporal path (found by preflight, not by running it)
`--temporal` widens netG by output_nc (the prev frame is concatenated for G but NOT for D), so a
v50 checkpoint's 73-channel first conv does not fit the 76 the temporal model builds — and
`load_network` random-inits it without raising. The run would have looked completely healthy
while having discarded v50 entirely. Same class as note 5. Also: the loader returned no
depth/normal/chroma/light; clip boundaries were being recovered from `nurec_raw` which is NOT
index-parallel to `nrhr_` (9,069 of 23,053); Dark Zurich needed content alignment; a bash
var-prefix on a function call leaked sunny flags into the night run; and `render_model.sh` had no
`--temporal`, so even a good checkpoint would have rendered garbage.

**Lesson: for anything that changes channel arithmetic, write the preflight before the run.**
`preflight_temporal.py` checks the arithmetic against the checkpoint, asserts the grafted columns
are exactly zero, verifies flow count == frames - clips, and pulls a real sample through the
loader to confirm the assembled width equals the first conv. All six would otherwise have cost
hours and been diagnosed only by eye.

### 23. The crosshatch that ruined v54 was EXPOSURE BIAS, not the loss and not the flow
The user reported "shiny white dots"; a frame showed a woven crosshatch over every flat surface.
Road texture against the parent (Town10HD, road-labelled pixels only):
```
v50 parent                            1.00x
v54  temporal (wrong-direction flow)  5.81x
v57a same, video discriminator GONE  12.99x   <- worse: refutes "the video disc did it"
v58  temporal, flow direction FIXED   7.98x   <- worse: refutes "the flow direction did it"
v58  with the prev channel ZEROED     0.18x   <- clean road, confirmed by eye
```
Training always feeds G the REAL previous frame; inference feeds G its own output (and a black
frame at t=0, wildly out of distribution). G never learned to consume its own output, so errors
compound into a texture. This is textbook exposure bias in an autoregressive model.

### 24. Every metric in the suite was blind to it, for a structural reason
`alt p99` said v54 was CALMER — a static weave is perfectly frame-to-frame consistent, so a
flicker metric REWARDS it; the defect and the success metric were aligned, and measuring flicker
harder could never have found it. Laplacian sharpness said +80% (a weave is high-frequency).
A median-filtered "clean sharpness" still said sharper (a median kills isolated specks, not a
structured pattern). An FFT peak-ratio detector barely moved (79.9 vs 87.3). CIPO was the ONLY
metric that objected (0.887 -> 0.849) and it was under-weighted as an acceptable trade.
`road_texture.py` works because it asks a different question: not "is there detail" but "is there
detail WHERE THE LABEL MAP SAYS THERE SHOULD BE NONE". **Always look at a frame.**

### 25. The flow direction WAS wrong — a genuine bug that was not this bug
`warp()` backward-warps and needs a cur->prev field; the corpus builder computed prev->cur.
Measured mean abs error of the warped previous frame against the real current frame:
`warp(prev,+flow) 4.266 · warp(prev,-flow) 1.872 · no warp at all 3.423` — the training target was
worse aligned than not warping. Fixed (now 48% better than unwarped) and kept, and
`build_temporal_corpus.py` now verifies the direction numerically and ABORTS on failure. Fixing it
made the crosshatch worse, which is what finally pointed at the feedback loop.

### 26. Temporal training works, but does not beat the post-processing it was meant to replace
At prev-scale 0.5 (the sweep's clean point: road 0.66x parent, no weave), delivered over 5 towns:
```
sunny  v50d  alt 24.95  sharp  810  CIPO 0.887      night  v51  alt 23.75  CIPO 0.869
       v58   alt 26.27  sharp 1012  CIPO 0.827             v59  alt 22.70  CIPO 0.865
```
v58's raw gain is real (alt 41.67 vs parent 56.75, -27%) but v50d reaches 24.95 using DVP +
de-shimmer, so the trained-in gain merely EQUALS what post-processing already bought — it did not
stack. Net: +25% sharpness and a cleaner road for -0.06 CIPO and +5% flicker. Baselines unchanged.
**The one durable win: v58 matches v50d-class flicker without DVP, which is 1 h 56 of GPU a clip.**

### 27. An existing checkpoint is not proof of a finished run
`deliver_fixed.sh` skipped night training because `latest_net_G.pth` existed — left by a run
killed 1900 iterations into epoch 1 of 3. It then "passed" the gate at 0.91x purely because it had
barely moved from its parent, and would have shipped as the night temporal model. pix2pixHD writes
`latest_net_*.pth` every save_latest_freq iterations, so it exists long before a run completes.
Gate on the FINAL epoch file (`3_net_G.pth` with --save_epoch_freq 1) and delete partial runs
rather than resuming them. Same family as "a completion marker is not a lock" (note 12).

### 28. The sky ghost defeated four hypotheses — the temporal fine-tune is abandoned
Faint building outlines appear in the sky of every temporally fine-tuned sunny model. Measured on
sky-labelled pixels at the TAIL of a full-length render, against the v50 parent's 0.913:
```
v58  temporal, corrected flow                 6.61   7.24x
v57a same but video discriminator REMOVED     7.12   7.80x   -> not the discriminator
v60  temporal loss TEXTURE-GATED              6.67   7.30x   -> not the temporal loss
v61  prev-channel amplitude randomised        9.60  10.51x   -> not input distribution; WORSE
```
Each hypothesis was eliminated by measurement, not argument, and the texture gate demonstrably
worked (sky gradient median 0.0088 against a 0.02 threshold, so 88% of sky was excluded from the
loss and nothing changed). v61 also tripled road texture (3.20x vs v58's 0.35x) and was confirmed
bad by eye — scaly road, background buildings reduced to woven line drawings.

Nothing that was ADDED by the fine-tune explains it, and removing each in turn does not help.
**Conclusion: the temporal fine-tune as implemented cannot beat v50d.** Delivered, v58 was already
not better (alt 26.27 vs 24.95, CIPO 0.827 vs 0.887); the ghost removes the remaining argument for
it. Baseline stays v50d. The untried lever is true student forcing — generating the previous frame
during training rather than perturbing a real one — but on this evidence the expected payoff no
longer justifies it.

### 29. Night finally received the cleanup sunny had, and it is a genuine trade
v51 never got any of the post-processing that produced v50d, and night is the WORSE case:
luminance 40.7 vs sunny's 93.5, alt p99 49.97-66.47 vs 34.86, and roughly twice the white specks.
What transfers was decided by measurement, not by copying: the de-shimmer gate fires on 75-84% of
night pixels (so it is not a no-op), while sharpening was REJECTED because it amplifies exactly
the specks being removed, and vehicle-colour correction was rejected because hue is meaningless on
a dark car. A night-specific despeckle removes only tiny, colourless, locally-bright blobs, so
lamps and signals survive. Result over 5 towns:
```
       alt p99   sharp    CIPO    rng      specks
v51      23.75   222.4   0.869   3.27   ~2100-3000
v51d     19.13   179.1   0.860   3.45   ~30% fewer
```
-19% flicker and ~30% fewer specks for -19% sharpness and +5% range error; CIPO unchanged within
noise. A real trade rather than a win, so v51 remains the baseline and v51d is a candidate.

### 30. Detecting a CARLA respawn: two detectors that fail, and one that is exact
The Town10HD "teleport" the user saw at 0:22 is frames 709-731 with the camera BURIED in geometry
(the whole frame is close-up concrete), after which the drive resumes elsewhere with a DIFFERENT
ego vehicle. 23 frames are unusable.
- **Speed trace: useless.** Town10HD's largest speed change is 1.67, SMALLER than clean Town04's,
  and "stopped then instantly moving" appears in no take at all. An earlier note claiming the
  speed trace confirmed the teleport was simply wrong.
- **Frame-to-frame image difference: not separable.** The cut is real (63.8 against neighbours at
  9-12) but Town10HD is a fast drive whose ordinary differences already run 15-25, so no threshold
  separates a cut from a fast turn — it caught Town03 and missed Town10HD.
- **Semantic map: exact, no threshold.** A buried camera sees exactly ONE class; any real driving
  view has many. `check_teleport.py` flags Town10HD 709-731 and clears all five other towns.

### 31. A second buried-camera take existed and nobody had seen it
Once check_teleport.py was exact (semantic map, note 30), scanning every recording found Town03
NIGHT frames 880-901 with the camera buried against a wall then fully black — the same defect as
Town10HD sunny, in a clip that had been delivered and scored for weeks. Every other take is clean.
**Lesson: build the detector, then run it over everything, not just the clip that was complained
about.** The user reported one instance; the tool found the population.

Re-recording Town03 night also showed that retrying the SAME strategy is not a retry: four
`buildings` spawns all aborted with "stuck 150 ticks" at frames 208-245 while NPCs went 110 -> 50,
so it was never traffic. `--spawn_mode highway` succeeded first try. When repeated attempts fail
at the same point, change the approach, not the parameter.

### 32. --multiframe CURES the sky ghost, proving its cause — but buys no flicker reduction
The last untried temporal mode. Instead of conditioning on the previous GENERATED frame
(--temporal), --multiframe stacks the LABEL MAPS of t-1, t, t+1 and predicts t. Labels are ground
truth at train and test alike, so there is no feedback loop to compound.

Tail of a full-length Town10HD render, vs the v50 parent:
```
                road tex   sky ghost   specks    raw flicker   raw sharp
v50 parent         29.66       0.913     8122          52.44       961.2
v62 multiframe     47.87       0.793     9740          55.82      1290.4
ratio               1.61x       0.87x    1.20x
```
**The sky ghost is GONE — 0.87x, better than the parent** — where every --temporal variant sat at
7.2-10.5x regardless of which term was removed. That settles note 28: the ghost came from the
autoregressive previous-frame input, not from the temporal loss, the video discriminator, or the
input distribution. Removing the channel removes the artifact. Confirmed by eye: clean sky, sharp
cars, no crosshatch.

But it does not do the job. Raw flicker is 55.82 against the parent's 52.44 -- no improvement at
all, and the whole point was less flicker. Road grain rises 61%. So v62 is NOT delivered.

**The temporal family is now exhausted**: --temporal produces artifacts that no configuration
removes, and --multiframe is artifact-free but confers no benefit. Sunny flicker is a
post-processing problem, and note 33 is where it was actually solved.

### 33. Flicker is class-structured, and the fix was aiming the filter, not strengthening it
Measuring flicker per SEMANTIC CLASS on the raw sunny render finally explained why the clips read
as blurry AND flickery at the same time:
```
class          % of frame   mean flicker   contribution
building             25.0          48.0          11.97   <- more than all others combined
car                   7.5          47.4           3.55
road                 38.0           8.6           3.27   <- already stable, biggest surface
vegetation            5.4          51.3           2.80
pole                  1.3         100.5           1.34
```
The global de-shimmer only smooths pixels optical flow calls STATIONARY. Buildings sweep past the
camera, so the class owning most of the flicker was the one it refused to touch, while the road --
38% of the frame and already stable -- got smoothed hard. Measured effect of the shipped filter:
road sharpness 64.6 -> 50.5 (-22%), building flicker 186 -> 183 (-2%).

`class_deshimmer.py` weights by class AND relaxes the motion gate per class (buildings and poles
move by smooth ego-motion over rigid surfaces, unlike the erratic flow that defeated motion
compensation in note 20; cars keep the tight gate so they do not trail). Delivered result:
```
                road sharp   road flick   bldg sharp   bldg flick
v50d (shipped)        50.5         28.0       1632.5        183.0
v50g (new)            73.7         41.0       1199.1        102.0
```
Building flicker -44%, road sharpness +46%, CIPO unchanged (0.884 vs 0.887), range slightly better
(1.52 vs 1.62). Whole-frame alt p99 reads slightly WORSE (26.44 vs 24.95) purely because the road
is no longer smoothed -- a whole-frame metric cannot see this trade, which is why the per-class
comparison exists.

### 34. protect_vehicle_colour turned a red bus magenta by reading hue off a grey surface
The user's "sunny looks worse than night" traced to v50b, the vehicle-colour stage. Replicating its
own decision per region on Town10HD frame 450: CARLA's vehicles have median saturation 5-31 out of
255 -- effectively neutral -- yet the circular mean reports a confident hue near 125 (blue) with
resultant length 0.94-0.99, because on a near-grey surface the angle is set by tiny channel
differences. The script adopted that hue for the render's red bus and blended, and red toward blue
is magenta. A control region with CARLA saturation 93 was correctly skipped, so the logic was right
whenever the colour was real; it simply never checked whether it was.
FIX: an achromatic guard (MIN_SAT). When CARLA's region is neutral, do not adopt a hue at all --
only pull the render's saturation toward neutral, which still removes the invented warm cast this
stage exists for. Shipped in v50f/v50g. **This bug was in every sunny delivery from v50b onward,
including the v50d baseline.**

### 35. v63: distance-weighted vegetation loss fixes the distance falloff and overshoots the road
The user's complaint was distant trees, and note 33's filtering work could not touch it: a filter
only removes. The deficit was in what the model never learned, so this was a training run.
20 epochs on the full 32,475-image corpus, `--veg_weight 4.0 --far_boost 3.0`, lr 5e-5, ~37h.

Vegetation, Town10HD, Laplacian variance on vegetation-labelled pixels:
```
                      near      far   far/near   % of photo ceiling
REAL training photos  1191     1246       1.05        100%
CARLA source           928     1030       1.11         80%
v50 parent             673      561       0.83         51%
v63                    673      670       1.00         55%
```
The distance falloff is GONE -- far/near 0.83 -> 1.00, far detail +19.5% -- and the effect is
visible: palm fronds resolve as fronds instead of mush, and v50's magenta/green speckle in the
crowns disappears (specks on vegetation-labelled pixels fall 22110 -> 15644 over 20 tail frames).
Near detail did not move, so the absolute figure only reaches 55% of the photographs. That is the
honest read: the run fixed the ratio, not the level.

But it raised high-frequency output everywhere, not only on vegetation, and tail_check FAILS:
road 1.81x the parent, sky 1.57x, specks 1.24x, against a 1.2x gate.

Comparing to the parent says a defect is new; it does not say which side of reality the parent sat
on. So `road_sky_ceiling.py` measures the same high-pass energy on the real training photos and the
CARLA source (photos are 2048x1536, renders 2048x1024, so the per-pixel scale is comparable):
```
                     road hp   sky hp (erode 9)   sky hp (erode 31)
REAL training photos   25.70             14.584              13.535
CARLA source           23.81              0.145               0.100
v50 parent             29.66              0.913               0.367
v63                    53.79              1.438               0.713
```
Two different answers from one gate failure:
- ROAD is a real regression. The parent was already slightly above both references; v63 is 2.1x a
  real photograph. That is invented grain on the one surface where lane geometry is read.
- SKY is not ghosting. Under a 31x31 erode -- hard enough to exclude the wires, fronds and banner
  edges the label map does not resolve -- v63 is still 1.9x the parent, so it is not edge spill
  either; but at 0.713 against a photograph's 13.5, both models render a sky 19x smoother than any
  real one. The metric is measuring the right thing on a scale where neither model is near it.
Extra specks are on buildings (+22k) and vehicles (+10k), not road (-778) or vegetation (-6466):
v63 renders facade and trim structure the parent smoothed away.

VERDICT: gate not loosened, baseline unchanged. The road overshoot is the blocker, not the sky.

Rendering every saved checkpoint (`sweep_v63.sh`, `epoch_sweep.py`) shows road is bad from the
start, so it is not late drift and no epoch is deliverable:
```
render      veg near  veg far  far/near   road    sky   specks   gate
parent           672      561      0.83  1.00x  1.00x   1.00x   PASS
ep8              811      723      0.89  1.88x  1.21x   1.12x   FAIL
ep12             570      611      1.07  1.87x  2.40x   1.31x   FAIL
ep16             627      591      0.94  2.89x  2.04x   1.25x   FAIL
ep20 (latest)    673      670      1.00  1.81x  1.57x   1.24x   FAIL
```
CAUSE: `VGGLoss.forward` with a weight map is a WEIGHTED MEAN -- `(w*d).sum()/w.sum()` -- so
emphasis on vegetation is dilution of everything else by 1/mean(w). The road lost perceptual
pressure and the GAN term, unweighted and rewarded for plausible texture, filled the gap. Note the
corollary: down-weighting road would make this WORSE, not better.
FOLLOW-UP: `--veg_extra` adds a vegetation-only perceptual term on top of the unmodified objective
instead of redistributing it (`VGGLoss.forward_plus`). v64 tests exactly that, 12 epochs.

### 36. v64: the additive form fixed the dilution and did not fix the road — the road is the corpus
`--veg_extra 3.0 --far_boost 3.0`, 12 epochs, ~23 h. Verified in `opt.txt` that training really ran
with `veg_extra: 3.0`. It removed the dilution mechanism exactly as designed: the plain VGG term
ran at full strength and vegetation got an extra term on top. Result:
```
source                 near      far   far/near   % of photo ceiling
REAL training photos   1191     1246       1.05        100%
v50 parent              673      561       0.83         51%
v63 (weight map)        673      670       1.00         55%
v64 (additive)          370      568       1.53         38%
```
Worse than the parent, and worse than v63. far/near 1.53 looks like the best ratio of the three and
is meaningless: FAR did not move (568 vs 561), NEAR COLLAPSED 673 -> 370. The ratio improved because
the near end fell out, which is the exact failure veg_report.py's two-column design exists to catch,
inverted. Absolute detail fell to 38% of the photographs. `check_v64.png` agrees by eye: muddier
grade, colour speckle on the van and bus, hazier buildings than the same frame from v63.

And the road did NOT improve: 58.78 vs v63's 53.79, still ~2x the parent and ~2.3x a real
photograph.

THE REAL CONCLUSION. Two runs with opposite loss geometries -- redistributive and additive -- both
land at road 1.8-2.0x the parent, and v63's ladder puts it there by epoch 8. The road grain is not
caused by the vegetation weighting. It comes from fine-tuning v50 on this corpus at lr 5e-5 at all:
the parent is a converged point whose road happens to sit right at the photographic reference
(29.66 vs 25.70), and further training on Mapillary-heavy data pushes the GAN to texture it. No
amount of vegetation loss shaping addresses that, so this line of attack is finished.

VERDICT: v64 rejected outright. v50d remains the sunny baseline. Gate never loosened.
STILL TRUE: v63 fixed the distance falloff (far/near 0.83 -> 1.00, far +19.5%) and looks better on
distant foliage. Its cost is road grain at 2.1x a real photograph. That is a trade for the user's
eye, not for a metric to decide, so a like-for-like delivery clip is being built for them to judge.

### 37. v63 delivered for the user's eye: better picture, mixed perception, road still the cost
Town10HD sunny through the full current chain (`render_model.sh sunny carla2real_semantic_v63_veg
v63 Town10HD`). Vision Pilot on the same recording, same GT:
```
                 CIPO recall   CIPO false alarm   range MAE   range p95   lane MAE   CTE jitter p99
v50d                   0.795              0.612      4.00 m       13.15      0.227            0.161
v50j                   0.783              0.699      4.02 m       13.15      0.213            0.160
v63                    0.744              0.306      3.26 m       12.03      0.237            0.200
```
Not a clean win either way, and more interesting than the image metrics implied: v63 finds ~5 points
fewer of the objects that are present but HALVES the false alarms and is 19% better on range. Lane
and stability are marginally worse.

Two comparison clips, because one of them lies:
- `COMPARE_town10hd_sunny_v50d_vs_v63.mp4` flatters v63 heavily, but v50d predates BOTH the
  achromatic-guard colour fix (note 34) and the building-structure injection, so it shows model plus
  chain fixes together. At frame 620 v50d's bus is magenta and v63's is red -- that is note 34, not
  the model.
- `COMPARE_town10hd_sunny_v50j_vs_v63.mp4` is the honest one: same chain, only the model differs.
  v63 still wins clearly on facade structure, vehicle panel definition and distant foliage.

A visual impression that did NOT survive measurement: v63's ego car looks bronze against v50j's
silver, which read as colour drift. Measured against the CARLA source on the same frame, CARLA's own
car is warm silver (R-B +15.0, sat 22.4); v63 is +8.5 / 23.2 and v50j is +7.2 / 31.9. v63 is CLOSER
to the source on both. Withdrawn -- and a reminder that the eye needs the reference too, not just
the two renders side by side.

STILL NOT PROMOTED. Road grain at 2.1x a real photograph is unchanged, the gate still fails, and
v50d remains the sunny baseline until the user looks.
