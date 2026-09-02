# WHERE WE ARE — 2026-09-01

Read this first after any context loss. Companion to `EXPERIMENTS.md` (26 numbered findings).

## THE CROSSHATCH: root cause found 2026-08-25 — EXPOSURE BIAS

The user reported "weird shiny white dots all over the screen" on v54 and a "faint ghost outline"
on night. Looking at a frame showed far worse than dots: a woven crosshatch over every flat
surface. It is in the RAW model output, not the delivery pipeline.

### The diagnosis chain, including two wrong turns

```
road texture vs parent, Town10HD, road-labelled pixels only
  v50 parent (clean)                          1.00x
  v54  temporal, wrong-direction flow         5.81x
  v57a same but video discriminator REMOVED  12.99x   <- WORSE, refuted that hypothesis
  v58  temporal, flow direction CORRECTED     7.98x   <- WORSE, refuted that one too
  v58  with the prev channel ZEROED           0.18x   <- clean, confirmed by eye
```

1. **Video discriminator?** No. Removing it doubled the weave. It was the only term RESISTING it.
2. **Flow direction?** It was genuinely wrong — `warp()` wants cur->prev, the builder computed
   prev->cur, and the warped target was worse aligned than not warping at all (4.266 vs 3.423).
   Fixed, verified (now 48% better than unwarped). **The weave got worse.** A real bug, not this
   bug. Keep the fix.
3. **Resampling artifacts in the warp target?** No. The target measures SMOOTHER than its source
   (0.58 vs 0.66 high-pass on flat regions), so it carries no weave to copy.
4. **Exposure bias. Yes.** Training always feeds G the REAL previous frame. Inference feeds G its
   own previous OUTPUT — and a black frame at t=0, which is wildly out of distribution. G never
   saw its own output during training, so small errors compound into a texture. Zeroing the prev
   channel removes the weave completely (0.18x, visually spotless road).

### Why every metric missed it, which is the durable lesson

- `alt p99` said v54 was CALMER. A static weave is perfectly frame-to-frame consistent, so the
  flicker metric REWARDS it. The defect and the success metric were aligned.
- Laplacian sharpness said +80%. A weave is high-frequency, so it reads as detail.
- "clean sharpness" (Laplacian after a median) still said sharper — a median removes isolated
  specks, not a structured pattern.
- An FFT peak-ratio detector barely moved (79.9 vs 87.3).
- CIPO was the ONLY metric that objected (0.887 -> 0.849) and it was under-weighted as a trade.

`road_texture.py` is the metric that works: high-pass energy on pixels the LABEL MAP says are
road, relative to the parent. It asks "is there detail where there should be none" instead of
"is there detail", which is the question the others were all asking. **Gate is 1.5x, and it is
enforced in fix_temporal.sh before any delivery.** And always LOOK at the crop — the user
identified this in seconds from the video.

### The fix

`test.py` already has `VID2VID_PREV_SCALE` (documented as an anti-drift knob) and it has never
been used. Sweeping it trades weave against flicker: 0 is single-frame (clean, no temporal
benefit), 1.0 is full autoregression (weave runs away). If an intermediate value gives a low
weave AND flicker well under the parent's 52.44, the model is usable as trained.

If not, the proper fix is training-time **student forcing**: feed G its own generated previous
frame for a fraction of steps so its inference-time input distribution is one it has actually
seen. That is a training-loop change in train.py, ~2 h to run.

## FINAL VERDICT 2026-08-26 00:00 — BASELINES UNCHANGED: sunny v50d, night v51

Both temporal models were fixed (no crosshatch), delivered on all 5 towns, and scored. Neither
earns the baseline.

### Sunny, 5 towns, delivered

```
tag      alt p99    sharp    CIPO    rng    road tex vs parent
v49        22.42    517.1   0.769   1.82
v50d       24.95    810.4   0.887   1.62         1.00 (it IS the reference)
v58        26.27   1011.9   0.827   1.62         0.35
```

v58 is **+25% sharper** and its road is **cleaner than v50d's** (0.35x), and it needs no DVP at
all — but it is **5% MORE flickery** than v50d and costs **0.06 CIPO**. The entire purpose of the
exercise was less flicker, and delivered it has slightly more. **v50d stays.**

Why: v58's RAW output at prev-scale 0.5 is alt 41.67 against the parent's 56.75 (-27%), a real
model-level gain. But v50d gets to 24.95 using DVP + de-shimmer, and the trained-in gain is
roughly equal to what post-processing was already buying. The model improvement did not stack on
top of the pipeline; it replaced it, at parity.

### Night, 5 towns, delivered

```
tag      alt p99    sharp    CIPO    rng    road tex vs parent
v51        23.75    222.4   0.869   3.27         1.00
v59        22.70    219.8   0.865   3.51         0.99
```

-4% flicker, -0.004 CIPO, +7% range error. A wash. **v51 stays.**

### The honest summary of the whole temporal effort

The flicker CAN be reduced at the source — that part is real and reproducible. But once the
crosshatch is suppressed (prev-scale 0.5), the surviving benefit is about the same size as what
`temporal_deshimmer.py` + DVP already deliver, and it comes with a CIPO cost. The one durable win
is that v58 reaches v50d-class flicker with +25% sharpness and **without DVP**, which is 1 h 56 of
GPU per clip. If throughput ever matters more than the 0.06 CIPO, v58 is the better engine.

Delivered for comparison, not promoted:
  `calibrated/v58/sunny/` (5 towns) · `calibrated/v59/night/` (5 towns)
  `calibrated/_REJECTED/` — v54/v55/v56/tsun/tnig, the crosshatch versions, with an explanation.

## TEMPORAL WORK CLOSED 2026-08-26 — sunny v50d unchanged; night moved to v59 on 2026-08-27

Four hypotheses for the sky ghost were each eliminated by measurement; see EXPERIMENTS.md note 28.
v61 (prev-channel augmentation) was the last and made things WORSE — sky 10.51x, road texture
3.20x, confirmed bad by eye. The temporal fine-tune does not beat v50d and is abandoned.

Candidates delivered for the user's eye, NOT promoted:
  `calibrated/v58/sunny/`  temporal sunny, 5 towns   (alt 26.27 vs v50d 24.95, CIPO 0.827 vs 0.887)
  `calibrated/v59/night/`  temporal night, 5 towns   (a wash against v51)
  `calibrated/v51d/night/` night cleanup, 5 towns    (-19% flicker, -30% specks, -19% sharpness)
  `calibrated/_REJECTED/`  the crosshatch versions, with an explanation

Open, and worth doing: Town10HD sunny frames 709-731 have the camera buried in geometry after a
respawn (the defect the user reported at 0:22). `reroll_town10hd.sh` re-records it with
--no_teleport and verifies with `check_teleport.py`, which is exact — a buried camera's semantic
map collapses to a single class. It delivers as a SEPARATE take (Town10HDb) so the existing
comparable set is untouched.

## OVERNIGHT 26->27 AUG: two buried-camera takes found and re-recorded

`check_teleport.py` (semantic-map based, exact) scanned every recording and found the defect the
user reported at 0:22 — plus a second one nobody had noticed:

```
Town10HD sunny   frames 709-731 buried in concrete, ego returns as a DIFFERENT vehicle
Town03   night   frames 880-901 buried against a wall then fully black
all other takes  clean (Town04/05/06/07 sunny+night, Town05dense, Town10HD night)
```

Both re-recorded with `--no_teleport` and verified clean, delivered ALONGSIDE the originals so the
comparable set is untouched:
```
town10hdb_sunny_vp55_v50clean_FINAL_1920_visionpilot.mp4   CIPO 0.993  rng 1.70 m  lane 0.24 m
town03b_night_vp55_clean_FINAL_1920_visionpilot.mp4        CIPO 1.000  rng 2.02 m  lane 0.12 m
```
(Different routes from the originals, so these scores are not directly comparable to the 5-town
means — they say the clips are clean and usable, not that the model improved.)

**Town03 night needed a different spawn strategy.** Four `buildings` attempts all aborted with
"stuck 150 ticks" at frames 208-245 regardless of NPC count (110 down to 50), which ruled out
traffic — the autopilot kept driving somewhere it could not leave. `--spawn_mode highway` (prefers
multi-lane road) succeeded on the first attempt. `reroll_town.sh <Town> <weather>` is the
generalised tool; `SPAWN=highway` and `NPCS=` are env knobs.

Three script bugs fixed while doing this, all of which had silently wasted time:
- `carla.Client(...).set_timeout(10.0).get_world()` — set_timeout returns None, so this readiness
  probe raises AttributeError EVERY time and reports "CARLA never came up" while CARLA runs fine.
  It orphaned a server holding 5.8 GB. Inherited from expand_coverage2.sh, still latent there.
- `wait_for_gpu` before starting CARLA deadlocks: CARLA renders on the GPU, so the script waited
  for the very resource it was about to use. Now waits for a trainer/renderer by name instead.
- `carla_down` sent a plain TERM which the Shipping binary survived; now escalates and verifies.

## SUNNY FIXED 2026-08-27 — v50g is the candidate; the temporal family is closed

Three findings, in order of how much they matter to the picture:

1. **protect_vehicle_colour was reading hue off grey surfaces** and turning a red bus magenta.
   Present in EVERY sunny delivery from v50b on, including the v50d baseline. Fixed with an
   achromatic guard (EXPERIMENTS.md note 34).
2. **The de-shimmer was aimed wrong.** Buildings own more flicker than all other classes combined
   (mean 48.0 over 25% of frame) but sweep past the camera, so the motion gate excluded them --
   while the road (38% of frame, already stable at 8.6) got smoothed hard. That is exactly why the
   clips read blurry AND flickery. `class_deshimmer.py` weights by class and relaxes the motion
   gate for rigid ego-motion classes only (note 33).
3. **--multiframe cured the sky ghost but gave no flicker benefit**, which closes the temporal
   family: --temporal is artifact-ridden in every configuration, --multiframe is clean but useless
   for this (note 32). Do not reopen it.

```
delivered, Town10HD    road sharp   road flick   bldg sharp   bldg flick
v50d (current baseline)      50.5         28.0       1632.5        183.0
v50g (new)                   73.7         41.0       1199.1        102.0
5-town: CIPO 0.884 vs 0.887, range 1.52 vs 1.62, ghost 3.93 vs 4.49
```

**v50g was a sunny candidate and was not chosen.** Superseded: the sunny baseline is **v50l** as
of 2026-09-02. Night remains v59.

## Baselines

| weather | baseline | why |
|---|---|---|
| night | **v59** (`carla2real_semantic_v59_tnight3`) | user's call 2026-08-27, BY EYE from COMPARE_town10hd_night_v51_vs_v59.mp4. Metrics were a near-tie (flicker 22.70 vs 23.75, CIPO 0.865 vs 0.869, range 3.51 vs 3.27) so this is a visual preference, which is how every baseline here has been chosen. |
| sunny | **v50m** | user's call 2026-09-02, replacing v50l the same day. v50l's delivery chain with three fixes: car de-shimmer tightened (trails), a robust + rate-limited colour grade (shade switching), unsharp 0.55→0.75 and CARLA facade injection 1.0→1.2 (sharpness). Measured on Town05: **real instability +1.2%** (warped residual 10.45 → 10.58) for **+15% detail**, less trailing (LAG 0.389 → 0.407), colour held, CIPO unchanged at 0.742. `make_v50m.sh`. |
| sunny (previous) | v50l | user's call 2026-09-02, BY EYE. v63's colour grade carried by the v50j render (`fuse_colour.py` / `make_v50kl.sh`) — NOT a trained model. Over 5 towns against v50d: flicker −10.2%, colourfulness +23.2%, sharpness −2.9%, CIPO 0.879 vs 0.887, lane MAE 0.196 vs 0.204. Reproducing it needs the v50j chain AND a v63 render of the same town. |
| sunny (previous) | v50d | user's call 2026-08-24, superseded 2026-09-02. Still the reference every v50x variant is measured against. |

Sunny/night are **separate lineages and always have been**. Night uses `--light_input`, sunny uses
`--chroma_input`, and neither has ever seen the other's data. There is no "v51 sunny" and there
never was — v50 is the sunny model of that generation.

## Delivered clips

- Masters (no HUD): `$CARLA2REAL_OUT/*_FINAL_1920_visionpilot.mp4`
  (the `_visionpilot` suffix on these is a misnomer — they are the plain renders)
- With VP HUD: `$CARLA2REAL_OUT/calibrated/<model>/<sunny|night>/`
- Organised tree: `pix2pixHD/results/mp4/NEW/<model>/town<N>/<weather>/`
- Re-file after any new delivery: `bash organise_calibrated.sh && bash refresh_new.sh`

Model tags: v44 (old, broken homography) · v47 night · v47r (v47 re-posted) · v49 sunny ·
v50 sunny · v50b (+colour+sharpen) · v50d (+de-shimmer) · v51 night · thingsunny/thingnight
(FAILED experiment, keep for comparison only) · `cov` was a JOB tag, not a model.

## Scores (5 matched towns, `python3 compare_models.py`)

```
night  v47   CIPO 0.694  rng 4.88 m  lane 0.23 m
night  v51   CIPO 0.869  rng 3.27 m  lane 0.24 m
sunny  v49   CIPO 0.769  rng 1.82 m  lane 0.19 m
sunny  v50   CIPO 0.899  rng 1.49 m  lane 0.19 m
```

## The three faults the user reported on v50, and their status

1. **"sandy" at 0:22 on town10hd** — FIXED. The generator repaints vehicles: CARLA has a beige
   Lincoln, v50 painted it maroon, and that car filling half the frame warmed the whole shot.
   21.3% of frames ran warm(R-B)>10 vs raw CARLA's 0.0%. Present in v49 too, so not new.
   Fix: `protect_vehicle_colour.py` — per-region hue from CARLA, render keeps luminance/texture,
   only acts where they disagree. Warm frames 21.3% -> 4.8%.

2. **Blur** — FIXED by an edge-masked unsharp in `fix_v50_pass.sh` (sharpness +85%, no flicker
   cost because flat regions are excluded).

3. **Flicker** — FIXED by `temporal_deshimmer.py`. CRITICAL LESSON: mean static-pixel luminance
   delta said v50 was CALMER than v49 (2.79 vs 3.37), contradicting the user. The eye responds to
   ALTERNATION, `|2*x[t] - x[t-1] - x[t+1]|`. On that metric the user was right:

```
                          alt p99   sharpness   (960x480, 150 frames, town10hd sunny)
v49  stable, muddled       27.51      440.9
v50  detailed, shimmery    43.85      464.2
v50b +sharpen              45.48      772.0     <- sharpening made shimmer worse
v50d +de-shimmer           26.72      634.3     <- calmer than v49 AND 44% sharper
```

   `temporal_deshimmer.py` = 3-frame temporal MEDIAN, gated to pixels that are both alternating
   and static by optical flow. Median not mean: a mean blurs edges, which is v49's muddiness.
   Tuned defaults `--alt 1 --flow 6.0` (~64% of pixels gated).

4. **Teleport at frame 709 on town10hd sunny** — NOT FIXED, needs a re-record. The jump is in the
   RAW CARLA recording (frame-delta 64.7 vs ~12 normal); `record_town_auto.py` respawned the ego
   after it got stuck, twice in that take. Post-processing cannot touch it. Fix would be
   re-record with fewer NPCs + `--no_teleport`, then re-render (~15 min + ~2.5 h GPU).
   User has not asked for this yet.

## RUNNING NOW — overnight queue (2026-08-24 21:30), strictly serialised on the GPU

```
1  train_temporal.sh   night render of `tnig` (v55 + DVP)         -> temporal_log.txt
2  epoch_ladder.sh     raw-renders every saved epoch, both models -> epoch_ladder_log.txt
3  clean_delivery.sh   SUNNY only: probes epoch 1 vs 3 on Town10HD by CIPO,
                       then delivers the winner over the other 4 towns, all DVP-free
                                                                   -> clean_delivery_log.txt
4  night_v2.sh         trains v56 on NuRec-night sequences only, delivers only if it
                       beats v55                                   -> night_v2_log.txt
```

Each waits on the previous one's DONE marker. **A completion marker is not a lock** — that lesson
cost two clips before; these also check the previous process is actually alive before waiting.

Tags: `tsun`/`tnig` = temporal models WITH DVP · `v54`/`v55`/`v56` = DVP-free ·
`v54e1`/`v54elatest` = the epoch probes.

## Why post-processing was abandoned, with numbers

Every de-shimmer setting just slides along a shimmer-vs-ghost curve; none of them moves it.

```
                          alt p99   sharp    ghost   (town10hd, 150 frames, 960x480)
v50b (no de-shimmer)        45.48   772.0     0.00
3-frame gated (= v50d)      26.72   634.3     4.01
3-frame no gate (= v50e)    18.82   625.4     5.87
5-frame no gate             12.76   525.0     8.79
motion-compensated 3-frame  37.20   486.9     3.89   <- WORSE, see below
v49 ("the stable one")      27.51   440.9       --
```

**Motion compensation failed and is a dead end.** Warping neighbours by optical flow before the
median should have broken the trade (a moving car sits on top of itself, so nothing smears). It
made things worse because the flow is estimated FROM the shimmering video: the shimmer corrupts
the flow, the warp adds its own error, and the consistency check then rejects so many pixels the
median runs out of votes.

### v50e — measured, delivered, and REJECTED

v50e = v50b + de-shimmer at maximum strength (no gating). 5-town means:

```
              alt p99   sharpness  |  CIPO   rng MAE
v49             24.25       542    | 0.769     1.82
v50             34.55       651    | 0.899     1.49
v50d            27.01       854    | 0.887     1.62
v50e            17.20       799    | 0.879     1.60
```

On flicker and perception it looks like a clear win — 36% calmer than v50d for 0.008 CIPO. But
ghosting, measured against un-deshimmered v50b on pixels optical flow says actually moved:

```
town03    v50d 4.83    v50e 7.92
town10hd  v50d 4.96    v50e 8.30
```

The v1 stabiliser scored **6.30**, and that is the build where the user could plainly see
car-outline trails. v50e is well past that line, so it buys its calm with the exact artifact the
user said must never happen again. **Not a baseline.** Left in `calibrated/v50e/` with
`READ_THIS_FIRST.txt` explaining the trade. v50d remains the sunny baseline.

## The temporal fine-tune — what was actually wrong and what was fixed

Root cause of flicker: pix2pixHD trains on SINGLE FRAMES. Nothing in its loss requires frame t to
agree with t-1, so two near-identical label maps yield different textures. `--temporal` adds an
occlusion-masked flow-warp consistency loss; `--video_disc` adds a discriminator over (prev,cur)
pairs, which is what stops the consistency loss from simply blurring everything. Both have
existed unused in this fork since v44.

### SIX blockers found and fixed before the run (all verified by `preflight_temporal.py`)

1. **`--temporal` widens netG by output_nc.** THE DANGEROUS ONE. The previous frame is
   concatenated for G, so v50's 73-channel first conv does not fit the 76 the temporal model
   builds — and `load_network` would have silently random-inited it, throwing v50 away while the
   run looked perfectly healthy. Fixed with a zero graft (`make_v50_init.py --g-only`, new flag):
   `w[:, :73] = v50; w[:, 73:] = 0`, so step 0 is bit-identical to v50 and training can only add.
   D is NOT widened by `--temporal`, hence `--g-only`; D is copied through untouched.
2. **`temporal_dataset.py` returned no depth/normal/chroma/light.** Passing those flags with a
   None-supplying loader skips the concat and triggers the same silent fallback. Extended to load
   all four with BILINEAR/normalize=False, matching `aligned_dataset.py` exactly; it now raises if
   a requested channel dir is missing rather than quietly returning 0.
3. **Clip boundaries were being recovered from the wrong source.** The old builder assumed
   `nrhr_%05d` ↔ `sorted(nurec_raw/*)` index-parallel. It is not: only 9,069 of 23,053 raws have
   a pair, so every boundary would have been mismapped and the flow paired across cuts.
   `build_v40_corpus.do_nurec()` actually enumerates
   `concat over SUNNY of sorted(test_mp4/<v>_work/frames/*.jpg)` — repeating that recovers
   (video, frame) exactly. Same scheme for `ngt_` via `build_night_corpus.py`.
4. **Dark Zurich cannot be recovered that way** — `dz_filter.py` kept a filtered subset. Rather
   than reproducing its thresholds, the builder aligns the two sorted lists by thumbnail
   correlation in a monotonic walk and refuses the result below 95% confident matches.
5. **`train_temporal.sh` passed arch as a var-prefix on a function call**, which bash keeps in the
   environment afterwards, leaking the sunny flags into the night run. Now a positional parameter.
6. **`render_model.sh` hardcoded arch with no `--temporal`**, so the render would have built a
   narrower netG than the checkpoint and silently random-inited it. Added `TEMPORAL=1` and a
   `grep 'not initialized'` abort after every render.

Flow stays `.npy` (quarter-res float16) — `TemporalDataset` does `np.load`. Do not switch to PNG.

### Corpus (both built and preflight-verified 2026-08-24)

```
sunny  datasets/training_temporal_sunny   9,069 frames /  10 clips / 9,059 flow
       source training_v49_chroma  (img,label,edge,depth,normal,chroma under one set of stems)
night  datasets/training_temporal_night   6,182 frames / 133 clips / 6,049 flow
       source training_v51_night   (img,label,edge,depth,normal,light)
```

flow == frames - clips in both, which is the invariant preflight checks (every frame gets a flow
map except the first of each clip). Channels are all symlinked; nothing was recomputed.

Two night-only defects found and fixed while building, both silent:
- **1,368 `GP*` frames were being dropped.** A GoPro splits a long recording into GOPR0351.MP4,
  GP010351.MP4, GP020351.MP4 ...; the old `GOPR\d+` regex matched none of the continuations.
  Each segment restarts its frame numbering, so they are now separate clips (which is also what
  they physically are). Recovered 882 usable frames.
- **`dz_filter.py` kept only dark-enough frames, so the "consecutive" frames were not.** Drive
  0375 contributes 64 frames spanning 1,061 — pairing those as t-1,t would feed the temporal loss
  a large-displacement Farneback estimate (noise) and show the video discriminator transitions no
  real video contains. The builder now splits a segment wherever the frame number jumps by more
  than `MAXGAP=2`, and drops runs shorter than 2. Dark Zurich went from 1,700 frames with
  fabricated adjacency to 2,582 frames in 129 genuinely contiguous runs.

Sunny needed neither fix: every source video is 100% kept with max gap 1, and video `20` has no
frames directory at all, which is why the total is 9,069 and not 9,900.

### Raw-model baselines (Town10HD, 300 frames, BEFORE any post-processing)

```
v49 raw   alt 54.63   sharp 1285
v50 raw   alt 52.44   sharp  961
```

The two parents are nearly identical in raw flicker, so the difference the user sees between
delivered v49 and v50 comes from POST-PROCESSING, not the models. Delivered v50d is alt 27.01 —
i.e. the pipeline (stabilise + DVP + de-shimmer) currently halves it. `raw_flicker.py` measures
this; use it, not the delivered clip, to answer "did the temporal training work?", because DVP
alone removes ~33% of flicker and would mask the entire effect.

### RESULT: the temporal fine-tune worked (sunny, 2026-08-24 17:35)

v54_tsunny trained clean — **zero `not initialized`**, so the graft held and v50's weights were
preserved. G_Temporal fell 1.976 -> 0.589 over 3 epochs (~36 min each).

Town10HD, 300 frames, RAW render with no post-processing of any kind:

```
                 alt p99   sharpness
v49 raw            54.63      1285
v50 raw            52.44       961
v54 raw            25.74      1908     <- 51% less flicker, 2.0x sharper than its own parent
delivered v50d     27.01       854     <- and this is WITH stabiliser + DVP + de-shimmer
```

**v54's raw output is calmer than fully post-processed v50d, and 2.2x sharper than it.**

Two checks before believing that:

1. **Autoregressive drift** — test.py feeds the previous GENERATED frame back in, so the classic
   failure is a clip that looks great early and turns to mush by frame 800. It does not:
```
   window      v50 alt  v50 sh  |  v54 alt  v54 sh
   0-200         49.82   934.9  |    24.26  1894.9
   300-500       52.84   767.5  |    29.84  2219.0
   600-800       37.54   988.7  |    19.71  2349.8
   800-1000      66.85  1369.0  |    37.82  2417.3
```
   v54 tracks v50's scene-driven variation while staying 40-50% below it throughout, and its
   sharpness RISES along the clip rather than degrading. No compounding error.

2. **Is sharpness 1908 real detail or high-frequency noise?** Sharpness split by whether optical
   flow says the pixel moved:
```
          static sh   moving sh   ratio
   v50        613.3      1333.9    2.18
   v54       1784.0      1987.2    1.11
```
   v54 is ~3x sharper on static regions. The decisive argument is simpler than the table though:
   **v54 has 2x the sharpness AND half the flicker. Noise cannot be both high-frequency and
   temporally stable**, so the extra detail is real. This is exactly what the video discriminator
   was there to enforce — it is what stops the consistency loss from just blurring everything.

**Independent cross-check.** `stabilize_frames_v2.py` computes its own unrelated "ground
flicker" metric and reports how much it managed to remove. Across every model in the render logs
that number starts at 7.33-14.34; on v54 it starts at **5.73**, the lowest ever logged here, and
the stabiliser could only find 6% to remove (against 8-20% on every other model). A second,
independently-written metric agreeing with `alt p99` is worth more than either alone.

### Consequence: DVP and the de-shimmer come off

`render_model.sh` gained `NO_TEMPORAL_POST=1`, which sets the stabiliser's `--alpha 0` (its
temporal blend term is `eff = alpha * trust * motion_scale`, so this zeroes the smoothing while
keeping its detail/saturation grade) and skips DVP entirely. `clean_delivery.sh` delivers v54/v55
that way across all 5 towns, as tags **`v54`/`v55`**, alongside the chain's **`tsun`/`tnig`**
which DO include DVP — that pair is the controlled measurement of what DVP is still worth here.
Saves 1 h 56 of GPU per clip.

STILL UNVERIFIED and the thing to check first: **CIPO from score_vp.py, and the user's eye.**
Every metric here says v54 is better; the user's eye has overruled my metrics before.

### NIGHT: the same recipe barely worked, and why

v55_tnight trained clean (zero `not initialized`, all epochs saved) but converged worse than
sunny — G_GAN 2.06 vs 0.83, G_Temporal 1.12 vs 0.59. The raw result matches that:

```
                    alt p99   sharpness
v51 raw (parent)      32.55       497.1
v55 raw               27.92       454.1     -14% flicker, -9% SHARPNESS
--- for contrast, sunny with identical code/hyperparameters/graft ---
v50 raw (parent)      52.44       961
v54 raw               25.74      1908       -51% flicker, +99% sharpness
```

**The corpora are the difference, and it is sequence QUALITY not volume.**

```
sunny  9,069 frames /  10 clips  = 907 frames per clip
night  6,182 frames / 133 clips  =  46 frames per clip
```

Night's 133 clips are the 4 good NuRec ones (900 each) plus 129 Dark Zurich fragments averaging
~20 frames, because `dz_filter.py` kept only frames dark enough to count as night. A flow-warp
consistency loss and a video discriminator learn from long consecutive runs; 20-frame shards
yield almost no usable pairs while dominating the clip count.

The obvious objection — dropping Dark Zurich loses appearance diversity — does not apply. This is
a 3-epoch fine-tune ON TOP OF v51, and **v51 already learned Dark Zurich's appearance from that
exact data**. The temporal stage only needs good motion.

`night_v2.sh` tests exactly that: same code, same hyperparameters, same graft, corpus = the 4
clean NuRec night clips only (3,600 frames, `training_temporal_nightseq`). It raw-renders Town03
and **only delivers if it actually beats v55** — otherwise night stays on v51 and we have a clean
negative result instead of hours spent shipping a worse model.

### DEFINITIVE SUNNY RESULT — 5 towns, v54 delivered DVP-free (2026-08-25 02:02)

```
tag      alt p99    sharp   ghost    CIPO    rng
v49        22.42    517.1       -   0.769   1.82
v50d       24.95    810.4    4.49   0.887   1.62
v50e       15.66    733.0    7.05   0.879   1.60   (rejected: ghost)
v54        18.45   1458.4       -   0.849   1.47
```

per town:
```
town       v50d CIPO  v54 CIPO  |  v50d rng  v54 rng
town03         0.929     0.819  |      1.35     0.70
town04         0.999     1.000  |      1.25     1.09
town05         0.746     0.750  |      1.08     1.51
town06         0.967     0.990  |      0.44     0.37
town10hd       0.795     0.686  |      4.00     3.68
MEAN           0.887     0.849  |      1.62     1.47
```

**v54 vs v50d: -26% flicker, +80% sharpness, -9% range error, -0.04 CIPO.** It ties or wins CIPO
on 3 of 5 towns and loses ~0.11 on two (town03, town10hd). This is a far better trade than the
Town10HD-only reading suggested (-0.13) — that town is v54's worst case, not its typical one.

**NEAR-MISS worth remembering:** the first 5-town run reported v54 CIPO **0.890** against v50d's
0.887 and looked like an outright win. It was averaged over 4 towns, because town10hd's v54 clip
was copied from the `v54elatest` probe and its score sat under a different tag — and the missing
town was v54's WORST. `flicker_report.py` now prints `towns` and `scored` counts per tag and
flags any incomplete mean, because a mean silently taken over a different denominator than the
row above it is indistinguishable from a real result.

### DVP is NOT the cause of the CIPO drop (Town10HD)

```
                              CIPO
v50d  (baseline)             0.795
tsun  (v54 + DVP)            0.665
v54elatest (v54, DVP-free)   0.686
```

Removing DVP recovered only +0.02 of a 0.13 gap. **So cause (a) is largely ruled out: the
detection loss is the MODEL, not the post-processing.** That leaves drift, which the epoch-1
probe was supposed to test.

**BUG — the epoch-1 probe silently produced nothing.** `render_model.sh` reads its output from a
hardcoded `${PHS}_latest/images`, but `--which_epoch 1` makes test.py write `${PHS}_1/images`.
I added the `EPOCH` variable to the `--which_epoch` flag and missed the output path, so the render
processed all 1000 frames and then counted 0, failed the completeness gate, and never scored.
`cipo_of` correctly returned empty and the picker defaulted to `latest` — which is the silent
default the probe existed to prevent, arriving through a different door.

FIX (apply when render_model.sh is NOT executing — editing a running bash script corrupts it,
which already happened once tonight):
```
DD=$R/$M/${PHS}_${EPOCH:-latest}/images       # was ${PHS}_latest
rm -rf "$R/$M/${PHS}_${EPOCH:-latest}"        # same
```
then re-run the epoch-1 delivery to get its CIPO. Until that number exists, **the drift question
is still open and v54 must not replace v50d.**

### Epoch ladder results (raw renders, 300 frames, no post)

```
sunny v54 (Town10HD)   alt p99   sharp   |   night v55 (Town03)   alt p99   sharp
  epoch 1                27.47  1703.6   |     epoch 1              28.77   450.9
  epoch 2                28.94  1688.8   |     epoch 2              28.98   456.6
  epoch 3 = latest       25.72  1908.8   |     epoch 3 = latest     27.93   454.2
  (parent v50)           52.44   961.0   |     (parent v51)         32.55   497.1
```

**Sunny: epoch 3 wins on BOTH axes, so there is no drift penalty to trade against** — the worry
that a NuRec-only corpus would pull the model off Mapillary detail did not materialise within 3
epochs. It was also still improving at the end, which suggests a LONGER run could go further.
Do not start one until the CIPO probe reports: if epoch 1 scores better CIPO than epoch 3, then
more temporal training costs perception and a longer run is exactly the wrong move.

**Night: flat across every epoch** (28.77 / 28.98 / 27.93). A model that is learning improves with
epochs. This one does not, which is independent evidence for the sequence-quality diagnosis above
— it is not under-trained, it has nothing to learn from 20-frame fragments.

### Night delivered (Town03, DVP included, tag `tnig`)

```
       alt p99   sharp    CIPO    rng
v51      34.54   328.7   0.617   8.16
tnig     30.51   303.5   0.589   9.07
```

-12% flicker bought with -8% sharpness, -0.03 CIPO and +11% range error. Matches the raw
measurement (-14% / -9%). **Night stays on v51.**

### THE OPEN PROBLEM: v54 looks better but DETECTS worse

First delivered comparison, Town10HD, 300 frames (`tsun` = v54 WITH DVP):

```
   tag       alt p99    sharp   ghost    CIPO    rng
   v49         32.15    514.9       -   0.650   3.80
   v50d        34.12    727.6    5.24   0.795   4.00
   v50e        19.36    605.7    9.49   0.774   3.90
   tsun        22.97   1107.5       -   0.665   3.83
```

Flicker down 33% and sharpness up 52% against v50d — and **CIPO down from 0.795 to 0.665**. Not
a wash either: at the same false-alarm rate (0.608 vs 0.612) tsun simply recalls less, so it is
strictly worse on detection, not trading recall for precision.

Two candidate causes, and they must be separated rather than guessed:
- **(a) DVP.** `tsun` includes it, and post-processing is already known to cost detection here.
- **(b) Drift.** 3 epochs on a NuRec-only corpus can pull the model off the 32k mixed corpus that
  gave v50 its Mapillary detail; perception would be the first casualty.

`clean_delivery.sh` separates them: every delivery it makes is DVP-free (removing (a)), and it
probes epoch 1 against epoch 3 on Town10HD, reading CIPO back from each, before committing the
other four towns to the winner (measuring (b)). Tags `v54e1` / `v54elatest` are those probes;
`v54` / `v55` are the final sets.

**Do not promote v54 over v50d on the flicker numbers alone until this resolves.** v50e was the
same shape of mistake — best-looking metric of its generation, rejected once the cost was
measured.

Note on the `ghost` column: it is only meaningful WITHIN a model family, because it asks what
post-processing did to a render and needs the same generator with less post as reference.
`flicker_report.py` now prints `-` across families instead of a large, meaningless model-vs-model
difference.

### Epoch ladder (`epoch_ladder.sh`, armed, waits for TEMPORAL DONE)

`--save_epoch_freq 1` keeps every epoch. Temporal consistency and Mapillary detail move in
OPPOSITE directions on a NuRec-only corpus, so there is no reason to assume epoch 3 is the best
trade. The ladder raw-renders 300 frames per epoch (~5 min each, vs ~2.5 h for a full delivery)
and reports alt/sharpness per epoch so the choice is measured. Log
`pix2pixHD/checkpoints/epoch_ladder_log.txt`.

### How to judge it

`alt p99` against v49 / v50d, plus ghost against v50b, plus CIPO from `score_vp.py`. The bar:
**beat v50d's 27.01 flicker without exceeding its ~4.9 ghost.** v50e proves that lowering flicker
alone is easy and meaningless; doing it without ghosting is the whole point. If the temporal model
clears that bar it also lets us drop DVP (1 h 56/clip) and the de-shimmer stage entirely.
**Risk:** temporal GAN training is finicky and the video discriminator can destabilise.

## Standing rules

- Full autonomy on carla2/pix2pix. **Vision Pilot is READ-AND-RUN ONLY** — never edit it; report
  any change it needs. Audit of past edits: `vision_pilot_data/VisionPilot_Change_Audit.pdf`.
  One repair still outstanding and deliberately unrun: `homography_C_matrix.yaml.openlane.bak`
  was overwritten with CARLA values.
- Feed Vision Pilot 1024x512, never 1920x960 (lane MAE 0.27 vs 0.87 m).
- Never run two GPU jobs at once — a 2048 render peaks at 27.5 GB of 32. Use `gpu_wait.sh`;
  a bare `pgrep -f` self-matches the caller (this trap has bitten four times).
- Deliver only `*_FINAL_1920.mp4`; clean per-stage intermediates.
- Use lossless `.avi` (FFV1) for intermediates — `vidcodec.py` picks by extension. Six mp4v
  re-encodes cost 8.5 points of object detection.


## VEGETATION: v63 done, v64 running (2026-08-30)

The user's last request was distant trees -- "the trees from a distance is still blurry n shit" --
and explicitly a training fix, not another filter.

### v63 (`train_v63_veg.sh`, --veg_weight 4.0 --far_boost 3.0, 20 epochs, ~37 h) -- FINISHED
Fixed what it aimed at:
```
                      near      far   far/near   % of photo ceiling
REAL training photos  1191     1246       1.05        100%
v50 parent             673      561       0.83         51%
v63                    673      670       1.00         55%
```
far/near 0.83 -> 1.00, far detail +19.5%, and visible in `cmp63_far.png`: fronds resolve as fronds,
and v50's magenta speckle in the crowns is gone. Near detail did not move, so the absolute level
only reached 55%. It fixed the ratio, not the level.

FAILED the tail gate: road 1.81x parent, sky 1.57x, specks 1.24x (gate 1.2x). `road_sky_ceiling.py`
splits that failure in two:
- ROAD is real. Parent 29.66 was already above the photographs (25.70) and CARLA (23.81); v63 is
  53.79 -- 2.1x a real photograph. Invented grain where lane geometry is read.
- SKY is not ghosting. Under a 31x31 erode (hard enough to drop wires and fronds the label map does
  not resolve) v63 is still 1.9x the parent, but 0.713 against a photograph's 13.5. Both models
  render a sky 19x smoother than any real one.
Extra specks are on buildings (+22k) and vehicles (+10k), NOT road (-778) or vegetation (-6466).

`sweep_v63.sh` + `epoch_sweep.py` rendered every saved checkpoint. No epoch passes; road is bad
from the start, so this is not late drift:
```
render      veg near  veg far  far/near   road    sky   specks   gate
parent           672      561      0.83  1.00x  1.00x   1.00x   PASS
ep8              811      723      0.89  1.88x  1.21x   1.12x   FAIL
ep12             570      611      1.07  1.87x  2.40x   1.31x   FAIL
ep16             627      591      0.94  2.89x  2.04x   1.25x   FAIL
ep20 (latest)    673      670      1.00  1.81x  1.57x   1.24x   FAIL
```

### Why, and what v64 changes
`VGGLoss.forward` with a weight map computes `(w*d).sum()/w.sum()` -- a WEIGHTED MEAN. Emphasis on
vegetation is therefore DILUTION of everything else by 1/mean(w). With less perceptual pressure to
match the real road, the GAN term -- unweighted, and rewarded for plausible texture -- decides it.

`--veg_extra` (new, `networks.VGGLoss.forward_plus`) keeps the original objective at full strength
and ADDS a vegetation-only perceptual term, so vegetation gains gradient without the road losing
any. `train_v64_veg.sh`: --veg_extra 3.0 --far_boost 3.0, 12 epochs (v63's own ladder shows the
vegetation gain is fully present by epoch 8), started 2026-08-30 05:53, ~22 h, log `v64_log.txt`.
One re-run, not a sweep: the mechanism is identified and this tests it.

Backups: `models/networks.py.pre_vegextra`, `models/pix2pixHD_model.py.pre_vegextra`,
`options/base_options.py.pre_vegextra`, `check_pipeline.sh.pre_v64`.

BASELINES UNCHANGED: sunny v50d, night v59. Gate not loosened.


## v64 REJECTED, VEGETATION-LOSS LINE CLOSED (2026-08-31)

`--veg_extra` removed the weighted-mean dilution exactly as intended, and the result was worse than
the parent: near vegetation detail collapsed 673 -> 370, far did not move (568 vs 561), absolute
level fell to 38% of the photographs. far/near reads 1.53 only because the near end fell out.
`check_v64.png` is muddier and specklier than v63's same frame. Road stayed at 1.98x.

Two runs with opposite loss geometries both land at road 1.8-2.0x, and v63 was already there at
epoch 8. The road grain is a property of fine-tuning v50 on this corpus at lr 5e-5, not of the
vegetation weighting. Loss shaping cannot fix it, so this line is closed -- do not run a third
variant of it.

WHAT SURVIVES: v63 genuinely fixed the distance falloff (far/near 0.83 -> 1.00, far detail +19.5%,
visible in `cmp63_far.png`) at the cost of road grain 2.1x a real photograph. Gate blocks automatic
promotion and it stays blocked; a Town10HD delivery clip vs v50d is being built so the user can
judge that trade by eye, which is how every call in this project has been made.

BASELINES UNCHANGED: sunny v50d, night v59.


## FOR THE USER, TUESDAY 2026-09-01

Watch `COMPARE_town10hd_sunny_v50j_vs_v63.mp4` (same post chain, only the model differs -- the
honest one). `COMPARE_town10hd_sunny_v50d_vs_v63.mp4` also exists but flatters v63, because v50d
predates the colour fix and the building injection.

THE CALL: v63 renders distant trees, facades and vehicle panels visibly better, halves Vision
Pilot's false alarms (0.61 -> 0.31) and improves range MAE 19% (4.00 -> 3.26 m). It costs ~5 points
of CIPO recall (0.795 -> 0.744), slightly worse lane MAE and jitter, and road grain at 2.1x a real
photograph -- which is why the automatic gate blocks it and why nothing was promoted.

Delivered: `CARLA/town10hd_sunny_vp55_v63_FINAL_1920_visionpilot.mp4`, score in `CARLA/logs_v63/`.
Also still awaiting a look: v50j (5 sunny towns, colour + building fixes on the v50 model) and v51d
(5 night towns).

CLOSED: vegetation loss shaping (notes 35, 36). Two opposite loss geometries both put road at
1.8-2.0x; it is fine-tuning v50 on this corpus that does it, not the vegetation weighting.


## FINISHED 2026-09-01 — nothing is running

Both chained jobs completed. `render_model.sh` delivered v63 for Town03/04/05/06 (18:05);
`make_v50kl.sh` finished 20:49 with all 5 towns x 2 variants fused, calibrated and scored.
Delivered and organised:
`calibrated/v63/sunny/`, `calibrated/v50k/sunny/`, `calibrated/v50l/sunny/` — 5 clips each.
Per-town Vision Pilot scores in `logs_v63/`, `logs_v50k/`, `logs_v50l/`. **Results in note 39.**

What they were (kept for the record), logged under `pix2pixHD/checkpoints/`:

1. **`render_model.sh sunny carla2real_semantic_v63_veg v63 Town03 Town04 Town05 Town06`**
   log `render_v63_log.txt`. ~2 h/town, DVP dominates.

2. **`make_v50kl.sh`** log `v50kl_log.txt`. Waited for job 1 to exit, then for all 5 towns x 2
   variants: fuse -> encode FINAL_1920 -> downscale to 1024x512 -> run Vision Pilot
   `record_carla.sh` -> `score_vp.py` -> `organise_calibrated.sh` into
   `calibrated/v50k/sunny/` and `calibrated/v50l/sunny/`.
   It waits on `pgrep -x VisionPilot` before each run so two VP jobs never overlap.
   Lossless fusion intermediates are ~1.3 GB each and are deleted immediately after encoding.

### The fusion, in one line
`fuse_colour.py <carrier> <colour_source> <out.avi> [strength] [window]` transfers Lab colour
statistics, SMOOTHED over a temporal window, so vibrancy crosses over without flicker.
**v50k** = v63 colour onto v50d. **v50l** = v63 colour onto v50j.

**v50l is the candidate**, now confirmed over all five towns: it keeps v63's colour while giving
back only 0.8 points of CIPO recall, where v63 alone costs 5.1 (0.887 -> 0.836). Best lane MAE of
the fused pair. **v50k is a dead end, confirmed by eye in two towns**: a global grade cannot fix a
per-object hue error, it amplifies it -- v50d's magenta bus and violet car both come out MORE
saturated. See notes 38 and 39.

### Still not fixed
Sharpness. v50l is 677 against v50d's 803. Colour and stability were separable; blur is a third axis.

BASELINES: **sunny v50m** (promoted 2026-09-02 by the user, replacing v50l the same day), night v59.
AWAITING THE USER'S EYE: v50l (5 sunny towns, the fusion candidate), v63 (5 sunny towns),
v50j (5 sunny), v51d (5 night).

## RELEASE REPO (separate from this tree)
`$CARLA2REAL_ROOT` -> https://github.com/Terbz98/CARLAtoreal (private).
75+ files, ~600 KB, code only: no data, no weights, no video, no Vision Pilot source.
Paths configurable via `config.sh` / `config.py`. pix2pixHD BSD licence restored verbatim.
Licensed **Apache-2.0** (`LICENSE`, `NOTICE`) as of 2026-09-02, scoped to this project's code only --
not the corpus, not weights derived from it, not the vendored pix2pixHD (BSD).
Also now ships the entry points it was missing: `record_town_auto.py`, `prepare_gt_test_label.py`,
`train_v50.sh`, `train_v51_night.sh`, `requirements.txt`.
`THIRD_PARTY_NOTICES.md` records the open item: the 21 training videos called "NuRec" are NOT
NVIDIA's NuRec, carry several different third-party creator watermarks, and have no provenance --
treat as not redistributable. Mapillary + Cityscapes (72% of corpus) are properly licensed and
linked in `datasets/README.md`. The user pushes manually from a real terminal.

## v50m — in flight (2026-09-02)

Three defects reported against v50l, each fixed at the stage that causes it. `make_v50m.sh`,
log `pix2pixHD/checkpoints/v50m_log.txt`. Pilot on Town05 first, then the other four.

1. **Car ghost trails (Town05).** Measured v50j 13.35 / v50l 13.42 on moving pixels — the fusion
   adds nothing, the trails are inherited from the carrier. `class_deshimmer.py`'s own table sets
   cars to 0.4 strength / 4.0 flow tolerance and warns they trail if loosened; v50j overrode both
   to 0.85 / 6.0 to fix a car-flicker regression. v50m takes 0.55 / 4.5. Expect a small car-flicker
   rise — flicker trades ~1:1 against ghosting and a trail is the more visible artefact.
2. **Colour switching between shades.** The grade used whole-frame Lab mean/std, so one large
   coloured object re-grades everything. `fuse_colour.py` gains `ROBUST=1` (median/MAD) and
   `SLEW` (per-frame travel cap); window 61 → 91. A window alone only turns a jump into a ramp.
3. **Not sharp enough.** Unsharp 0.55 → 0.75, CARLA facade injection 1.0 → 1.2. Both guarded, and
   road grain must still be checked against the real-photograph ceiling (`road_sky_ceiling.py`).

## The flicker metric misled a decision (2026-09-02)

v50m was reported as 14% flickerier than v50l on `alt p99` (141.8 vs 124.0) and that number was
acted on. It was wrong. Three separate knobs were swept to find the cause and NONE moved it:
unsharp 0.75→0.60 (v50n, 139.1) and the car de-shimmer restored to v50j's 0.85/4.0 (v50p, 141.3).

When three unrelated changes cannot shift a number, the number is not measuring the claimed thing.
`true_instability.py` — which exists for exactly this — separates real shimmer from detail moving
past the camera:

```
              warped resid (real)   plain alt (misleading)   detail
v50l                    10.45                     13.27       806
v50m                    10.58                     13.89       924
v50p                    10.55                     13.80       928
```

**Real instability is +1.2%, not +14%.** The rest is v50m's extra detail being counted as
alternation. Rule: never report `alt p99` across versions with different sharpness — use
`true_instability.py`, or the comparison is meaningless.
