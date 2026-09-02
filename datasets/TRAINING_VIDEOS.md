# The 21 training videos ("NuRec") — identification manifest

**These files are not distributed with this repository, and this project cannot grant you a licence
to them.** This page exists so that anyone holding a copy can verify it is the same material, and so
that the provenance question is documented rather than left implicit.

## Why they are not here

They are internally called "NuRec", but they are **not** NVIDIA's NuRec — that is a set of
3D-reconstructed USDZ scenes, a different thing entirely. What these actually are:

- **different third-party creator watermarks on different videos** (visible on 00, 03, 07, 12, 17, 20
  among others), so they do not share a single source
- **audio tracks present**, which a rendered or reconstructed dataset would not carry
- **re-encoded** (Lavf58.76.100 / x264 crf=15), so these are not originals
- **no manifest, no licence file, no attribution** arrived with them

The most likely explanation is collected web footage. On that basis this project treats them as
**not redistributable**, and ships neither the videos nor model weights trained on them.

If you know the original source of any of these, that resolves a real open question — please open an
issue.

## What this means for the weights

These 21 videos are **9,069 of the 32,475 training pairs (28%)**. Any checkpoint trained on this
corpus inherits their unresolved status. That is why this repository ships code and training recipes
but no `.pth` files.

The other 72% is properly licensed and downloadable — see `README.md` in this directory:

| Source | Pairs | Share | Licence |
|---|---|---|---|
| Mapillary Vistas | 19,293 | 59% | research / non-commercial, free registration |
| Cityscapes | 4,113 | 13% | research / non-commercial, free registration |
| these videos | 9,069 | 28% | **unresolved** |

Note that Mapillary and Cityscapes are themselves **research/non-commercial**. Commercial use of a
model trained on this corpus is not cleared by resolving the video question alone.

## Manifest

All 21 are 2560x1440, 30 fps, H.264 in MP4. SHA-256 is truncated to 16 hex characters, which is
ample to confirm a match.

| file | frames | size (MB) | sha256[:16] | weather | used for |
|---|---|---|---|---|---|
| 00.mp4 | 966 | 83.0 | d6e9d5b3f2dfa0b4 | sunny | v6 |
| 01.mp4 | 899 | 71.3 | 5eddaec0749f6693 | sunny | v6 |
| 02.mp4 | 902 | 63.6 | ad701155808953e6 | sunny | v6 |
| 03.mp4 | 901 | 63.2 | 6cb5e8a824dd4c70 | sunny | v6 |
| 04.mp4 | 901 | 78.6 | 3ad5c0b6e4ca8609 | sunny | v6 |
| 05.mp4 | 900 | 42.3 | c750390b63209fe7 | sunny | v6 |
| 06.mp4 | 900 | 42.8 | 311bb4c4cd3ac19f | sunny | v6 |
| 07.mp4 | 900 | 34.1 | ce0f183ab14927e8 | sunny | v6 |
| 08.mp4 | 900 | 20.3 | 9a35d2abc2597310 | sunny | v6 |
| 09.mp4 | 900 | 19.5 | 4a81151981b94235 | sunny | v6 |
| 10.mp4 | 900 | 10.8 | 8a92771f858fd8fc | night | night model |
| 11.mp4 | 900 | 15.7 | 561097aaeaaed1cd | night | night model |
| 12.mp4 | 900 | 9.7 | b175acc69caefb22 | night | night model |
| 13.mp4 | 900 | 5.5 | a7075e8cadf037cd | night | night model |
| 14.mp4 | 900 | 11.3 | f74fe93f29ebbb01 | rain | rain model |
| 15.mp4 | 900 | 15.8 | 2de60ea9ebf64225 | rain | rain model |
| 16.mp4 | 900 | 13.4 | afa8754c7baca08d | rain | rain model |
| 17.mp4 | 600 | 47.5 | 5b6ef7772dcbafdc | snow | snow (untrained) |
| 18.mp4 | 750 | 44.5 | c082953aa7446c1b | snow | snow (untrained) |
| 19.mp4 | 900 | 11.2 | 9fcbf408d7c5dfd6 | snow | snow (untrained) |
| 20.mp4 | 900 | 5.0 | b08f50ec37c37d56 | sunny | v6 |

Total 11 GB, 18,819 frames. Sampled to 9,069 training pairs.

## Training without them

Every recipe in this repository runs on a corpus you assemble yourself. Mapillary Vistas alone is
19,293 pairs — larger than the video contribution — and `train_v50.sh` documents the schedule. The
videos add real-world night and rain, which Mapillary covers less densely; expect weaker night and
rain models without an equivalent substitute, and no difference in the sunny path that this
project's baselines actually use.
