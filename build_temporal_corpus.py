#!/usr/bin/env python3
"""Build a sequence corpus for temporal (vid2vid-style) training, with optical flow.

WHY. Every model v44-v53 was trained on single frames. Nothing in pix2pixHD's loss requires
frame t to agree with frame t-1, so two near-identical label maps produce different textures --
that difference IS the flicker. DVP and temporal_deshimmer.py smooth the symptom afterwards;
--temporal / --video_disc attack the cause.

WHAT TemporalDataset NEEDS
  {phase}_{label,img,edge,flow}/ plus one dir per enabled conditioning flag
  ({phase}_{depth,normal,chroma,light}/), and filenames prefixed <VID>_<FRAME>. It derives the
  video id as basename.split('_')[0], so each source clip MUST get its own prefix -- otherwise
  frames from different clips are treated as consecutive and the flow between them is garbage.
  Prefixes therefore must not themselves contain an underscore.

RECOVERING CLIP BOUNDARIES. The training corpora renumbered everything flat (nrhr_%05d,
ngt_%05d), which throws the boundaries away. But both were written by enumerating a
deterministic sorted list, and the index IS the position in that list:

  build_v40_corpus.do_nurec():      frames = concat over SUNNY of sorted(test_mp4/<v>_work/frames/*.jpg)
                                    then write_pair('nrhr', i, ...)
  build_night_corpus.py:            same, over NIGHT_VIDS

so repeating that enumeration recovers (video, frame) for every index exactly. Note this is the
frames/ directory, NOT nurec_raw -- an earlier version of this script assumed nurec_raw and
would have mismapped every boundary, because only 9069 of 23053 raws have an nrhr_ pair.

Dark Zurich is different: dz_filter.py kept a FILTERED subset in sorted order, so index i does
not map to raw i. Rather than trying to reproduce the filter's thresholds, we align the two
sorted lists by image content (monotonic walk, thumbnail correlation) and refuse to use the
result unless nearly all frames match confidently.

We relink the EXISTING labels/edges/depth/normal/chroma/light rather than recomputing them, so
no Mask2Former / depth / normal pass is needed.

Usage:
  build_temporal_corpus.py sunny            # NuRec sunny clips, from training_v49_chroma
  build_temporal_corpus.py night            # NuRec night + Dark Zurich, from training_v51_night
  build_temporal_corpus.py <w> --flow-only  # just (re)compute the flow field
"""
from config import DATA, ROOT
import os, sys, glob, re
import cv2
import numpy as np

ROOT = DATA
WEATHER = sys.argv[1] if len(sys.argv) > 1 else 'sunny'
FLOW_ONLY = '--flow-only' in sys.argv
DST = f'{ROOT}/training_temporal_{WEATHER}'

# Source corpora chosen because each already holds EVERY channel its model needs, under one set
# of stems -- so nothing can fall out of alignment between channels.
#   sunny -> v50 was trained with --depth --normal --chroma
#   night -> v51 was trained with --depth --normal --light
SUNNY_VIDS = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '20']
NIGHT_VIDS = ['10', '11', '12', '13']
_NIGHT = dict(src=f'{ROOT}/training_v51_night',
              chans=('img', 'label', 'edge', 'depth', 'normal', 'light'))
CFG = {
    'sunny': dict(src=f'{ROOT}/training_v49_chroma',
                  chans=('img', 'label', 'edge', 'depth', 'normal', 'chroma')),
    'night': _NIGHT,
    # 'nightseq' = the NuRec night videos only, no Dark Zurich.
    # v55 (trained on both) barely moved: alt 32.55 -> 27.92 and sharpness DOWN 9%, against
    # sunny's -51% / +99%. The difference is sequence QUALITY, not data volume. Sunny got 10
    # contiguous 900-frame clips; night got those 4 plus 129 Dark Zurich fragments averaging ~20
    # frames, because dz_filter kept only frames dark enough to be night. A temporal loss and a
    # video discriminator learn from long consecutive runs -- 20-frame shards teach very little
    # and dominate the clip count. Appearance diversity is NOT what this stage needs: the parent
    # v51 already learned Dark Zurich's look. This stage only needs good motion.
    'nightseq': _NIGHT,
}[WEATHER]
SRC, CHANS = CFG['src'], CFG['chans']
SUBS = tuple(f'train_{c}' for c in CHANS) + ('train_flow',)

for s in SUBS:
    os.makedirs(f'{DST}/{s}', exist_ok=True)


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(src), dst)


def link_frame(stem, newstem):
    """Link every channel for one frame. All-or-nothing: a frame missing any channel is skipped
    entirely, because a half-linked frame would make the loader raise mid-epoch."""
    srcs = {}
    for c in CHANS:
        ext = 'jpg' if c == 'img' else 'png'
        p = f'{SRC}/train_{c}/{stem}.{ext}'
        if not os.path.exists(p):
            return False
        srcs[c] = (p, ext)
    for c, (p, ext) in srcs.items():
        link(p, f'{DST}/train_{c}/{newstem}.{ext}')
    return True



MAXGAP = 2      # frames this far apart still count as consecutive


def emit(items):
    """items: list of (segment, frame, stem). Assign final prefixes, splitting a segment wherever
    the frame numbers jump.

    This matters because the source corpora are FILTERED subsets, not whole videos. dz_filter.py
    kept only frames dark enough to be night, so drive 0375 contributes 64 frames spanning 1061 --
    two "adjacent" entries can be 50 frames apart. TemporalDataset decides has_prev purely from the
    filename prefix, so without this split it would pair those as t-1,t and the flow between them
    would be a large-displacement Farneback estimate, i.e. noise. The occlusion mask suppresses
    most of the resulting loss, but the pair is still wasted and the video discriminator would be
    shown a "transition" that no real video contains.

    Prefixes must contain no underscore -- the loader takes basename.split('_')[0] as the video id.
    """
    by = {}
    for seg, fr, stem in items:
        by.setdefault(seg, []).append((fr, stem))
    kept, clips = 0, 0
    for seg in sorted(by):
        runs, cur, prev = [], [], None
        for fr, stem in sorted(by[seg]):
            if prev is not None and fr - prev > MAXGAP:
                runs.append(cur); cur = []
            cur.append((fr, stem)); prev = fr
        runs.append(cur)
        for ri, run in enumerate(runs):
            if len(run) < 2:            # a lone frame yields no temporal pair
                continue
            vid = f'{seg}s{ri:02d}'
            n = 0
            for fi, (fr, stem) in enumerate(run):
                if link_frame(stem, f'{vid}_{fi:06d}'):
                    n += 1
            if n:
                kept += n; clips += 1
    print(f'    -> {kept} frames in {clips} gap-free clips')
    return kept


def carry_flat(vids, prefix_fmt, stem_fmt):
    """Recover (video, frame) for a flat-numbered corpus by repeating the build's enumeration."""
    items, idx = [], 0
    for v in vids:
        frames = sorted(glob.glob(f'{ROOT}/test_mp4/{v}_work/frames/*.jpg'))
        for fi, _ in enumerate(frames):
            stem = stem_fmt % idx
            if os.path.exists(f'{SRC}/train_img/{stem}.jpg'):
                items.append((prefix_fmt % v, fi, stem))
            idx += 1
    return emit(items)


def _thumbs(paths, note):
    """Tiny normalized grayscale signatures for content matching."""
    out = []
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            out.append(None); continue
        t = cv2.resize(im, (32, 16)).astype(np.float32)
        t -= t.mean()
        s = t.std()
        out.append(t / s if s > 1e-6 else None)
        if (i + 1) % 500 == 0:
            print(f'    {note} {i+1}/{len(paths)}', flush=True)
    return out


def carry_dark_zurich():
    """Align the filtered dz_ subset back onto the raw drives by content, then link.

    dz_filter.py kept frames in sorted order but dropped some, so index i is NOT raw i. Matching
    on content is immune to whatever the filter's thresholds were; matching on count is not.
    """
    raws = sorted(glob.glob(f'{ROOT}/dark_zurich/raw/*.png'))
    dz = sorted(glob.glob(f'{SRC}/train_img/dz_*.jpg'))
    if not raws or not dz:
        print('  dark zurich: nothing to do'); return 0
    print(f'  dark zurich: aligning {len(dz)} kept frames against {len(raws)} raws by content')
    tr, td = _thumbs(raws, 'raw'), _thumbs(dz, 'dz')

    pairs, r = {}, 0
    for j, a in enumerate(td):
        if a is None:
            continue
        best, bi = -1.0, -1
        # monotonic walk: the kept subset preserves sorted order, so the match for dz[j] is at or
        # after the match for dz[j-1]. The window bounds the cost and blocks wild mismatches.
        for k in range(r, min(r + 400, len(tr))):
            b = tr[k]
            if b is None:
                continue
            c = float((a * b).mean())
            if c > best:
                best, bi = c, k
        if best > 0.90:
            pairs[j] = bi
            r = bi + 1
    frac = len(pairs) / len(dz)
    print(f'  dark zurich: matched {len(pairs)}/{len(dz)} ({frac:.1%}) at corr>0.90')
    if frac < 0.95:
        print('  dark zurich: SKIPPED -- alignment too weak to trust clip boundaries')
        return 0

    # Segment, not drive. A GoPro splits a long recording into GOPR0351.MP4, GP010351.MP4,
    # GP020351.MP4 ... and each restarts its frame numbering at 1. Treating them as one clip would
    # collide those numberings; treating each as its own clip is also what they physically are.
    # The earlier GOPR-only regex silently dropped all 1368 GP frames.
    items, unparsed = [], 0
    for j, k in pairs.items():
        b = os.path.basename(raws[k])
        m = re.match(r'((?:GOPR|GP)\d+)_frame_(\d+)', b)
        if not m:
            unparsed += 1
            continue
        seg = 'dz' + m.group(1).replace('GOPR', 'r').replace('GP', 'p')   # no underscore allowed
        items.append((seg, int(m.group(2)), os.path.splitext(os.path.basename(dz[j]))[0]))
    if unparsed:
        print(f'  dark zurich: {unparsed} filenames did not parse')
    return emit(items)


def compute_flow():
    """Farneback flow t -> t-1 (BACKWARD), saved as .npy -- what TemporalDataset np.load()s.

    DIRECTION IS CRITICAL AND WAS WRONG UNTIL 2026-08-25. pix2pixHD_model.warp() backward-warps:
    for each pixel of the CURRENT frame it looks up where that pixel came from in prev, so it
    needs the CUR->PREV displacement. Its own comment says so:
        # backward-warp img by flow (cur->prev displacement)
    This function originally computed calcOpticalFlowFarneback(prev_g, g), i.e. PREV->CUR, the
    opposite. Feeding that to a backward warp displaces prev the wrong way, misaligning it by
    roughly twice the true motion. Measured on this corpus (mean abs error against the real
    current frame):
        warp(prev, +flow)  4.266     <- what v54/v55/v56 trained against
        warp(prev, -flow)  1.872
        no warp at all     3.423
    The training target was WORSE ALIGNED THAN NOT WARPING AT ALL, so the temporal loss
    |fake - warped_prev| taught the generator to paint a displaced ghost of the previous frame
    over the current one. That is the crosshatch weave on road surfaces and the ghost outline the
    user reported at night, and it is why removing the video discriminator made it worse -- that
    discriminator was the only term resisting the corrupted target.

    Argument order below is (current, previous), which yields the cur->prev field warp() wants.

    Stored at quarter resolution in float16. TemporalDataset resizes the field to the transformed
    image size and rescales the vectors itself, so full resolution buys nothing; quarter res in
    fp16 is ~130 KB a frame instead of 1 MB, which matters at ~9k frames.

    The first frame of each clip is skipped entirely: has_prev is False there, so the loader never
    reads it and writing zeros would just waste space.
    """
    imgs = sorted(glob.glob(f'{DST}/train_img/*.jpg'))
    print(f'  computing flow for {len(imgs)} frames')
    prev_vid, prev_g = None, None
    written = skipped = 0
    for k, ip in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(ip))[0]
        vid = stem.split('_')[0]
        out = f'{DST}/train_flow/{stem}.npy'
        im = cv2.imread(ip)
        if im is None:
            prev_vid, prev_g = None, None
            continue
        h, w = im.shape[:2]
        g = cv2.cvtColor(cv2.resize(im, (w // 4, h // 4)), cv2.COLOR_BGR2GRAY)
        first_of_clip = (prev_g is None or vid != prev_vid)
        if first_of_clip:
            skipped += 1
        elif not os.path.exists(out):
            if prev_g.shape != g.shape:      # resolution change inside a clip: treat as a break
                skipped += 1
            else:
                # (current, previous) -> the cur->prev field warp() wants. DO NOT SWAP.
                fl = cv2.calcOpticalFlowFarneback(g, prev_g, None, 0.5, 3, 21, 3, 5, 1.2, 0)
                np.save(out, fl.astype(np.float16))
                written += 1
        prev_vid, prev_g = vid, g
        if (k + 1) % 2000 == 0:
            print(f'    {k + 1}/{len(imgs)}', flush=True)
    have = len(glob.glob(f'{DST}/train_flow/*.npy'))
    print(f'  flow: {written} written this run, {have} total, {skipped} clip-first frames skipped')


def verify_flow_direction():
    """Assert the stored flow actually IMPROVES alignment when used the way training uses it.

    This gate exists because the direction was wrong for v54/v55/v56 and nothing caught it: the
    corpus built, the preflight passed, training converged, and the loss went down -- it was
    simply descending toward a corrupted target. The only symptom was in the pictures.

    So: warp prev by the stored flow exactly as pix2pixHD_model.warp() does, and compare against
    the real current frame. If warping is not clearly better than not warping, the field is
    unusable and building on it would waste hours and produce ghosting.
    """
    import torch
    import torch.nn.functional as Fnn

    def warp(img, flow):
        B, C, H, W = img.size()
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        base = torch.stack((xx, yy), 0).float().unsqueeze(0)
        grid = base + flow
        gx = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        gy = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        return Fnn.grid_sample(img, torch.stack((gx, gy), dim=3), align_corners=True,
                               padding_mode='border')

    imgs = sorted(glob.glob(f'{DST}/train_img/*.jpg'))
    warped_err, plain_err, n = 0.0, 0.0, 0
    for k in range(min(len(imgs) - 1, 300), min(len(imgs) - 1, 360), 6):
        stem = os.path.splitext(os.path.basename(imgs[k]))[0]
        vid, fr = stem.rsplit('_', 1)
        pp = f'{DST}/train_img/{vid}_{int(fr)-1:06d}.jpg'
        fp = f'{DST}/train_flow/{stem}.npy'
        if not (os.path.exists(pp) and os.path.exists(fp)):
            continue
        cur, prev = cv2.imread(imgs[k]), cv2.imread(pp)
        if cur is None or prev is None:
            continue
        H, W = 256, 512
        cur, prev = cv2.resize(cur, (W, H)), cv2.resize(prev, (W, H))
        fl = np.load(fp).astype(np.float32)
        h0, w0 = fl.shape[:2]
        fl = cv2.resize(fl, (W, H)); fl[..., 0] *= W / w0; fl[..., 1] *= H / h0
        t = lambda im: torch.from_numpy(im.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)
        ct, pt = t(cur), t(prev)
        ft = torch.from_numpy(fl.transpose(2, 0, 1)).unsqueeze(0)
        warped_err += float((warp(pt, ft) - ct).abs().mean())
        plain_err += float((pt - ct).abs().mean())
        n += 1
    if n == 0:
        print('  flow check: no sample pairs found, SKIPPED')
        return
    w, p_ = warped_err / n, plain_err / n
    print(f'  flow check: warped err {w:.3f} vs unwarped {p_:.3f} over {n} pairs')
    if w >= p_:
        raise SystemExit(
            f'  FLOW DIRECTION IS WRONG: warping ({w:.3f}) is no better than not warping '
            f'({p_:.3f}).\n  warp() wants a CUR->PREV field; compute_flow must call '
            f'calcOpticalFlowFarneback(current, previous).')
    print(f'  flow check OK -- warping reduces misalignment by {100*(1-w/p_):.0f}%')


if __name__ == '__main__':
    print(f'=== temporal corpus: {WEATHER} -> {DST}')
    print(f'    source {SRC}, channels {" ".join(CHANS)}')
    if not FLOW_ONLY:
        if WEATHER == 'sunny':
            n = carry_flat(SUNNY_VIDS, 's%s', 'nrhr_%05d')
        else:
            n = carry_flat(NIGHT_VIDS, 'n%s', 'ngt_%05d')
            if WEATHER != 'nightseq':
                n += carry_dark_zurich()
        print(f'  linked {n} frames')
        if n == 0:
            raise SystemExit('no frames linked -- refusing to build an empty corpus')
    compute_flow()
    verify_flow_direction()
    for s in SUBS:
        print(f'  {s:<14} {len(os.listdir(f"{DST}/{s}"))}')
