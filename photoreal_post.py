"""Camera-realism post-processing: make clean pix2pix output look like real camera footage.
Adds the physical imperfections real cameras have (and synthetic renders lack): tone/color
grade, bloom, chromatic aberration, vignette, and sensor grain. Subtle by default.

Usage: python photoreal_post.py <in> <out> [strength=1.0]   (in/out = image or mp4)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from vidcodec import fourcc_for   # .avi -> lossless FFV1, anything else -> mp4v
import sys, cv2, numpy as np, os

# Night needs different numbers to the daylight default. The S-curve pivots on mid-grey, so on
# a frame whose pixels sit around 0.1 it does not add contrast, it crushes: 0.10 -> 0.068, a
# third of the remaining shadow detail gone. Combined with the vignette that is most of why the
# night clip read as "patches of black" rather than a dark scene. These knobs default to the
# original daylight values, so every existing pipeline is unchanged.
GRADE_CONTRAST = float(os.environ.get('GRADE_CONTRAST', '1.08'))
SHADOW_LIFT    = float(os.environ.get('SHADOW_LIFT',    '0.0'))   # try 0.06 for night
VIGNETTE_SCALE = float(os.environ.get('VIGNETTE_SCALE', '1.0'))


def _grade(img):
    f = img.astype(np.float32) / 255
    # gentle S-curve contrast + tiny warm lift
    f = np.clip((f - 0.5) * GRADE_CONTRAST + 0.5, 0, 1)
    if SHADOW_LIFT > 0:
        # raise the black point without touching highlights: strongest at f=0, zero by f~0.5
        f = np.clip(f + SHADOW_LIFT * (1.0 - np.clip(f / 0.5, 0, 1)) ** 2, 0, 1)
    f[..., 2] *= 1.03   # a touch warmer (BGR: R up)
    f[..., 0] *= 0.99
    return np.clip(f, 0, 1)

def _bloom(f, s):
    bright = np.clip(f - 0.72, 0, 1)
    blur = cv2.GaussianBlur(bright, (0, 0), 9)
    return np.clip(f + blur * 0.6 * s, 0, 1)

def _chroma(f, s):
    h, w = f.shape[:2]
    sh = max(1, int(1.2 * s))
    b, g, r = cv2.split((f * 255).astype(np.uint8))
    M = np.float32([[1, 0, sh], [0, 1, 0]]); r = cv2.warpAffine(r, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    M = np.float32([[1, 0, -sh], [0, 1, 0]]); b = cv2.warpAffine(b, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([b, g, r]).astype(np.float32) / 255

def _vignette(f, s):
    h, w = f.shape[:2]
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2)
    v = np.clip(1 - (d ** 2) * 0.28 * s, 0.5, 1)
    return f * v[..., None]

# Each effect is separately scalable, because they do NOT cost the same thing.
# Measured on v41 Town10HD (sharpness = Laplacian variance at 1024 wide):
#     v33_sharp in            1209
#     after full photoreal     565   (-53%)
# _grade is the one that buys the saturation (37.6 -> 53.6) and costs no sharpness.
# _bloom (sigma-9 Gaussian added back at 0.6) and _chroma (channel shift) are what soften
# the image. Setting BLOOM_SCALE=0 CHROMA_SCALE=0 keeps the colour grade and the vignette
# while retaining the detail the local enhancer was built to produce.
GRAIN_SCALE  = float(os.environ.get('GRAIN_SCALE',  '1.0'))
BLOOM_SCALE  = float(os.environ.get('BLOOM_SCALE',  '1.0'))
CHROMA_SCALE = float(os.environ.get('CHROMA_SCALE', '1.0'))

def _grain(f, s):
    if GRAIN_SCALE <= 0:
        return f
    n = np.random.randn(*f.shape[:2]).astype(np.float32) * 0.022 * s * GRAIN_SCALE
    return np.clip(f + n[..., None], 0, 1)

def process(bgr, s=1.0):
    f = _grade(bgr)
    if BLOOM_SCALE > 0:
        f = _bloom(f, s * BLOOM_SCALE)
    if CHROMA_SCALE > 0:
        f = _chroma(f, s * CHROMA_SCALE)
    if VIGNETTE_SCALE > 0:
        f = _vignette(f, s * VIGNETTE_SCALE)
    f = _grain(f, s)
    return (f * 255).astype(np.uint8)

def main():
    inp, out = sys.argv[1], sys.argv[2]
    s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    if inp.lower().endswith(('.png', '.jpg', '.jpeg')):
        cv2.imwrite(out, process(cv2.imread(inp), s)); print('wrote', out); return
    c = cv2.VideoCapture(inp); W = int(c.get(3)); H = int(c.get(4)); fps = c.get(5) or 30
    vw = cv2.VideoWriter(out, fourcc_for(out), fps, (W, H)); k = 0
    while True:
        ok, fr = c.read()
        if not ok: break
        vw.write(process(fr, s)); k += 1
    vw.release(); print('wrote %s (%d frames)' % (out, k))

if __name__ == '__main__':
    main()
