"""Panoptic instance IDs, for turning object boundaries into an edge-channel signal.

Why this exists: the reported defect is "some parts painted red some isn't" on cars, and the
same blotching on buildings. That is the classic symptom of a generator that cannot tell one
object from the next -- three overlapping cars are a single connected region labelled "car", so
the model paints regions rather than objects. pix2pixHD has an instance input for exactly this,
and this project has always run --no_instance.

Rather than switch on pix2pixHD's instance path (which needs an input-dimension change and a
matching map for every corpus image), the boundaries are OR'd into the EXISTING edge channel.
No architecture change, and the edge channel is already the most reliable lever here -- v17
buildings, v19 walls, v44 vehicles all landed through it.

Semantic segmentation cannot produce this: two touching cars share a label. Panoptic can, so
this runs the panoptic head of the same Mask2Former checkpoint family.

Output: 16-bit PNG of instance ids (0 = stuff/no instance). Only THING classes get ids.

Usage: gen_instance_maps.py --src_dir <images> --out_dir <inst> [--batch 2]
"""
import argparse, numpy as np, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
TARGET_W, TARGET_H = 1024, 512

ap = argparse.ArgumentParser()
ap.add_argument('--src_dir', required=True)
ap.add_argument('--out_dir', required=True)
ap.add_argument('--batch', type=int, default=2)
a = ap.parse_args()

SRC, OUT = Path(a.src_dir), Path(a.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
paths = sorted(SRC.glob('*.png')) + sorted(SRC.glob('*.jpg'))
todo = [p for p in paths if not (OUT / p.stem).with_suffix('.png').exists()]
print('%d images, %d to process' % (len(paths), len(todo)), flush=True)

if todo:
    proc = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME, ignore_index=255, do_resize=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME).eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev)

    @torch.no_grad()
    def run(images):
        inp = proc(images=images, return_tensors='pt')
        inp = {k: v.to(dev) for k, v in inp.items()}
        out = model(**inp)
        return proc.post_process_panoptic_segmentation(
            out, target_sizes=[(im.height, im.width) for im in images])

    for i in tqdm(range(0, len(todo), a.batch)):
        batch = todo[i:i + a.batch]
        images = [Image.open(p).convert('RGB') for p in batch]
        for p, res in zip(batch, run(images)):
            seg = res['segmentation'].cpu().numpy()
            # remap to compact ids so the 16-bit PNG never overflows
            ids = np.zeros(seg.shape, dtype=np.uint16)
            for k, info in enumerate(res['segments_info'], start=1):
                ids[seg == info['id']] = k
            im = Image.fromarray(ids)
            if im.size != (TARGET_W, TARGET_H):
                im = im.resize((TARGET_W, TARGET_H), Image.NEAREST)
            im.save(str((OUT / p.stem).with_suffix('.png')))

print('Done: %d instance maps in %s' % (len(list(OUT.glob('*.png'))), OUT), flush=True)
