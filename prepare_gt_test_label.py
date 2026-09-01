"""
Convert CARLA GT semantic images (recorded_Town03/semantic/) to
grayscale Cityscapes-19 trainId label maps for pix2pixHD inference.

Output: datasets/training_semantic_v6/test_Town03_gt_label/
Usage:
  conda run -n carla_env python3 prepare_gt_test_label.py
"""
from config import DATA
import os
import numpy as np
from PIL import Image
from pathlib import Path

SEM_DIR = Path(DATA) / "recorded_Town03" / "semantic"
OUT_DIR = Path(DATA) / "training_semantic_v6" / "test_Town03_gt_label"

CARLA_BGR_FILE_TO_ID = {
    (0,   0,   0):   0,
    (70,  70,  70):  1,
    (40,  40, 100):  2,
    (80,  90,  55):  3,
    (60,  20, 220):  4,
    (153,153, 153):  5,
    (50, 234, 157):  6,
    (128, 64, 128):  7,
    (232, 35, 244):  8,
    (35, 142, 107):  9,
    (142,  0,   0): 10,
    (156,102, 102): 11,
    (0,  220, 220): 12,
    (180,130,  70): 13,
    (81,   0,  81): 14,
    (100,100, 150): 15,
    (140,150, 230): 16,
    (180,165, 180): 17,
    (30, 170, 250): 18,
    (160,190, 110): 19,
    (50, 120, 170): 20,
    (150, 60,  45): 21,
    (100,170, 145): 22,
    (230,   0,   0): 10,
}

CARLA_TO_CITY19 = {
    0:  0,
    1:  2,
    2:  4,
    3:  2,
    4:  11,
    5:  5,
    6:  0,
    7:  0,
    8:  1,
    9:  8,
    10: 13,
    11: 3,
    12: 7,
    13: 10,
    14: 0,
    15: 2,
    16: 0,
    17: 3,
    18: 6,
    19: 2,
    20: 13,
    21: 0,
    22: 9,
}

def build_lut():
    lut = {}
    for color, cid in CARLA_BGR_FILE_TO_ID.items():
        lut[color] = CARLA_TO_CITY19.get(cid, 0)
    return lut

def convert(sem_path, lut):
    arr = np.array(Image.open(sem_path).convert("RGB"))
    pixels = arr.reshape(-1, 3)
    result = np.zeros(len(pixels), dtype=np.uint8)
    for color, city_id in lut.items():
        mask = np.all(pixels == np.array(color, dtype=np.uint8), axis=1)
        result[mask] = city_id
    return Image.fromarray(result.reshape(arr.shape[:2]), mode="L")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lut = build_lut()
    files = sorted(SEM_DIR.glob("*.png"))
    print(f"Converting {len(files)} files...")
    for i, p in enumerate(files):
        label = convert(p, lut)
        label.save(OUT_DIR / p.name)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"Done → {OUT_DIR}")

if __name__ == "__main__":
    main()
