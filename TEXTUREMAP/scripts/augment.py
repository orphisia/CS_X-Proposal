#!/usr/bin/env python3
"""Pad a class with rotated / flipped variants — offline data augmentation.

Most useful for **mandalas**: they're radially symmetric, so rotating them is a
natural, label-preserving transform that multiplies a small set (e.g. 166 unique
mandalas) into ~1000 training images. StyleGAN2-ADA *also* augments on the fly
(`--aug=ada`), so this is a complement, not a replacement — and it can't invent
new unique content, it just helps a tiny set converge.

Rotations use the largest centered square that stays inside the rotated frame
(no black corners), then resize back to the original edge.

Usage
-----
    python scripts/augment.py --in data/processed/mandalas \
        --out data/processed/mandalas_aug --factor 6
    # icons have a clear "up" — use a small angle if you augment them:
    python scripts/augment.py --in data/processed/icons \
        --out data/processed/icons_aug --factor 3 --max-angle 12 --no-hflip
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def rotate_crop(img, angle):
    """Rotate by `angle` deg and crop the largest centered square with no corners."""
    s = img.size[0]
    rot = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    a = math.radians(abs(angle) % 90)
    inner = s / (math.cos(a) + math.sin(a))      # largest inscribed axis-aligned square
    off = (s - inner) / 2
    crop = rot.crop((off, off, off + inner, off + inner))
    return crop.resize((s, s), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=6, help="output count multiplier (incl. original)")
    ap.add_argument("--max-angle", type=float, default=180.0, help="max abs rotation (deg); 180 = any orientation")
    ap.add_argument("--no-hflip", action="store_true", help="disable random horizontal flips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    files = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        print(f"No images in {in_dir}")
        return 1

    n = 0
    for p in tqdm(files, desc=f"augment:{in_dir.name}", unit="img"):
        base = Image.open(p).convert("RGB")
        base.save(out_dir / f"aug_{n:06d}.png"); n += 1          # keep the original
        for _ in range(args.factor - 1):
            img = base
            if args.max_angle > 0:
                img = rotate_crop(img, rng.uniform(-args.max_angle, args.max_angle))
            if not args.no_hflip and rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(out_dir / f"aug_{n:06d}.png"); n += 1

    print(f"\nWrote {n} images to {out_dir} (from {len(files)} originals, factor {args.factor}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
