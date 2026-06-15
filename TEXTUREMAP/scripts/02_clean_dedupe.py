#!/usr/bin/env python3
"""Clean raw images into a StyleGAN-ready set: dedupe, square-crop, resize.

Per input directory (one class each), every image is:
  1. opened and validated (corrupt / too-small / extreme-aspect images dropped),
  2. perceptual-hash deduped (phash, Hamming distance < threshold = duplicate),
  3. center-cropped to a square (keeps the central figure for portrait icons),
  4. resized to SIZE x SIZE (LANCZOS),
  5. written as data/processed/<class>/img_NNNNN.png.

Usage
-----
    python scripts/02_clean_dedupe.py \
        --in data/raw/mandalas data/raw/icons \
        --out data/processed --size 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None          # large museum scans are fine
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly-truncated downloads

REPO = Path(__file__).resolve().parents[1]


def center_crop_square(img):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def color_stats(img):
    """(mean_saturation, white_fraction) in [0,1] from an RGB PIL image.

    Used to drop near-grayscale archival photos (low saturation) and line-art /
    coloring-book pages (mostly white, near-zero saturation) — both of which
    survive dedupe but ruin the gold/ochre-palette training signal. Gold leaf is
    bright but warm (saturation > 0.1), so it is not counted as white.
    """
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    return float(s.mean()), float(((v > 0.94) & (s < 0.10)).mean())


def process_dir(in_dir, out_dir, size, min_px, max_aspect, hash_thresh,
                min_sat, max_white, delete_source=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    kept_hashes = []
    n_written = stats_corrupt = stats_small = stats_aspect = stats_dup = 0
    stats_mono = stats_lineart = stats_deleted = 0

    for p in tqdm(files, desc=f"clean:{in_dir.name}", unit="img"):
        try:
            try:
                img = Image.open(p)
                img.load()
                img = img.convert("RGB")
            except Exception:
                stats_corrupt += 1
                continue

            w, h = img.size
            if min(w, h) < min_px:
                stats_small += 1
                continue
            if max(w, h) / min(w, h) > max_aspect:
                stats_aspect += 1
                continue

            # Downscale + format FIRST, then judge/dedupe the final 256px image.
            out = center_crop_square(img).resize((size, size), Image.LANCZOS)
            mean_sat, white_frac = color_stats(out)
            if mean_sat < min_sat:           # near-grayscale (B&W archival photo)
                stats_mono += 1
                continue
            if white_frac > max_white:       # line-art / coloring page on white
                stats_lineart += 1
                continue

            ph = imagehash.phash(out)
            if any((ph - k) < hash_thresh for k in kept_hashes):
                stats_dup += 1
                continue
            kept_hashes.append(ph)

            out.save(out_dir / f"img_{n_written:05d}.png")
            n_written += 1
        finally:
            # Reclaim disk: drop the large original once we're done with it
            # (kept or not). Re-downloadable from the provenance log if needed.
            if delete_source:
                try:
                    p.unlink()
                    stats_deleted += 1
                except OSError:
                    pass

    return {
        "input": len(files), "written": n_written, "corrupt": stats_corrupt,
        "too_small": stats_small, "bad_aspect": stats_aspect, "duplicate": stats_dup,
        "grayscale": stats_mono, "lineart": stats_lineart, "deleted_source": stats_deleted,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="one or more raw class dirs")
    ap.add_argument("--out", default=str(REPO / "data" / "processed"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--min-px", type=int, default=300, help="drop images smaller than this on either axis")
    ap.add_argument("--max-aspect", type=float, default=3.0, help="drop images more lopsided than this ratio")
    ap.add_argument("--hash-thresh", type=int, default=6, help="phash Hamming distance below which = duplicate")
    ap.add_argument("--min-saturation", type=float, default=0.08,
                    help="drop near-grayscale images below this mean saturation (0..1)")
    ap.add_argument("--max-white", type=float, default=0.55,
                    help="drop images whose white-background fraction exceeds this (line art)")
    ap.add_argument("--delete-source", action="store_true",
                    help="unlink each raw file after processing it (reclaim disk; "
                         "originals are re-downloadable from the provenance log)")
    args = ap.parse_args()

    out_root = Path(args.out)
    grand_total = 0
    for raw in args.inputs:
        in_dir = Path(raw)
        if not in_dir.is_dir():
            print(f"SKIP (not a dir): {in_dir}")
            continue
        s = process_dir(in_dir, out_root / in_dir.name, args.size,
                        args.min_px, args.max_aspect, args.hash_thresh,
                        args.min_saturation, args.max_white, args.delete_source)
        grand_total += s["written"]
        print(f"\n[{in_dir.name}] {s['input']} in -> {s['written']} kept "
              f"(dropped: {s['corrupt']} corrupt, {s['too_small']} small, "
              f"{s['bad_aspect']} aspect, {s['grayscale']} grayscale, "
              f"{s['lineart']} line-art, {s['duplicate']} dup)")
        if args.delete_source:
            print(f"  reclaimed disk: deleted {s['deleted_source']} source files from {in_dir}")

    print(f"\nTotal kept: {grand_total} images at {args.size}x{args.size} in {out_root}")
    if grand_total == 0:
        print("Nothing was written — did you run 01_scrape_data.py first?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
