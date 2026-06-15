#!/usr/bin/env python3
"""Build a labelled contact sheet of processed images for visual culling.

A broad scrape always leaves some off-target images (Western paintings among the
icons, photos among the thangkas). This tiles a class directory into one big
labelled grid so you can eyeball it, note the bad tiles by filename, and delete
them before packing the dataset:

    python scripts/make_contact_sheet.py --in data/processed/mandalas
    open outputs/samples/mandalas_contact.jpg          # review
    rm data/processed/mandalas/img_00042.png ...        # cull the bad ones
    python scripts/03_prepare_dataset.py                # repack what remains

Each tile is labelled with its filename so deletion is unambiguous.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True, help="dir of processed images")
    ap.add_argument("--out", default=None, help="output jpg (default outputs/samples/<name>_contact.jpg)")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--thumb", type=int, default=130, help="thumbnail edge in px")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        print(f"No images in {in_dir}")
        return 1

    out = Path(args.out) if args.out else REPO / "outputs" / "samples" / f"{in_dir.name}_contact.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)

    t, label_h, pad = args.thumb, 14, 3
    cell_w, cell_h = t + 2 * pad, t + label_h + 2 * pad
    cols = min(args.cols, len(files))
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, p in enumerate(files):
        r, c = divmod(i, cols)
        x, y = c * cell_w + pad, r * cell_h + pad
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im.thumbnail((t, t))
        sheet.paste(im, (x, y))
        draw.text((x, y + t + 1), p.name, fill=(20, 20, 20), font=font)

    sheet.save(out, quality=85)
    print(f"Contact sheet: {out}  ({len(files)} images, {cols}x{rows} grid)")
    print(f"Review it, then delete unwanted files from {in_dir} and repack with 03_prepare_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
