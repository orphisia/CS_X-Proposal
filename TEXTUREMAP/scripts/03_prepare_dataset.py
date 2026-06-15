#!/usr/bin/env python3
"""Pack data/processed/ into the .zip StyleGAN's trainer expects.

Thin wrapper around third_party/stylegan3/dataset_tool.py with a pre-flight
sanity check (per-class counts) and a post-run summary.

Usage
-----
    python scripts/03_prepare_dataset.py                       # 512x512 default
    python scripts/03_prepare_dataset.py --resolution 1024x1024
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET_TOOL = REPO / "third_party" / "stylegan3" / "dataset_tool.py"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(REPO / "data" / "processed"))
    ap.add_argument("--dest", default=None, help="output .zip (default data/datasets/csx-<res>.zip)")
    ap.add_argument("--resolution", default="512x512")
    ap.add_argument("--min-per-class", type=int, default=400)
    args = ap.parse_args()

    source = Path(args.source)
    res_tag = args.resolution.split("x")[0]
    dest = Path(args.dest) if args.dest else REPO / "data" / "datasets" / f"csx-{res_tag}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not source.is_dir():
        print(f"ERROR: source not found: {source}\nRun 02_clean_dedupe.py first.")
        return 1

    # Pre-flight: count per class.
    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    if not class_dirs:  # flat dir of images, still valid (unconditional)
        n = len([p for p in source.iterdir() if p.is_file()])
        print(f"Flat source: {n} images")
    else:
        print("Pre-flight image counts:")
        for d in sorted(class_dirs):
            n = len([p for p in d.iterdir() if p.is_file()])
            flag = "  <-- LOW" if n < args.min_per_class else ""
            print(f"  {d.name:12s} {n}{flag}")

    if not DATASET_TOOL.exists():
        print(f"\nERROR: {DATASET_TOOL} not found.")
        print("Add the engine first:  git submodule update --init --recursive")
        return 1

    cmd = [sys.executable, str(DATASET_TOOL),
           "--source", str(source), "--dest", str(dest),
           "--resolution", args.resolution]
    print("\nRunning:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("dataset_tool.py failed.")
        return rc

    size_mb = dest.stat().st_size / 1e6 if dest.exists() else 0
    print(f"\nDataset written: {dest} ({size_mb:.1f} MB)")
    print(f"Train with:  bash scripts/04_train.sh   (uses {dest.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
