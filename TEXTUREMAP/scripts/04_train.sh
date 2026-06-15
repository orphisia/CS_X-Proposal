#!/usr/bin/env bash
# StyleGAN2-ADA training launch (run on a CUDA GPU — RunPod / Vast.ai / Kaggle).
# ADA augmentation is the key to training on a ~1k-image set without mode collapse.
#
#   bash scripts/04_train.sh
#
# Override the dataset zip:  DATA=data/datasets/csx-1024.zip bash scripts/04_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/datasets/csx-512.zip}"
OUTDIR="${OUTDIR:-checkpoints}"

if [ ! -f "$DATA" ]; then
  echo "Dataset not found: $DATA  — run scripts/03_prepare_dataset.py first." >&2
  exit 1
fi
if [ ! -f third_party/stylegan3/train.py ]; then
  echo "Engine missing — run: git submodule update --init --recursive" >&2
  exit 1
fi

python third_party/stylegan3/train.py \
  --outdir="$OUTDIR" \
  --data="$DATA" \
  --cfg=stylegan2 \
  --gpus=1 \
  --batch=32 \
  --gamma=8 \
  --mirror=1 \
  --aug=ada \
  --target=0.6 \
  --snap=10 \
  --metrics=fid50k_full
