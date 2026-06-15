#!/usr/bin/env bash
# Resume training from the most recent snapshot (spot-instance friendly).
#
#   bash scripts/05_resume.sh
#
# Override which checkpoint to resume from:  RESUME=path/to/snap.pkl bash scripts/05_resume.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/datasets/csx-512.zip}"
OUTDIR="${OUTDIR:-checkpoints}"

RESUME="${RESUME:-}"
if [ -z "$RESUME" ]; then
  RESUME="$(ls -t "$OUTDIR"/*/network-snapshot-*.pkl 2>/dev/null | head -1 || true)"
fi
if [ -z "$RESUME" ] || [ ! -f "$RESUME" ]; then
  echo "No checkpoint found under $OUTDIR/. Start fresh with scripts/04_train.sh." >&2
  exit 1
fi
echo "Resuming from: $RESUME"

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
  --metrics=fid50k_full \
  --resume="$RESUME"
