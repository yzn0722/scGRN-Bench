#!/usr/bin/env bash
# Protocol A: GRNFormer pretrained transfer on BEELINE, FBEval metrics.
#
# Setup once:
#   git clone https://github.com/BioinfoMachineLearning/GRNformer.git /mnt/10T/yzn/GRNformer
#   cd /mnt/10T/yzn/GRNformer && bash setup.sh
#   # place/download official .ckpt, set CKPT below
#
# Then:
#   bash scripts/run_grnformer_beeline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GT_ROOT="${GT_ROOT:-/mnt/10T/yzn/benchmark_GRN/input_process}"
TF_ROOT="${TF_ROOT:-/mnt/10T/yzn/Beeline-master}"
GRNFORMER_ROOT="${GRNFORMER_ROOT:-/mnt/10T/yzn/GRNformer}"
CKPT="${CKPT:-${GRNFORMER_ROOT}/Trainings/GRNFormer_epoch=26_valid_loss=0.645546.ckpt}"
PRED_DIR="${PRED_DIR:-outputs/grnformer/preds}"
OUTDIR="${OUTDIR:-outputs/grnformer_fbeval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS=(hESC hHep mESC mDC mHSC-E mHSC-GM mHSC-L)

echo "== GRNFormer Protocol A =="
echo "GT_ROOT=$GT_ROOT"
echo "GRNFORMER_ROOT=$GRNFORMER_ROOT"
echo "CKPT=$CKPT"

if [[ ! -f "$GRNFORMER_ROOT/infer_grn.py" ]]; then
  echo "ERROR: clone GRNformer first: git clone https://github.com/BioinfoMachineLearning/GRNformer.git $GRNFORMER_ROOT"
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "Download/place official ckpt and export CKPT=/path/to.ckpt"
  exit 1
fi

"$PYTHON_BIN" -u FBEval/run_grnformer_baseline.py \
  --gt-root "$GT_ROOT" \
  --tf-root "$TF_ROOT" \
  --datasets "${DATASETS[@]}" \
  --gt-types STRING Non_CHIP CHIP \
  --expr-source STRING \
  --grnformer-root "$GRNFORMER_ROOT" \
  --ckpt "$CKPT" \
  --run-infer \
  --pred-dir "$PRED_DIR" \
  --outdir "$OUTDIR" \
  --protocol A \
  --python-bin "$PYTHON_BIN"

echo "Done. See $OUTDIR/grnformer_summary_mean_std.csv"
