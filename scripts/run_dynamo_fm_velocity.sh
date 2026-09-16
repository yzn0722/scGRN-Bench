#!/usr/bin/env bash
# Use foundation models to benchmark "velocity direction" on Dynamo datasets.
#
# Step 1: export h5ad -> CSV + PseudoTime (+ optional Dynamo velocity truth)
# Step 2: run scGPT (or unified runner) with datasets_dynamo.json
#
# Usage:
#   bash scripts/run_dynamo_fm_velocity.sh zebrafish
#   bash scripts/run_dynamo_fm_velocity.sh zebrafish_dyn /path/to/processed.h5ad

set -euo pipefail

ROOT="/mnt/10T/yzn/scGRN-Bench"
DYN_DATA="/mnt/10T/yzn/dynamo-release/data"
EXPORT_ROOT="${ROOT}/data/dynamo_export"
SCGPT_MODEL="/mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human"
OUTDIR="${ROOT}/outputs/dynamo_fm_velocity"

NAME="${1:-zebrafish}"
H5AD="${2:-${DYN_DATA}/zebrafish.h5ad}"
PT_SOURCE="${PT_SOURCE:-umap_1}"
VELOCITY_LAYER="${VELOCITY_LAYER:-}"

cd "${ROOT}"
source /mnt/10T/yzn/anconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dynamo-env 2>/dev/null || conda activate base 2>/dev/null || true

echo "=== [1/2] Export Dynamo h5ad for FM benchmark: ${NAME} ==="
PREP_ARGS=(
  --h5ad "${H5AD}"
  --name "${NAME}"
  --out-root "${EXPORT_ROOT}"
  --layer spliced
  --pt-source "${PT_SOURCE}"
  --gene-transform upper
  --species zebrafish
)
if [[ -n "${VELOCITY_LAYER}" ]]; then
  PREP_ARGS+=(--velocity-layer "${VELOCITY_LAYER}")
fi

python src/GRN_inferance/dyn/prepare_dynamo_for_fm.py "${PREP_ARGS[@]}"

echo "=== [2/2] scGPT direction accuracy (FM predicts pseudotime delta) ==="
mkdir -p "${OUTDIR}"
python src/GRN_inferance/dyn/scgpt_dyn.py \
  --model-dir "${SCGPT_MODEL}" \
  --outdir "${OUTDIR}/scgpt_${NAME}" \
  --datasets-json "${EXPORT_ROOT}/datasets_dynamo.json" \
  --pt-quantile 0.2 \
  --top-percent 30 \
  --gen-iters 16 \
  --batch-size 16

echo "Done. Results: ${OUTDIR}/scgpt_${NAME}"
echo "If dynamo_velocity_truth.csv exists, run evaluate_fm_vs_dynamo_velocity.py for Dynamo-vs-FM comparison."
