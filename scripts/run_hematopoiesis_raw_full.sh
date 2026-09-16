#!/usr/bin/env bash
# Full pipeline: hematopoiesis_raw.h5ad -> Dynamo velocity -> FM benchmark export -> scGPT
set -euo pipefail

DYN_ROOT="/mnt/10T/yzn/dynamo-release"
BENCH_ROOT="/mnt/10T/yzn/scGRN-Bench"
RAW="${DYN_ROOT}/data/hematopoiesis_raw.h5ad"
PROC="${DYN_ROOT}/results/hematopoiesis_raw/hematopoiesis_processed.h5ad"
SCGPT="/mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human"
EXPORT="${BENCH_ROOT}/data/dynamo_export"
OUT="${BENCH_ROOT}/outputs/dynamo_fm_velocity/scgpt_hematopoiesis"

source /mnt/10T/yzn/anconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dynamo-env

echo "=== Step 1: Dynamo on hematopoiesis_raw ==="
cd "${DYN_ROOT}"
python run_hematopoiesis_raw.py --input "${RAW}" --output "${PROC}" --cores 4

echo "=== Step 2: Export for scGRN-Bench (human genes) ==="
cd "${BENCH_ROOT}"
# velocity layer for raw one-shot workflow (NOT velocity_S)
PT_SOURCE="${PT_SOURCE:-pseudotime_fp}"
python src/GRN_inferance/dyn/prepare_dynamo_for_fm.py \
  --h5ad "${PROC}" \
  --name hematopoiesis \
  --out-root "${EXPORT}" \
  --layer total \
  --pt-source "${PT_SOURCE}" \
  --velocity-layer velocity_alpha_minus_gamma_s \
  --species human \
  --gene-transform none

echo "=== Step 3: scGPT direction accuracy ==="
python src/GRN_inferance/dyn/scgpt_dyn.py \
  --model-dir "${SCGPT}" \
  --outdir "${OUT}" \
  --datasets-json "${EXPORT}/datasets_dynamo.json" \
  --cores 4

echo "All done."
echo "  Processed: ${PROC}"
echo "  FM results: ${OUT}"
