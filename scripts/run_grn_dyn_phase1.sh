#!/usr/bin/env bash
# Phase-1 GRN-aware dynamic prediction sweep (six datasets).
set -euo pipefail

ROOT="/mnt/10T/yzn"
BENCH="${ROOT}/benchmark_GRN"
SCGPT="${BENCH}/model/weights/scgpt/scGPT_human"
OUT="${ROOT}/scGRN-Bench/FBplot/fig4/grn_dyn_phase1"
SCRIPT="${ROOT}/scGRN-Bench/src/GRN_inferance/dyn/scgpt_dyn_grn.py"

export PYTHONPATH="${BENCH}/pre_scgpt/scGPT:${PYTHONPATH:-}"

# Prefer conda env with scGPT + CUDA (override via PYTHON_BIN).
PYTHON_BIN="${PYTHON_BIN:-/mnt/10T/yzn/anconda3/envs/singlecell/bin/python3}"

mkdir -p "$OUT"

"$PYTHON_BIN" "$SCRIPT" \
  --model-dir "$SCGPT" \
  --expr-root "${BENCH}/input_process" \
  --pt-root "${BENCH}/PseudoTime" \
  --outdir "$OUT/sweep_sources_beta0.25" \
  --grn-beta 0.25 \
  --sweep-sources \
  --gen-iters 16 \
  --batch-size 16

echo "Done. See ${OUT}/sweep_sources_beta0.25/grn_dyn_summary.csv"
