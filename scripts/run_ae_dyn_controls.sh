#!/usr/bin/env bash
# Run AE / DAE controls mirroring scgpt_dyn evaluation protocol.
#
# Usage:
#   bash scripts/run_ae_dyn_controls.sh
#   EXPR_ROOT=/path/to/input_process PT_ROOT=/path/to/PseudoTime bash scripts/run_ae_dyn_controls.sh
#   DATASET=hESC bash scripts/run_ae_dyn_controls.sh   # single dataset smoke test
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EXPR_ROOT="${EXPR_ROOT:-${ROOT}/data/input_process}"
PT_ROOT="${PT_ROOT:-${ROOT}/data/PseudoTime}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/ae_dyn}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ARG=()
if [[ -n "${DATASET:-}" ]]; then
  DATASET_ARG=(--dataset "${DATASET}")
fi

# Prefer CHIP subfolder layout used by scgpt_dyn; fall back if CSVs are flat.
if [[ ! -d "${EXPR_ROOT}/CHIP" ]]; then
  echo "[WARN] ${EXPR_ROOT}/CHIP not found; ensure --expr-root points to a directory"
  echo "       that contains CHIP/{dataset}_chip_matched-ExpressionData.csv"
fi

common_args=(
  --expr-root "${EXPR_ROOT}"
  --pt-root "${PT_ROOT}"
  --gen-iters 16
  --epochs 50
  --batch-size 64
  --seed 42
  "${DATASET_ARG[@]}"
)

echo "==> AE / self_holdout_early"
"${PYTHON_BIN}" -u src/GRN_inferance/dyn/ae_dyn.py \
  --model-type ae \
  --train-mode self_holdout_early \
  --outdir "${OUT_ROOT}/ae_self_holdout_early" \
  "${common_args[@]}"

echo "==> DAE / self_holdout_early"
"${PYTHON_BIN}" -u src/GRN_inferance/dyn/ae_dyn.py \
  --model-type dae \
  --train-mode self_holdout_early \
  --outdir "${OUT_ROOT}/dae_self_holdout_early" \
  "${common_args[@]}"

echo "==> AE / self_mid (exclude early+late; stricter control)"
"${PYTHON_BIN}" -u src/GRN_inferance/dyn/ae_dyn.py \
  --model-type ae \
  --train-mode self_mid \
  --outdir "${OUT_ROOT}/ae_self_mid" \
  "${common_args[@]}"

echo "==> DAE / self_mid"
"${PYTHON_BIN}" -u src/GRN_inferance/dyn/ae_dyn.py \
  --model-type dae \
  --train-mode self_mid \
  --outdir "${OUT_ROOT}/dae_self_mid" \
  "${common_args[@]}"

echo "==> AE / leave-one-out"
"${PYTHON_BIN}" -u src/GRN_inferance/dyn/ae_dyn.py \
  --model-type ae \
  --train-mode loo \
  --outdir "${OUT_ROOT}/ae_loo" \
  "${common_args[@]}"

echo "==> DAE / leave-one-out"
"${PYTHON_BIN}" -u src/GRN_inferance/dyn/ae_dyn.py \
  --model-type dae \
  --train-mode loo \
  --outdir "${OUT_ROOT}/dae_loo" \
  "${common_args[@]}"

echo "==> AE / untrained (noise floor)"
"${PYTHON_BIN}" -u src/GRN_inferance/dyn/ae_dyn.py \
  --model-type ae \
  --train-mode untrained \
  --outdir "${OUT_ROOT}/ae_untrained" \
  "${common_args[@]}"

echo "Done. Results under ${OUT_ROOT}"
echo "Compare final_direction_acc against scgpt_dyn outputs (chance ≈ 0.5)."
echo "Note: self_holdout_early can be high if late cells are in the train set;"
echo "      prefer self_mid / loo / untrained for the main control narrative."
