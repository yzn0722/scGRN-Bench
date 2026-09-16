#!/usr/bin/env bash
# 单数据集导出示例；先 source env_paths.sh
# 用法:
#   source .../env_paths.sh
#   MODEL=sccello DATASET=hESC HEAD_INDICES=all bash .../run_export_heads.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${ENV_PATHS:-$_SCRIPT_DIR/env_paths.sh}"

MODEL="${MODEL:-sccello}"
DATASET="${DATASET:-hESC}"
DATA_TYPE="${DATA_TYPE:-CHIP}"
HEAD_INDICES="${HEAD_INDICES:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# scPRINT: 与 scCello 4 head 对齐时设 SCPRINT_LAST_LAYER_ONLY=1
if [[ "$MODEL" == "scprint" && "${SCPRINT_LAST_LAYER_ONLY:-0}" == "1" ]]; then
  HEAD_INDICES="$(scprint_last_layer_indices "$SCPRINT_NLAYERS" "$SCPRINT_NHEAD")"
  SCPRINT_EXTRA=(--last-layer-only)
  echo "SCPRINT_LAST_LAYER_ONLY=1 -> HEAD_INDICES=$HEAD_INDICES"
else
  SCPRINT_EXTRA=()
fi

if [[ ! -f "$EXPORT_HEADS_UNIFIED" ]]; then
  echo "Missing: $EXPORT_HEADS_UNIFIED" >&2
  exit 1
fi

echo "UNIFIED=$EXPORT_HEADS_UNIFIED"
echo "MODEL=$MODEL DATASET=$DATASET HEAD_INDICES=$HEAD_INDICES"
echo "WEIGHTS_ROOT=$WEIGHTS_ROOT"
echo "INPUT_ROOT=$INPUT_ROOT OUTPUT_ROOT=$OUTPUT_ROOT"

case "$MODEL" in
  scgpt)
    python -u "$EXPORT_HEADS_UNIFIED" --model scgpt -- \
      --data-type "$DATA_TYPE" --dataset "$DATASET" \
      --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" \
      --scgpt-repo-dir "$SCGPT_REPO" --model-dir "$SCGPT_MODEL" \
      --batch-size "$BATCH_SIZE" --head-indices "$HEAD_INDICES"
    ;;
  geneformer)
    python -u "$EXPORT_HEADS_UNIFIED" --model geneformer -- \
      --data-type "$DATA_TYPE" --dataset "$DATASET" \
      --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" \
      --weights-root "$GENEFORMER_ROOT" \
      --batch-size "$BATCH_SIZE" --head-indices "$HEAD_INDICES"
    ;;
  langcell)
    python -u "$EXPORT_HEADS_UNIFIED" --model langcell -- \
      --data-type "$DATA_TYPE" --dataset "$DATASET" \
      --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" \
      --parent-model-dir "$LANGCELL_PARENT" \
      --batch-size "$BATCH_SIZE" --head-indices "$HEAD_INDICES"
    ;;
  sccello)
    python -u "$EXPORT_HEADS_UNIFIED" --model sccello -- \
      --data-type "$DATA_TYPE" --dataset "$DATASET" \
      --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" \
      --model-path "$SCCELLO_MODEL" \
      --dict-dir "$SCCELLO_DICT" \
      --sccello-repo-dir "$SCCELLO_REPO" \
      --batch-size "$BATCH_SIZE" --head-indices "$HEAD_INDICES"
    ;;
  scprint)
    expr="$INPUT_ROOT/CHIP/${DATASET}_chip_matched-ExpressionData.csv"
    python -u "$EXPORT_HEADS_UNIFIED" --model scprint -- \
      --expr-csv "$expr" \
      --out-prefix "$OUTPUT_ROOT/scprint/${DATASET}" \
      --ckpt "$SCPRINT_CKPT" \
      --token-pkl "$SCPRINT_TOKEN_PKL" \
      --nlayers "$SCPRINT_NLAYERS" \
      --nhead "$SCPRINT_NHEAD" \
      --head-indices "$HEAD_INDICES" \
      "${SCPRINT_EXTRA[@]}"
    ;;
  *)
    echo "MODEL must be: scgpt|geneformer|langcell|sccello|scprint" >&2
    exit 1
    ;;
esac
