#!/usr/bin/env bash
# 批处理 CHIP 六数据集
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${ENV_PATHS:-$_SCRIPT_DIR/env_paths.sh}"

MODEL="${MODEL:-scgpt}"
HEAD_INDICES="${HEAD_INDICES:-all}"
DATASETS="${DATASETS:-hESC,hHep,mDC,mHSC-E,mHSC-GM,mHSC-L}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "MODEL=$MODEL HEAD_INDICES=$HEAD_INDICES"
echo "EXPORT_HEADS_UNIFIED=$EXPORT_HEADS_UNIFIED"
echo "WEIGHTS_ROOT=$WEIGHTS_ROOT"
echo "INPUT_ROOT=$INPUT_ROOT OUTPUT_ROOT=$OUTPUT_ROOT"

IFS=',' read -r -a DS_ARR <<< "$DATASETS"
for ds in "${DS_ARR[@]}"; do
  ds="$(echo "$ds" | xargs)"
  [[ -z "$ds" ]] && continue
  echo "======== $MODEL / $ds ========"
  MODEL="$MODEL" DATASET="$ds" HEAD_INDICES="$HEAD_INDICES" BATCH_SIZE="$BATCH_SIZE" \
    bash "$_SCRIPT_DIR/run_export_heads.sh"
done

echo "Done. Check $OUTPUT_ROOT"
