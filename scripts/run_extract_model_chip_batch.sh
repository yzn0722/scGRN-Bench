#!/usr/bin/env bash
set -euo pipefail

# One-click batch runner for CHIP datasets in scGRN-Bench/src/extract_model.
#
# Usage:
#   bash scripts/run_extract_model_chip_batch.sh
#   INPUT_ROOT=/path/to/input_process1000 OUTPUT_ROOT=/path/to/out bash scripts/run_extract_model_chip_batch.sh
#
# Optional env vars:
#   DATA_ROOT                 default: $PROJECT_ROOT/data
#   INPUT_ROOT                default: $DATA_ROOT/input_process1000
#   INPUT_ROOT_ATT            default: $DATA_ROOT/input_process
#   OUTPUT_ROOT               default: $PROJECT_ROOT/outputs
#   MODEL_WEIGHTS_ROOT        default: $DATA_ROOT/model_weights
#   SCGPT_MODEL_DIR           default: $DATA_ROOT/scgpt/scgpt_human
#   SCFOUNDATION_ROOT         default: $DATA_ROOT/scfoundation
#   SCFOUNDATION_MODEL_PATH   default: $SCFOUNDATION_ROOT/model/models/models.ckpt
#   SCFOUNDATION_VOCAB_PATH   default: $SCFOUNDATION_ROOT/model/OS_scRNA_gene_index.19264.tsv
#   SCCELLO_REPO_DIR          default: $PROJECT_ROOT/models/sc_foundation_evals
#   DYNAMIC_EXPR_ROOT         default: $DATA_ROOT/input_process
#   DYNAMIC_PT_ROOT           default: $DATA_ROOT/PseudoTime
#   RUN_EMBEDDING             default: 1
#   RUN_HIDDEN                default: 1
#   RUN_ATTENTION             default: 0  (heavy + dependency-sensitive; off by default)
#   RUN_DYNAMIC               default: 0

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
INPUT_ROOT="${INPUT_ROOT:-$DATA_ROOT/input_process1000}"
INPUT_ROOT_ATT="${INPUT_ROOT_ATT:-$DATA_ROOT/input_process}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
MODEL_WEIGHTS_ROOT="${MODEL_WEIGHTS_ROOT:-$DATA_ROOT/model_weights}"
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-$DATA_ROOT/scgpt/scgpt_human}"
SCFOUNDATION_ROOT="${SCFOUNDATION_ROOT:-$DATA_ROOT/scfoundation}"
SCFOUNDATION_MODEL_PATH="${SCFOUNDATION_MODEL_PATH:-$SCFOUNDATION_ROOT/model/models/models.ckpt}"
SCFOUNDATION_VOCAB_PATH="${SCFOUNDATION_VOCAB_PATH:-$SCFOUNDATION_ROOT/model/OS_scRNA_gene_index.19264.tsv}"
SCCELLO_REPO_DIR="${SCCELLO_REPO_DIR:-$PROJECT_ROOT/models/sc_foundation_evals}"
DYNAMIC_EXPR_ROOT="${DYNAMIC_EXPR_ROOT:-$DATA_ROOT/input_process}"
DYNAMIC_PT_ROOT="${DYNAMIC_PT_ROOT:-$DATA_ROOT/PseudoTime}"

RUN_EMBEDDING="${RUN_EMBEDDING:-1}"
RUN_HIDDEN="${RUN_HIDDEN:-1}"
RUN_ATTENTION="${RUN_ATTENTION:-0}"
RUN_DYNAMIC="${RUN_DYNAMIC:-0}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

run_step() {
  local name="$1"
  shift
  local start_ts
  start_ts="$(date '+%F %T')"
  local t0
  t0="$(date +%s)"
  echo "[START] ${name} @ ${start_ts}"
  "$@"
  local t1
  t1="$(date +%s)"
  echo "[DONE ] ${name} | elapsed=$((t1 - t0))s"
  echo
}

echo "== scGRN-Bench CHIP batch =="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo

mkdir -p "$OUTPUT_ROOT"

if [[ "$RUN_EMBEDDING" == "1" ]]; then
  echo "[1/4] embedding/*"
  run_step "embedding/scgpt_all.py" python -u src/extract_model/embedding/scgpt_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/scgpt" \
    --model-dir "$SCGPT_MODEL_DIR" \
    --folders CHIP

  run_step "embedding/scfoundation_all.py" python -u src/extract_model/embedding/scfoundation_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/scfoundation" \
    --ckpt-path "$SCFOUNDATION_MODEL_PATH" \
    --vocab-path "$SCFOUNDATION_VOCAB_PATH" \
    --folders CHIP

  run_step "embedding/geneformer_all.py" python -u src/extract_model/embedding/geneformer_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/geneformer" \
    --model-dir "$MODEL_WEIGHTS_ROOT/Geneformer/default/12L" \
    --dict-dir "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --folders CHIP

  run_step "embedding/Langcell_all.py" python -u src/extract_model/embedding/Langcell_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/langcell" \
    --parent-model-dir "$MODEL_WEIGHTS_ROOT" \
    --folders CHIP

  run_step "embedding/sccello_all.py" python -u src/extract_model/embedding/sccello_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/sccello" \
    --parent-model-dir "$MODEL_WEIGHTS_ROOT" \
    --folders CHIP
fi

if [[ "$RUN_HIDDEN" == "1" ]]; then
  echo "[2/4] hidden_emb/*"
  run_step "hidden_emb/scgpt_hidden.py" python -u src/extract_model/hidden_emb/scgpt_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/scgpt" \
    --model-dir "$SCGPT_MODEL_DIR" \
    --folders CHIP

  run_step "hidden_emb/scfoundation_hidden.py" python -u src/extract_model/hidden_emb/scfoundation_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/scfoundation" \
    --scfoundation-root "$SCFOUNDATION_ROOT" \
    --model-path "$SCFOUNDATION_MODEL_PATH" \
    --vocab-path "$SCFOUNDATION_VOCAB_PATH"

  run_step "hidden_emb/Genfoemer_hidden.py" python -u src/extract_model/hidden_emb/Genfoemer_hidden.py \
    --model-dir "$MODEL_WEIGHTS_ROOT/Geneformer/default/12L" \
    --dict-dir "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/geneformer" \
    --folders CHIP

  run_step "hidden_emb/Langcell_hidden.py" python -u src/extract_model/hidden_emb/Langcell_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/langcell" \
    --dict-path "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --langcell-model-path "$MODEL_WEIGHTS_ROOT/LangCell/cell_bert" \
    --folders CHIP

  run_step "hidden_emb/sccello_hidden.py" python -u src/extract_model/hidden_emb/sccello_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/sccello" \
    --model-dir "$MODEL_WEIGHTS_ROOT/scCello" \
    --dict-dir "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --repo-root "$DATA_ROOT" \
    --folders CHIP
fi

if [[ "$RUN_ATTENTION" == "1" ]]; then
  echo "[3/4] attention/*"
  for ds in hESC hHep mDC mHSC-E mHSC-GM mHSC-L; do
    run_step "attention/scCello.py CHIP/${ds}" python -u src/extract_model/attention/scCello.py \
      CHIP "$ds" \
      --input-root "$INPUT_ROOT_ATT" \
      --output-root "$OUTPUT_ROOT/attention/sccello" \
      --model-path "$MODEL_WEIGHTS_ROOT/scCello" \
      --dict-dir "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
      --sccello-repo-dir "$SCCELLO_REPO_DIR"
  done
fi

if [[ "$RUN_DYNAMIC" == "1" ]]; then
  echo "[4/4] GRN_inferance/dyn/*"
  run_step "dyn/scgpt_dyn.py" python -u src/GRN_inferance/dyn/scgpt_dyn.py \
    --model-dir "$SCGPT_MODEL_DIR" \
    --outdir "$OUTPUT_ROOT/dyn/scgpt" \
    --expr-root "$DYNAMIC_EXPR_ROOT" \
    --pt-root "$DYNAMIC_PT_ROOT"

  run_step "dyn/scFoundation_dyn.py" python -u src/GRN_inferance/dyn/scFoundation_dyn.py \
    --scfoundation-root "$SCFOUNDATION_ROOT" \
    --ckpt-path "$SCFOUNDATION_MODEL_PATH" \
    --gene-index-tsv "$SCFOUNDATION_VOCAB_PATH" \
    --outdir "$OUTPUT_ROOT/dyn/scfoundation" \
    --expr-root "$DYNAMIC_EXPR_ROOT" \
    --pt-root "$DYNAMIC_PT_ROOT"
fi

echo
echo "Done. Outputs under: $OUTPUT_ROOT"
