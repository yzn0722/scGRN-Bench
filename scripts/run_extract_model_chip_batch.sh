#!/usr/bin/env bash
set -euo pipefail

# One-click batch runner for CHIP datasets in scGRN-Bench/src/extract_model.
#
# Usage:
#   bash scripts/run_extract_model_chip_batch.sh
#   INPUT_ROOT=/path/to/input_process1000 OUTPUT_ROOT=/path/to/out bash scripts/run_extract_model_chip_batch.sh
#
# Optional env vars:
#   INPUT_ROOT                default: /mnt/10T/yzn/benchmark_GRN/input_process1000
#   INPUT_ROOT_ATT            default: /mnt/10T/yzn/benchmark_GRN/input_process
#   OUTPUT_ROOT               default: /mnt/10T/yzn/scGRN-Bench/outputs
#   MODEL_WEIGHTS_ROOT        default: /mnt/10T/yzn/benchmark_GRN/model/weights
#   SCGPT_MODEL_DIR           default: /mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human
#   SCFOUNDATION_ROOT         default: /mnt/10T/yzn/scFoundation-main
#   SCFOUNDATION_MODEL_PATH   default: $SCFOUNDATION_ROOT/model/models/models.ckpt
#   SCFOUNDATION_VOCAB_PATH   default: $SCFOUNDATION_ROOT/model/OS_scRNA_gene_index.19264.tsv
#   SCCELLO_REPO_DIR          default: /mnt/10T/yzn/benchmark_GRN/sc_foundation_evals
#   RUN_EMBEDDING             default: 1
#   RUN_HIDDEN                default: 1
#   RUN_ATTENTION             default: 0  (heavy + dependency-sensitive; off by default)
#   RUN_DYNAMIC               default: 0

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_ROOT="${INPUT_ROOT:-/mnt/10T/yzn/benchmark_GRN/input_process1000}"
INPUT_ROOT_ATT="${INPUT_ROOT_ATT:-/mnt/10T/yzn/benchmark_GRN/input_process}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/10T/yzn/scGRN-Bench/outputs}"
MODEL_WEIGHTS_ROOT="${MODEL_WEIGHTS_ROOT:-/mnt/10T/yzn/benchmark_GRN/model/weights}"
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human}"
SCFOUNDATION_ROOT="${SCFOUNDATION_ROOT:-/mnt/10T/yzn/scFoundation-main}"
SCFOUNDATION_MODEL_PATH="${SCFOUNDATION_MODEL_PATH:-$SCFOUNDATION_ROOT/model/models/models.ckpt}"
SCFOUNDATION_VOCAB_PATH="${SCFOUNDATION_VOCAB_PATH:-$SCFOUNDATION_ROOT/model/OS_scRNA_gene_index.19264.tsv}"
SCCELLO_REPO_DIR="${SCCELLO_REPO_DIR:-/mnt/10T/yzn/benchmark_GRN/sc_foundation_evals}"

RUN_EMBEDDING="${RUN_EMBEDDING:-1}"
RUN_HIDDEN="${RUN_HIDDEN:-1}"
RUN_ATTENTION="${RUN_ATTENTION:-0}"
RUN_DYNAMIC="${RUN_DYNAMIC:-0}"

echo "== scGRN-Bench CHIP batch =="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo

mkdir -p "$OUTPUT_ROOT"

if [[ "$RUN_EMBEDDING" == "1" ]]; then
  echo "[1/4] embedding/*"
  python src/extract_model/embedding/scgpt_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/scgpt" \
    --model-dir "$SCGPT_MODEL_DIR" \
    --folders CHIP

  python src/extract_model/embedding/scfoundation_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/scfoundation" \
    --ckpt-path "$SCFOUNDATION_MODEL_PATH" \
    --vocab-path "$SCFOUNDATION_VOCAB_PATH" \
    --folders CHIP

  python src/extract_model/embedding/geneformer_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/geneformer" \
    --model-dir "$MODEL_WEIGHTS_ROOT/Geneformer/default/12L" \
    --dict-dir "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --folders CHIP

  python src/extract_model/embedding/Langcell_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/langcell" \
    --parent-model-dir "$MODEL_WEIGHTS_ROOT" \
    --folders CHIP

  python src/extract_model/embedding/sccello_all.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/embedding/sccello" \
    --parent-model-dir "$MODEL_WEIGHTS_ROOT" \
    --folders CHIP
fi

if [[ "$RUN_HIDDEN" == "1" ]]; then
  echo "[2/4] hidden_emb/*"
  python src/extract_model/hidden_emb/scgpt_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/scgpt" \
    --model-dir "$SCGPT_MODEL_DIR" \
    --folders CHIP

  python src/extract_model/hidden_emb/scfoundation_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/scfoundation" \
    --scfoundation-root "$SCFOUNDATION_ROOT" \
    --model-path "$SCFOUNDATION_MODEL_PATH" \
    --vocab-path "$SCFOUNDATION_VOCAB_PATH"

  python src/extract_model/hidden_emb/Genfoemer_hidden.py \
    --model-dir "$MODEL_WEIGHTS_ROOT/Geneformer/default/12L" \
    --dict-dir "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/geneformer" \
    --folders CHIP

  python src/extract_model/hidden_emb/Langcell_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/langcell" \
    --dict-path "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --langcell-model-path "$MODEL_WEIGHTS_ROOT/LangCell/cell_bert" \
    --folders CHIP

  python src/extract_model/hidden_emb/sccello_hidden.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/hidden/sccello" \
    --model-dir "$MODEL_WEIGHTS_ROOT/scCello" \
    --dict-dir "$MODEL_WEIGHTS_ROOT/Geneformer/dicts" \
    --repo-root /mnt/10T/yzn/benchmark_GRN \
    --folders CHIP
fi

if [[ "$RUN_ATTENTION" == "1" ]]; then
  echo "[3/4] attention/*"
  for ds in hESC hHep mDC mHSC-E mHSC-GM mHSC-L; do
    python src/extract_model/attention/scCello.py \
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
  python src/GRN_inferance/dyn/scgpt_dyn.py \
    --model-dir "$SCGPT_MODEL_DIR" \
    --outdir "$OUTPUT_ROOT/dyn/scgpt" \
    --expr-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --pt-root /mnt/10T/yzn/benchmark_GRN/PseudoTime

  python src/GRN_inferance/dyn/scFoundation_dyn.py \
    --scfoundation-root "$SCFOUNDATION_ROOT" \
    --ckpt-path "$SCFOUNDATION_MODEL_PATH" \
    --gene-index-tsv "$SCFOUNDATION_VOCAB_PATH" \
    --outdir "$OUTPUT_ROOT/dyn/scfoundation" \
    --expr-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --pt-root /mnt/10T/yzn/benchmark_GRN/PseudoTime
fi

echo
echo "Done. Outputs under: $OUTPUT_ROOT"
