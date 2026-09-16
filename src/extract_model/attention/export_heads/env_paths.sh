# 用法: source /mnt/10T/yzn/scGRN-Bench/src/extract_model/attention/export_heads/env_paths.sh
# 之后可直接运行 run_export_heads.sh 或下面的 python 命令（无需 cd 到项目根）

export PROJECT_ROOT="/mnt/10T/yzn/scGRN-Bench"
export WEIGHTS_ROOT="${WEIGHTS_ROOT:-$PROJECT_ROOT/models/weights}"

# 表达矩阵 / CHIP 网络（benchmark_GRN；若你有 data/input_process 可改 INPUT_ROOT）
export INPUT_ROOT="${INPUT_ROOT:-/mnt/10T/yzn/benchmark_GRN/input_process}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/attention_heads}"

# scGPT：权重在 models/weights，源码在 benchmark_GRN/pre_scgpt
export SCGPT_REPO="${SCGPT_REPO:-/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT}"
export SCGPT_MODEL="${SCGPT_MODEL:-$WEIGHTS_ROOT/scgpt/scGPT_human}"

# Geneformer / LangCell / scCello 权重均在 WEIGHTS_ROOT 下
export GENEFORMER_ROOT="${GENEFORMER_ROOT:-$WEIGHTS_ROOT}"
export LANGCELL_PARENT="${LANGCELL_PARENT:-$WEIGHTS_ROOT}"
export SCCELLO_MODEL="${SCCELLO_MODEL:-$WEIGHTS_ROOT/scCello}"
export SCCELLO_DICT="${SCCELLO_DICT:-$WEIGHTS_ROOT/Geneformer/dicts}"
# scCello 推理代码（含 sccello/src/model_prototype_contrastive.py）
export SC_FOUNDATION_EVALS="/mnt/10T/yzn/scGRN-Bench/models/sc_foundation_evals"
export SCCELLO_REPO="${SCCELLO_REPO:-$SC_FOUNDATION_EVALS}"

# scPRINT（可选；varp['GRN'] 第三维 = nlayers * nhead 个 slice）
export SCPRINT_CKPT="${SCPRINT_CKPT:-$WEIGHTS_ROOT/1lnm8pgh_geneslist_9606.ckpt}"
export SCPRINT_TOKEN_PKL="${SCPRINT_TOKEN_PKL:-/mnt/10T/yzn/scPRINT/data/main/token_dictionary.pkl}"
export SCPRINT_NLAYERS="${SCPRINT_NLAYERS:-8}"
export SCPRINT_NHEAD="${SCPRINT_NHEAD:-4}"
# 1 = 只导出最后一层 4 个 slice（默认 ckpt 下为 28,29,30,31，可与 scCello 4 head 对比）
export SCPRINT_LAST_LAYER_ONLY="${SCPRINT_LAST_LAYER_ONLY:-0}"

scprint_last_layer_indices() {
  local nl="${1:-$SCPRINT_NLAYERS}"
  local nh="${2:-$SCPRINT_NHEAD}"
  local start=$(( (nl - 1) * nh ))
  local i
  local out=""
  for ((i = 0; i < nh; i++)); do
    [[ -n "$out" ]] && out+=","
    out+=$((start + i))
  done
  echo "$out"
}

# 统一入口（绝对路径，任意 cwd 均可）
export EXPORT_HEADS_UNIFIED="${EXPORT_HEADS_UNIFIED:-$PROJECT_ROOT/src/extract_model/attention/export_heads/export_heads_unified.py}"
