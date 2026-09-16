#!/usr/bin/env bash
# Remaining STRING noleak sweeps (hESC/hHep already done).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/mnt/10T/yzn/anconda3/envs/singlecell/bin/python}"
MODEL_DIR="${SCGPT_MODEL_DIR:-/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human}"
REPO_DIR="${SCGPT_REPO_DIR:-/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT}"
OUTDIR="${OUTDIR:-outputs/gene_count_sweep_noleak}"
NETWORK="${NETWORK:-string}"
ORDERS="${ORDERS:-net_degree,net_degree_asc,net_random}"
N_NULL="${N_NULL:-10}"
DATASETS="${DATASETS:-mDC mHSC-E mHSC-GM mHSC-L}"

echo "[batch] network=$NETWORK datasets=$DATASETS"

for ds in $DATASETS; do
  csv="$OUTDIR/$ds/$NETWORK/noleak_sweep.csv"
  if [[ -f "$csv" ]]; then
    have=$(cut -d, -f1 "$csv" | sort -u | tr '\n' ' ')
    need_run=0
    for o in ${ORDERS//,/ }; do
      echo "$have" | grep -qw "$o" || need_run=1
    done
    echo "$have" | grep -qw baseline_all || need_run=1
    if [[ $need_run -eq 0 ]]; then
      echo "[skip] $ds/$NETWORK already complete"
      continue
    fi
  fi
  echo "===== $ds / $NETWORK ====="
  PYTHONUNBUFFERED=1 "$PY" run_c1_gene_count_sweep_noleak.py \
    --dataset "$ds" \
    --network "$NETWORK" \
    --outdir "$OUTDIR" \
    --orders "$ORDERS" \
    --n-null "$N_NULL" \
    --scgpt-model-dir "$MODEL_DIR" \
    --scgpt-repo-dir "$REPO_DIR" \
    --device cuda \
    --gen-iters 16 \
    --batch-size 64 \
    --max-early-cells 64 \
    --seed 0
done

"$PY" plot_noleak_six_datasets.py --network "$NETWORK" --outdir "$OUTDIR"
echo "Done."
