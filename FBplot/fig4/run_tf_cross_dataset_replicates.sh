#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/10T/yzn/scGRN-Bench"
python_bin="/mnt/10T/yzn/anconda3/envs/singlecell/bin/python"
runner="$repo_root/FBplot/fig4/run_dynamic_grn_validation.py"
output_root="$repo_root/outputs/tf_cross_dataset_replicates"
model_root="/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT"
datasets=(hHep mDC mHSC-E mHSC-GM mHSC-L)

mkdir -p "$output_root/logs"
cd "$repo_root"

for dataset in "${datasets[@]}"; do
    for seed in 0 1 2 3 4 5 6; do
        run_root="$output_root/$dataset/seed$seed"
        result="$run_root/$dataset/dynamic_grn_validation.json"
        log="$output_root/logs/${dataset}_seed${seed}.log"
        if [[ -s "$result" ]]; then
            echo "[$(date --iso-8601=seconds)] skipping completed $dataset seed=$seed"
            continue
        fi
        echo "[$(date --iso-8601=seconds)] starting $dataset seed=$seed" | tee "$log"
        "$python_bin" "$runner" \
            --dataset "$dataset" \
            --outdir "$run_root" \
            --expr-root /mnt/10T/yzn/benchmark_GRN/input_process \
            --pt-root /mnt/10T/yzn/benchmark_GRN/PseudoTime \
            --scgpt-model-dir "$model_root/scgpt_human" \
            --scgpt-repo-dir "$model_root" \
            --device cuda \
            --execute \
            --gen-iters 2 \
            --max-cells 64 \
            --batch-size 4 \
            --n-rewired 1 \
            --n-tf-probes 34 \
            --probe-iters 4 \
            --n-random-target-sets 1000 \
            --max-forward-passes 5000 \
            --memory-limit-gb 4 \
            --cpu-threads 2 \
            --seed "$seed" 2>&1 | tee -a "$log"
        echo "[$(date --iso-8601=seconds)] completed $dataset seed=$seed" | tee -a "$log"
    done
done

echo "[$(date --iso-8601=seconds)] all replicate runs completed"
