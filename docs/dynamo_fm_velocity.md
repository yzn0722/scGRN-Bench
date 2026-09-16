# 用大模型在 Dynamo 数据集上测「速度/方向」

## 思路

| 角色 | 来源 | 含义 |
|------|------|------|
| **金标准 A** | early/late 表达差分 `true_delta` | scGRN-Bench 默认（与 BEELINE 一致） |
| **金标准 B** | Dynamo `velocity_S` 等 | RNA velocity 方向（需先跑 Dynamo） |
| **预测** | scGPT / scFoundation 等 | 迭代生成后的 `pred_delta` |
| **指标** | direction accuracy | Top30% 动态基因上符号一致率 |

## 推荐数据集

| Dynamo 数据 | 大模型适配 | 说明 |
|-------------|------------|------|
| **hematopoiesis** | scGPT / scFoundation 最好 | 人源，与论文一致 |
| **zebrafish** | 需 `--gene-transform upper`，匹配率低 | 练 Dynamo；FM 仅作探索 |
| **zebrafish_processed** | 加 `--velocity-layer velocity_S` | 可和 Dynamo velocity 比 |

## 命令

### 1. 仅 zebrafish（用 umap_1 代理伪时间）

```bash
cd /mnt/10T/yzn/scGRN-Bench
bash scripts/run_dynamo_fm_velocity.sh zebrafish
```

### 2. Dynamo 跑完后（真实 velocity + 伪时间）

```bash
# 先完成 dynamo-release/run_zebrafish.py，得到 zebrafish_processed.h5ad

export VELOCITY_LAYER=velocity_S
export PT_SOURCE=latent_time   # 或 palantir_pseudotime，视 adata.obs 列而定
bash scripts/run_dynamo_fm_velocity.sh zebrafish_dyn \
  /mnt/10T/yzn/dynamo-release/results/zebrafish/zebrafish_processed.h5ad
```

### 3. 多模型（统一入口）

```bash
python src/GRN_inferance/run_unified_multidataset_pseudotime.py \
  --model scgpt \
  --datasets-json data/dynamo_export/datasets_dynamo.json \
  --scgpt-model-dir /mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human \
  --outdir outputs/dynamo_fm_velocity/unified_scgpt
```

### 4. FM 预测 vs Dynamo velocity

```bash
python src/GRN_inferance/dyn/evaluate_fm_vs_dynamo_velocity.py \
  --dynamo-velocity-csv data/dynamo_export/zebrafish_dyn/dynamo_velocity_truth.csv \
  --fm-gene-delta outputs/dynamo_fm_velocity/scgpt_zebrafish_dyn/gene_delta_compare.csv \
  --out-json outputs/dynamo_fm_velocity/scgpt_zebrafish_dyn/vs_dynamo_velocity.json
```

## 文件结构

```text
data/dynamo_export/
├── datasets_dynamo.json
├── zebrafish/
│   ├── CHIP/zebrafish_chip_matched-ExpressionData.csv
│   ├── PseudoTime.csv
│   └── dynamo_velocity_truth.csv   # 可选
└── zebrafish_dyn/
    └── ...
```
