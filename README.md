## scGRN-Bench

scGRN-Bench 是一个用于 **单细胞基础模型（foundation model）表征 → GRN 推断 → 基准评测 → 论文图表复现** 的代码与数据组织目录。

本目录当前包含三部分：

- **`src/`**：从不同基础模型中抽取表征，并构建 GRN 预测边（`Gene1`, `Gene2`, `EdgeWeight`）。
- **`FBEval/`**：对预测网络进行基准评测（AUPR / EPR）。
- **`FBplot/`**：复现论文 Figure 2/3/4 的绘图脚本与产物（PDF/PNG/CSV 等）。

> 重要说明  
> 如果你要在另一台机器复现，请按下文 “路径与数据组织” 修改为你自己的路径，或在运行前用环境变量/软链接把数据放到对应位置。

---

## 环境安装

建议使用独立 conda 环境（Python 3.9+）。

1) 安装通用依赖（评测 + 绘图必需）：

```bash
cd /mnt/10T/yzn/scGRN-Bench
python -m pip install -r requirements.txt
```

2) 深度模型相关依赖（可选）：

- `src/` 中抽取 scGPT / scFoundation / Geneformer / Langcell / scCello 等模型表征时，可能需要额外安装对应模型代码与权重（例如 `scgpt`、`torch`、`scanpy` 等），并准备好模型权重目录。
- 这部分依赖在不同模型间差异较大，建议按你论文实验实际用到的模型逐个补齐环境。

### scPRINT / scprint（可选）

`src/extract_model/attention/scPRINT.py` 依赖 `scprint`（Python 包名为 `scprint`），并且会额外用到 `mygene` 做基因 symbol→Ensembl 的映射。

```bash
python -m pip install scprint mygene
```

### singlecell（本地 tar.gz，可选）
 `singlecell`（路径：`https://zenodo.org/records/17062099`），可以用下面方式安装：

```bash
python -m pip install "/mnt/10T/yzn/singlecell.tar.gz"
```

---

## 数据下载与准备

本项目用到的公开数据来源（来自 `data/readme.txt`）：

- **OmniPath**：`omnipath.parquet`  
  来源：`https://github.com/cantinilab/scPRINT/tree/main/data/main`
- **GroundTruth (CHIP / NonCHIP / STRING)**：BEELINE Networks  
  来源：`https://zenodo.org/records/3701939`（文件：`BEELINE-Networks.zip`）
- **scRNA-Seq 表达矩阵**：BEELINE data  
  来源：`https://zenodo.org/records/3701939`（文件：`BEELINE-data.zip`）
- **process.py**：BEELINE 数据预处理脚本  
  来源：`https://github.com/murali-group/BEELINE`

建议的本地组织方式（和评测脚本默认假设一致）：

```text
scGRN-Bench/data/
├── Groundtruth/
│   ├── CHIP/
│   │   ├── hESC_chip_matched-network.csv
│   │   ├── hHep_chip_matched-network.csv
│   │   └── ...
│   ├── Non_CHIP/
│   │   ├── hESC_processed-network.csv
│   │   └── ...
│   └── STRING/
│       ├── hESC_processed-network.csv
│       └── ...
└── row_data/                # 原始数据（若你需要从头预处理）
```

---

## 预测文件（GRN edges）格式要求

评测脚本要求 **预测网络** 为 TSV/CSV，至少包含以下列（大小写不敏感，但推荐统一）：

- `Gene1`：调控源（TF）
- `Gene2`：靶基因
- `EdgeWeight`：边权重（数值，评测时会取绝对值）

推荐的预测文件目录结构（来自 `FBEval/readme.txt`）：

```text
/path/to/predictions/
├── scgpt_hidden/
│   ├── scGPT_hESC.tsv
│   ├── scGPT_hHep.tsv
│   └── ...
├── Geneformer_hidden/
│   └── Geneformer_hESC.tsv
└── ...
```

---

## 评测（AUPR / EPR）

评测脚本在 `scGRN-Bench/FBEval/`：

- `AUPR.py`：批量计算 AUPR 与 AUPR Ratio（支持多模型 × 多数据集 × 多 GT 类型）
- `EPR.py`：批量计算 Early Precision Ratio（EPR）

### AUPR

```bash
cd /mnt/10T/yzn/scGRN-Bench/FBEval
python AUPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/groundtruth \
  --output results/aupr_results.csv
```

### EPR

```bash
cd /mnt/10T/yzn/scGRN-Bench/FBEval
python EPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/groundtruth \
  --output results/epr_results.csv
```

### GroundTruth 路径规则（评测脚本内置）

- `CHIP`：`${true_root}/CHIP/{dataset}_chip_matched-network.csv`
- `Non_CHIP` / `STRING`：`${true_root}/{gt_type}/{dataset}_processed-network.csv`

默认评测的数据集列表：

- `hESC`, `hHep`, `mDC`, `mHSC-E`, `mHSC-GM`, `mHSC-L`

---

## 绘图（Figure 2/3/4）

绘图脚本位于 `scGRN-Bench/FBplot/fig2`、`scGRN-Bench/FBplot/fig3`、`scGRN-Bench/FBplot/fig4`。

这些脚本通常会读取：

- `FBEval/` 产生的结果 CSV（或中间统计 CSV）
- `FBplot/**/output/`、`FBplot/**/network/`、`FBplot/**/curve/` 等目录下的中间产物

由于不同图脚本依赖的输入文件名和路径不完全一致，建议按论文复现顺序逐个运行，并在脚本顶部配置你自己的数据路径/输出路径。

---

## 路径与数据组织（重要）

当前 `src/` 中多处脚本使用类似下面的硬编码路径（示例）：

- 输入表达矩阵：`/mnt/10T/yzn/benchmark_GRN/input_process1000/...`
- 模型权重：`/mnt/10T/yzn/benchmark_GRN/model/weights/...`
- 输出预测边：`/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_.../...`

如果你要在其他机器复现，建议采用以下任一方式：

- **方式 A（推荐）**：在脚本中把 `INPUT_ROOT` / `OUTPUT_ROOT` / `MODEL_DIR` 改成相对路径（相对 `scGRN-Bench/`）或命令行参数。
- **方式 B**：在你的机器上创建与上述路径一致的目录结构（或用软链接映射）。

---

## 复现建议（最小可复现链路）

如果你只需要从“预测边文件”开始复现论文评测与图表，最小链路是：

1. 准备 `pred_root/`（预测边文件，按上面的格式与命名组织）
2. 准备 `true_root/`（GroundTruth：CHIP/Non_CHIP/STRING）
3. 跑评测：
   - `python scGRN-Bench/FBEval/AUPR.py ...`
   - `python scGRN-Bench/FBEval/EPR.py ...`
4. 按需运行 `FBplot/fig2` / `fig3` / `fig4` 的绘图脚本生成最终 PDF/PNG。

---

## 可直接运行的主入口（已去硬编码）

下面这些脚本已经改为路径参数驱动，不再依赖固定 `/mnt/...`：

- `src/GRN_inferance/run_unified_multidataset_pseudotime.py`
- `src/GRN_inferance/dyn/scgpt_dyn.py`
- `src/GRN_inferance/dyn/scFoundation_dyn.py`
- `src/extract_model/embedding/*.py`
- `src/extract_model/hidden_emb/scgpt_hidden.py`
- `src/extract_model/hidden_emb/scfoundation_hidden.py`
- `src/extract_model/hidden_emb/Langcell_hidden.py`
- `src/extract_model/hidden_emb/sccello_hidden.py`
- `src/extract_model/hidden_emb/Genfoemer_hidden.py`
- `src/extract_model/attention/scPRINT.py`
- `src/extract_model/attention/scCello.py`

### 统一 pseudotime 主入口示例

```bash
python src/GRN_inferance/run_unified_multidataset_pseudotime.py \
  --model scgpt \
  --outdir results_unified \
  --expr-root /path/to/input_process \
  --pt-root /path/to/PseudoTime \
  --scgpt-model-dir /path/to/scgpt_model
```

说明：

- `--datasets-json` 与 `--expr-root + --pt-root` 二选一。
- 如果用 `--datasets-json`，格式为：
  - 顶层 key 是数据集名（如 `hESC`）
  - 每个值包含 `expr_csv`、`pt_csv`、`species`

### 一键批量跑 CHIP（推荐）

项目已提供批量脚本：

```bash
bash scripts/run_extract_model_chip_batch.sh
```

默认会运行：

- `src/extract_model/embedding/*`（5 个）
- `src/extract_model/hidden_emb/*`（5 个）

默认不会运行（可选开启）：

- `attention`（依赖更重，且更耗时）
- `dyn`（pseudotime 动态脚本）

可选环境变量（覆盖默认路径）：

```bash
INPUT_ROOT=/path/to/input_process1000 \
OUTPUT_ROOT=/path/to/output_root \
MODEL_WEIGHTS_ROOT=/path/to/model/weights \
SCGPT_MODEL_DIR=/path/to/scgpt_human \
SCFOUNDATION_ROOT=/path/to/scFoundation-main \
RUN_ATTENTION=1 \
RUN_DYNAMIC=1 \
bash scripts/run_extract_model_chip_batch.sh
```

脚本默认输出目录：`/mnt/10T/yzn/scGRN-Bench/outputs`

---

scFM_bench:https://zenodo.org/records/17062099
scGPT:https://github.com/bowang-lab/scGPT
scFoundation:https://github.com/biomap-research/scFoundation
Geneformer:https://huggingface.co/ctheodoris/Geneformer
Langcell:https://github.com/PharMolix/LangCell
scPRINGT:https://github.com/cantinilab/scPRINT
scCello:











## 许可与引用

- 数据集与 GroundTruth 的许可与引用请遵循其原始来源（BEELINE / OmniPath 等）。
- 论文投稿时，建议在补充材料中明确：预测边文件格式、评测命令、以及对应的 GroundTruth 版本与下载链接。

