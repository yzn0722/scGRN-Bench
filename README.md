## scGRN-Bench

scGRN-Bench is a benchmark codebase for:

1. extracting representations from single-cell foundation models,
2. generating GRN edge files with columns `Gene1`, `Gene2`, and `EdgeWeight`,
3. evaluating predictions with AUPR and EPR,
4. reproducing figure panels used in analysis.

## Repository Structure

- `src/`: extraction, inference, and model-running scripts
- `FBEval/`: evaluation scripts
- `FBplot/`: plotting scripts
- `scripts/`: helper shell scripts
- `data/`: default local data root used by the batch runner
- `outputs/`: default output directory

## Environment Setup

Recommended Python: **3.9–3.11** (3.9+). GPU (CUDA) is optional but recommended for foundation-model scripts.

### Option A — Create a fresh conda / venv (recommended for reviewers)

```bash
# 1) Create and activate an environment
conda create -n scgrn-bench python=3.10 -y
conda activate scgrn-bench

# 2) Enter the repo
cd scGRN-Bench

# 3) Install base Python packages
python -m pip install -U pip
python -m pip install -r requirements.txt

# 4) (Recommended if you run model extraction / dyn scripts) PyTorch + scanpy
#    Pick the CUDA build that matches your driver from https://pytorch.org
python -m pip install torch torchvision torchaudio
python -m pip install scanpy

# 5) (Optional) model-specific packages — install only what you need
python -m pip install scgpt          # scGPT scripts
# Geneformer / LangCell / scCello usually need transformers + their local checkpoints
# scPRINT is already listed in requirements.txt (scprint, mygene)
```

Minimal check (no GPU / no model weights required):

```bash
python -c "import numpy, pandas, sklearn, networkx, yaml; print('base OK')"
python scripts/download_data.py --skip-scrna   # small subset; omit --skip-scrna for full data
```

### Option B — Unpack a conda-packed `singlecell` environment

If you already have a conda-pack tarball `singlecell.tar.gz` (used by some scFM-Bench / local setups):

```bash
# Example: unpack into your conda envs directory
mkdir -p "$HOME/miniconda3/envs/singlecell"
tar -xzf /path/to/singlecell.tar.gz -C "$HOME/miniconda3/envs/singlecell"
conda activate singlecell

# First activation after unpack often needs:
conda-unpack   # if available inside the packed env

cd scGRN-Bench
python -c "import torch, scanpy, pandas; print('singlecell env OK')"
```

Do **not** `pip install singlecell.tar.gz` — it is a packed conda environment, not a Python wheel.

Related external assets:

- scFM-Bench / related Zenodo notes: https://zenodo.org/records/17062099
- Model weights / datasets used by related scFM work: https://zenodo.org/records/17054467

### Download benchmark data

```bash
cd scGRN-Bench
python scripts/download_data.py
# or: python scripts/download_data.py --skip-scrna   # skip ~250MB BEELINE-data.zip
```

This places raw files under `data/` (see `data/readme.txt`). Processed matrices under `data/input_process/` are produced by BEELINE-style preprocessing (`data/process_data/process.py`), not by the download script.

### Model checkpoints (large files, optional)

Place checkpoints under `data/` so default batch scripts can find them:

```text
data/
├── model_weights/          # Geneformer / LangCell / scCello (+ dicts)
├── scgpt/scgpt_human/      # scGPT checkpoint + vocab
└── scfoundation/           # scFoundation checkpoint + vocab
```

Sources (examples): [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [scGPT](https://github.com/bowang-lab/scGPT), [scFoundation](https://github.com/biomap-research/scFoundation), [LangCell](https://github.com/PharMolix/LangCell), [scPRINT](https://github.com/cantinilab/scPRINT), Zenodo weights https://zenodo.org/records/17054467 .

## Default Data Layout

The batch runner assumes all local resources are placed under `data/` inside this repository.

```text
scGRN-Bench/
├── data/
│   ├── input_process1000/
│   │   └── CHIP/
│   │       ├── hESC_chip_matched-ExpressionData.csv
│   │       ├── hHep_chip_matched-ExpressionData.csv
│   │       └── ...
│   ├── input_process/
│   ├── PseudoTime/
│   ├── model_weights/
│   │   ├── Geneformer/
│   │   ├── LangCell/
│   │   └── scCello/
│   ├── scgpt/
│   │   └── scgpt_human/
│   ├── scfoundation/
│   └── sc_foundation_evals/
├── outputs/
└── scripts/
```

Expected pseudotime layout:

```text
data/PseudoTime/
├── hESC/PseudoTime.csv
├── hHep/PseudoTime.csv
└── ...
```

Expected ground-truth layout for evaluation:

```text
<ground_truth_root>/
├── CHIP/{dataset}_chip_matched-network.csv
├── Non_CHIP/{dataset}_processed-network.csv
└── STRING/{dataset}_processed-network.csv
```

## Prediction File Format

Prediction files should contain exactly these columns:

- `Gene1`
- `Gene2`
- `EdgeWeight`

TSV format is recommended.

## Main CLI Scripts

The following scripts are path-driven and do not require editing hardcoded machine-specific paths:

- `src/GRN_inferance/run_unified_multidataset_pseudotime.py`
- `src/GRN_inferance/dyn/scgpt_dyn.py`
- `src/GRN_inferance/dyn/scFoundation_dyn.py`
- `src/extract_model/embedding/*.py`
- `src/extract_model/hidden_emb/*.py`
- `src/extract_model/attention/scPRINT.py`
- `src/extract_model/attention/scCello.py`

## Quick Start

### Run the default CHIP batch pipeline

```bash
cd scGRN-Bench
bash scripts/run_extract_model_chip_batch.sh
```

By default, the batch runner:

- runs `embedding/*` and `hidden_emb/*`,
- writes outputs under `outputs/`,
- reads inputs from `data/`,
- does not run `attention/*` or `dyn/*` unless explicitly enabled.

Enable the optional sections:

```bash
RUN_ATTENTION=1 RUN_DYNAMIC=1 bash scripts/run_extract_model_chip_batch.sh
```

Override any default paths when needed:

```bash
DATA_ROOT=/path/to/data \
INPUT_ROOT=/path/to/input_process1000 \
OUTPUT_ROOT=/path/to/output_root \
MODEL_WEIGHTS_ROOT=/path/to/model_weights \
SCGPT_MODEL_DIR=/path/to/scgpt_human \
SCFOUNDATION_ROOT=/path/to/scfoundation \
bash scripts/run_extract_model_chip_batch.sh
```

## Run a Single Embedding Script

Example: run Geneformer embedding extraction on CHIP datasets and export `Gene1/Gene2/EdgeWeight` TSV files.

```bash
python -u src/extract_model/embedding/geneformer_all.py \
  --input-root data/input_process1000 \
  --output-root outputs/embedding/geneformer \
  --model-dir data/model_weights/Geneformer/default/12L \
  --dict-dir data/model_weights/Geneformer/dicts \
  --folders CHIP
```

Example: run scGPT embedding extraction.

```bash
python -u src/extract_model/embedding/scgpt_all.py \
  --input-root data/input_process1000 \
  --output-root outputs/embedding/scgpt \
  --model-dir data/scgpt/scgpt_human \
  --folders CHIP
```

## Unified Pseudotime Runner

Example:

```bash
python src/GRN_inferance/run_unified_multidataset_pseudotime.py \
  --model scgpt \
  --outdir outputs/results_unified \
  --expr-root data/input_process \
  --pt-root data/PseudoTime \
  --scgpt-model-dir data/scgpt/scgpt_human
```

Use either:

- `--datasets-json <file>`
- or `--expr-root` together with `--pt-root`

## Evaluation

Run AUPR:

```bash
cd FBEval
python AUPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/ground_truth \
  --output results/aupr_results.csv
```

Run EPR:

```bash
cd FBEval
python EPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/ground_truth \
  --output results/epr_results.csv
```

## Plotting

Plot scripts are located in:

- `FBplot/fig2`
- `FBplot/fig3`
- `FBplot/fig4`

These scripts read intermediate CSV and TSV outputs. Adjust their input paths as needed before running them.

## External References

- [scFM_bench](https://zenodo.org/records/17062099)
- [scGPT](https://github.com/bowang-lab/scGPT)
- [scFoundation](https://github.com/biomap-research/scFoundation)
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- [LangCell](https://github.com/PharMolix/LangCell)
- [scPRINT](https://github.com/cantinilab/scPRINT)

## Citation and Reproducibility Notes

- Follow the original licenses and citation requirements for external datasets, checkpoints, and model repositories.
- For paper submission or release, record the exact versions of model checkpoints and ground-truth files used in each experiment.

