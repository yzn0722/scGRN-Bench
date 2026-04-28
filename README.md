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

Recommended Python version: 3.9 or newer.

Install the base dependencies:

```bash
cd scGRN-Bench
python -m pip install -r requirements.txt
```

Some models require additional optional packages such as `torch`, `scanpy`, `scgpt`, `scprint`, and `mygene`.

Optional packages used by some scripts:

```bash
python -m pip install scprint mygene
python -m pip install /path/to/singlecell.tar.gz
```

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

