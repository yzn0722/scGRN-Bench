FBEval evaluates predicted GRN edge files against reference networks.

Prediction directory example:

/path/to/predictions/
├── scgpt_hidden/
│   ├── scGPT_hESC.tsv
│   ├── scGPT_hHep.tsv
│   └── ...
├── geneformer_hidden/
│   ├── Geneformer_hESC.tsv
│   └── ...
└── ...

Ground-truth directory example:

/path/to/ground_truth/
├── CHIP/
│   ├── hESC_chip_matched-network.csv
│   ├── hHep_chip_matched-network.csv
│   └── ...
├── Non_CHIP/
│   └── hESC_processed-network.csv
└── STRING/
    └── hESC_processed-network.csv

Prediction files should contain the following columns:

- Gene1
- Gene2
- EdgeWeight

Run AUPR evaluation:

python AUPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/ground_truth \
  --output results/aupr_results.csv

Run EPR evaluation:

python EPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/ground_truth \
  --output results/epr_results.csv