omnipath.parquet       download: https://github.com/cantinilab/scPRINT/tree/main/data/main
GroundTruth (CHIP/Non_CHIP/STRING) download: https://zenodo.org/records/3701939   BEELINE-Networks.zip
scRNA-Seq download: https://zenodo.org/records/3701939   BEELINE-data.zip
process.py download: https://github.com/murali-group/BEELINE

singlecell (conda-packed environment and related assets) source:
- scFM_bench (Zenodo): https://zenodo.org/records/17062099
- Environment install steps: see ../README.md → "Environment Setup"
  (create conda env + pip install -r requirements.txt, OR unpack singlecell.tar.gz)

One-click download (places files under this data/ tree):

    cd scGRN-Bench
    python scripts/download_data.py

Useful options:

    python scripts/download_data.py --force              # overwrite existing files
    python scripts/download_data.py --skip-scrna          # skip ~250MB BEELINE-data.zip
    python scripts/download_data.py --cache-dir /tmp/dl   # keep downloaded zips

After download, layout is:

    data/
    ├── row_data/
    │   ├── omnipath.parquet
    │   └── scRNA-Seq/{dataset}/ExpressionData.csv | GeneOrdering.csv | PseudoTime.csv
    ├── Groundtruth/
    │   ├── CHIP/          # ChIP-seq (+ mESC-lofgof) networks from BEELINE-Networks.zip
    │   ├── NonCHIP/       # Non-specific ChIP networks
    │   └── STRING/        # human_STRING-network.csv, mouse_STRING-network.csv
    ├── PseudoTime/{dataset}/PseudoTime.csv
    └── process_data/
        ├── process.py     # BEELINE generateExpInputs.py
        ├── human-tfs.csv
        └── mouse-tfs.csv

Notes:
- This repository does not download or commit large files by default.
- Place the downloaded files under `scGRN-Bench/data/` following the layout
  described in the main `README.md`.
- Processed inputs used by most runners (`data/input_process/`,
  `data/input_process1000/` with CHIP / Non_CHIP / STRING) are generated from
  the raw scRNA-Seq + Groundtruth files via BEELINE-style preprocessing
  (`data/process_data/process.py`), not by the download script.
