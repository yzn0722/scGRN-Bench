
# 文件结构

预测文件结构:
/path/to/predictions/
├── scgpt_hidden/
│   ├── scGPT_hESC.tsv
│   ├── scGPT_hHep.tsv
│   └── ...
├── Geneformer_hidden/
│   └── Geneformer_hESC.tsv
└── ...

真实文件结构:
/path/to/groundtruth/
├── CHIP/
│   ├── hESC_chip_matched-network.csv
│   ├── hHep_chip_matched-network.csv
│   └── ...
├── Non_CHIP/
│   └── hESC_processed-network.csv
└── STRING/
    └── hESC_processed-network.csv



# 评估代码运行方式
使用默认参数
python AUPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/groundtruth \
  --output results/aupr_results.csv



使用默认参数
python EPR.py \
  --pred_root /path/to/predictions \
  --true_root /path/to/groundtruth \
  --output results/epr_results.csv