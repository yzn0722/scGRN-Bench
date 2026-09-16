#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6 models × 3 scGPT-style extractions — prediction path registry (aligned with fig2)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

BENCH_ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
EVL_ROOT = BENCH_ROOT / "evl_omipath"
INPUT_ROOT = BENCH_ROOT / "input_process"

# ground truth 来源：子目录 + 文件名模板
GT_SOURCES: Dict[str, Tuple[str, str]] = {
    "STRING": ("STRING", "{dataset}_processed-network.csv"),
    "omnipath": ("omnipath", "{dataset}_processed-network.csv"),
    "Non_CHIP": ("Non_CHIP", "{dataset}_processed-network.csv"),
    "CHIP": ("CHIP", "{dataset}_chip_matched-network.csv"),
}

GT_DISPLAY: Dict[str, str] = {
    "STRING": "STRING",
    "omnipath": "OmniPath",
    "Non_CHIP": "Non-CHIP",
    "CHIP": "CHIP-seq",
}

MODELS: Sequence[str] = (
    "Geneformer",
    "LangCell",
    "scGPT",
    "scCello",
    "scFoundation",
    "scPrint",
)
EXTRACTIONS: Sequence[str] = ("emb500", "att500", "embhidden500")

MODEL_DIRS: Dict[str, Dict[str, str]] = {
    "Geneformer": {"emb500": "geneformer", "att500": "geneformer", "embhidden500": "geneformer"},
    "LangCell": {"emb500": "langcell", "att500": "langcell", "embhidden500": "Langcell"},
    "scGPT": {"emb500": "scgpt", "att500": "scgpt", "embhidden500": "scgpt"},
    "scCello": {"emb500": "sccello", "att500": "sccello", "embhidden500": "sccello"},
    "scFoundation": {"emb500": "scFoundation", "att500": "scFoundation", "embhidden500": "scFoundation"},
    "scPrint": {"emb500": "scprint", "att500": "scprint", "embhidden500": "scprint"},
}

MODEL_STEMS: Dict[str, Tuple[str, ...]] = {
    "Geneformer": ("geneformer", "Geneformer"),
    "LangCell": ("LangCell", "langcell", "Langcell"),
    "scGPT": ("scgpt", "scGPT", "scGPT2"),
    "scCello": ("scCello", "sccello"),
    "scFoundation": ("scFoundation", "scfoundation"),
    "scPrint": ("scprint", "scPrint", "scPRINT"),
}

WEIGHT_COLS = (
    "EdgeWeight",
    "edgeweight",
    "edge_weight",
    "Attention score",
    "Weight",
    "Score",
    "Importance",
)


def resolve_gt_path(
    gt_source: str,
    dataset: str,
    input_root: Path = INPUT_ROOT,
) -> Path:
    if gt_source not in GT_SOURCES:
        raise ValueError(f"Unknown gt_source {gt_source!r}, choose from {list(GT_SOURCES)}")
    subdir, pattern = GT_SOURCES[gt_source]
    path = input_root / subdir / pattern.format(dataset=dataset)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_pred_path(model: str, extraction: str, dataset: str, evl_root: Path = EVL_ROOT) -> Optional[Path]:
    if model not in MODEL_DIRS or extraction not in EXTRACTIONS:
        return None
    base = evl_root / f"output_{extraction}" / MODEL_DIRS[model][extraction]
    if not base.is_dir():
        return None
    for stem in MODEL_STEMS[model]:
        cand = base / f"{stem}_{dataset}.tsv"
        if cand.is_file():
            return cand
    for p in sorted(base.glob("*.tsv")):
        if dataset.lower() in p.name.lower():
            return p
    return None
