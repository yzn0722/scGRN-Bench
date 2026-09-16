#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TF family annotation: curated table + symbol-prefix rules (AnimalTFDB-style)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from tf_static.utils import norm_gene

DEFAULT_FAMILY_CSV = Path(__file__).resolve().parent / "data" / "human_tf_family_curated.csv"

# Order matters: first match wins
PREFIX_RULES: tuple[tuple[str, str], ...] = (
    (r"^HOX[A-Z]?", "Homeobox"),
    (r"^PAX\d", "Homeobox"),
    (r"^NKX", "Homeobox"),
    (r"^FOX[A-Z]", "Forkhead"),
    (r"^SOX\d", "HMG (SOX)"),
    (r"^TCF\d", "HMG (TCF/LEF)"),
    (r"^LEF\d", "HMG (TCF/LEF)"),
    (r"^STAT\d", "STAT"),
    (r"^IRF\d", "IRF"),
    (r"^GATA\d", "GATA"),
    (r"^ETS\d", "ETS"),
    (r"^ELF\d", "ETS"),
    (r"^ELK\d", "ETS"),
    (r"^FLI\d", "ETS"),
    (r"^ERG$", "ETS"),
    (r"^ETV\d", "ETS"),
    (r"^JUN", "bZIP"),
    (r"^FOS", "bZIP"),
    (r"^ATF\d", "bZIP"),
    (r"^CREB\d", "bZIP"),
    (r"^CEBP", "bZIP"),
    (r"^DDIT", "bZIP"),
    (r"^MYC$", "bHLH"),
    (r"^MAX$", "bHLH"),
    (r"^MITF$", "bHLH"),
    (r"^TFE\d", "bHLH"),
    (r"^TFEB$", "bHLH"),
    (r"^TFE3$", "bHLH"),
    (r"^TWIST", "bHLH"),
    (r"^HAND\d", "bHLH"),
    (r"^ASCL\d", "bHLH"),
    (r"^OLIG\d", "bHLH"),
    (r"^NEUROD", "bHLH"),
    (r"^BHLH", "bHLH"),
    (r"^ZNF\d", "C2H2 zinc finger"),
    (r"^ZBTB", "C2H2 zinc finger"),
    (r"^KLF\d", "C2H2 zinc finger"),
    (r"^SP\d$", "C2H2 zinc finger"),
    (r"^WT\d$", "C2H2 zinc finger"),
    (r"^EGR\d", "C2H2 zinc finger"),
    (r"^NR\d[A-Z]", "Nuclear receptor"),
    (r"^ESR\d", "Nuclear receptor"),
    (r"^AR$", "Nuclear receptor"),
    (r"^RARA?$", "Nuclear receptor"),
    (r"^RARB$", "Nuclear receptor"),
    (r"^RARG$", "Nuclear receptor"),
    (r"^PPAR", "Nuclear receptor"),
    (r"^RXR", "Nuclear receptor"),
    (r"^VDR$", "Nuclear receptor"),
    (r"^THR", "Nuclear receptor"),
    (r"^RFX\d", "RFX"),
    (r"^TBX\d", "T-box"),
    (r"^MEF2", "MADS"),
    (r"^SRF$", "MADS"),
    (r"^POLR2", "General transcription machinery"),
    (r"^POLR1", "General transcription machinery"),
    (r"^TBP$", "General transcription machinery"),
    (r"^GTF", "General transcription machinery"),
    (r"^SUPT", "General transcription machinery"),
    (r"^TP53$", "p53"),
    (r"^TP63$", "p53"),
    (r"^TP73$", "p53"),
    (r"^E2F\d", "E2F"),
    (r"^RB1$", "E2F"),
    (r"^SMAD\d", "SMAD"),
    (r"^NFKB", "Rel"),
    (r"^RELA$", "Rel"),
    (r"^RELB$", "Rel"),
    (r"^SNAI\d", "Zinc finger (SNAIL)"),
    (r"^ZEB\d", "Zinc finger (ZEB)"),
    (r"^PRDM", "PRDM"),
    (r"^RUNX", "Runt"),
    (r"^CBF", "Runt"),
)


def infer_family_from_symbol(symbol: str) -> str:
    s = norm_gene(symbol)
    if not s:
        return "Other"
    for pat, fam in PREFIX_RULES:
        if re.match(pat, s):
            return fam
    return "Other"


def load_family_table(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DEFAULT_FAMILY_CSV
    if path.is_file() and path.stat().st_size > 20:
        try:
            df = pd.read_csv(path, comment="#")
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["gene", "family"])
        if "gene" in df.columns and "family" in df.columns and len(df) > 0:
            df["gene"] = df["gene"].map(norm_gene)
            df["family"] = df["family"].astype(str)
            return df.dropna(subset=["gene"])
    return pd.DataFrame(columns=["gene", "family"])


def annotate_tf_families(
    genes: list[str],
    curated_csv: Optional[Path] = None,
) -> pd.DataFrame:
    curated = load_family_table(curated_csv)
    override: Dict[str, str] = {}
    if not curated.empty:
        override = dict(zip(curated["gene"], curated["family"]))

    rows = []
    for g in sorted({norm_gene(x) for x in genes if norm_gene(x)}):
        fam = override.get(g) or infer_family_from_symbol(g)
        rows.append({"TF": g, "family": fam})
    return pd.DataFrame(rows)
