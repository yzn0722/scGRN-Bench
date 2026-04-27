#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified color palette for FoundBench paper figures (fig2).

This palette follows the user's provided pastel 7-color scheme:
  #8FB4DC #FFDD8E #70CDBE #AC99D2 #7AC3DF #F5AA61 #EB7E60
These are used as fixed, publication-ready colors for all plots under fig2.

Keep model-to-color mapping stable across all plots.
"""

from __future__ import annotations

from typing import Dict


# --- Base palette (given; 7 colors) ---
PALETTE: Dict[str, str] = {
    # models (fixed)
    "scGPT": "#8FB4DC",
    "Geneformer": "#FFDD8E",
    "GENIE3": "#70CDBE",
    "LangCell": "#AC99D2",
    "scCello": "#7AC3DF",
    "scFoundation": "#F5AA61",
    "scPrint": "#EB7E60",
    # Reference / axes (separate neutral so it doesn't steal a model color)
    "STRING": "#6E6E6E",
}


def model_color(name: str, default: str = "#808080") -> str:
    """Return fixed color for a model name (case-insensitive, tolerant to aliases)."""
    key = str(name).strip()
    if not key:
        return default

    aliases = {
        "langcell": "LangCell",
        "langcelll": "LangCell",
        "sccello": "scCello",
        "sccelo": "scCello",
        "scprint": "scPrint",
        "scfoundation": "scFoundation",
        "scgpt": "scGPT",
        "string": "STRING",
    }
    lk = key.lower()
    if lk in aliases:
        key = aliases[lk]
    return PALETTE.get(key, default)


MODEL_COLORS: Dict[str, str] = {
    # Fixed mapping (use only the provided scheme colors; no gray fallback)
    "scFoundation": model_color("scFoundation"),
    "LangCell": model_color("LangCell"),
    "scCello": model_color("scCello"),
    "Geneformer": model_color("Geneformer"),
    "GENIE3": model_color("GENIE3"),
    "scGPT": model_color("scGPT"),
    "scPrint": model_color("scPrint"),
    "STRING": model_color("STRING"),
}

