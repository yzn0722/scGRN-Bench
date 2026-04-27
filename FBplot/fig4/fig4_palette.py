#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified model-color mapping for FoundBench fig4.
Keep consistent with fig2/fig3 palette used in this session.
"""

from __future__ import annotations

from typing import Dict
import matplotlib as mpl


MODEL_COLORS: Dict[str, str] = {
    "hvg": "#8A9A9B",
    "scGPT": "#4EA3F1",
    "Geneformer": "#FFD447",
    "uce": "#70CDBE",
    "genecompass": "#7AC3DF",
    "GENIE3": "#70CDBE",
    "LangCell": "#AC99D2",
    "scCello": "#7AC3DF",
    "scFoundation": "#FF9A3D",
    "scPRINT": "#EB7E60",
    "scPrint": "#EB7E60",
    "STRING": "#6E6E6E",
}

FIG4_FONT_FAMILY = "DejaVu Sans"
FIG4_FIGSIZE = (5.0, 5.0)
FIG4_DPI = 600

# scGPT 3-extraction display names (keep consistent with fig2/fig3)
METHOD_DISPLAY: Dict[str, str] = {
    "emb500": r"COS$_{tok}$",
    "embhidden500": r"COS$_{hid}$",
    "att500": "attn",
}


def model_color(name: str, default: str = "#808080") -> str:
    key = str(name).strip()
    if not key:
        return default
    aliases = {
        "langcell": "LangCell",
        "sccello": "scCello",
        "scfoundation": "scFoundation",
        "scprint": "scPrint",
        "scgpt": "scGPT",
        "geneformer": "Geneformer",
        "genie3": "GENIE3",
        "string": "STRING",
    }
    lk = key.lower()
    if lk in aliases:
        key = aliases[lk]
    return MODEL_COLORS.get(key, default)


def method_label(method: str) -> str:
    key = str(method).strip()
    if not key:
        return ""
    return METHOD_DISPLAY.get(key, key)


def apply_fig4_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": FIG4_FONT_FAMILY,
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.major.size": 0.0,
            "ytick.major.size": 0.0,
            "xtick.minor.size": 0.0,
            "ytick.minor.size": 0.0,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": FIG4_DPI,
            "savefig.dpi": FIG4_DPI,
            "figure.figsize": FIG4_FIGSIZE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

