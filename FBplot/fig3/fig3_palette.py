#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified palette for FoundBench paper figures (fig3).

Keep colors consistent with fig2 (current working scheme):
  scGPT       #8FB4DC
  Geneformer  #FFDD8E
  GENIE3      #70CDBE
  LangCell    #AC99D2
  scCello     #7AC3DF
  scFoundation#F5AA61
  scPrint     #EB7E60
  STRING      #6E6E6E (neutral)

For scGPT representation methods (emb500/embhidden500/att500), we also provide fixed method colors.
"""

from __future__ import annotations

from typing import Dict


MODEL_COLORS: Dict[str, str] = {
    "scGPT": "#8FB4DC",
    "Geneformer": "#FFDD8E",
    "GENIE3": "#70CDBE",
    "LangCell": "#AC99D2",
    "scCello": "#7AC3DF",
    "scFoundation": "#F5AA61",
    "scPrint": "#EB7E60",
    "STRING": "#6E6E6E",
}

# scGPT representation colors (distinct, but still within the same family)
METHOD_COLORS: Dict[str, str] = {
    "emb500": MODEL_COLORS["scGPT"],
    "embhidden500": MODEL_COLORS["LangCell"],
    "att500": MODEL_COLORS["GENIE3"],
}

METHOD_DISPLAY: Dict[str, str] = {
    "emb500": r"cos$_{tok}$",
    "embhidden500": r"cos$_{hid}$",
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
        "string": "STRING",
    }
    lk = key.lower()
    if lk in aliases:
        key = aliases[lk]
    return MODEL_COLORS.get(key, default)


def method_color(method: str, default: str = "#808080") -> str:
    key = str(method).strip()
    if not key:
        return default
    return METHOD_COLORS.get(key, default)


def method_label(method: str) -> str:
    key = str(method).strip()
    if not key:
        return ""
    return METHOD_DISPLAY.get(key, key)

