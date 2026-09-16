#!/usr/bin/env python3
"""Regenerate cross-head module figures from existing detect_att_modules CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_attention_heads as ath  # noqa: E402
import detect_att_modules as dam  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="scgpt_hESC")
    p.add_argument("--input-dir", default=str(_SCRIPT_DIR / "output" / "att_modules" / "scgpt_hESC"))
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--display", default="scGPT")
    args = p.parse_args()

    out_dir = Path(args.input_dir)
    tag = args.tag
    key_df = pd.read_csv(out_dir / f"{tag}_keygene_module_across_heads.csv")

    dam.plot_keygene_across_heads(key_df, args.display, args.dataset, out_dir / f"{tag}_keygene_module_heatmap.pdf")
    dam.plot_keygene_strength_heatmap(
        key_df, "in_strength", "Key genes: in-hub strength",
        args.display, args.dataset, out_dir / f"{tag}_keygene_in_strength_heatmap.pdf",
    )
    dam.plot_keygene_strength_heatmap(
        key_df, "out_strength", "Key genes: out-strength",
        args.display, args.dataset, out_dir / f"{tag}_keygene_out_strength_heatmap.pdf",
    )
    print(f"[INFO] Updated cross-head heatmaps in {out_dir}")
    print("  (Re-run detect_att_modules.py for 8-panel PCA / module stats)")


if __name__ == "__main__":
    main()
