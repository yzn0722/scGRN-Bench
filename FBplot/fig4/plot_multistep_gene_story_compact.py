#!/usr/bin/env python3
"""紧凑对比图：1 个非靶 + 1 个 CHIP 靶，并排 + 大字号示意。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_multistep_gene_story import (  # noqa: E402
    build_figure,
    SCRIPT_DIR,
)

if __name__ == "__main__":
    out = SCRIPT_DIR / "error_biology" / "multistep_pt" / "hESC" / "gene_story_compact_NRP1_DNMT3B.png"
    build_figure("hESC", ["NRP1", "DNMT3B"], out)
