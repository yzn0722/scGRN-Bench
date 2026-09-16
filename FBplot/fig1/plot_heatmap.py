#!/usr/bin/env python3
"""Heatmap for fig1 data (画图.xlsx) using AUPR.py color scheme."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Style (matches benchmark_GRN/plot_paper/指标图/AUPR.py)
# ---------------------------------------------------------------------------
COLORS = [
    "#0d0829",
    "#4b0c6b",
    "#7d1d6d",
    "#bb3754",
    "#cf533b",
    "#f18518",
    "#f9c74f",
    "#fcffa4",
]
CMAP = LinearSegmentedColormap.from_list("nature_magma", COLORS, N=256)
CMAP.set_bad(color="black")
VMIN = 1.0
FIG_SIZE = 10.0

GT_DISPLAY = {
    "STRING": "STRING",
    "CHIP": "Cell-type-specific\nChIP-seq",
    "Non_CHIP": "Nonspecific\nChIP-seq",
    "omnipath": "OmniPath",
    "Omnipath": "OmniPath",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 4,
        "axes.unicode_minus": False,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    }
)


def _find_section_starts(df: pd.DataFrame) -> list[int]:
    starts = []
    for i in range(len(df)):
        val = df.iloc[i, 1]
        if isinstance(val, str) and val == "Panel_Name":
            starts.append(i)
    return starts


def _parse_column_layout(df: pd.DataFrame, start_row: int) -> tuple[list[str], list[int]]:
    """Infer metric + ground-truth labels from the header rows."""
    gt_row = df.iloc[start_row + 1]

    gt_labels: list[str] = []
    for j in range(2, df.shape[1]):
        cell = gt_row.iloc[j]
        if pd.isna(cell) or str(cell).strip() == "":
            continue
        gt_labels.append(str(cell).strip())

    if not gt_labels:
        raise ValueError(f"Could not parse ground-truth labels at row {start_row + 1}")

    half = len(gt_labels) // 2
    col_names: list[str] = []
    col_indices: list[int] = []
    for k, gt in enumerate(gt_labels):
        metric = "EPR" if k < half else "AUPR"
        col_names.append(f"{metric}_{gt}")
        col_indices.append(2 + k)

    return col_names, col_indices


def parse_panel(
    df: pd.DataFrame, start_row: int, end_row: int | None = None
) -> tuple[str, pd.DataFrame, list[dict]]:
    panel_name = str(df.iloc[start_row, 2])
    col_names, col_indices = _parse_column_layout(df, start_row)
    data_rows: list[dict] = []
    row_groups: list[dict] = []
    current_group: str | None = None
    group_start = 0

    stop = end_row if end_row is not None else len(df)
    for i in range(start_row + 2, stop):
        row = df.iloc[i]
        group_label = row.iloc[0]
        model_name = row.iloc[1]

        if isinstance(model_name, str) and model_name.strip() in ("Panel_Name", "Group_Name", "评价指标"):
            break

        if pd.isna(model_name) or str(model_name).strip() == "":
            continue

        if isinstance(group_label, str) and group_label.strip():
            if current_group is not None:
                row_groups.append(
                    {"name": current_group, "start": group_start, "end": len(data_rows)}
                )
            current_group = group_label.strip()
            group_start = len(data_rows)

        values = pd.to_numeric(row.iloc[col_indices], errors="coerce").values.astype(float)
        data_rows.append({"model": str(model_name), "values": values})

    if current_group is not None and len(data_rows) > group_start:
        row_groups.append({"name": current_group, "start": group_start, "end": len(data_rows)})

    if not data_rows:
        raise ValueError(f"No data rows found for panel starting at row {start_row}")

    models = [r["model"] for r in data_rows]
    mat = np.vstack([r["values"] for r in data_rows])
    mat[mat < VMIN] = np.nan

    panel_df = pd.DataFrame(mat, index=models, columns=col_names)
    return panel_name, panel_df, row_groups


def build_rgba(df_vals: pd.DataFrame, row_groups: list[dict]) -> np.ndarray:
    n_rows, n_cols = df_vals.shape
    rgba_img = np.zeros((n_rows, n_cols, 4), dtype=float)

    for g in row_groups:
        r0, r1 = g["start"], g["end"]
        block = df_vals.iloc[r0:r1, :].values
        block_max = np.nanmax(block)
        if (not np.isfinite(block_max)) or block_max <= VMIN:
            block_max = VMIN + 1e-6
        norm_g = Normalize(vmin=VMIN, vmax=block_max)
        block_rgba = CMAP(norm_g(block))
        nan_mask = np.isnan(block)
        block_rgba[nan_mask] = (0, 0, 0, 1)
        rgba_img[r0:r1, :, :] = block_rgba

    return rgba_img


def _clean_model_name(name: str) -> str:
    return (
        name.replace("_emb", "")
        .replace("_att", "")
        .replace("_hidden", "")
        .replace("scgpt", "scGPT")
        .replace("scello", "scCello")
    )


def _gt_label(col: str) -> str:
    gt = col.split("_", 1)[1]
    return GT_DISPLAY.get(gt, gt)


def _draw_heatmap_block(
    ax: plt.Axes,
    df_vals: pd.DataFrame,
    rgba_img: np.ndarray,
    *,
    show_ylabels: bool,
) -> None:
    n_rows, n_cols = df_vals.shape
    ax.imshow(rgba_img, aspect="auto")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.tick_top()

    xlabels = [_gt_label(c) for c in df_vals.columns]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(xlabels, rotation=45, ha="left", fontweight="bold", fontsize=8)

    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=1)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=1)

    for i in range(n_rows):
        for j in range(n_cols):
            val = df_vals.iat[i, j]
            if pd.isna(val):
                continue
            bg = rgba_img[i, j, :3]
            luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            text_color = "white" if luminance < 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
            )

    if show_ylabels:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([_clean_model_name(n) for n in df_vals.index], fontsize=8)
        ax.tick_params(axis="y", pad=2)


def plot_panel(
    panel_name: str,
    df_vals: pd.DataFrame,
    row_groups: list[dict],
    outpath: Path,
) -> None:
    epr_cols = [c for c in df_vals.columns if c.startswith("EPR_")]
    aupr_cols = [c for c in df_vals.columns if c.startswith("AUPR_")]
    df_epr = df_vals[epr_cols].copy()
    df_aupr = df_vals[aupr_cols].copy()

    rgba_epr = build_rgba(df_epr, row_groups)
    rgba_aupr = build_rgba(df_aupr, row_groups)
    vmax = np.nanmax(df_vals.values)
    n_rows = len(df_vals)

    margin_left = 1.95
    margin_right = 0.25
    margin_top = 1.85
    margin_bottom = 1.55
    gap = 0.18

    usable_w = FIG_SIZE - margin_left - margin_right
    block_w = (usable_w - gap) / 2
    heatmap_h = FIG_SIZE - margin_top - margin_bottom

    fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), dpi=300)
    ax_epr = fig.add_axes(
        [margin_left / FIG_SIZE, margin_bottom / FIG_SIZE, block_w / FIG_SIZE, heatmap_h / FIG_SIZE]
    )
    ax_aupr = fig.add_axes(
        [
            (margin_left + block_w + gap) / FIG_SIZE,
            margin_bottom / FIG_SIZE,
            block_w / FIG_SIZE,
            heatmap_h / FIG_SIZE,
        ]
    )

    _draw_heatmap_block(ax_epr, df_epr, rgba_epr, show_ylabels=True)
    _draw_heatmap_block(ax_aupr, df_aupr, rgba_aupr, show_ylabels=False)

    gap_lw = max(4, int(n_rows * 0.18))
    for ax in (ax_epr, ax_aupr):
        for g in row_groups:
            if g["end"] < n_rows:
                ax.axhline(g["end"] - 0.5, color="white", linewidth=gap_lw)

    trans = mtransforms.blended_transform_factory(fig.transFigure, ax_epr.transData)
    ax_left_x = ax_epr.get_position().x0
    grp_x = ax_left_x - 0.11

    for g in row_groups:
        y_center = (g["start"] + g["end"] - 1) / 2
        fig.text(
            grp_x,
            y_center,
            g["name"],
            va="center",
            ha="center",
            fontweight="bold",
            fontsize=8,
            transform=trans,
        )

    header_y = -1.2
    fig.text(
        ax_left_x - 0.01,
        header_y,
        "Method",
        va="center",
        ha="right",
        fontweight="bold",
        fontsize=9,
        transform=trans,
    )
    fig.text(
        grp_x,
        header_y,
        "Category",
        va="center",
        ha="center",
        fontweight="bold",
        fontsize=9,
        transform=trans,
    )

    for ax, title in ((ax_epr, "EPR"), (ax_aupr, "AUPR")):
        ax.text(
            (len(ax.get_xticks()) - 1) / 2,
            -3.2,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transData,
        )

    cax = fig.add_axes([0.22, 0.06, 0.56, 0.018])
    norm_global = Normalize(vmin=VMIN, vmax=vmax if np.isfinite(vmax) else VMIN + 1)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_global, cmap=CMAP),
        cax=cax,
        orientation="horizontal",
    )
    cax.set_xlabel(f"Score ({panel_name})", fontsize=8, labelpad=3)
    cax.tick_params(labelsize=6)

    cbar_x, cbar_y, cbar_w, cbar_h = (1 - 0.25) / 2, 0.05, 0.25, 0.015
    block_x = cbar_x + cbar_w
    rect = Rectangle(
        (block_x, cbar_y),
        0.02,
        cbar_h,
        transform=fig.transFigure,
        facecolor="black",
        edgecolor="none",
    )
    fig.add_artist(rect)
    fig.text(
        block_x + 0.03,
        cbar_y + cbar_h / 2,
        "Random predictor (<1)",
        ha="left",
        va="center",
        fontsize=8,
    )

    fig.savefig(outpath, dpi=600, bbox_inches=None, pad_inches=0)
    pdf_path = outpath.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path(__file__).resolve().parent / "画图.xlsx",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_excel(args.xlsx, sheet_name="Sheet1", header=None)
    section_starts = _find_section_starts(df_raw)

    for idx, start in enumerate(section_starts):
        end = section_starts[idx + 1] if idx + 1 < len(section_starts) else len(df_raw)
        panel_name, panel_df, row_groups = parse_panel(df_raw, start, end)
        safe_name = panel_name.replace("+", "_")

        plot_panel(
            panel_name,
            panel_df,
            row_groups,
            args.outdir / f"{safe_name}_EPR_AUPR_heatmap.png",
        )


if __name__ == "__main__":
    main()
