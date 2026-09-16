#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRN-aware Phase-1 dynamic prediction for scGPT.

Fuses Transformer MLM output with directed 1-hop propagation on a GRN:

    fused = (1 - beta) * mlm_out + beta * gnn_out     (genes with incoming edges)
    fused = mlm_out                                   (no incoming edge)

Then applies the same EMA update as scgpt_dyn.py.

GRN sources (--grn-source):
  none          baseline (beta forced to 0)
  embhidden500  cos_hid / high-quality scGPT GRN
  att500        attention-derived GRN
  emb500        cos token embedding GRN
  string        STRING reference network
  shuffle       shuffle Gene2 on embhidden500 (negative control)

Example:
  python scgpt_dyn_grn.py \\
    --model-dir /mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human \\
    --expr-root /mnt/10T/yzn/benchmark_GRN/input_process \\
    --pt-root /mnt/10T/yzn/benchmark_GRN/PseudoTime \\
    --outdir /mnt/10T/yzn/scGRN-Bench/FBplot/fig4/grn_dyn_phase1 \\
    --grn-source embhidden500 --grn-beta 0.25 --datasets hESC mDC
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

# ---------------------------------------------------------------------------
# Paths / registry (minimal copy of fig3 model_registry)
# ---------------------------------------------------------------------------
BENCH_ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
EVL_ROOT = BENCH_ROOT / "evl_omipath"
INPUT_ROOT = BENCH_ROOT / "input_process"

EXTRACTION_DIRS = {
    "emb500": EVL_ROOT / "output_emb500" / "scgpt",
    "embhidden500": EVL_ROOT / "output_embhidden500" / "scgpt",
    "att500": EVL_ROOT / "output_att500" / "scgpt",
}
PRED_STEMS = ("scgpt", "scGPT", "scGPT2")

WEIGHT_COLS = (
    "EdgeWeight",
    "edgeweight",
    "edge_weight",
    "Attention score",
    "Weight",
    "Score",
)

DATASET_SPECS = {
    "hESC": "human",
    "hHep": "human",
    "mDC": "mouse",
    "mHSC-E": "mouse",
    "mHSC-GM": "mouse",
    "mHSC-L": "mouse",
}


def parse_args():
    p = argparse.ArgumentParser(description="scGPT GRN-aware iterative dynamic prediction (Phase 1).")
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--expr-root", default=str(INPUT_ROOT), type=str)
    p.add_argument("--pt-root", default=str(BENCH_ROOT / "PseudoTime"), type=str)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS.keys()),
        choices=list(DATASET_SPECS.keys()),
    )
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--log1p", action="store_true")
    p.add_argument(
        "--grn-source",
        default="embhidden500",
        choices=["none", "emb500", "embhidden500", "att500", "string", "shuffle"],
    )
    p.add_argument(
        "--shuffle-base",
        default="embhidden500",
        choices=["emb500", "embhidden500", "att500"],
        help="Base GRN for --grn-source shuffle.",
    )
    p.add_argument("--grn-beta", type=float, default=0.25, help="Weight on directed propagation branch.")
    p.add_argument("--grn-max-edges", type=int, default=200_000, help="Cap edges after filtering.")
    p.add_argument("--grn-topk-per-target", type=int, default=50, help="Keep top-K regulators per target.")
    p.add_argument("--grn-shuffle-seed", type=int, default=42)
    p.add_argument(
        "--sweep-beta",
        action="store_true",
        help="Run beta in {0, 0.1, 0.25, 0.5} (grn-source must be set).",
    )
    p.add_argument(
        "--sweep-sources",
        action="store_true",
        help="Run none, embhidden500, att500, string, shuffle at fixed --grn-beta.",
    )
    return p.parse_args()


ARGS = parse_args()

PT_QUANTILE = float(ARGS.pt_quantile)
TOP_PERCENT = int(ARGS.top_percent)
GEN_ITERS = int(ARGS.gen_iters)
BATCH_SIZE = int(ARGS.batch_size)
EMA_ALPHA = float(ARGS.ema_alpha)
NO_LOG1P = not bool(ARGS.log1p)

import sys as _sys

_SCGPT_REPO = Path(__file__).resolve().parents[3] / "pre_scgpt" / "scGPT"
if not _SCGPT_REPO.is_dir():
    _SCGPT_REPO = BENCH_ROOT / "pre_scgpt" / "scGPT"
if _SCGPT_REPO.is_dir() and str(_SCGPT_REPO) not in _sys.path:
    _sys.path.insert(0, str(_SCGPT_REPO))

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def bin_expr_to_0_50(x, do_log1p=True):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(np.percentile(x, 99.5), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def convert_mouse_to_human_gene(gene_name: str) -> str:
    return gene_name.upper()


def detect_weight_col(df: pd.DataFrame) -> Optional[str]:
    for c in WEIGHT_COLS:
        if c in df.columns:
            return c
    return None


def normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Gene1" not in d.columns or "Gene2" not in d.columns:
        if d.shape[1] < 2:
            raise ValueError(f"GRN needs Gene1/Gene2 columns, got {list(d.columns)}")
        d = d.iloc[:, :2].copy()
        d.columns = ["Gene1", "Gene2"]
    d["Gene1"] = d["Gene1"].astype(str).str.strip()
    d["Gene2"] = d["Gene2"].astype(str).str.strip()
    d = d[(d["Gene1"] != "") & (d["Gene2"] != "") & (d["Gene1"] != d["Gene2"])]
    return d.drop_duplicates(subset=["Gene1", "Gene2"])


def resolve_pred_tsv(extraction: str, dataset: str) -> Path:
    base = EXTRACTION_DIRS.get(extraction)
    if base is None or not base.is_dir():
        raise FileNotFoundError(f"GRN dir missing for extraction={extraction}: {base}")
    for stem in PRED_STEMS:
        cand = base / f"{stem}_{dataset}.tsv"
        if cand.is_file():
            return cand
    for p in sorted(base.glob("*.tsv")):
        if dataset.lower() in p.name.lower():
            return p
    raise FileNotFoundError(f"No pred TSV for {dataset} under {base}")


def resolve_string_network(dataset: str, expr_root: Path) -> Path:
    p = expr_root / "STRING" / f"{dataset}_processed-network.csv"
    if not p.is_file():
        p = INPUT_ROOT / "STRING" / f"{dataset}_processed-network.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_grn_edges(
    source: str,
    dataset: str,
    gene_universe: Set[str],
    *,
    expr_root: Path,
    max_edges: int,
    topk_per_target: int,
    shuffle_seed: int,
    shuffle_base: str,
) -> pd.DataFrame:
    if source == "string":
        path = resolve_string_network(dataset, expr_root)
        df = pd.read_csv(path)
        df = normalize_edges(df)
        df["EdgeWeight"] = 1.0
    elif source == "shuffle":
        path = resolve_pred_tsv(shuffle_base, dataset)
        df = _read_pred_grn(path, gene_universe, max_edges, topk_per_target)
        rng = np.random.default_rng(shuffle_seed)
        df = df.copy()
        df["Gene2"] = rng.permutation(df["Gene2"].to_numpy())
        df = df[df["Gene1"] != df["Gene2"]]
    elif source in EXTRACTION_DIRS:
        path = resolve_pred_tsv(source, dataset)
        df = _read_pred_grn(path, gene_universe, max_edges, topk_per_target)
    else:
        raise ValueError(f"Unknown GRN source: {source}")

    df = df[df["Gene1"].isin(gene_universe) & df["Gene2"].isin(gene_universe)]
    return df.reset_index(drop=True)


def _read_pred_grn(
    path: Path,
    gene_universe: Set[str],
    max_edges: int,
    topk_per_target: int,
) -> pd.DataFrame:
    usecols = ["Gene1", "Gene2"] + list(WEIGHT_COLS)
    try:
        df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols)
    except (ValueError, KeyError):
        df = pd.read_csv(path, sep="\t")
    df = normalize_edges(df)
    wcol = detect_weight_col(df)
    if wcol is None:
        df["EdgeWeight"] = 1.0
        wcol = "EdgeWeight"
    else:
        df[wcol] = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0)
        df = df.rename(columns={wcol: "EdgeWeight"})

    df = df[df["Gene1"].isin(gene_universe) & df["Gene2"].isin(gene_universe)]
    if df.empty:
        return df

    if topk_per_target > 0:
        df = (
            df.sort_values("EdgeWeight", ascending=False)
            .groupby("Gene2", sort=False, group_keys=False)
            .head(topk_per_target)
        )
    if max_edges > 0 and len(df) > max_edges:
        df = df.nlargest(max_edges, "EdgeWeight")
    return df[["Gene1", "Gene2", "EdgeWeight"]].reset_index(drop=True)


@dataclass
class GrnGraph:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_w: torch.Tensor
    has_incoming: torch.Tensor
    n_edges: int
    n_targets_with_edges: int


def build_grn_graph(
    genes: Sequence[str],
    grn_df: pd.DataFrame,
    device: torch.device,
) -> GrnGraph:
    """Map genes to tensor columns [0]=<cls>, [1..G]=genes; build normalized incoming edges."""
    n_pos = len(genes) + 1
    name_to_pos: Dict[str, int] = {g: i + 1 for i, g in enumerate(genes)}

    incoming: Dict[int, List[Tuple[int, float]]] = {}
    for row in grn_df.itertuples(index=False):
        j = name_to_pos.get(str(row.Gene2))
        i = name_to_pos.get(str(row.Gene1))
        if j is None or i is None:
            continue
        w = float(row.EdgeWeight)
        if w <= 0:
            continue
        incoming.setdefault(j, []).append((i, w))

    src_list: List[int] = []
    dst_list: List[int] = []
    w_list: List[float] = []
    for j, edges in incoming.items():
        s = sum(w for _, w in edges)
        if s <= 0:
            continue
        for i, w in edges:
            src_list.append(i)
            dst_list.append(j)
            w_list.append(w / s)

    has_incoming = torch.zeros(n_pos, dtype=torch.bool)
    for j in incoming:
        has_incoming[j] = True
    has_incoming[0] = False

    if not src_list:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return GrnGraph(z, z, torch.zeros(0, device=device), has_incoming.to(device), 0, 0)

    return GrnGraph(
        edge_src=torch.tensor(src_list, dtype=torch.long, device=device),
        edge_dst=torch.tensor(dst_list, dtype=torch.long, device=device),
        edge_w=torch.tensor(w_list, dtype=torch.float32, device=device),
        has_incoming=has_incoming.to(device),
        n_edges=len(src_list),
        n_targets_with_edges=len(incoming),
    )


def directed_propagate(vals: torch.Tensor, graph: GrnGraph) -> torch.Tensor:
    """vals: [B, L]; aggregate regulators into targets (incoming, TF -> target)."""
    if graph.n_edges == 0:
        return vals
    bsz = vals.shape[0]
    contrib = vals[:, graph.edge_src] * graph.edge_w.unsqueeze(0)
    out = torch.zeros_like(vals)
    out.scatter_add_(1, graph.edge_dst.unsqueeze(0).expand(bsz, -1), contrib)
    return out


def build_model(model_dir, device):
    with open(Path(model_dir) / "args.json") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)

    use_fast = bool(cfg.get("fast_transformer", True)) and device.type == "cuda"
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=use_fast,
    )
    ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()
    return model, vocab


@torch.no_grad()
def iterative_direction_accuracy_grn(
    model,
    gene_ids_tensor,
    values_tensor,
    pad_mask,
    update_mask_1d,
    early_mean,
    true_delta,
    top_idx,
    graph: Optional[GrnGraph],
    grn_beta: float,
):
    device = next(model.parameters()).device
    vals_all = values_tensor.clone()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()
    beta = float(grn_beta)
    use_grn = graph is not None and graph.n_edges > 0 and beta > 0

    if use_grn:
        has_in = graph.has_incoming.unsqueeze(0)
    acc_curve = []

    for _ in range(GEN_ITERS):
        for start in range(0, vals_all.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals_all.shape[0])
            bs = end - start

            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(bs, -1).to(device)
            mask = pad_mask[start:end].to(device)
            freeze = mask | (~update_mask.expand(bs, -1))

            out = model(src=src, values=vals, src_key_padding_mask=mask)
            mlm = out["mlm_output"]

            if use_grn:
                gnn = directed_propagate(vals, graph)
                fused = torch.where(
                    has_in.expand(bs, -1),
                    (1.0 - beta) * mlm + beta * gnn,
                    mlm,
                )
            else:
                fused = mlm

            vals = torch.where(freeze, vals, EMA_ALPHA * vals + (1.0 - EMA_ALPHA) * fused)
            vals_all[start:end] = vals.detach().cpu()

        pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
        pred_delta = pred_mean - early_mean
        acc = float((np.sign(pred_delta[top_idx]) == np.sign(true_delta[top_idx])).mean())
        acc_curve.append(acc)

    return acc_curve


def coverage_on_top_genes(
    genes: Sequence[str],
    top_idx: np.ndarray,
    graph: Optional[GrnGraph],
) -> Dict[str, float]:
    if graph is None or graph.n_targets_with_edges == 0:
        return {
            "frac_top_with_incoming": 0.0,
            "n_top_with_incoming": 0,
            "n_top": int(len(top_idx)),
        }
    has = graph.has_incoming.cpu().numpy()
    # top_idx indexes into gene array (0..G-1), tensor positions are top_idx+1
    covered = sum(1 for i in top_idx if has[i + 1])
    n_top = len(top_idx)
    return {
        "frac_top_with_incoming": covered / max(n_top, 1),
        "n_top_with_incoming": covered,
        "n_top": n_top,
    }


def run_dataset(
    name: str,
    cfg: dict,
    model,
    vocab,
    device: torch.device,
    grn_source: str,
    grn_beta: float,
    expr_root: Path,
    shuffle_base: str,
):
    print(f"\n{'=' * 60}\nRunning {name} | grn={grn_source} beta={grn_beta}\n{'=' * 60}")

    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    genes_original = expr.index.astype(str).tolist()
    species = cfg.get("species", "human")
    genes = (
        [convert_mouse_to_human_gene(g) for g in genes_original]
        if species == "mouse"
        else genes_original
    )

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=not NO_LOG1P)

    lo, hi = np.quantile(pt, [PT_QUANTILE, 1 - PT_QUANTILE])
    early = pt <= lo
    late = pt >= hi

    early_mean = X_bin[early].mean(axis=0)
    late_mean = X_bin[late].mean(axis=0)
    true_delta = late_mean - early_mean

    total_genes = len(true_delta)
    top_n = max(int(total_genes * TOP_PERCENT / 100), 1)
    top_idx = np.argsort(np.abs(true_delta))[::-1][:top_n].copy()

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    values_tensor = torch.tensor(X_in, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    update_mask_1d = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask_1d[1:] = True

    graph = None
    grn_meta = {"grn_source": grn_source, "grn_beta": grn_beta, "n_edges": 0, "n_targets_with_edges": 0}
    if grn_source != "none" and grn_beta > 0:
        universe = set(genes)
        grn_df = load_grn_edges(
            grn_source,
            name,
            universe,
            expr_root=expr_root,
            max_edges=ARGS.grn_max_edges,
            topk_per_target=ARGS.grn_topk_per_target,
            shuffle_seed=ARGS.grn_shuffle_seed,
            shuffle_base=ARGS.shuffle_base,
        )
        graph = build_grn_graph(genes, grn_df, device)
        grn_meta.update(
            n_edges=graph.n_edges,
            n_targets_with_edges=graph.n_targets_with_edges,
            n_grn_rows=len(grn_df),
        )
        print(f"  [GRN] edges={graph.n_edges}, targets_with_incoming={graph.n_targets_with_edges}")

    acc_curve = iterative_direction_accuracy_grn(
        model,
        gene_ids_tensor,
        values_tensor[early],
        pad_mask[early],
        update_mask_1d,
        early_mean,
        true_delta,
        top_idx,
        graph,
        grn_beta,
    )

    cov = coverage_on_top_genes(genes, top_idx, graph)
    print(f"  [RESULT] final acc={acc_curve[-1]:.2%} | top-{TOP_PERCENT}% GRN coverage={cov['frac_top_with_incoming']:.1%}")

    diag = {
        "n_genes": len(genes),
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "eval_genes_count": top_n,
        **grn_meta,
        **cov,
    }
    return acc_curve, diag


def build_dataset_cfg(name: str, expr_root: Path, pt_root: Path) -> dict:
    return {
        "expr_csv": str(expr_root / "CHIP" / f"{name}_chip_matched-ExpressionData.csv"),
        "pt_csv": str(pt_root / name / "PseudoTime.csv"),
        "species": DATASET_SPECS[name],
    }


def run_experiment_grid(
    model,
    vocab,
    device: torch.device,
    outdir: Path,
    expr_root: Path,
    pt_root: Path,
):
    runs: List[Tuple[str, float]] = []
    if ARGS.sweep_sources:
        runs = [
            ("none", 0.0),
            ("embhidden500", ARGS.grn_beta),
            ("att500", ARGS.grn_beta),
            ("string", ARGS.grn_beta),
            ("shuffle", ARGS.grn_beta),
        ]
    elif ARGS.sweep_beta:
        src = ARGS.grn_source
        if src == "none":
            runs = [("none", 0.0)]
        else:
            runs = [(src, b) for b in (0.0, 0.1, 0.25, 0.5)]
    else:
        beta = 0.0 if ARGS.grn_source == "none" else ARGS.grn_beta
        runs = [(ARGS.grn_source, beta)]

    all_results: Dict[str, Dict] = {}
    for grn_source, grn_beta in runs:
        key = f"{grn_source}_beta{grn_beta:g}"
        curves = {}
        diagnostics = {}
        for ds in ARGS.datasets:
            cfg = build_dataset_cfg(ds, expr_root, pt_root)
            curve, diag = run_dataset(
                ds, cfg, model, vocab, device, grn_source, grn_beta, expr_root, ARGS.shuffle_base
            )
            curves[ds] = curve
            diagnostics[ds] = diag
        all_results[key] = {"curves": curves, "diagnostics": diagnostics}

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "grn_dyn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # summary table
    rows = []
    for key, payload in all_results.items():
        for ds, curve in payload["curves"].items():
            d = payload["diagnostics"][ds]
            rows.append(
                {
                    "run": key,
                    "dataset": ds,
                    "final_acc": curve[-1],
                    "frac_top_grn_covered": d.get("frac_top_with_incoming", 0.0),
                    "n_edges": d.get("n_edges", 0),
                    "grn_source": d.get("grn_source"),
                    "grn_beta": d.get("grn_beta"),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "grn_dyn_summary.csv", index=False)
    print(f"\n[OK] Wrote {outdir / 'grn_dyn_results.json'} and grn_dyn_summary.csv")
    return all_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, vocab = build_model(ARGS.model_dir, device)

    expr_root = Path(ARGS.expr_root)
    pt_root = Path(ARGS.pt_root)
    outdir = Path(ARGS.outdir)

    run_experiment_grid(model, vocab, device, outdir, expr_root, pt_root)


if __name__ == "__main__":
    main()
