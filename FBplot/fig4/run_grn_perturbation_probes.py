#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 / C2 perturbation probes: does scGPT iteration reflect GRN structure?

C1 — GRN-guided vs destroyed iteration (same early cells):
  - baseline: update all genes (current pipeline)
  - chip / string / pred neighborhoods as update masks
  - random:   degree-matched random gene sets of matching sizes

C2 — In silico TF knockout:
  - zero/scale a TF in early-cell inputs, iterate, measure CHIP-target
    |pred_delta| attenuation vs wild-type and random-gene knockouts

Examples
--------
  cd /mnt/10T/yzn/scGRN-Bench/FBplot/fig4

  # Both C1 + C2 on hESC (recommended)
  python run_grn_perturbation_probes.py \
    --dataset hESC \
    --mode both \
    --gen-iters 16 \
    --max-early-cells 64 \
    --scgpt-model-dir /mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human \
    --scgpt-repo-dir /mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT \
    --expr-root /mnt/10T/yzn/benchmark_GRN/input_process \
    --pt-root /mnt/10T/yzn/benchmark_GRN/PseudoTime \
    --grn-tsv /mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt/scgpt_hESC.tsv \
    --chip-network /mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-network.csv \
    --outdir outputs/grn_perturbation_probes

  # C1 only / C2 only
  python run_grn_perturbation_probes.py --dataset hESC --mode c1 ...
  python run_grn_perturbation_probes.py --dataset hESC --mode c2 --ko-tfs SOX2,POU5F1 ...
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
# Prefer full source under benchmark_GRN (scGRN-Bench unified copy is a broken pyc launcher).
_UNIFIED_CANDIDATES = [
    Path("/mnt/10T/yzn/benchmark_GRN"),
    SCRIPT_DIR.parents[1] / "src" / "GRN_inferance",
]
for _u in _UNIFIED_CANDIDATES:
    _py = _u / "run_unified_multidataset_pseudotime.py"
    if _py.is_file() and _py.stat().st_size > 10000:
        if str(_u) not in sys.path:
            sys.path.insert(0, str(_u))
        break
from run_unified_multidataset_pseudotime import (  # noqa: E402
    bin_expr_to_0_50,
    direction_accuracy_top_genes,
    load_scgpt_model,
    read_pt_file,
    set_seed,
)


def split_pt_masks(pt, q: float = 0.2, mid_lo: float = 0.4, mid_hi: float = 0.6):
    """Early/late/mid masks + quantile cuts (compatible with prior C1 runs)."""
    pt = np.asarray(pt, dtype=np.float64)
    lo, hi = np.quantile(pt, [q, 1.0 - q])
    mlo, mhi = np.quantile(pt, [mid_lo, mid_hi])
    early = pt <= lo
    late = pt >= hi
    mid = (pt >= mlo) & (pt <= mhi) & (~early) & (~late)
    return early, late, mid, float(lo), float(hi), float(mlo), float(mhi)



# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="C1/C2 GRN perturbation probes for scGPT iteration")
    p.add_argument("--dataset", default="hESC")
    p.add_argument("--mode", choices=["c1", "c2", "both"], default="both")
    p.add_argument("--outdir", default="outputs/grn_perturbation_probes")
    p.add_argument("--expr-root", default="/mnt/10T/yzn/benchmark_GRN/input_process")
    p.add_argument("--pt-root", default="/mnt/10T/yzn/benchmark_GRN/PseudoTime")
    p.add_argument(
        "--chip-network",
        default="",
        help="CHIP network CSV (Gene1,Gene2). Default: {expr-root}/CHIP/{ds}_chip_matched-network.csv",
    )
    p.add_argument(
        "--string-network",
        default="",
        help="STRING network CSV (Gene1,Gene2). Default: {expr-root}/STRING/{ds}_processed-network.csv",
    )
    p.add_argument(
        "--grn-tsv",
        default="",
        help="Predicted GRN TSV (Gene1,Gene2,EdgeWeight). Optional; used for C1 neighborhood if set.",
    )
    p.add_argument("--grn-source", choices=["chip", "string", "pred", "both", "chip_string", "chip_tf_nontf", "string_tf_nontf"], default="chip_tf_nontf",
                   help="C1 graphs. *_tf_nontf: TF vs non-TF split without top-dynamic filter (TF = Gene1 in that network).")
    p.add_argument("--c1-intersect-top-dynamic", action="store_true",
                   help="Legacy leakage-prone filter: intersect network nodes with top-|Δ| genes for update masks.")
    p.add_argument("--top-grn-edges", type=int, default=5000,
                   help="Keep top-|weight| predicted edges when --grn-source uses pred.")
    p.add_argument("--scgpt-model-dir", required=True)
    p.add_argument("--scgpt-repo-dir", required=True)
    p.add_argument("--scgpt-bin-log1p", action="store_true", default=False)
    p.add_argument("--scgpt-legacy-pt", action="store_true", default=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--max-early-cells", type=int, default=64)
    p.add_argument("--print-every", type=int, default=4)
    # C2
    p.add_argument("--ko-tfs", default="",
                   help="Comma-separated TF symbols for knockout. Empty = auto top out-degree CHIP TFs.")
    p.add_argument("--n-auto-tfs", type=int, default=5)
    p.add_argument("--n-random-ko", type=int, default=5,
                   help="Number of random non-TF gene knockouts as negative controls.")
    p.add_argument("--ko-scale", type=float, default=0.0,
                   help="Multiply TF expression by this scale (0 = hard zero).")
    p.add_argument("--c2-gen-iters", type=int, default=0,
                   help="Override gen-iters for C2 only (0 = use --gen-iters).")
    return p.parse_args()


def _norm(g: str) -> str:
    return str(g).strip().upper()


def resolve_paths(args) -> Tuple[Path, Path, Path, Path, Path]:
    ds = args.dataset
    expr = Path(args.expr_root) / "CHIP" / f"{ds}_chip_matched-ExpressionData.csv"
    pt = Path(args.pt_root) / ds / "PseudoTime.csv"
    chip = Path(args.chip_network) if args.chip_network else (
        Path(args.expr_root) / "CHIP" / f"{ds}_chip_matched-network.csv"
    )
    string = Path(args.string_network) if args.string_network else (
        Path(args.expr_root) / "STRING" / f"{ds}_processed-network.csv"
    )
    grn = Path(args.grn_tsv) if str(args.grn_tsv).strip() else None
    for p, name in [(expr, "expr"), (pt, "pt"), (chip, "chip-network")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")
    if args.grn_source in ("string", "chip_string") and not string.exists():
        raise FileNotFoundError(f"Missing string-network: {string}")
    if args.grn_source in ("pred", "both") and (grn is None or not grn.is_file()):
        raise FileNotFoundError(f"--grn-source={args.grn_source} needs existing --grn-tsv")
    return expr, pt, chip, string, grn


def load_edges_csv(path: Path, top_n: int = 0, has_weight: bool = False) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    g1 = cols.get("gene1") or cols.get("tf") or df.columns[0]
    g2 = cols.get("gene2") or cols.get("target") or df.columns[1]
    out = pd.DataFrame({"Gene1": df[g1].map(_norm), "Gene2": df[g2].map(_norm)})
    if has_weight or "edgeweight" in cols:
        wcol = cols.get("edgeweight") or cols.get("weight")
        if wcol is not None:
            out["EdgeWeight"] = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0).abs()
            out = out.sort_values("EdgeWeight", ascending=False)
            if top_n > 0:
                out = out.head(top_n)
        else:
            out["EdgeWeight"] = 1.0
    else:
        out["EdgeWeight"] = 1.0
    out = out[(out["Gene1"] != "") & (out["Gene2"] != "") & (out["Gene1"] != out["Gene2"])]
    return out.reset_index(drop=True)


def build_adjacency(edges: pd.DataFrame) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b in zip(edges["Gene1"], edges["Gene2"]):
        adj[a].add(b)
    return adj


def degree_matched_random_genes(
    genes: Sequence[str],
    keep: Sequence[str],
    degrees: Dict[str, int],
    rng: np.random.Generator,
) -> List[str]:
    """Sample |keep| genes with roughly matched degree histogram (fallback: uniform)."""
    keep_set = set(keep)
    pool = [g for g in genes if g not in keep_set]
    if len(pool) < len(keep):
        return list(rng.choice(genes, size=len(keep), replace=False))
    # Bucket by log2(degree+1)
    def bucket(g: str) -> int:
        return int(np.floor(np.log2(degrees.get(g, 0) + 1)))

    by_b: Dict[int, List[str]] = defaultdict(list)
    for g in pool:
        by_b[bucket(g)].append(g)
    chosen: List[str] = []
    for g in keep:
        b = bucket(g)
        # search nearby buckets
        cand = None
        for db in range(0, 8):
            for bb in {b - db, b + db}:
                if bb in by_b and by_b[bb]:
                    cand = by_b[bb].pop()
                    break
            if cand is not None:
                break
        if cand is None:
            # leftover
            leftovers = [x for xs in by_b.values() for x in xs]
            if not leftovers:
                break
            cand = leftovers[rng.integers(0, len(leftovers))]
            by_b[bucket(cand)].remove(cand)
        chosen.append(cand)
    if len(chosen) < len(keep):
        extra = [g for g in pool if g not in chosen]
        need = len(keep) - len(chosen)
        chosen.extend(list(rng.choice(extra, size=need, replace=False)))
    return chosen


# ---------------------------------------------------------------------------
# Data + model prep
# ---------------------------------------------------------------------------

class ScgptBundle:
    def __init__(self, args):
        self.args = args
        if args.device == "cpu":
            self.device = torch.device("cpu")
        elif args.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Minimal namespace for load_scgpt_model
        class _A:
            pass

        a = _A()
        a.scgpt_model_dir = args.scgpt_model_dir
        a.scgpt_repo_dir = args.scgpt_repo_dir
        self.model, self.vocab = load_scgpt_model(a, self.device)

        expr_path, pt_path, chip_path, string_path, grn_path = resolve_paths(args)
        expr = pd.read_csv(expr_path, index_col=0)
        if args.scgpt_legacy_pt:
            pt_df = pd.read_csv(pt_path)
            pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
        else:
            pt_df = read_pt_file(pt_path)
        common = expr.columns.intersection(pt_df.index)
        if len(common) == 0:
            raise ValueError("No overlapping cells between expression and pseudotime")
        expr = expr[common]
        pt = pt_df.loc[common, "pt"].to_numpy()

        genes_raw = expr.index.astype(str).tolist()
        # CHIP files are human gene symbols for hESC/hHep; keep upper for matching.
        self.genes = [_norm(g) for g in genes_raw]
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}

        X = expr.T.to_numpy(dtype=np.float32)
        self.X_bin = bin_expr_to_0_50(X, do_log1p=args.scgpt_bin_log1p)

        early, late, mid, lo, hi, mlo, mhi = split_pt_masks(pt, args.pt_quantile)
        if early.sum() == 0 or late.sum() == 0:
            raise ValueError("Empty early/late after quantile split")
        self.early = early
        self.late = late
        self.mid = mid
        self.early_mean = self.X_bin[early].mean(axis=0)
        self.late_mean = self.X_bin[late].mean(axis=0)
        self.global_mean = self.X_bin.mean(axis=0)
        self.true_delta = self.late_mean - self.early_mean
        n = len(self.true_delta)
        top_n = max(int(n * args.top_percent / 100), 1)
        self.top_idx = np.argsort(np.abs(self.true_delta))[::-1][:top_n]

        init_idx = np.where(early)[0]
        if args.max_early_cells > 0:
            init_idx = init_idx[: args.max_early_cells]
        self.init_idx = init_idx
        self.init_mean = self.X_bin[init_idx].mean(axis=0)

        gene_ids = np.array(
            [self.vocab[g] if g in self.vocab else self.vocab["<pad>"] for g in self.genes],
            dtype=np.int64,
        )
        gene_ids = np.concatenate([[self.vocab["<cls>"]], gene_ids])
        self.gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)
        values = np.concatenate(
            [np.zeros((self.X_bin.shape[0], 1), dtype=np.float32), self.X_bin], axis=1
        )
        self.values_tensor = torch.tensor(
            values, dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.pad_mask = self.gene_ids_tensor.eq(self.vocab["<pad>"]).expand(values.shape[0], -1)

        # Graphs
        self.chip_edges = load_edges_csv(chip_path, has_weight=False)
        self.chip_adj = build_adjacency(self.chip_edges)
        self.string_edges = None
        self.string_adj = None
        if string_path.exists():
            self.string_edges = load_edges_csv(string_path, has_weight=False)
            self.string_adj = build_adjacency(self.string_edges)
        self.pred_edges = None
        self.pred_adj = None
        if grn_path is not None and grn_path.exists():
            self.pred_edges = load_edges_csv(grn_path, top_n=args.top_grn_edges, has_weight=True)
            self.pred_adj = build_adjacency(self.pred_edges)

        self.meta = {
            "dataset": args.dataset,
            "n_genes": n,
            "n_early_used": int(len(init_idx)),
            "n_late": int(late.sum()),
            "pt_lo": float(lo),
            "pt_hi": float(hi),
            "top_n": int(top_n),
            "chip_edges": int(len(self.chip_edges)),
            "string_edges": int(len(self.string_edges)) if self.string_edges is not None else 0,
            "pred_edges": int(len(self.pred_edges)) if self.pred_edges is not None else 0,
        }
        print(
            f"[data] {args.dataset}: genes={n}, early_used={len(init_idx)}, "
            f"chip_edges={self.meta['chip_edges']}, string_edges={self.meta['string_edges']}, "
            f"pred_edges={self.meta['pred_edges']}, device={self.device}"
        )


def gene_update_mask(bundle: ScgptBundle, gene_names: Sequence[str]) -> np.ndarray:
    """Boolean mask over model tokens: [cls] + genes. True = allow update."""
    m = np.zeros(bundle.gene_ids_tensor.shape[1], dtype=bool)
    for g in gene_names:
        i = bundle.gene_to_idx.get(_norm(g))
        if i is not None:
            m[i + 1] = True  # +1 for CLS
    return m


def run_iteration(
    bundle: ScgptBundle,
    update_mask_genes: np.ndarray,
    n_iters: int,
    values_init: Optional[torch.Tensor] = None,
    label: str = "",
) -> dict:
    """
    update_mask_genes: bool array length = n_tokens (incl CLS), True = update.
    values_init: optional [n_cells, 1+G] starting state; default = early cells.
    """
    args = bundle.args
    if values_init is None:
        vals_all = bundle.values_tensor[bundle.init_idx].clone()
    else:
        vals_all = values_init.clone()
    init_pad = bundle.pad_mask[bundle.init_idx]
    update_mask_t = torch.tensor(update_mask_genes[None, :], device=bundle.device).bool()

    acc_curve: List[float] = []
    pred_mean_by_iter: List[np.ndarray] = []
    pred_delta_by_iter: List[np.ndarray] = []

    with torch.no_grad():
        for it in range(n_iters):
            for start in range(0, vals_all.shape[0], args.batch_size):
                end = min(start + args.batch_size, vals_all.shape[0])
                bs = end - start
                vals = vals_all[start:end].to(bundle.device)
                src = bundle.gene_ids_tensor.expand(bs, -1).to(bundle.device)
                mask = init_pad[start:end].to(bundle.device)
                freeze = mask | (~update_mask_t.expand(bs, -1))
                out = bundle.model(src=src, values=vals, src_key_padding_mask=mask)
                new_vals = out["mlm_output"]
                vals = torch.where(
                    freeze, vals, args.ema_alpha * vals + (1.0 - args.ema_alpha) * new_vals
                )
                vals_all[start:end] = vals.detach().cpu()

            pred_mean = vals_all[:, 1:].numpy().mean(axis=0).astype(np.float32)
            pred_delta = pred_mean - bundle.early_mean
            pred_mean_by_iter.append(pred_mean)
            pred_delta_by_iter.append(pred_delta)
            acc, _ = direction_accuracy_top_genes(pred_delta, bundle.true_delta, bundle.top_idx)
            acc_curve.append(acc)
            if args.print_every > 0 and ((it + 1) % args.print_every == 0 or it == 0):
                print(f"    [{label}] iter {it+1}/{n_iters} acc={acc:.2%}")

    pm = np.stack(pred_mean_by_iter, 0)
    pdlt = np.stack(pred_delta_by_iter, 0)
    x0 = bundle.init_mean
    xT = pm[-1]
    late = bundle.late_mean
    glob = bundle.global_mean
    axis = late - bundle.early_mean
    axis_n = float(np.linalg.norm(axis)) + 1e-12
    axis_u = axis / axis_n
    # Restrict geometry to top genes for fair comparison with probes
    top = bundle.top_idx

    def _sl(v):
        return v[top]

    return {
        "label": label,
        "acc_curve": acc_curve,
        "final_acc": float(acc_curve[-1]),
        "pred_mean_by_iter": pm,
        "pred_delta_by_iter": pdlt,
        "pred_delta_final": pdlt[-1],
        "dist_late_final": float(np.linalg.norm(_sl(xT) - _sl(late))),
        "dist_global_final": float(np.linalg.norm(_sl(xT) - _sl(glob))),
        "proj_init": float(np.dot(_sl(x0) - _sl(bundle.early_mean), _sl(axis_u))),
        "proj_final": float(np.dot(_sl(xT) - _sl(bundle.early_mean), _sl(axis_u))),
        "n_update_genes": int(update_mask_genes[1:].sum()),
        "vals_final": vals_all[:, 1:].numpy().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------

def neighborhood_from_edges(
    bundle: ScgptBundle,
    edges: pd.DataFrame,
    intersect_top_dynamic: bool = False,
) -> Tuple[List[str], Dict[str, int]]:
    """Genes allowed to update under a given edge table + undirected degree map.

    By default keep ALL in-matrix network nodes (no |true_delta| filter).
    Pass intersect_top_dynamic=True only for the legacy (leakage-prone) protocol.
    """
    nodes: Set[str] = set()
    deg: Dict[str, int] = defaultdict(int)
    for a, b in zip(edges["Gene1"], edges["Gene2"]):
        if a in bundle.gene_to_idx and b in bundle.gene_to_idx:
            nodes.add(a)
            nodes.add(b)
            deg[a] += 1
            deg[b] += 1
    if intersect_top_dynamic:
        top_genes = {bundle.genes[i] for i in bundle.top_idx}
        keep = sorted(nodes & top_genes)
        if len(keep) < max(20, int(0.05 * len(bundle.top_idx))):
            keep = sorted(nodes)
    else:
        keep = sorted(nodes)
    return keep, dict(deg)


def chip_tf_nontf_gene_sets(bundle: ScgptBundle) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """Split CHIP nodes in the expression matrix into TF vs non-TF (no dynamic filter).

    TF = appears as Gene1; non-TF = in CHIP but never Gene1 (target-only).
    Degree map = undirected CHIP degree for degree-matched random controls.
    """
    edges = bundle.chip_edges
    tfs: Set[str] = set()
    targets: Set[str] = set()
    deg: Dict[str, int] = defaultdict(int)
    for a, b in zip(edges["Gene1"], edges["Gene2"]):
        if a in bundle.gene_to_idx:
            tfs.add(a)
        if b in bundle.gene_to_idx:
            targets.add(b)
        if a in bundle.gene_to_idx and b in bundle.gene_to_idx:
            deg[a] += 1
            deg[b] += 1
    tf_list = sorted(tfs)
    nontf_list = sorted(targets - tfs)  # target-only
    all_chip = sorted(tfs | targets)
    gene_sets = {
        "chip_tf": tf_list,
        "chip_nontf": nontf_list,
        "chip_all_nodes": all_chip,
    }
    deg_maps = {k: dict(deg) for k in gene_sets}
    return gene_sets, deg_maps



def string_tf_nontf_gene_sets(bundle: ScgptBundle) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """Split STRING nodes by STRING's own Gene1 (= TF) vs non-Gene1.

    Same definition as CHIP split: TF = appears as Gene1; non-TF = in network but never Gene1.
    No |true_delta| filter.
    """
    if bundle.string_edges is None:
        raise ValueError("STRING edges not loaded")
    tfs: Set[str] = set()
    targets: Set[str] = set()
    deg: Dict[str, int] = defaultdict(int)
    for a, b in zip(bundle.string_edges["Gene1"], bundle.string_edges["Gene2"]):
        if a in bundle.gene_to_idx:
            tfs.add(a)
        if b in bundle.gene_to_idx:
            targets.add(b)
        if a in bundle.gene_to_idx and b in bundle.gene_to_idx:
            deg[a] += 1
            deg[b] += 1
    tf_list = sorted(tfs)
    nontf_list = sorted(targets - tfs)  # Gene2-only
    all_list = sorted(tfs | targets)
    gene_sets = {
        "string_tf": tf_list,
        "string_nontf": nontf_list,
        "string_all_nodes": all_list,
    }
    deg_maps = {k: dict(deg) for k in gene_sets}
    return gene_sets, deg_maps


def c1_neighborhood_genes(bundle: ScgptBundle) -> Tuple[List[str], Dict[str, int]]:
    """Single neighborhood for chip/pred/both/string union modes."""
    args = bundle.args
    frames = []
    if args.grn_source in ("chip", "both"):
        frames.append(bundle.chip_edges)
    if args.grn_source == "string":
        if bundle.string_edges is None:
            raise ValueError("STRING edges not loaded")
        frames.append(bundle.string_edges)
    if args.grn_source in ("pred", "both") and bundle.pred_edges is not None:
        frames.append(bundle.pred_edges)
    if not frames:
        raise ValueError(f"No edges for grn_source={args.grn_source}")
    if len(frames) == 1:
        flag = bool(getattr(bundle.args, "c1_intersect_top_dynamic", False))
    if len(frames) == 1:
        return neighborhood_from_edges(bundle, frames[0], intersect_top_dynamic=flag)
    return neighborhood_from_edges(bundle, pd.concat(frames, ignore_index=True), intersect_top_dynamic=flag)


def run_c1(bundle: ScgptBundle, outdir: Path) -> dict:
    print("\n===== C1: GRN-guided vs random vs baseline =====")
    args = bundle.args
    rng = np.random.default_rng(args.seed)

    gene_sets: Dict[str, List[str]] = {}
    deg_maps: Dict[str, Dict[str, int]] = {}
    top_flag = bool(getattr(args, "c1_intersect_top_dynamic", False))

    if args.grn_source == "chip_tf_nontf":
        # No |true_delta| filter: compare CHIP TFs vs CHIP non-TFs (+ all CHIP nodes).
        gene_sets, deg_maps = chip_tf_nontf_gene_sets(bundle)
        cond_order = ["chip_tf", "chip_nontf", "chip_all_nodes"]
        print(
            f"  [C1 mode=chip_tf_nontf] "
            f"TF={len(gene_sets['chip_tf'])}, nonTF={len(gene_sets['chip_nontf'])}, "
            f"all_CHIP={len(gene_sets['chip_all_nodes'])} (no top-dynamic intersect)"
        )
    elif args.grn_source == "string_tf_nontf":
        gene_sets, deg_maps = string_tf_nontf_gene_sets(bundle)
        cond_order = ["string_tf", "string_nontf", "string_all_nodes"]
        print(
            f"  [C1 mode=string_tf_nontf] "
            f"STRING-TF(Gene1)={len(gene_sets['string_tf'])}, "
            f"STRING nonTF={len(gene_sets['string_nontf'])}, "
            f"all_STRING={len(gene_sets['string_all_nodes'])} (no top-dynamic intersect)"
        )
    elif args.grn_source == "chip_string":
        chip_genes, chip_deg = neighborhood_from_edges(bundle, bundle.chip_edges, top_flag)
        if bundle.string_edges is None:
            raise ValueError("STRING edges required for --grn-source chip_string")
        string_genes, string_deg = neighborhood_from_edges(bundle, bundle.string_edges, top_flag)
        gene_sets["chip_neighborhood"] = chip_genes
        deg_maps["chip_neighborhood"] = chip_deg
        gene_sets["string_neighborhood"] = string_genes
        deg_maps["string_neighborhood"] = string_deg
        cond_order = ["chip_neighborhood", "string_neighborhood"]
    elif args.grn_source == "string":
        g, d = neighborhood_from_edges(bundle, bundle.string_edges, top_flag)
        gene_sets["string_neighborhood"] = g
        deg_maps["string_neighborhood"] = d
        cond_order = ["string_neighborhood"]
    elif args.grn_source == "chip":
        g, d = neighborhood_from_edges(bundle, bundle.chip_edges, top_flag)
        gene_sets["chip_neighborhood"] = g
        deg_maps["chip_neighborhood"] = d
        cond_order = ["chip_neighborhood"]
    elif args.grn_source == "pred":
        g, d = c1_neighborhood_genes(bundle)
        gene_sets["pred_neighborhood"] = g
        deg_maps["pred_neighborhood"] = d
        cond_order = ["pred_neighborhood"]
    else:  # both = chip+pred union
        g, d = c1_neighborhood_genes(bundle)
        gene_sets["chip_pred_neighborhood"] = g
        deg_maps["chip_pred_neighborhood"] = d
        cond_order = ["chip_pred_neighborhood"]

    masks = {"baseline_all": np.ones(bundle.gene_ids_tensor.shape[1], dtype=bool)}
    for name in cond_order:
        masks[name] = gene_update_mask(bundle, gene_sets[name])
        # random control name: chip_tf -> random_matched_chip_tf
        rand_name = f"random_matched_{name}"
        rand_genes = degree_matched_random_genes(bundle.genes, gene_sets[name], deg_maps[name], rng)
        gene_sets[rand_name] = rand_genes
        masks[rand_name] = gene_update_mask(bundle, rand_genes)

    for m in masks.values():
        m[0] = False

    results = {}
    for name, umask in masks.items():
        print(f"  -> condition={name}, n_update={int(umask[1:].sum())}")
        results[name] = run_iteration(bundle, umask, args.gen_iters, label=name)

    rows = []
    for name, r in results.items():
        rows.append(
            {
                "condition": name,
                "n_update_genes": r["n_update_genes"],
                "final_acc": r["final_acc"],
                "dist_late_final": r["dist_late_final"],
                "dist_global_final": r["dist_global_final"],
                "closer_to_late_than_global": r["dist_late_final"] < r["dist_global_final"],
                "proj_init": r["proj_init"],
                "proj_final": r["proj_final"],
                "proj_gain": r["proj_final"] - r["proj_init"],
                "acc_curve": r["acc_curve"],
            }
        )
    df = pd.DataFrame(rows)
    df_csv = df.drop(columns=["acc_curve"])
    outdir.mkdir(parents=True, exist_ok=True)
    df_csv.to_csv(outdir / "c1_summary.csv", index=False)

    curve_df = pd.DataFrame({name: results[name]["acc_curve"] for name in results})
    curve_df.insert(0, "iter", np.arange(1, args.gen_iters + 1))
    curve_df.to_csv(outdir / "c1_acc_curves.csv", index=False)

    gene_sets_out = {k: v for k, v in gene_sets.items()}
    gene_sets_out["grn_source"] = args.grn_source
    gene_sets_out["n"] = {k: len(v) for k, v in gene_sets.items()}
    with open(outdir / "c1_gene_sets.json", "w") as f:
        json.dump(gene_sets_out, f, indent=2)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), dpi=160)
    for name, r in results.items():
        axes[0].plot(range(1, args.gen_iters + 1), r["acc_curve"], "-o", ms=3, label=name)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Direction accuracy (top genes)")
    axes[0].set_title("C1: accuracy under update masks")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    x = np.arange(len(rows))
    axes[1].bar(x - 0.15, [r["dist_late_final"] for r in rows], 0.3, label="→ late", color="#d95f02")
    axes[1].bar(x + 0.15, [r["dist_global_final"] for r in rows], 0.3, label="→ global", color="#7570b3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r["condition"] for r in rows], rotation=20, ha="right", fontsize=7)
    axes[1].set_ylabel("Distance (top-gene space)")
    axes[1].set_title("C1: final distance to centroids")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "c1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    supports = {}
    for name in cond_order:
        rand_name = f"random_matched_{name}"
        if rand_name not in results:
            continue
        acc_g = results[name]["final_acc"]
        acc_r = results[rand_name]["final_acc"]
        proj_g = results[name]["proj_final"] - results[name]["proj_init"]
        proj_r = results[rand_name]["proj_final"] - results[rand_name]["proj_init"]
        supports[name] = bool(acc_g > acc_r) or bool(proj_g > proj_r)

    summary = {
        "supports_over_random": supports,
        "supports_grn_over_random": bool(any(supports.values())),
        "final_acc": {k: float(results[k]["final_acc"]) for k in results},
        "proj_gain": {
            k: float(results[k]["proj_final"] - results[k]["proj_init"]) for k in results
        },
        "n_update_genes": {k: int(results[k]["n_update_genes"]) for k in results},
        "interpretation": (
            "At least one true network neighborhood outperforms its degree-matched random set."
            if any(supports.values())
            else "No clear advantage of true network neighborhoods over random masks."
        ),
    }
    with open(outdir / "c1_verdict.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  [C1] final_acc:")
    for k, v in summary["final_acc"].items():
        print(f"       {k}: {v:.4f} (n={summary['n_update_genes'][k]})")

    for name, r in results.items():
        np.save(outdir / f"c1_pred_delta_{name}.npy", r["pred_delta_final"])
    return summary


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------

def pick_ko_tfs(bundle: ScgptBundle, args) -> List[str]:
    if args.ko_tfs.strip():
        tfs = [_norm(x) for x in args.ko_tfs.split(",") if x.strip()]
        return [t for t in tfs if t in bundle.gene_to_idx and t in bundle.chip_adj]
    # Auto: CHIP TFs with most in-matrix targets, prefer those in top dynamic genes
    scored = []
    top_set = {bundle.genes[i] for i in bundle.top_idx}
    for tf, targets in bundle.chip_adj.items():
        if tf not in bundle.gene_to_idx:
            continue
        t_in = [t for t in targets if t in bundle.gene_to_idx]
        if len(t_in) < 3:
            continue
        score = len(t_in) + (5 if tf in top_set else 0)
        scored.append((score, len(t_in), tf))
    scored.sort(reverse=True)
    return [tf for _, _, tf in scored[: args.n_auto_tfs]]


def pick_random_ko_genes(bundle: ScgptBundle, tfs: Sequence[str], n: int, rng: np.random.Generator) -> List[str]:
    tf_set = set(tfs) | set(bundle.chip_adj.keys())
    # Match mean expression of TF set roughly
    tf_idx = [bundle.gene_to_idx[t] for t in tfs if t in bundle.gene_to_idx]
    if not tf_idx:
        pool = [g for g in bundle.genes if g not in tf_set]
        return list(rng.choice(pool, size=min(n, len(pool)), replace=False))
    tf_mean_expr = float(bundle.init_mean[tf_idx].mean())
    pool = []
    for g in bundle.genes:
        if g in tf_set:
            continue
        i = bundle.gene_to_idx[g]
        pool.append((abs(float(bundle.init_mean[i]) - tf_mean_expr), g))
    pool.sort()
    return [g for _, g in pool[: max(n * 5, n)]][:n] if False else [
        g for _, g in pool[:n]
    ]  # take closest expression matches


def apply_knockout(values: torch.Tensor, gene_idx: int, scale: float) -> torch.Tensor:
    """values: [n_cells, 1+G]; gene_idx is gene index (not including CLS)."""
    out = values.clone()
    out[:, gene_idx + 1] = out[:, gene_idx + 1] * float(scale)
    return out


def target_effect_stats(
    bundle: ScgptBundle,
    pred_delta_wt: np.ndarray,
    pred_delta_ko: np.ndarray,
    targets: Sequence[str],
) -> dict:
    t_idx = [bundle.gene_to_idx[t] for t in targets if t in bundle.gene_to_idx]
    if not t_idx:
        return {
            "n_targets": 0,
            "mean_abs_delta_wt": float("nan"),
            "mean_abs_delta_ko": float("nan"),
            "attenuation_ratio": float("nan"),
            "frac_abs_decreased": float("nan"),
            "mean_signed_change": float("nan"),
        }
    wt = pred_delta_wt[t_idx]
    ko = pred_delta_ko[t_idx]
    abs_wt = np.abs(wt)
    abs_ko = np.abs(ko)
    # attenuation: ko/wt (<1 means knockout weakened target response)
    ratio = float(np.mean(abs_ko) / (np.mean(abs_wt) + 1e-12))
    return {
        "n_targets": int(len(t_idx)),
        "mean_abs_delta_wt": float(np.mean(abs_wt)),
        "mean_abs_delta_ko": float(np.mean(abs_ko)),
        "attenuation_ratio": ratio,
        "frac_abs_decreased": float(np.mean(abs_ko < abs_wt)),
        "mean_signed_change": float(np.mean(ko - wt)),
    }


def run_c2(bundle: ScgptBundle, outdir: Path) -> dict:
    print("\n===== C2: in silico TF knockout =====")
    args = bundle.args
    rng = np.random.default_rng(args.seed + 7)
    n_iters = int(args.c2_gen_iters) if int(args.c2_gen_iters) > 0 else args.gen_iters
    tfs = pick_ko_tfs(bundle, args)
    if not tfs:
        raise RuntimeError("No valid CHIP TFs found for knockout. Pass --ko-tfs explicitly.")
    rand_genes = pick_random_ko_genes(bundle, tfs, args.n_random_ko, rng)
    print(f"  TFs: {tfs}")
    print(f"  random KO controls: {rand_genes}")

    # Update all genes (baseline dynamics); only the *input* is perturbed
    umask = np.ones(bundle.gene_ids_tensor.shape[1], dtype=bool)
    umask[0] = False

    print("  -> wild-type")
    wt = run_iteration(bundle, umask, n_iters, label="wt")
    pred_wt = wt["pred_delta_final"]

    rows = []
    # Non-target background genes for specificity
    all_targets = set()
    for tf in tfs:
        all_targets |= set(bundle.chip_adj.get(tf, ()))
    nontargets = [g for g in bundle.genes if g not in all_targets and g not in tfs]
    nontargets = list(rng.choice(nontargets, size=min(100, len(nontargets)), replace=False))

    for kind, genes in [("tf", tfs), ("random", rand_genes)]:
        for g in genes:
            gi = bundle.gene_to_idx[g]
            vals0 = apply_knockout(
                bundle.values_tensor[bundle.init_idx], gi, args.ko_scale
            )
            print(f"  -> KO {kind}={g}")
            r = run_iteration(bundle, umask, n_iters, values_init=vals0, label=f"ko_{g}")
            targets = sorted(t for t in bundle.chip_adj.get(g, ()) if t in bundle.gene_to_idx)
            # For random genes, use CHIP targets of a matched TF? No — use empty / or same-size random "pseudo-targets"
            if kind == "random":
                # Specificity control: effect on real CHIP targets of the TF panel (should be weak)
                # Also report effect on random gene's co-expression neighbors: skip; use shared TF-target set
                stats_targets = target_effect_stats(bundle, pred_wt, r["pred_delta_final"], sorted(all_targets))
                stats_targets["target_set"] = "all_chip_targets_of_panel"
            else:
                stats_targets = target_effect_stats(bundle, pred_wt, r["pred_delta_final"], targets)
                stats_targets["target_set"] = "chip_targets_of_this_tf"
            stats_bg = target_effect_stats(bundle, pred_wt, r["pred_delta_final"], nontargets)
            rows.append(
                {
                    "ko_kind": kind,
                    "ko_gene": g,
                    "n_chip_targets": len(targets) if kind == "tf" else stats_targets["n_targets"],
                    "final_acc": r["final_acc"],
                    **{f"tgt_{k}": v for k, v in stats_targets.items()},
                    **{f"bg_{k}": v for k, v in stats_bg.items()},
                }
            )
            np.save(outdir / f"c2_pred_delta_ko_{g}.npy", r["pred_delta_final"])

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "c2_pred_delta_wt.npy", pred_wt)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "c2_knockout_effects.csv", index=False)

    # Aggregate TF vs random
    def _agg(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {}
        return {
            "mean_tgt_attenuation_ratio": float(sub["tgt_attenuation_ratio"].mean()),
            "mean_bg_attenuation_ratio": float(sub["bg_attenuation_ratio"].mean()),
            "mean_tgt_frac_abs_decreased": float(sub["tgt_frac_abs_decreased"].mean()),
            "n": int(len(sub)),
        }

    tf_agg = _agg(df[df["ko_kind"] == "tf"])
    rand_agg = _agg(df[df["ko_kind"] == "random"])
    supports = False
    if tf_agg and rand_agg:
        # TF KO should attenuate CHIP targets more than random KO does
        supports = bool(
            tf_agg["mean_tgt_attenuation_ratio"] < rand_agg["mean_tgt_attenuation_ratio"]
            and tf_agg["mean_tgt_attenuation_ratio"] < 1.0
            and tf_agg["mean_tgt_attenuation_ratio"] < tf_agg["mean_bg_attenuation_ratio"]
        )

    verdict = {
        "tfs": tfs,
        "random_ko_genes": rand_genes,
        "tf_aggregate": tf_agg,
        "random_aggregate": rand_agg,
        "supports_tf_specific_effect": supports,
        "interpretation": (
            "Knocking out CHIP TFs attenuates |pred_delta| of their targets more than "
            "random gene KOs / non-target background — consistent with GRN-linked dynamics."
            if supports
            else "No clear TF-specific attenuation vs random KO; in-silico KO evidence is weak "
            "under current settings."
        ),
    }
    with open(outdir / "c2_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=160)
    plot_df = df.copy()
    colors = {"tf": "#d95f02", "random": "#7570b3"}
    for kind, sub in plot_df.groupby("ko_kind"):
        ax.scatter(
            sub["tgt_attenuation_ratio"],
            sub["bg_attenuation_ratio"],
            s=60,
            c=colors.get(kind, "gray"),
            label=kind,
            edgecolors="white",
            linewidths=0.5,
        )
        for _, row in sub.iterrows():
            ax.text(row["tgt_attenuation_ratio"], row["bg_attenuation_ratio"], row["ko_gene"], fontsize=7)
    ax.axvline(1.0, color="0.6", ls="--", lw=1)
    ax.axhline(1.0, color="0.6", ls="--", lw=1)
    ax.set_xlabel("Target |Δ| attenuation (KO / WT)")
    ax.set_ylabel("Non-target background |Δ| (KO / WT)")
    ax.set_title("C2: in silico knockout specificity")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "c2_knockout_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"  [C2] tf_attn={tf_agg.get('mean_tgt_attenuation_ratio')} "
        f"rand_attn={rand_agg.get('mean_tgt_attenuation_ratio')} supports={supports}"
    )
    return verdict


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    # stash args on a simple namespace used by bundle
    bundle = ScgptBundle(args)
    with open(outdir / "meta.json", "w") as f:
        json.dump(bundle.meta, f, indent=2)

    report = {"dataset": args.dataset, "mode": args.mode}
    if args.mode in ("c1", "both"):
        report["c1"] = run_c1(bundle, outdir / "c1")
    if args.mode in ("c2", "both"):
        report["c2"] = run_c2(bundle, outdir / "c2")

    with open(outdir / "perturbation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nDone.")
    print(f"Outputs: {outdir}")
    if "c1" in report:
        print("  C1 supports_grn_over_random:", report["c1"].get("supports_grn_over_random"))
    if "c2" in report:
        print("  C2 supports_tf_specific_effect:", report["c2"].get("supports_tf_specific_effect"))


if __name__ == "__main__":
    main()
