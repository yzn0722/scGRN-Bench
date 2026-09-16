#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RegVelo-style learnable GRN head for scGPT pseudotime direction prediction (MVP).

Frozen scGPT encoder + small trainable head:
  rate_g = MLP( H_g, aggregate_TF(H), x_g )
  pred_delta = mean(rate | late segment) - mean(rate | early segment)

Training uses adjacent pseudotime segment pairs (default K=5).
Evaluation matches fig4: early/late quantile endpoints, top-30% dynamic genes.

Init modes (--init-mode):
  fm_only       frozen FM iterative baseline (no head training)
  none          train head without GRN aggregation
  grn_init      trainable edge weights initialized from FM GRN (embhidden500)
  random_init   trainable random edge weights (same sparsity as GRN)
  shuffle_init  trainable weights with shuffled GRN topology
  grn_frozen    GRN weights fixed, only MLP trains

Example:
  python scgpt_dyn_reghead.py \\
    --model-dir /mnt/10T/yzn/benchmark_GRN/model/weights/scgpt/scGPT_human \\
    --expr-root /mnt/10T/yzn/benchmark_GRN/input_process \\
    --pt-root /mnt/10T/yzn/benchmark_GRN/PseudoTime \\
    --outdir /mnt/10T/yzn/scGRN-Bench/FBplot/fig4/reghead_mvp \\
    --datasets hESC --sweep-inits

  python scgpt_dyn_reghead.py ... --init-mode grn_init --epochs 80
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
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

# ---------------------------------------------------------------------------
# Paths
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

INIT_MODES = (
    "fm_only",
    "none",
    "grn_init",
    "random_init",
    "shuffle_init",
    "grn_frozen",
)


def parse_args():
    p = argparse.ArgumentParser(description="RegVelo-style learnable GRN head (scGPT dynamic MVP).")
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--expr-root", default=str(INPUT_ROOT), type=str)
    p.add_argument("--pt-root", default=str(BENCH_ROOT / "PseudoTime"), type=str)
    p.add_argument("--chip-dir", default=str(INPUT_ROOT / "CHIP"), type=str)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["hESC"],
        choices=list(DATASET_SPECS.keys()),
    )
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--n-segments", type=int, default=5, help="Pseudotime bins for training pairs.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log1p", action="store_true")
    p.add_argument(
        "--init-mode",
        default="grn_init",
        choices=INIT_MODES,
        help="Weight initialization / baseline mode.",
    )
    p.add_argument(
        "--grn-extraction",
        default="embhidden500",
        choices=["emb500", "embhidden500", "att500"],
        help="GRN TSV used for grn_init / shuffle_init / random_init sparsity.",
    )
    p.add_argument("--grn-max-edges", type=int, default=200_000)
    p.add_argument("--grn-topk-per-target", type=int, default=50)
    p.add_argument("--grn-shuffle-seed", type=int, default=42)
    p.add_argument(
        "--sweep-inits",
        action="store_true",
        help="Run fm_only, none, grn_init, random_init, shuffle_init.",
    )
    p.add_argument(
        "--fm-iters",
        type=int,
        default=16,
        help="Iterations for fm_only baseline (matches scgpt_dyn.py).",
    )
    p.add_argument("--fm-ema-alpha", type=float, default=0.1)
    return p.parse_args()


ARGS = parse_args()
PT_QUANTILE = float(ARGS.pt_quantile)
TOP_PERCENT = int(ARGS.top_percent)
N_SEGMENTS = max(int(ARGS.n_segments), 2)
NO_LOG1P = not bool(ARGS.log1p)

_SCGPT_REPO = Path(__file__).resolve().parents[3] / "pre_scgpt" / "scGPT"
if not _SCGPT_REPO.is_dir():
    _SCGPT_REPO = BENCH_ROOT / "pre_scgpt" / "scGPT"
if _SCGPT_REPO.is_dir() and str(_SCGPT_REPO) not in sys.path:
    sys.path.insert(0, str(_SCGPT_REPO))

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab


# ---------------------------------------------------------------------------
# Utils (shared with scgpt_dyn_grn.py)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _read_pred_grn(path: Path, gene_universe: Set[str], max_edges: int, topk_per_target: int) -> pd.DataFrame:
    usecols = ["Gene1", "Gene2"] + list(WEIGHT_COLS)
    try:
        df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols)
    except (ValueError, KeyError):
        df = pd.read_csv(path, sep="\t")
    df = normalize_edges(df)
    wcol = detect_weight_col(df)
    if wcol is None:
        df["EdgeWeight"] = 1.0
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


def load_grn_edges(
    dataset: str,
    gene_universe: Set[str],
    *,
    expr_root: Path,
    extraction: str,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> pd.DataFrame:
    path = resolve_pred_tsv(extraction, dataset)
    df = _read_pred_grn(path, gene_universe, ARGS.grn_max_edges, ARGS.grn_topk_per_target)
    if shuffle and not df.empty:
        rng = np.random.default_rng(shuffle_seed)
        df = df.copy()
        df["Gene2"] = rng.permutation(df["Gene2"].to_numpy())
        df = df[df["Gene1"] != df["Gene2"]]
    return df.reset_index(drop=True)


@dataclass
class GrnTensors:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_w: torch.Tensor
    has_incoming: torch.Tensor
    n_edges: int


def build_grn_tensors(genes: Sequence[str], grn_df: pd.DataFrame) -> GrnTensors:
    n_pos = len(genes) + 1
    name_to_pos = {g: i + 1 for i, g in enumerate(genes)}

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

    if not src_list:
        z = torch.zeros(0, dtype=torch.long)
        return GrnTensors(z, z, torch.zeros(0), has_incoming, 0)

    return GrnTensors(
        edge_src=torch.tensor(src_list, dtype=torch.long),
        edge_dst=torch.tensor(dst_list, dtype=torch.long),
        edge_w=torch.tensor(w_list, dtype=torch.float32),
        has_incoming=has_incoming,
        n_edges=len(src_list),
    )


def load_chip_tf_targets(chip_dir: Path, dataset: str, genes: Sequence[str]) -> Set[str]:
    path = chip_dir / f"{dataset}_chip_matched-network.csv"
    if not path.is_file():
        return set()
    net = pd.read_csv(path)
    cols = {c.lower(): c for c in net.columns}
    g2 = cols.get("gene2", net.columns[1])
    targets = set(net[g2].astype(str).str.strip())
    gene_set = set(genes)
    return targets & gene_set


def segment_masks(pt: np.ndarray, n_segments: int) -> List[np.ndarray]:
    edges = np.quantile(pt, np.linspace(0, 1, n_segments + 1))
    edges[-1] += 1e-9
    masks: List[np.ndarray] = []
    for i in range(n_segments):
        if i < n_segments - 1:
            masks.append((pt >= edges[i]) & (pt < edges[i + 1]))
        else:
            masks.append((pt >= edges[i]) & (pt <= edges[i + 1]))
    return masks


def direction_accuracy(pred_delta: np.ndarray, true_delta: np.ndarray, top_idx: np.ndarray) -> float:
    pd_ = pred_delta[top_idx]
    td_ = true_delta[top_idx]
    return float((np.sign(pd_) == np.sign(td_)).mean())


def direction_accuracy_mask(
    pred_delta: np.ndarray, true_delta: np.ndarray, gene_idx: np.ndarray
) -> Optional[float]:
    if len(gene_idx) == 0:
        return None
    return float((np.sign(pred_delta[gene_idx]) == np.sign(true_delta[gene_idx])).mean())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SparseGRNAgg(nn.Module):
    """Weighted sum of regulator hidden states into each target position."""

    def __init__(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_w_init: torch.Tensor,
        train_edge_w: bool = True,
    ):
        super().__init__()
        self.register_buffer("edge_src", edge_src.long())
        self.register_buffer("edge_dst", edge_dst.long())
        self.edge_w = nn.Parameter(edge_w_init.clone())
        if not train_edge_w:
            self.edge_w.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.edge_src.numel() == 0:
            return torch.zeros_like(hidden)
        w = torch.relu(self.edge_w)
        w = w / (w.sum() + 1e-8) * w.numel()  # keep scale stable
        contrib = hidden[:, self.edge_src] * w.view(1, -1, 1)
        out = torch.zeros_like(hidden)
        idx = self.edge_dst.view(1, -1, 1).expand(hidden.shape[0], -1, hidden.shape[2])
        out.scatter_add_(1, idx, contrib)
        return out


class RegVeloHead(nn.Module):
    def __init__(self, d_model: int, grn_agg: Optional[SparseGRNAgg] = None):
        super().__init__()
        self.grn_agg = grn_agg
        in_dim = d_model * 2 + 1 if grn_agg is not None else d_model + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )

    def forward(self, hidden: torch.Tensor, expr: torch.Tensor) -> torch.Tensor:
        """Return per-gene rate [B, G] (excludes <cls> token)."""
        if self.grn_agg is not None:
            h_reg = self.grn_agg(hidden)
            feat = torch.cat([hidden, h_reg, expr.unsqueeze(-1)], dim=-1)
        else:
            feat = torch.cat([hidden, expr.unsqueeze(-1)], dim=-1)
        rate = self.mlp(feat).squeeze(-1)
        return rate[:, 1:]


def build_regvelo_head(
    d_model: int,
    grn: Optional[GrnTensors],
    init_mode: str,
    device: torch.device,
) -> RegVeloHead:
    if init_mode in ("none", "fm_only"):
        return RegVeloHead(d_model, grn_agg=None).to(device)

    if grn is None or grn.n_edges == 0:
        return RegVeloHead(d_model, grn_agg=None).to(device)

    edge_src = grn.edge_src.to(device)
    edge_dst = grn.edge_dst.to(device)
    w_init = grn.edge_w.to(device)

    if init_mode == "random_init":
        w_init = torch.rand_like(w_init) + 0.1
        w_init = w_init / w_init.sum()
    elif init_mode == "shuffle_init":
        perm = torch.randperm(w_init.numel())
        w_init = w_init[perm]

    train_w = init_mode != "grn_frozen"
    agg = SparseGRNAgg(edge_src, edge_dst, w_init, train_edge_w=train_w)
    return RegVeloHead(d_model, grn_agg=agg).to(device)


# ---------------------------------------------------------------------------
# scGPT I/O
# ---------------------------------------------------------------------------
def build_model(model_dir: str, device: torch.device):
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
    return model, vocab, int(cfg["embsize"])


@torch.no_grad()
def encode_cells(
    model,
    gene_ids_tensor: torch.Tensor,
    values_tensor: torch.Tensor,
    pad_mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return hidden [N,L,D] and expr [N,L] on CPU float32."""
    n = values_tensor.shape[0]
    hidden_chunks: List[torch.Tensor] = []
    expr_chunks: List[torch.Tensor] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        vals = values_tensor[start:end].to(device)
        src = gene_ids_tensor.expand(end - start, -1).to(device)
        mask = pad_mask[start:end].to(device)
        hidden = model._encode(src, vals, mask)
        hidden_chunks.append(hidden.float().cpu())
        expr_chunks.append(vals.float().cpu())

    return torch.cat(hidden_chunks, dim=0), torch.cat(expr_chunks, dim=0)


@torch.no_grad()
def fm_iterative_baseline(
    model,
    gene_ids_tensor: torch.Tensor,
    values_tensor: torch.Tensor,
    pad_mask: torch.Tensor,
    update_mask_1d: np.ndarray,
    early_mean: np.ndarray,
    true_delta: np.ndarray,
    top_idx: np.ndarray,
    device: torch.device,
    n_iters: int,
    ema_alpha: float,
    batch_size: int,
) -> Tuple[float, np.ndarray]:
    """Match scgpt_dyn.py iterative MLM baseline."""
    vals_all = values_tensor.clone()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()
    final_pred_delta = np.zeros_like(early_mean)

    for _ in range(n_iters):
        for start in range(0, vals_all.shape[0], batch_size):
            end = min(start + batch_size, vals_all.shape[0])
            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(end - start, -1).to(device)
            mask = pad_mask[start:end].to(device)
            freeze = mask | (~update_mask.expand(end - start, -1))
            out = model(src=src, values=vals, src_key_padding_mask=mask)
            new_vals = out["mlm_output"]
            vals = torch.where(freeze, vals, ema_alpha * vals + (1.0 - ema_alpha) * new_vals)
            vals_all[start:end] = vals.detach().cpu()

        pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
        final_pred_delta = pred_mean - early_mean

    acc = direction_accuracy(final_pred_delta, true_delta, top_idx)
    return acc, final_pred_delta


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
@dataclass
class DatasetBundle:
    name: str
    genes: List[str]
    X_bin: np.ndarray
    pt: np.ndarray
    gene_ids_tensor: torch.Tensor
    values_tensor: torch.Tensor
    pad_mask: torch.Tensor
    early_mask: np.ndarray
    late_mask: np.ndarray
    early_mean: np.ndarray
    true_delta: np.ndarray
    top_idx: np.ndarray
    chip_targets: Set[str]
    seg_masks: List[np.ndarray]


def load_dataset_bundle(
    name: str,
    expr_root: Path,
    pt_root: Path,
    chip_dir: Path,
    vocab,
    device: torch.device,
) -> DatasetBundle:
    expr_csv = expr_root / "CHIP" / f"{name}_chip_matched-ExpressionData.csv"
    pt_csv = pt_root / name / "PseudoTime.csv"

    expr = pd.read_csv(expr_csv, index_col=0)
    pt_df = pd.read_csv(pt_csv)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    genes_original = expr.index.astype(str).tolist()
    if DATASET_SPECS[name] == "mouse":
        genes = [convert_mouse_to_human_gene(g) for g in genes_original]
    else:
        genes = genes_original

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=not NO_LOG1P)

    lo, hi = np.quantile(pt, [PT_QUANTILE, 1 - PT_QUANTILE])
    early_mask = pt <= lo
    late_mask = pt >= hi
    early_mean = X_bin[early_mask].mean(axis=0)
    true_delta = X_bin[late_mask].mean(axis=0) - early_mean

    top_n = max(int(len(true_delta) * TOP_PERCENT / 100), 1)
    top_idx = np.argsort(np.abs(true_delta))[::-1][:top_n].copy()

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    values_tensor = torch.tensor(X_in, dtype=dtype)

    chip_targets = load_chip_tf_targets(chip_dir, name, genes)
    seg_masks = segment_masks(pt, N_SEGMENTS)

    return DatasetBundle(
        name=name,
        genes=genes,
        X_bin=X_bin,
        pt=pt,
        gene_ids_tensor=gene_ids_tensor,
        values_tensor=values_tensor,
        pad_mask=pad_mask,
        early_mask=early_mask,
        late_mask=late_mask,
        early_mean=early_mean,
        true_delta=true_delta,
        top_idx=top_idx,
        chip_targets=chip_targets,
        seg_masks=seg_masks,
    )


def segment_true_delta(X_bin: np.ndarray, early_mask: np.ndarray, late_mask: np.ndarray) -> np.ndarray:
    return X_bin[late_mask].mean(axis=0) - X_bin[early_mask].mean(axis=0)


def build_training_pairs(seg_masks: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for i in range(len(seg_masks) - 1):
        e, l = seg_masks[i], seg_masks[i + 1]
        if e.sum() >= 3 and l.sum() >= 3:
            pairs.append((e, l))
    return pairs


def predict_delta_from_masks(
    head: RegVeloHead,
    hidden: torch.Tensor,
    expr: torch.Tensor,
    early_mask: np.ndarray,
    late_mask: np.ndarray,
) -> np.ndarray:
    h_e = hidden[torch.from_numpy(early_mask)].mean(dim=0, keepdim=True)
    h_l = hidden[torch.from_numpy(late_mask)].mean(dim=0, keepdim=True)
    x_e = expr[torch.from_numpy(early_mask)].mean(dim=0, keepdim=True)
    x_l = expr[torch.from_numpy(late_mask)].mean(dim=0, keepdim=True)
    rate_e = head(h_e, x_e)
    rate_l = head(h_l, x_l)
    return (rate_l - rate_e).detach().cpu().numpy().reshape(-1)


def direction_loss(pred: torch.Tensor, true: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
    p = pred[top_idx]
    t = true[top_idx]
    return F.binary_cross_entropy_with_logits(p, (t > 0).float())


def train_reghead(
    head: RegVeloHead,
    hidden: torch.Tensor,
    expr: torch.Tensor,
    X_bin: np.ndarray,
    seg_masks: List[np.ndarray],
    device: torch.device,
) -> List[float]:
    pairs = build_training_pairs(seg_masks)
    if not pairs:
        return []

    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
    losses: List[float] = []

    for epoch in range(ARGS.epochs):
        epoch_loss = 0.0
        for early_mask, late_mask in pairs:
            true_d = torch.tensor(
                segment_true_delta(X_bin, early_mask, late_mask),
                dtype=torch.float32,
                device=device,
            )
            top_n = max(int(len(true_d) * TOP_PERCENT / 100), 1)
            top_idx = torch.tensor(
                np.argsort(np.abs(true_d.detach().cpu().numpy()))[::-1][:top_n].copy(),
                device=device,
                dtype=torch.long,
            )

            h_e = hidden[torch.from_numpy(early_mask)].mean(dim=0, keepdim=True).to(device)
            h_l = hidden[torch.from_numpy(late_mask)].mean(dim=0, keepdim=True).to(device)
            x_e = expr[torch.from_numpy(early_mask)].mean(dim=0, keepdim=True).to(device)
            x_l = expr[torch.from_numpy(late_mask)].mean(dim=0, keepdim=True).to(device)

            pred_d = head(h_l, x_l) - head(h_e, x_e)
            loss = direction_loss(pred_d.reshape(-1), true_d, top_idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        losses.append(epoch_loss / max(len(pairs), 1))

    head.eval()
    return losses


def evaluate_head_on_endpoint(
    head: RegVeloHead,
    hidden: torch.Tensor,
    expr: torch.Tensor,
    bundle: DatasetBundle,
) -> Dict[str, Optional[float]]:
    pred_delta = predict_delta_from_masks(
        head, hidden, expr, bundle.early_mask, bundle.late_mask
    )
    chip_idx = np.array([i for i, g in enumerate(bundle.genes) if g in bundle.chip_targets], dtype=int)
    chip_in_top = np.array([i for i in bundle.top_idx if i in set(chip_idx.tolist())], dtype=int)

    return {
        "direction_acc_top": direction_accuracy(pred_delta, bundle.true_delta, bundle.top_idx),
        "direction_acc_all": direction_accuracy(
            pred_delta, bundle.true_delta, np.arange(len(bundle.genes))
        ),
        "direction_acc_chip_targets": direction_accuracy_mask(pred_delta, bundle.true_delta, chip_idx),
        "direction_acc_chip_in_top": direction_accuracy_mask(pred_delta, bundle.true_delta, chip_in_top),
        "n_chip_targets": int(len(chip_idx)),
        "n_chip_in_top": int(len(chip_in_top)),
    }


@dataclass
class DatasetCache:
    hidden: torch.Tensor
    expr: torch.Tensor
    fm_acc: float
    fm_pred: np.ndarray


def build_dataset_cache(
    bundle: DatasetBundle,
    model,
    device: torch.device,
) -> DatasetCache:
    update_mask = np.zeros(bundle.gene_ids_tensor.shape[1], dtype=bool)
    update_mask[1:] = True

    print(f"  [cache] encoding {bundle.values_tensor.shape[0]} cells ...", flush=True)
    hidden, expr = encode_cells(
        model,
        bundle.gene_ids_tensor,
        bundle.values_tensor,
        bundle.pad_mask,
        device,
        ARGS.batch_size,
    )
    print(f"  [cache] running fm iterative baseline ({ARGS.fm_iters} iters) ...", flush=True)
    fm_acc, fm_pred = fm_iterative_baseline(
        model,
        bundle.gene_ids_tensor,
        bundle.values_tensor[bundle.early_mask],
        bundle.pad_mask[bundle.early_mask],
        update_mask,
        bundle.early_mean,
        bundle.true_delta,
        bundle.top_idx,
        device,
        ARGS.fm_iters,
        ARGS.fm_ema_alpha,
        ARGS.batch_size,
    )
    print(f"  [cache] fm baseline direction acc = {fm_acc:.2%}", flush=True)
    return DatasetCache(hidden=hidden, expr=expr, fm_acc=fm_acc, fm_pred=fm_pred)


def run_one_mode(
    init_mode: str,
    bundle: DatasetBundle,
    cache: DatasetCache,
    device: torch.device,
    expr_root: Path,
) -> Dict:
    print(f"\n{'=' * 60}\n{bundle.name} | init_mode={init_mode}\n{'=' * 60}", flush=True)

    fm_acc = cache.fm_acc
    if init_mode == "fm_only":
        chip_idx = np.array([i for i, g in enumerate(bundle.genes) if g in bundle.chip_targets], dtype=int)
        chip_in_top = np.array([i for i in bundle.top_idx if i in set(chip_idx.tolist())], dtype=int)
        metrics = {
            "direction_acc_top": fm_acc,
            "direction_acc_chip_targets": direction_accuracy_mask(cache.fm_pred, bundle.true_delta, chip_idx),
            "direction_acc_chip_in_top": direction_accuracy_mask(cache.fm_pred, bundle.true_delta, chip_in_top),
            "n_chip_targets": int(len(chip_idx)),
            "n_chip_in_top": int(len(chip_in_top)),
        }
        return {
            "init_mode": init_mode,
            "metrics": metrics,
            "train_losses": [],
            "grn_edges": 0,
            "fm_baseline_acc": fm_acc,
        }

    grn_tensors: Optional[GrnTensors] = None
    if init_mode != "none":
        grn_df = load_grn_edges(
            bundle.name,
            set(bundle.genes),
            expr_root=expr_root,
            extraction=ARGS.grn_extraction,
            shuffle=(init_mode == "shuffle_init"),
            shuffle_seed=ARGS.grn_shuffle_seed,
        )
        grn_tensors = build_grn_tensors(bundle.genes, grn_df)

    d_model = cache.hidden.shape[-1]
    head = build_regvelo_head(d_model, grn_tensors, init_mode, device)
    train_losses = train_reghead(head, cache.hidden, cache.expr, bundle.X_bin, bundle.seg_masks, device)
    metrics = evaluate_head_on_endpoint(head, cache.hidden, cache.expr, bundle)

    print(
        f"  [reghead] top={metrics['direction_acc_top']:.2%} | "
        f"chip={metrics['direction_acc_chip_targets']} | "
        f"chip∩top={metrics['direction_acc_chip_in_top']} | "
        f"fm_base={fm_acc:.2%}",
        flush=True,
    )

    return {
        "init_mode": init_mode,
        "metrics": metrics,
        "train_losses": train_losses,
        "grn_edges": int(grn_tensors.n_edges if grn_tensors else 0),
        "fm_baseline_acc": fm_acc,
        "final_train_loss": train_losses[-1] if train_losses else None,
    }


def build_dataset_cfg(name: str, expr_root: Path, pt_root: Path) -> dict:
    return {
        "expr_csv": str(expr_root / "CHIP" / f"{name}_chip_matched-ExpressionData.csv"),
        "pt_csv": str(pt_root / name / "PseudoTime.csv"),
        "species": DATASET_SPECS[name],
    }


def run_experiment(
    model,
    vocab,
    device: torch.device,
    outdir: Path,
    expr_root: Path,
    pt_root: Path,
    chip_dir: Path,
) -> Dict:
    init_modes = ["fm_only", "none", "grn_init", "random_init", "shuffle_init"] if ARGS.sweep_inits else [ARGS.init_mode]

    all_results: Dict[str, Dict] = {}
    rows: List[Dict] = []

    for ds in ARGS.datasets:
        print(f"\n>>> Dataset: {ds}", flush=True)
        bundle = load_dataset_bundle(ds, expr_root, pt_root, chip_dir, vocab, device)
        cache = build_dataset_cache(bundle, model, device)
        ds_payload: Dict[str, Dict] = {}
        for mode in init_modes:
            payload = run_one_mode(mode, bundle, cache, device, expr_root)
            ds_payload[mode] = payload
            m = payload["metrics"]
            rows.append(
                {
                    "dataset": ds,
                    "init_mode": mode,
                    "direction_acc_top": m.get("direction_acc_top"),
                    "direction_acc_chip_targets": m.get("direction_acc_chip_targets"),
                    "direction_acc_chip_in_top": m.get("direction_acc_chip_in_top"),
                    "fm_baseline_acc": payload.get("fm_baseline_acc"),
                    "delta_vs_fm": (
                        (m.get("direction_acc_top") or 0) - (payload.get("fm_baseline_acc") or 0)
                        if m.get("direction_acc_top") is not None
                        else None
                    ),
                    "n_chip_targets": m.get("n_chip_targets"),
                    "n_chip_in_top": m.get("n_chip_in_top"),
                    "grn_edges": payload.get("grn_edges"),
                    "final_train_loss": payload.get("final_train_loss"),
                    "n_segments": N_SEGMENTS,
                    "grn_extraction": ARGS.grn_extraction,
                }
            )
        all_results[ds] = ds_payload

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "reghead_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "reghead_summary.csv", index=False)
    print(f"\n[OK] Wrote {outdir / 'reghead_results.json'}")
    print(f"[OK] Wrote {outdir / 'reghead_summary.csv'}")
    return all_results


def main():
    set_seed(ARGS.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, vocab, d_model = build_model(ARGS.model_dir, device)
    print(f"d_model={d_model} | segments={N_SEGMENTS} | epochs={ARGS.epochs}")

    run_experiment(
        model,
        vocab,
        device,
        Path(ARGS.outdir),
        Path(ARGS.expr_root),
        Path(ARGS.pt_root),
        Path(ARGS.chip_dir),
    )


if __name__ == "__main__":
    main()
