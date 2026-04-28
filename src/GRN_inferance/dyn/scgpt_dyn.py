


import os
import re
import json
import warnings
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

# =====================================================
# CLI ()
# =====================================================
def parse_args():
    p = argparse.ArgumentParser(description="scGPT pseudotime direction-accuracy benchmark (dynamic).")
    p.add_argument("--model-dir", required=True, type=str, help="scGPT model dir containing args.json/vocab.json/best_model.pt.")
    p.add_argument("--outdir", required=True, type=str, help="Output directory.")
    p.add_argument("--datasets-json", default="", type=str, help="Datasets JSON path. If empty, build from --expr-root and --pt-root.")
    p.add_argument("--expr-root", default="", type=str, help="Expression root directory (CHIP/*.csv).")
    p.add_argument("--pt-root", default="", type=str, help="Pseudotime root directory (<dataset>/PseudoTime.csv).")
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--log1p", action="store_true", help="Use log1p in binning (default off to match previous NO_LOG1P=True).")
    return p.parse_args()


ARGS = parse_args()

MODEL_DIR = ARGS.model_dir
OUTDIR = ARGS.outdir

DATASET_SPECS = {
    "hESC": "human",
    "hHep": "human",
    "mDC": "mouse",
    "mHSC-E": "mouse",
    "mHSC-GM": "mouse",
    "mHSC-L": "mouse",
}


def build_datasets_from_roots(expr_root: str, pt_root: str):
    if not expr_root or not pt_root:
        raise ValueError("When --datasets-json is not set, both --expr-root and --pt-root are required.")
    datasets = {}
    for ds, species in DATASET_SPECS.items():
        datasets[ds] = {
            "expr_csv": str(Path(expr_root) / "CHIP" / f"{ds}_chip_matched-ExpressionData.csv"),
            "pt_csv": str(Path(pt_root) / ds / "PseudoTime.csv"),
            "species": species,
        }
    return datasets


def load_datasets_config(args):
    if args.datasets_json:
        with open(args.datasets_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict) or not cfg:
            raise ValueError("datasets-json must be a non-empty object.")
        return cfg
    return build_datasets_from_roots(args.expr_root, args.pt_root)


DATASETS = load_datasets_config(ARGS)

# =====================================================
# (:TOP_PERCENTTOPK)
# =====================================================
PT_QUANTILE = float(ARGS.pt_quantile)
TOP_PERCENT = int(ARGS.top_percent)
GEN_ITERS = int(ARGS.gen_iters)
BATCH_SIZE = int(ARGS.batch_size)
EMA_ALPHA = float(ARGS.ema_alpha)
NO_LOG1P = (not bool(ARGS.log1p))

# =====================================================
# scGPT
# =====================================================
import sys
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab


# =====================================================
# Utils
# =====================================================
def bin_expr_to_0_50(x, do_log1p=True):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(np.percentile(x, 99.5), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def convert_mouse_to_human_gene(gene_name):
    """
    :
    :  (e.g., Gapdh)
    :  (e.g., GAPDH)
    """
    return gene_name.upper()


# =====================================================
# Build model
# =====================================================
def build_model(model_dir, device):
    with open(Path(model_dir) / "args.json") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=cfg.get("fast_transformer", True),
    )

    ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        model.half()

    return model, vocab


# =====================================================
# Iterative generation + accuracy curve
# =====================================================
@torch.no_grad()
def iterative_direction_accuracy(
    model,
    gene_ids_tensor,
    values_tensor,
    pad_mask,
    update_mask_1d,
    early_mean,
    true_delta,
    top_idx,  # DatasetN%
):
    device = next(model.parameters()).device
    vals_all = values_tensor.clone()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()

    acc_curve = []

    for it in range(GEN_ITERS):
        for start in range(0, vals_all.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals_all.shape[0])
            bs = end - start

            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(bs, -1).to(device)
            mask = pad_mask[start:end].to(device)

            freeze = mask | (~update_mask.expand(bs, -1))
            out = model(src=src, values=vals, src_key_padding_mask=mask)
            new_vals = out["mlm_output"]

            vals = torch.where(
                freeze,
                vals,
                EMA_ALPHA * vals + (1 - EMA_ALPHA) * new_vals
            )

            vals_all[start:end] = vals.detach().cpu()

        # N%
        pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
        pred_delta = pred_mean - early_mean
        acc = float((np.sign(pred_delta[top_idx]) == np.sign(true_delta[top_idx])).mean())
        acc_curve.append(acc)

    return acc_curve


# =====================================================
# Run one dataset
# =====================================================
def run_dataset(name, cfg, model, vocab, device):
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")

    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    # 
    genes_original = expr.index.astype(str).tolist()
    
    # ,
    species = cfg.get("species", "human")
    if species == "mouse":
        genes = [convert_mouse_to_human_gene(g) for g in genes_original]
        print(f"  [INFO] Mouse data detected, converting gene names to uppercase")
    else:
        genes = genes_original
    
    # vocab
    matched = sum(1 for g in genes if g in vocab)
    match_rate = matched / len(genes) * 100
    print(f"  [INFO] Total genes: {len(genes)}")
    print(f"  [INFO] Vocab matched: {matched} ({match_rate:.1f}%)")
    print(f"  [INFO] Total cells: {len(common)}")

    X = expr.T.to_numpy(dtype=np.float32)
    X_bin = bin_expr_to_0_50(X, do_log1p=(not NO_LOG1P))

    lo, hi = np.quantile(pt, [PT_QUANTILE, 1 - PT_QUANTILE])
    early = pt <= lo
    late = pt >= hi
    print(f"  [INFO] Early cells (pt <= {lo:.3f}): {early.sum()}")
    print(f"  [INFO] Late cells (pt >= {hi:.3f}): {late.sum()}")

    early_mean = X_bin[early].mean(axis=0)
    late_mean = X_bin[late].mean(axis=0)
    true_delta = late_mean - early_mean

    # ==============================================
    # :DatasetN%(Timestamp)
    # ==============================================
    total_genes = len(true_delta)
    # TOP_PERCENT%,1(Dataset0)
    top_n = max(int(total_genes * TOP_PERCENT / 100), 1)
    # top_n
    top_idx = np.argsort(np.abs(true_delta))[::-1][:top_n].copy()
    
    print(f"  [INFO] Top-{TOP_PERCENT}% genes selected for evaluation:")
    print(f"         - Total genes in dataset: {total_genes}")
    print(f"         - Actual evaluated genes: {top_n} (Top-{TOP_PERCENT}%)")

    # N%vocab
    top_genes = [genes[i] for i in top_idx]
    top_matched = sum(1 for g in top_genes if g in vocab)
    top_match_rate = top_matched / len(top_genes) * 100
    print(f"  [INFO] Top-{TOP_PERCENT}% genes vocab matched: {top_matched} ({top_match_rate:.1f}%)")

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    values_tensor = torch.tensor(X_in, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    # ()
    update_mask_1d = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask_1d[1:] = True  # <cls> token,
    print(f"  [INFO] Iteration covers ALL {len(genes)} genes (Top-{TOP_PERCENT}% for evaluation only)")

    acc_curve = iterative_direction_accuracy(
        model,
        gene_ids_tensor,
        values_tensor[early],
        pad_mask[early],
        update_mask_1d,
        early_mean,
        true_delta,
        top_idx,
    )
    
    print(f"  [RESULT] Final accuracy (Top-{TOP_PERCENT}% genes): {acc_curve[-1]:.2%}")
    
    # ()
    diagnostics = {
        "n_genes": len(genes),
        "n_cells": len(common),
        "vocab_match_rate": match_rate,
        "top_vocab_match_rate": top_match_rate,
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "iter_genes_count": len(genes),
        "eval_percent": TOP_PERCENT,       # 
        "eval_genes_count": top_n,         # 
    }
    
    return acc_curve, diagnostics


# =====================================================
# Nature-style plotting
# =====================================================
def plot_nature_style(all_curves, diagnostics, outdir):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Nature
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    # Nature
    colors = {
        'hHep': '#E64B35',      # 
        'mDC': '#4DBBD5',       # 
        'mHSC-E': '#00A087',    # 
        'mHSC-GM': '#3C5488',   # 
    }
    
    # ==================== Figure 1:  ====================
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    for name, acc in all_curves.items():
        ax.plot(range(1, GEN_ITERS + 1), acc, 
                marker='o', markersize=4, linewidth=1.5,
                color=colors.get(name, '#666666'),
                label=name)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xlim(0.5, GEN_ITERS + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(1, GEN_ITERS + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(outdir / "fig1_convergence.pdf")
    plt.savefig(outdir / "fig1_convergence.png", dpi=300)
    plt.close()
    
    # ==================== Figure 2:  ====================
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    
    names = list(all_curves.keys())
    final_acc = [all_curves[n][-1] for n in names]
    bar_colors = [colors.get(n, '#666666') for n in names]
    
    bars = ax.bar(range(len(names)), final_acc, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)
    
    # 
    for i, (bar, v) in enumerate(zip(bars, final_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(outdir / "fig2_final_accuracy.pdf")
    plt.savefig(outdir / "fig2_final_accuracy.png", dpi=300)
    plt.close()
    
    # ==================== Figure 3:  - Vocab vs  ====================
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    
    for name in names:
        match_rate = diagnostics[name]['top_vocab_match_rate']
        acc = all_curves[name][-1]
        ax.scatter(match_rate, acc, s=60, c=colors.get(name, '#666666'), 
                   edgecolor='black', linewidth=0.5, label=name, zorder=3)
    
    ax.set_xlabel(f'Top-{TOP_PERCENT}% genes vocab match rate (%)')
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xlim(0, 105)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(outdir / "fig3_vocab_vs_accuracy.pdf")
    plt.savefig(outdir / "fig3_vocab_vs_accuracy.png", dpi=300)
    plt.close()
    
    # ==================== Figure 4:  () ====================
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    
    # Panel A: 
    ax = axes[0]
    for name, acc in all_curves.items():
        ax.plot(range(1, GEN_ITERS + 1), acc, 
                marker='o', markersize=4, linewidth=1.5,
                color=colors.get(name, '#666666'),
                label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xlim(0.5, GEN_ITERS + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(1, GEN_ITERS + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=7)
    ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    # Panel B: 
    ax = axes[1]
    bars = ax.bar(range(len(names)), final_acc, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)
    for i, (bar, v) in enumerate(zip(bars, final_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outdir / "fig_combined.pdf")
    plt.savefig(outdir / "fig_combined.png", dpi=300)
    plt.close()
    
    print(f"\n[OK] Nature-style figures saved to: {outdir}")


# =====================================================
# Main
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, vocab = build_model(MODEL_DIR, device)
    print(f"Vocab size: {len(vocab)}")
    # ()
    print(f"Run mode: Iterate ALL genes, Evaluate Top-{TOP_PERCENT}% genes (largest expression change)")

    all_curves = {}
    all_diagnostics = {}

    for name, cfg in DATASETS.items():
        acc_curve, diag = run_dataset(name, cfg, model, vocab, device)
        all_curves[name] = acc_curve
        all_diagnostics[name] = diag

    outdir = Path(OUTDIR)
    outdir.mkdir(exist_ok=True, parents=True)

    # Save results
    with open(outdir / "accuracy_curves.json", "w") as f:
        json.dump(all_curves, f, indent=2)
    
    with open(outdir / "diagnostics.json", "w") as f:
        json.dump(all_diagnostics, f, indent=2)

    # Nature
    plot_nature_style(all_curves, all_diagnostics, outdir)
    
    # ()
    print("\n" + "="*80)
    print(f"SUMMARY (Top-{TOP_PERCENT}% genes evaluation)")
    print("="*80)
    # ,Dataset
    print(f"{'Dataset':<12} {'Total Genes':<12} {'Eval Genes':<12} {'Vocab%':<10} {'Top% Vocab%':<12} {'Accuracy':<10}")
    print("-"*80)
    for name in all_curves:
        d = all_diagnostics[name]
        acc = all_curves[name][-1]
        print(f"{name:<12} {d['n_genes']:<12} {d['eval_genes_count']:<12} {d['vocab_match_rate']:<10.1f} {d['top_vocab_match_rate']:<12.1f} {acc:<10.2%}")
    print("="*80)


if __name__ == "__main__":
    main()