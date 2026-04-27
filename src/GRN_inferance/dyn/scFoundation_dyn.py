
#
# Output:
#   OUTDIR/
#     accuracy_curves.json
#     diagnostics.json
#     convergence_curves.png
#     <dataset_name>/
#        preds_910_mae.npy
#        preds_full_mae.npy
#        acc_curve.npy
#        gene_mapping_to_model.csv
#        gene_delta_compare.csv
#        norm_confusion_matrix.png
#        meta.json

import os
import json
import random
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

def parse_args():
    p = argparse.ArgumentParser(description="scFoundation pseudotime direction-accuracy benchmark (dynamic).")
    p.add_argument("--scfoundation-root", required=True, type=str, help="Directory containing pretrainmodels module.")
    p.add_argument("--ckpt-path", required=True, type=str, help="scFoundation checkpoint path.")
    p.add_argument("--gene-index-tsv", required=True, type=str, help="scFoundation gene index TSV.")
    p.add_argument("--outdir", required=True, type=str, help="Output directory.")
    p.add_argument("--datasets-json", default="", type=str, help="Datasets JSON path. If empty, build from --expr-root and --pt-root.")
    p.add_argument("--expr-root", default="", type=str, help="Expression root directory (CHIP/*.csv).")
    p.add_argument("--pt-root", default="", type=str, help="Pseudotime root directory (<dataset>/PseudoTime.csv).")
    return p.parse_args()


ARGS = parse_args()

# =========================================================
# Paths from CLI
# =========================================================
SCFOUNDATION_ROOT = ARGS.scfoundation_root
CKPT_PATH = ARGS.ckpt_path
MMF_KEY = "gene"
GENE_INDEX_TSV = ARGS.gene_index_tsv

# =========================================================
# Multi-dataset config
# =========================================================
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

# =========================================================
# Key params
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODE = "mae"  # "zero" or "mae"

# pseudotime split
PT_QUANTILE = 0.2

# evaluation
EVAL_MODE = "topk"  # ONLY: "topk" | "all_mapped"
TOP_PERCENT = 30  # MODIFIED: 评估前5%的mapped基因（替代固定TOPK）
EPS_DIR = 1e-3  # direction threshold for defining up/down/zero

# iterative inference
N_ITERS = 100
BATCH_CELLS = 16
UPDATE_ALPHA = 0.1

# MAE mask params
VALUE_MASK_PROB = 0.3
ZERO_MASK_PROB = 0.0
SEED = 1234

# scGPT-like option A
REFRESH_ENCODER_EACH_ITER = True
RESAMPLE_MASK_EACH_ITER = True

# update scope: "mask" | "zero" | "present_all"
UPDATE_SCOPE = "present_all"

# calibration
ENABLE_MEAN_MATCH_CALIBRATION = False
CALIBRATE_ON = "present"  # "present" or "present_nonzero"

# input transform:
# your data is already log: keep identity (no log1p, no 0-50 scaling)
USE_IDENTITY_INPUT = True

# outputs
SAVE_GENE_CSV = True
PLOT_NORM_CONFUSION = True

# =========================================================
# Model config template
# =========================================================
MODEL_CFG = {
    "model": "mae_autobin",
    "seq_len": 19266,
    "n_class": 100,
    "bin_alpha": 1.0,
    "bin_num": 100,
    "pad_token_id": 0,
    "mask_token_id": 1,
    "encoder": {
        "module_type": "transformer",
        "hidden_dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "ff_dropout": 0.0,
        "attn_dropout": 0.0,
    },
    "decoder": {
        "module_type": "transformer",
        "hidden_dim": 512,
        "depth": 6,
        "heads": 8,
        "dim_head": 64,
        "ff_dropout": 0.0,
        "attn_dropout": 0.0,
    },
    "ppi_edge": None,
}

# =========================================================
# Import scFoundation select_model
# =========================================================
import sys

if SCFOUNDATION_ROOT not in sys.path:
    sys.path.insert(0, SCFOUNDATION_ROOT)

from pretrainmodels import select_model  # noqa: E402


# =========================================================
# Utils
# =========================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_mouse_to_human_gene(g):
    # simple rule consistent with your other scripts
    return str(g).upper()


def _strip_prefix(sd: dict, prefix: str):
    out = {}
    for k, v in sd.items():
        out[k[len(prefix) :] if k.startswith(prefix) else k] = v
    return out


def gatherData(data: torch.Tensor, labels: torch.Tensor, pad_token_id: int):
    value_nums = labels.sum(1)
    max_num = int(value_nums.max().item())

    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device)
    data2 = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1, device=labels.device)

    none_labels = ~labels
    labels_f = labels.float()
    labels_f[none_labels] = torch.tensor(-float("Inf"), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels_f = labels_f + tmp_data

    labels_f = torch.hstack([labels_f, fake_label])
    idx = labels_f.topk(max_num).indices

    new_data = torch.gather(data2, 1, idx)
    padding_labels = new_data.eq(pad_token_id)
    return new_data, padding_labels


def load_model_mmf_gene(ckpt_path: str, device: torch.device, mmf_key: str = "gene"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if mmf_key not in ckpt:
        raise RuntimeError(f"Top keys={list(ckpt.keys())}, missing key='{mmf_key}'")

    gene_block = ckpt[mmf_key]
    if not isinstance(gene_block, dict) or "state_dict" not in gene_block:
        raise RuntimeError(f"ckpt['{mmf_key}'] should contain 'state_dict'")

    sd = _strip_prefix(gene_block["state_dict"], "model.")

    if "pos_emb.weight" in sd:
        seq_len = int(sd["pos_emb.weight"].shape[0] - 1)
    else:
        seq_len = int(MODEL_CFG["seq_len"])

    has_enc_performer = any(k.startswith("encoder.performer.") for k in sd.keys())
    has_dec_performer = any(k.startswith("decoder.performer.") for k in sd.keys())
    enc_type = "performer" if has_enc_performer else "transformer"
    dec_type = "performer" if has_dec_performer else "transformer"

    config = dict(MODEL_CFG)
    config["seq_len"] = seq_len
    config["encoder"] = dict(config["encoder"])
    config["decoder"] = dict(config["decoder"])
    config["encoder"]["module_type"] = enc_type
    config["decoder"]["module_type"] = dec_type

    print(f"[INFO] encoder.module_type={enc_type}, decoder.module_type={dec_type}, seq_len={seq_len}")

    model = select_model(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()

    if missing:
        print(f"[WARN] missing keys (first 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] unexpected keys (first 20): {unexpected[:20]}")

    return model, config


def read_gene_index_tsv(path: str, G_model: int):
    df = pd.read_csv(path, sep="\t", header=0)
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("gene_name", df.columns[0])
    idx_col = cols.get("index", None)
    if idx_col is None:
        raise RuntimeError(f"TSV missing 'index' column. Columns={list(df.columns)}")

    df = df[[gene_col, idx_col]].dropna()
    df[gene_col] = df[gene_col].astype(str)
    df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce")
    df = df.dropna()
    df[idx_col] = df[idx_col].astype(int)
    df = df[(df[idx_col] >= 0) & (df[idx_col] < G_model)].copy()

    gene2idx = dict(zip(df[gene_col].tolist(), df[idx_col].tolist()))
    print(f"[INFO] gene_index loaded rows={len(df)}, index range=({df[idx_col].min()}..{df[idx_col].max()})")
    return gene2idx


def build_aligned_matrix(X: np.ndarray, genes: list, gene2idx_model: dict, G_model: int):
    """
    X: [cells, n_genes]
    return:
      X_full [cells, G_model]
      present_mask [G_model] bool
      map_idx [n_genes] -> model idx or -1
    """
    n_cells, n_genes = X.shape
    X_full = np.zeros((n_cells, G_model), dtype=np.float32)
    present_mask = np.zeros((G_model,), dtype=bool)
    map_idx = np.full((n_genes,), -1, dtype=np.int64)

    found = 0
    for j, g in enumerate(genes):
        idx = gene2idx_model.get(g, None)
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < G_model:
            X_full[:, idx] = X[:, j]
            present_mask[idx] = True
            map_idx[j] = idx
            found += 1

    return X_full, present_mask, map_idx, found


def make_masks_present_only(
    raw_full: torch.Tensor,
    present_mask: torch.Tensor,
    mode: str,
    value_mask_prob: float,
    zero_mask_prob: float,
    seed: int,
):
    """
    returns encoder_visible, update_pos (all confined to present genes only)
    """
    assert mode in ("zero", "mae")
    device = raw_full.device
    B, G = raw_full.shape
    present = present_mask.view(1, G).expand(B, G)
    nonzero_present = (raw_full > 0) & present

    if mode == "zero":
        encoder_visible = nonzero_present
        update_pos = (~nonzero_present) & present
        bad = encoder_visible.sum(dim=1) == 0
        if bad.any():
            first_present = torch.nonzero(present_mask, as_tuple=False).min().item()
            encoder_visible[bad, first_present] = True
            update_pos[bad, first_present] = False
        return encoder_visible, update_pos

    seed_all(seed)
    rnd = torch.rand((B, G), device=device)
    masked_nonzero = nonzero_present & (rnd < value_mask_prob)

    if zero_mask_prob > 0:
        masked_zero = ((~nonzero_present) & present) & (torch.rand((B, G), device=device) < zero_mask_prob)
    else:
        masked_zero = torch.zeros((B, G), dtype=torch.bool, device=device)

    update_pos = masked_nonzero | masked_zero
    encoder_visible = (~update_pos) & nonzero_present

    bad = encoder_visible.sum(dim=1) == 0
    if bad.any():
        encoder_visible[bad] = nonzero_present[bad]
        update_pos[bad] = masked_nonzero[bad]
        bad2 = encoder_visible.sum(dim=1) == 0
        if bad2.any():
            first_present = torch.nonzero(present_mask, as_tuple=False).min().item()
            encoder_visible[bad2, first_present] = True
            update_pos[bad2, first_present] = False

    return encoder_visible, update_pos


def build_io(vals: torch.Tensor, encoder_visible: torch.Tensor, config: dict):
    device = vals.device
    B, G = vals.shape

    decoder_data = vals
    decoder_data_padding = torch.full_like(decoder_data, False, dtype=torch.bool, device=device)

    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_visible, config["pad_token_id"])

    gene_ids = torch.arange(G, device=device).unsqueeze(0).repeat(B, 1)
    encoder_pos, _ = gatherData(gene_ids, encoder_visible, config["pad_token_id"])
    decoder_pos = gene_ids

    encoder_pos[encoder_data_padding] = config["seq_len"]
    decoder_pos[decoder_data_padding] = config["seq_len"]

    return encoder_data, encoder_data_padding, encoder_pos, encoder_visible, decoder_data, decoder_data_padding, decoder_pos


def mean_match_calibration(
    vals: torch.Tensor,
    inp: torch.Tensor,
    present_mask: torch.Tensor,
    raw_full: torch.Tensor,
    mode: str = "present",
):
    B, G = vals.shape
    present = present_mask.view(1, G).expand(B, G)
    if mode == "present_nonzero":
        sel = present & (raw_full > 0)
    else:
        sel = present

    denom = sel.sum(dim=1).clamp_min(1)
    m_in = (inp * sel).sum(dim=1) / denom
    m_out = (vals * sel).sum(dim=1) / denom
    shift = (m_in - m_out).view(B, 1)
    vals = vals + shift * present.float()
    return vals


@torch.no_grad()
def iterative_predict_curve(
    model,
    config: dict,
    values_full_init: torch.Tensor,  # [B,G]
    raw_full: torch.Tensor,  # [B,G]
    present_mask: torch.Tensor,  # [G]
    update_scope: str,
    eval_model_idx: np.ndarray,  # indices in model space (length K_mapped)
    early_mean_eval: np.ndarray,  # [K_mapped] in model space
    true_delta_eval: np.ndarray,  # [K_mapped] in model space
    n_iters: int,
    seed0: int,
):
    """
    Run iterative inference and return accuracy curve (direction acc) on eval_model_idx and final preds (full G).
    """
    device = next(model.parameters()).device
    vals = values_full_init.to(device)
    inp = values_full_init.to(device)
    raw_full = raw_full.to(device)
    present_mask = present_mask.to(device)
    B, G = vals.shape

    acc_curve = []

    encoder_visible, update_pos = make_masks_present_only(
        raw_full=raw_full,
        present_mask=present_mask,
        mode=MODE,
        value_mask_prob=VALUE_MASK_PROB,
        zero_mask_prob=ZERO_MASK_PROB,
        seed=seed0,
    )

    for it in range(n_iters):
        if MODE == "mae" and RESAMPLE_MASK_EACH_ITER:
            encoder_visible, update_pos = make_masks_present_only(
                raw_full=raw_full,
                present_mask=present_mask,
                mode=MODE,
                value_mask_prob=VALUE_MASK_PROB,
                zero_mask_prob=ZERO_MASK_PROB,
                seed=seed0 + it + 1,
            )

        if it == 0 or REFRESH_ENCODER_EACH_ITER:
            encoder_data, padding_label, enc_pos, enc_labels, dec_data, dec_pad, dec_pos = build_io(
                vals, encoder_visible, config
            )

        use_cuda = device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_cuda):
            pred = model(
                x=encoder_data,
                padding_label=padding_label,
                encoder_position_gene_ids=enc_pos,
                encoder_labels=enc_labels,
                decoder_data=dec_data,
                mask_gene_name=False,
                mask_labels=None,
                decoder_position_gene_ids=dec_pos,
                decoder_data_padding_labels=dec_pad,
            )  # [B,G]

        present = present_mask.view(1, G).expand(B, G)
        if update_scope == "mask":
            pos = update_pos
        elif update_scope == "zero":
            pos = ((raw_full <= 0) & present)
        else:  # present_all
            pos = present

        # EMA update
        if UPDATE_ALPHA >= 1.0:
            vals[pos] = pred[pos]
        else:
            vals[pos] = (1.0 - UPDATE_ALPHA) * vals[pos] + UPDATE_ALPHA * pred[pos]

        if ENABLE_MEAN_MATCH_CALIBRATION:
            vals = mean_match_calibration(vals, inp, present_mask, raw_full, mode=CALIBRATE_ON)

        # direction accuracy on eval_model_idx
        pred_mean_eval = vals[:, eval_model_idx].detach().float().mean(dim=0).cpu().numpy()  # [K]
        pred_delta_eval = pred_mean_eval - early_mean_eval

        true_dir = np.where(true_delta_eval > EPS_DIR, 1, np.where(true_delta_eval < -EPS_DIR, -1, 0))
        pred_dir = np.where(pred_delta_eval > EPS_DIR, 1, np.where(pred_delta_eval < -EPS_DIR, -1, 0))
        valid = true_dir != 0
        acc = float((pred_dir[valid] == true_dir[valid]).mean()) if valid.any() else float("nan")
        acc_curve.append(acc)

    return acc_curve, vals.detach().float().cpu().numpy()


def plot_norm_confusion(true_dir, pred_dir, outpath_png: Path, title: str = ""):
    import matplotlib.pyplot as plt

    true_dir = np.asarray(true_dir)
    pred_dir = np.asarray(pred_dir)

    def m(x):
        return np.where(x == -1, 0, np.where(x == 0, 1, 2))

    t = m(true_dir)
    p = m(pred_dir)

    cm = np.zeros((3, 3), dtype=np.float32)
    for i in range(len(t)):
        cm[t[i], p[i]] += 1

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1e-6))

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)

    labels = ["Down(-1)", "Zero(0)", "Up(+1)"]
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted direction")
    ax.set_ylabel("True direction")
    if title:
        ax.set_title(title)

    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=300)
    plt.close(fig)


# =========================================================
# Dataset runner
# =========================================================
def run_one_dataset(name: str, cfg: dict, model, config, gene2idx_model, out_root: Path):
    print(f"\n{'='*70}\nRunning {name}\n{'='*70}")

    # ----- load expr -----
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)  # genes x cells
    genes_original = expr.index.astype(str).tolist()

    # ----- load pseudotime -----
    pt_df = pd.read_csv(cfg["pt_csv"])
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")

    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    # ----- gene name normalize -----
    species = cfg.get("species", "human")
    if species == "mouse":
        genes = [convert_mouse_to_human_gene(g) for g in genes_original]
        print("  [INFO] Mouse data detected, converting gene names to uppercase.")
    else:
        genes = genes_original

    # ----- matrix [cells, n_genes] -----
    X = expr.T.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    print(f"  [INFO] Cells: {X.shape[0]}  Genes: {X.shape[1]}")
    print(f"  [SANITY] X min/max: {float(X.min()):.4f} / {float(X.max()):.4f}")

    # identity input for log-already
    X_proc = X.astype(np.float32) if USE_IDENTITY_INPUT else X.astype(np.float32)
    X_raw = X_proc.copy()  # for nonzero detection

    # ----- early/late by pseudotime quantiles -----
    lo, hi = np.quantile(pt, [PT_QUANTILE, 1 - PT_QUANTILE])
    early = pt <= lo
    late = pt >= hi
    print(f"  [INFO] pt quantiles: lo={lo:.3f} hi={hi:.3f}")
    print(f"  [INFO] Early cells: {int(early.sum())}  Late cells: {int(late.sum())}")
    if early.sum() == 0 or late.sum() == 0:
        print("  [ERROR] Early or Late split empty. Skip.")
        return None, None

    # ----- align into model gene space -----
    G_model = int(config["seq_len"])
    X_full, present_mask_np, map_idx, mapped = build_aligned_matrix(X_proc, genes, gene2idx_model, G_model)
    X_full_raw, _, _, _ = build_aligned_matrix(X_raw, genes, gene2idx_model, G_model)
    present_mask = torch.tensor(present_mask_np, dtype=torch.bool)

    mapped_rate = mapped / max(1, len(genes))
    print(f"  [INFO] Mapped genes into model space: {mapped}/{len(genes)} ({mapped_rate*100:.1f}%)  (G_model={G_model})")
    if mapped < 50:
        print("  [WARN] Very low mapping count; results likely unreliable.")

    # ----- true stats in ORIGINAL gene space -----
    early_mean_910 = X_proc[early].mean(axis=0)
    late_mean_910 = X_proc[late].mean(axis=0)
    true_delta_910 = late_mean_910 - early_mean_910

    # ----- choose evaluation set in ORIGINAL gene space (then map to model indices) -----
    mapped_mask_910 = map_idx >= 0
    if EVAL_MODE == "topk":
        idx_pool = np.where(mapped_mask_910)[0]
        if len(idx_pool) == 0:
            print("  [ERROR] No mapped genes. Skip.")
            return None, None
        
        # MODIFIED: 动态计算前TOP_PERCENT%的mapped基因（至少1个）
        total_mapped = len(idx_pool)
        top_n = max(int(total_mapped * TOP_PERCENT / 100), 1)
        print(f"  [INFO] EVAL_MODE=topk: selecting top {TOP_PERCENT}% of mapped genes ({top_n}/{total_mapped})")
        
        order = np.argsort(np.abs(true_delta_910[idx_pool]))[::-1]
        eval_910 = idx_pool[order[:top_n]].copy()  # 取前top_n个
        
    elif EVAL_MODE == "all_mapped":
        eval_910 = np.where(mapped_mask_910)[0].copy()
    else:
        raise ValueError("EVAL_MODE must be 'topk' or 'all_mapped'")

    eval_model_idx = map_idx[eval_910]
    keep = eval_model_idx >= 0
    eval_910 = eval_910[keep]
    eval_model_idx = eval_model_idx[keep].astype(np.int64)

    if len(eval_model_idx) == 0:
        print("  [ERROR] No mapped genes in evaluation set. Skip.")
        return None, None

    # model-space true stats for per-iter scoring
    early_mean_eval = X_full[early][:, eval_model_idx].mean(axis=0)
    late_mean_eval = X_full[late][:, eval_model_idx].mean(axis=0)
    true_delta_eval = (late_mean_eval - early_mean_eval).astype(np.float32)

    # ----- run iterative inference on EARLY cells only -----
    early_idx = np.where(early)[0]
    values_early_full = X_full[early_idx]
    raw_early_full = X_full_raw[early_idx]

    acc_curve_accum = []
    preds_full_all = np.zeros_like(values_early_full, dtype=np.float32)

    for s in range(0, len(early_idx), BATCH_CELLS):
        batch = slice(s, min(s + BATCH_CELLS, len(early_idx)))
        vals0 = torch.tensor(
            values_early_full[batch],
            dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        rawb = torch.tensor(raw_early_full[batch], dtype=torch.float32)

        curve, preds_full = iterative_predict_curve(
            model=model,
            config=config,
            values_full_init=vals0,
            raw_full=rawb,
            present_mask=present_mask,
            update_scope=UPDATE_SCOPE,
            eval_model_idx=eval_model_idx,
            early_mean_eval=early_mean_eval,
            true_delta_eval=true_delta_eval,
            n_iters=N_ITERS,
            seed0=SEED + s,
        )

        preds_full_all[batch] = preds_full
        acc_curve_accum.append(curve)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  [INFO] done batch {s}..{min(s+BATCH_CELLS, len(early_idx))}")

    acc_curve = np.mean(np.array(acc_curve_accum, dtype=np.float32), axis=0).tolist()
    
    # MODIFIED: 更新结果打印，显示百分比信息
    if EVAL_MODE == "topk":
        print(f"  [RESULT] Final acc({EVAL_MODE}, Top-{TOP_PERCENT}%) = {acc_curve[-1]:.3f}")
    else:
        print(f"  [RESULT] Final acc({EVAL_MODE}) = {acc_curve[-1]:.3f}")

    # ----- map predictions back to input gene order for saving -----
    preds_910 = np.zeros((values_early_full.shape[0], X_proc.shape[1]), dtype=np.float32)  # [n_early, n_genes_input]
    for j in range(len(genes)):
        idx = map_idx[j]
        if idx >= 0:
            preds_910[:, j] = preds_full_all[:, idx]
        else:
            preds_910[:, j] = X_proc[early_idx, j]  # unmapped: keep original early values

    # ----- save per-dataset outputs -----
    outdir = out_root / name
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / f"preds_910_{MODE}.npy", preds_910)
    np.save(outdir / f"preds_full_{MODE}.npy", preds_full_all)
    np.save(outdir / "acc_curve.npy", np.array(acc_curve, dtype=np.float32))

    map_df = pd.DataFrame(
        {
            "gene": genes_original,
            "gene_used": genes,
            "model_index": map_idx,
            "mapped": (map_idx >= 0),
        }
    )
    map_df.to_csv(outdir / "gene_mapping_to_model.csv", index=False)

    # ----- gene-level CSV comparison -----
    pred_mean_910 = preds_910.mean(axis=0)
    pred_delta_910 = pred_mean_910 - early_mean_910

    true_dir_910 = np.where(true_delta_910 > EPS_DIR, 1, np.where(true_delta_910 < -EPS_DIR, -1, 0))
    pred_dir_910 = np.where(pred_delta_910 > EPS_DIR, 1, np.where(pred_delta_910 < -EPS_DIR, -1, 0))
    dir_match = true_dir_910 == pred_dir_910

    in_eval = np.zeros(len(genes), dtype=bool)
    in_eval[eval_910] = True

    if SAVE_GENE_CSV:
        df_cmp = pd.DataFrame(
            {
                "gene": genes_original,
                "gene_used": genes,
                "mapped": (map_idx >= 0),
                "model_index": map_idx,
                "in_eval_set": in_eval,
                "early_mean": early_mean_910,
                "late_mean": late_mean_910,
                "true_delta": true_delta_910,
                "pred_mean": pred_mean_910,
                "pred_delta": pred_delta_910,
                "true_dir": true_dir_910,
                "pred_dir": pred_dir_910,
                "dir_match": dir_match,
            }
        )
        df_cmp.to_csv(outdir / "gene_delta_compare.csv", index=False)

    # ----- normalized confusion matrix (only on eval set) -----
    if PLOT_NORM_CONFUSION:
        td = true_dir_910[in_eval]
        pd_ = pred_dir_910[in_eval]
        # MODIFIED: 更新混淆矩阵标题，显示百分比
        if EVAL_MODE == "topk":
            title = f"{name} (Top-{TOP_PERCENT}%)"
        else:
            title = f"{name} ({EVAL_MODE})"
        plot_norm_confusion(td, pd_, outdir / "norm_confusion_matrix.png", title=title)

    # MODIFIED: 更新meta信息，记录百分比和实际评估基因数
    meta = {
        "dataset": name,
        "expr_csv": cfg["expr_csv"],
        "pt_csv": cfg["pt_csv"],
        "species": species,
        "device": str(DEVICE),
        "mode": MODE,
        "ckpt": CKPT_PATH,
        "gene_index_tsv": GENE_INDEX_TSV,
        "G_model": int(config["seq_len"]),
        "n_cells_total": int(X.shape[0]),
        "n_genes_input": int(X.shape[1]),
        "mapped_genes": int(mapped),
        "mapped_rate": float(mapped_rate),
        "pt_quantile": PT_QUANTILE,
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "eval_mode": EVAL_MODE,
        "top_percent": float(TOP_PERCENT) if EVAL_MODE == "topk" else None,  # 新增：百分比参数
        "top_n_actual": int(top_n) if EVAL_MODE == "topk" else None,        # 新增：实际评估基因数
        "eval_genes_mapped_used": int(len(eval_model_idx)),
        "iters": N_ITERS,
        "batch_cells": BATCH_CELLS,
        "update_alpha": UPDATE_ALPHA,
        "refresh_encoder_each_iter": REFRESH_ENCODER_EACH_ITER,
        "resample_mask_each_iter": RESAMPLE_MASK_EACH_ITER,
        "update_scope": UPDATE_SCOPE,
        "value_mask_prob": VALUE_MASK_PROB,
        "zero_mask_prob": ZERO_MASK_PROB,
        "calibration": ENABLE_MEAN_MATCH_CALIBRATION,
        "calibrate_on": CALIBRATE_ON if ENABLE_MEAN_MATCH_CALIBRATION else None,
        "final_acc": float(acc_curve[-1]),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # MODIFIED: 更新diagnostics，记录百分比相关信息
    diagnostics = {
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "mapped_genes": int(mapped),
        "mapped_rate": float(mapped_rate),
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "eval_mode": EVAL_MODE,
        "top_percent": float(TOP_PERCENT) if EVAL_MODE == "topk" else None,
        "top_n_actual": int(top_n) if EVAL_MODE == "topk" else None,
        "eval_genes_used": int(len(eval_model_idx)),
        "final_acc": float(acc_curve[-1]),
    }
    return acc_curve, diagnostics


# =========================================================
# Plot curves across datasets
# =========================================================
def plot_curves(all_curves: dict, outdir: Path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.2, 3.2))
    ax = fig.add_subplot(111)
    for name, curve in all_curves.items():
        ax.plot(range(1, len(curve) + 1), curve, marker="o", linewidth=1.5, label=name)
    
    # MODIFIED: 更新Y轴标签，显示百分比
    if EVAL_MODE == "topk":
        ax.set_ylabel(f"Direction accuracy (Top-{TOP_PERCENT}%)")
    else:
        ax.set_ylabel(f"Direction accuracy ({EVAL_MODE})")
        
    ax.set_xlabel("Iteration")
    ax.set_xlim(0.5, max(len(v) for v in all_curves.values()) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "convergence_curves.png", dpi=200)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    print(f"Using device: {DEVICE}")
    # MODIFIED: 打印百分比参数
    if EVAL_MODE == "topk":
        print(f"[INFO] EVAL_MODE={EVAL_MODE} (Top-{TOP_PERCENT}% of mapped genes)")
    else:
        print(f"[INFO] EVAL_MODE={EVAL_MODE} (only 'topk' or 'all_mapped')")

    model, config = load_model_mmf_gene(CKPT_PATH, DEVICE, mmf_key=MMF_KEY)
    G_model = int(config["seq_len"])
    print(f"[INFO] G_model={G_model} MODE={MODE} iters={N_ITERS} batch={BATCH_CELLS}")
    print(f"[INFO] REFRESH={REFRESH_ENCODER_EACH_ITER} RESAMPLE={RESAMPLE_MASK_EACH_ITER} UPDATE_SCOPE={UPDATE_SCOPE}")
    print(f"[INFO] CALIBRATION={ENABLE_MEAN_MATCH_CALIBRATION} CALIBRATE_ON={CALIBRATE_ON}")

    gene2idx_model = read_gene_index_tsv(GENE_INDEX_TSV, G_model)

    out_root = Path(OUTDIR)
    out_root.mkdir(parents=True, exist_ok=True)

    all_curves = {}
    all_diags = {}

    for name, cfg in DATASETS.items():
        curve, diag = run_one_dataset(name, cfg, model, config, gene2idx_model, out_root)
        if curve is None:
            continue
        all_curves[name] = curve
        all_diags[name] = diag

    (out_root / "accuracy_curves.json").write_text(json.dumps(all_curves, indent=2), encoding="utf-8")
    (out_root / "diagnostics.json").write_text(json.dumps(all_diags, indent=2), encoding="utf-8")

    if len(all_curves) > 0:
        plot_curves(all_curves, out_root)

    # summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    # MODIFIED: 扩展汇总表格，显示百分比和实际评估基因数
    if EVAL_MODE == "topk":
        print(f"{'Dataset':<12} {'Mapped%':<10} {'EvalGenes':<10} {'Early':<8} {'Late':<8} {'FinalAcc':<10}")
        print("-" * 80)
        for name, d in all_diags.items():
            print(f"{name:<12} {100*d['mapped_rate']:<10.1f} {d['top_n_actual']:<10d} {d['n_early']:<8d} {d['n_late']:<8d} {d['final_acc']:<10.3f}")
    else:
        print(f"{'Dataset':<12} {'Mapped%':<10} {'Early':<8} {'Late':<8} {'FinalAcc':<10}")
        print("-" * 80)
        for name, d in all_diags.items():
            print(f"{name:<12} {100*d['mapped_rate']:<10.1f} {d['n_early']:<8d} {d['n_late']:<8d} {d['final_acc']:<10.3f}")
    print("=" * 80)
    print(f"[OK] Saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()