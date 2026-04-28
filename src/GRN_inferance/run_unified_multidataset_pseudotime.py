#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pseudotime benchmark entry for Geneformer / LangCell / scGPT / scFoundation.

(Model):
  true_delta = late  − early ;
  pred_delta = (early )−  early ;
   |true_delta|  top%  sign(pred_delta)  sign(true_delta).

Token Model( direction_accuracy_top_genes ).

Examples:
  python run_unified_multidataset_pseudotime.py --model geneformer
  python run_unified_multidataset_pseudotime.py --model langcell
  python run_unified_multidataset_pseudotime.py --model scgpt
  python run_unified_multidataset_pseudotime.py --model scfoundation
"""

import argparse
import json
import os
import pickle
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertModel

warnings.filterwarnings("ignore")


DATASET_SPECS = {
    "hESC": "human",
    "hHep": "human",
    "mDC": "mouse",
    "mHSC-E": "mouse",
    "mHSC-GM": "mouse",
    "mHSC-L": "mouse",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified pseudotime benchmark")
    parser.add_argument("--model", required=True, choices=["geneformer", "langcell", "sccello", "scgpt", "scfoundation"])
    parser.add_argument("--outdir", default="results_unified_pseudotime")
    parser.add_argument("--datasets-json", default="", help="Datasets JSON file. If empty, build from --expr-root and --pt-root.")
    parser.add_argument("--expr-root", default="", help="Expression root directory (expects CHIP/<dataset>_chip_matched-ExpressionData.csv).")
    parser.add_argument("--pt-root", default="", help="Pseudotime root directory (expects <dataset>/PseudoTime.csv).")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names to run (default: run all). Example: --datasets mDC,hESC",
    )
    parser.add_argument(
        "--save-iterations",
        action="store_true",
        default=True,
        help="Save per-iteration outputs and per-gene final changes (default: enabled).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-cell-preds-by-iter",
        action="store_true",
        default=False,
        help="Save per-iteration per-cell prediction states (disk-heavy). "
        "Token models save token-id states; continuous models save full predicted expression/state matrices.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device. auto: prefer CUDA if available.",
    )
    parser.add_argument("--pt-quantile", type=float, default=0.2)
    parser.add_argument("--top-percent", type=int, default=30)
    parser.add_argument("--gen-iters", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-early-cells",
        type=int,
        default=0,
        help="Token models only:  early (0= early, true_delta  early )."
        " batch_size , late−early .",
    )
    parser.add_argument(
        "--acc-eps",
        type=float,
        default=0.0,
        help=": >0, |true_delta|>eps (0 , |Δ|≈0 ).",
    )
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--topk-sample", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-unique", action="store_true", default=False)
    parser.add_argument(
        "--token-update",
        choices=["replace", "swap"],
        default="replace",
        help="Token models only: 'replace' (MLM sampling) or 'swap' (reorder by swapping existing tokens).",
    )
    parser.add_argument(
        "--swap-trials",
        type=int,
        default=32,
        help="Token models only, swap mode: number of swap proposals per iteration per cell.",
    )
    parser.add_argument(
        "--force-nondecreasing-acc",
        action="store_true",
        default=False,
        help="Token models only: revert iteration updates when accuracy decreases.",
    )
    parser.add_argument(
        "--restrict-to-dataset-genes",
        action="store_true",
        default=False,
        help="For token models: only sample token_ids that correspond to mapped genes in the current dataset.",
    )
    parser.add_argument("--use-log1p", action="store_true", default=True)
    parser.add_argument("--no-log1p", action="store_true", default=False)
    parser.add_argument("--geneformer-model-dir", default="")
    parser.add_argument("--geneformer-dicts-dir", default="")
    parser.add_argument("--langcell-model-dir", default="")
    parser.add_argument("--sccello-model-dir", default="")
    parser.add_argument("--sccello-repo-dir", default="", help="Contains sccello/src/*.py")
    parser.add_argument(
        "--sccello-cuda",
        action="store_true",
        default=False,
        help="Use CUDA for sccello. Default off because this model path frequently triggers CUDA index asserts in this workflow.",
    )
    parser.add_argument("--scgpt-model-dir", default="")
    parser.add_argument("--scgpt-repo-dir", default="")
    parser.add_argument(
        "--scgpt-bin-log1p",
        action="store_true",
        default=False,
        help="scGPT only: use log1p inside 0–50 binning. Default off (matches pre_scgpt/run_multidataset_pseudotime NO_LOG1P=True).",
    )
    parser.add_argument(
        "--scgpt-legacy-pt",
        action="store_true",
        default=False,
        help="scGPT only: read pseudotime exactly like pre_scgpt/run_multidataset_pseudotime.py "
        "(pd.read_csv + rename; no read_pt_file dropna/coerce). Use if numbers must match old script.",
    )
    parser.add_argument(
        "--scf-root",
        type=str,
        default="",
        help="scFoundation: repo root (pretrainmodels lives here).",
    )
    parser.add_argument(
        "--scf-ckpt",
        type=str,
        default="",
        help="scFoundation checkpoint path.",
    )
    parser.add_argument(
        "--scf-gene-index-tsv",
        type=str,
        default="",
        help="scFoundation gene index TSV.",
    )
    parser.add_argument("--scf-mmf-key", type=str, default="gene", help="Checkpoint sub-key for weights.")
    parser.add_argument("--scf-mode", choices=["mae", "zero"], default="mae", help="Masking mode (see pre_scfoundation/final.py).")
    parser.add_argument("--scf-value-mask-prob", type=float, default=0.3)
    parser.add_argument("--scf-zero-mask-prob", type=float, default=0.0)
    parser.add_argument("--scf-update-scope", choices=["present_all", "mask", "zero"], default="present_all")
    parser.add_argument("--scf-eps-dir", type=float, default=1e-3, help="Threshold for up/down vs zero in direction accuracy.")
    parser.add_argument(
        "--scf-no-refresh-encoder",
        action="store_true",
        default=False,
        help="If set, reuse encoder_io from iter0 (default: refresh each iter, like final.py).",
    )
    parser.add_argument(
        "--scf-no-resample-mask",
        action="store_true",
        default=False,
        help="If set, keep the same MAE mask every iteration.",
    )
    parser.add_argument(
        "--scf-no-identity-input",
        action="store_true",
        default=False,
        help="If set, disable identity input pass-through (unusual; default matches final.py).",
    )
    parser.add_argument("--scf-calibration", action="store_true", default=False)
    parser.add_argument("--scf-calibrate-on", choices=["present", "present_nonzero"], default="present")
    parser.add_argument("--ema-alpha", type=float, default=0.1)
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print iteration progress every N iterations.",
    )
    parser.add_argument(
        "--save-trajectory-plot",
        action="store_true",
        default=False,
        help="Save trajectory PNG and raw plotting data for each dataset.",
    )
    parser.add_argument(
        "--trajectory-embed",
        choices=["pca", "umap"],
        default="umap",
        help="Embedding method for trajectory plot.",
    )
    return parser.parse_args()


def build_datasets_from_roots(expr_root: str, pt_root: str) -> Dict[str, Dict[str, str]]:
    if not expr_root or not pt_root:
        raise ValueError("Please provide --datasets-json OR both --expr-root and --pt-root.")
    out = {}
    for ds, species in DATASET_SPECS.items():
        out[ds] = {
            "expr_csv": str(Path(expr_root) / "CHIP" / f"{ds}_chip_matched-ExpressionData.csv"),
            "pt_csv": str(Path(pt_root) / ds / "PseudoTime.csv"),
            "species": species,
        }
    return out


def resolve_datasets(args) -> Dict[str, Dict[str, str]]:
    if args.datasets_json:
        with open(args.datasets_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict) or not cfg:
            raise ValueError("datasets-json must be a non-empty JSON object.")
        return cfg
    return build_datasets_from_roots(args.expr_root, args.pt_root)


def require_paths(args, keys: List[str]):
    missing = [k for k in keys if not getattr(args, k, "")]
    if missing:
        flags = ", ".join(f"--{k.replace('_', '-')}" for k in missing)
        raise ValueError(f"Missing required path arguments for model={args.model}: {flags}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def direction_accuracy_top_genes(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    top_idx: np.ndarray,
    eps: float = 0.0,
) -> Tuple[float, float]:
    """
    :「 =  − 」「 =  − 」.

    - pred_delta: 「 early 」( scGPT  pred_mean − early_mean ).
    - true_delta:  late_mean − early_mean.
    - top_idx:  |true_delta|  top% ().
    - eps:  >0, |true_delta| > eps ( 0 「」, scFoundation  eps_dir ).
    """
    td = true_delta[top_idx]
    pd = pred_delta[top_idx]
    if eps > 0.0:
        m = np.abs(td) > eps
        if not np.any(m):
            return float("nan"), float("nan")
        td = td[m]
        pd = pd[m]
    #  *_gene_result.csv :
    #   dir = "Up" if delta > 0 else "Down"
    # :delta==0  Down( 0).
    true_dir = np.where(td > 0, 1, -1)
    pred_dir = np.where(pd > 0, 1, -1)
    acc = float((pred_dir == true_dir).mean())
    inv = float((pred_dir == (-true_dir)).mean())
    return acc, inv


def normalize_symbol(s: str) -> str:
    return str(s).strip().upper()


def is_ensembl_id(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("ENSG") or s.startswith("ENSMUSG"))


def read_pt_file(path: str) -> pd.DataFrame:
    try:
        pt_df = pd.read_csv(path, header=None)
        _ = float(pt_df.iloc[0, 1])
    except (ValueError, IndexError, TypeError):
        pt_df = pd.read_csv(path)
    if pt_df.shape[1] < 2:
        raise ValueError(f"Invalid pseudotime file: {path}")
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
    pt_df = pt_df.dropna(subset=["pt"])
    return pt_df


def load_geneformer_dicts(dicts_dir: Path):
    with open(dicts_dir / "token_dictionary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(dicts_dir / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_id = pickle.load(f)
    return vocab, gene_name_id, int(vocab["<pad>"]), int(vocab["<mask>"])


def load_sccello_ensembl_to_token_id(repo_dir: Path) -> Optional[Dict[str, int]]:
    """
    scCello pretraining maps Ensembl gene id -> integer token id (see sccello data_loading).
    If this file is missing, the runner falls back to Geneformer's token_dictionary, which
    can assign different ids to the same gene — logits rows then do not match embeddings.
    """
    pkl = repo_dir / "sccello" / "data" / "token_vocabulary" / "token_dictionary.pkl"
    if not pkl.is_file():
        return None
    with open(pkl, "rb") as f:
        raw = pickle.load(f)
    out: Dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if k.startswith("ENS"):
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                continue
    return out if out else None


def merge_vocab_with_sccello_ids(
    vocab: Dict,
    gene_name_id: Dict[str, str],
    sc_ens2t: Dict[str, int],
) -> Dict:
    """Overwrite symbol / Ensembl entries with scCello token ids where available."""
    merged = dict(vocab)
    sym2ens = build_symbol_to_ensembl_map(gene_name_id)
    for ens, tid in sc_ens2t.items():
        merged[ens] = tid
    for sym, ens in sym2ens.items():
        if ens in sc_ens2t:
            merged[sym] = sc_ens2t[ens]
    return merged


def build_symbol_to_ensembl_map(gene_name_id: Dict[str, str]) -> Dict[str, str]:
    items = list(gene_name_id.items())
    sample = items[: min(2000, len(items))]
    k_ens = sum(is_ensembl_id(str(k)) for k, _ in sample)
    v_ens = sum(is_ensembl_id(str(v)) for _, v in sample)
    if v_ens > k_ens:
        return {normalize_symbol(k): str(v) for k, v in gene_name_id.items()}
    return {normalize_symbol(v): str(k) for k, v in gene_name_id.items()}


class LangCellModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
        self.cls = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.cls(out.last_hidden_state)


class ScCelloMaskedLMWrapper(nn.Module):
    """
    Return [B, L, V] MLM scores for iterative_token_sampling (same interface as LangCell).

    Mirrors `PrototypeContrastiveForMaskedLM.calculate_masked_gene_prediction`:
    h_mask = cls(sequence_output); h_truth = _get_ground_truth_word_reprs();
    logits = normalize(h_mask) @ normalize(h_truth).T (with tau scaling).
    Do not call `inner.forward()` — it expects training labels/cell_type.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask):
        # Avoid `inner.forward()` because that path expects training-time `labels/cell_type`
        # and can crash in inference with NoneType errors.
        # Reconstruct MLM logits directly from encoder outputs and tied prediction head.
        outputs = self.inner.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = outputs[0]  # [B, L, H]
        h_mask = self.inner.cls(sequence_output)  # [B, L, H]
        h_truth = self.inner._get_ground_truth_word_reprs()  # [V, H]
        tau_sqrt = torch.sqrt(torch.tensor(float(self.inner.tau), device=h_mask.device))
        h_mask = F.normalize(h_mask, p=2, dim=-1) / tau_sqrt
        h_truth = F.normalize(h_truth, p=2, dim=-1) / tau_sqrt
        logits = torch.matmul(h_mask, h_truth.t())  # [B, L, V]
        if logits.ndim != 3:
            raise ValueError(f"Unexpected scCello logits shape: {tuple(logits.shape)} (expect [B,L,V])")
        return logits


@torch.no_grad()
def iterative_token_sampling(
    model,
    seq,
    length,
    pad_id,
    mask_id,
    device,
    args,
    is_geneformer: bool,
    allowed_token_ids: Optional[torch.Tensor] = None,
    n_steps: int = 1,
    vocab_size_limit: Optional[int] = None,
):
    x = seq.clone().to(device).unsqueeze(0)
    attn = torch.zeros((1, x.size(1)), dtype=torch.long, device=device)
    attn[0, :length] = 1
    valid_pos = torch.arange(0, length, device=device)
    banned = torch.tensor([pad_id, mask_id], device=device, dtype=torch.long)
    allowed_set = None
    if allowed_token_ids is not None and allowed_token_ids.numel() > 0:
        allowed_set = set(int(x) for x in allowed_token_ids.detach().cpu().tolist())

    for _ in range(max(1, int(n_steps))):
        # Defensive check: catch invalid token ids before GPU kernels crash.
        if vocab_size_limit is not None:
            bad = (x[0, :length] < 0) | (x[0, :length] >= int(vocab_size_limit))
            if torch.any(bad):
                bad_vals = x[0, :length][bad].detach().cpu().tolist()
                raise ValueError(
                    f"Token id out of range for vocab_size={int(vocab_size_limit)}; "
                    f"examples={bad_vals[:10]}"
                )
        if valid_pos.numel() == 0:
            break
        if args.token_update == "replace":
            n_mask = max(1, int(args.mask_ratio * valid_pos.numel()))
            # Avoid CUDA RNG ops (can fail under CUDA graph/capture in some envs).
            perm = torch.randperm(valid_pos.numel(), device="cpu")[:n_mask].to(device)
            idx = valid_pos[perm]
            x_masked = x.clone()
            x_masked[0, idx] = mask_id

            if is_geneformer:
                logits = model(input_ids=x_masked, attention_mask=attn).logits[0, idx, :].float()
            else:
                logits = model(x_masked, attn)[0, idx, :].float()

            logits /= max(args.temperature, 1e-6)
            # Guard in case pad/mask ids exceed vocab size (or differ across models).
            banned2 = banned[banned >= 0]
            banned2 = banned2[banned2 < logits.size(-1)]
            if banned2.numel() > 0:
                logits[:, banned2] = -1e9

            # Optional restriction: only allow tokens from current dataset's mapped genes.
            if allowed_token_ids is not None and allowed_token_ids.numel() > 0:
                masked_logits = torch.full_like(logits, -1e9)
                allowed2 = allowed_token_ids[(allowed_token_ids >= 0) & (allowed_token_ids < logits.size(-1))]
                if allowed2.numel() == 0:
                    raise ValueError(
                        f"All allowed_token_ids are out of range for logits vocab={logits.size(-1)}"
                    )
                masked_logits[:, allowed2] = logits[:, allowed2]
                logits = masked_logits

            if args.enforce_unique:
                present = set(x[0, :length].detach().cpu().tolist())
                present.discard(mask_id)
                present.discard(pad_id)
                if present:
                    present_t = torch.tensor(list(present), device=device, dtype=torch.long)
                    present_t = present_t[(present_t >= 0) & (present_t < logits.size(-1))]
                    if present_t.numel() > 0:
                        logits[:, present_t] = -1e9

            k = min(args.topk_sample, logits.size(-1))
            topv, topi = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(topv, dim=-1)
            # If probs contain NaN/Inf (e.g., model produced NaNs), fall back to uniform over top-k.
            if not torch.isfinite(probs).all():
                probs = torch.full_like(probs, 1.0 / probs.size(-1))
            # multinomial on CUDA can also be problematic under capture; do it on CPU.
            choice = torch.multinomial(probs.detach().to("cpu"), 1).squeeze(-1).to(device)
            sampled = topi[torch.arange(len(topi), device=device), choice]
            x[0, idx] = sampled

        elif args.token_update == "swap":
            # Reorder-only: keep token multiset fixed, propose swaps and accept if they increase MLM log-prob.
            # We score a swap (i,j) by masking both positions and comparing:
            #   logP(a at i)+logP(b at j)  vs  logP(b at i)+logP(a at j)
            # where a=x[i], b=x[j]. Masking both means logits depend on the same context for both options.
            trials = max(1, int(args.swap_trials))
            for _t in range(trials):
                if length < 2:
                    break
                # Avoid CUDA RNG ops under capture: sample swap indices on CPU.
                ij = torch.randperm(length, device="cpu")[:2]
                i = int(ij[0].item())
                j = int(ij[1].item())
                if i == j:
                    continue
                a = int(x[0, i].item())
                b = int(x[0, j].item())
                # Skip swapping banned tokens.
                if a in (pad_id, mask_id) or b in (pad_id, mask_id):
                    continue
                x_masked = x.clone()
                x_masked[0, i] = mask_id
                x_masked[0, j] = mask_id

                if is_geneformer:
                    logits2 = model(input_ids=x_masked, attention_mask=attn).logits[0, [i, j], :].float()
                else:
                    logits2 = model(x_masked, attn)[0, [i, j], :].float()
                logits2 /= max(args.temperature, 1e-6)
                banned2 = banned[banned >= 0]
                banned2 = banned2[banned2 < logits2.size(-1)]
                if banned2.numel() > 0:
                    logits2[:, banned2] = -1e9

                # Restrict candidate tokens if requested (still consistent with "swap" only when both are allowed).
                if allowed_set is not None:
                    if (a not in allowed_set) or (b not in allowed_set):
                        continue

                logp = torch.log_softmax(logits2, dim=-1)
                # logits2[0] corresponds to position i, logits2[1] to position j
                if (a < 0) or (b < 0) or (a >= logits2.size(-1)) or (b >= logits2.size(-1)):
                    continue
                cur = float(logp[0, a].item() + logp[1, b].item())
                swp = float(logp[0, b].item() + logp[1, a].item())
                if swp > cur:
                    x[0, i], x[0, j] = x[0, j].clone(), x[0, i].clone()
        else:
            raise ValueError(f"Unknown token_update: {args.token_update}")

    return x.squeeze(0).detach().cpu().numpy()


def positions_dict_from_seq(seq_ids: np.ndarray, length: int) -> Dict[int, int]:
    pos = {}
    for i in range(length):
        t = int(seq_ids[i])
        if t not in pos:
            pos[t] = i
    return pos


def run_token_model_dataset(
    name,
    cfg,
    model,
    vocab,
    gene_name_id,
    pad_id,
    mask_id,
    device,
    args,
    is_geneformer: bool,
    *,
    vocab_size_limit: Optional[int] = None,
    max_position_embeddings: Optional[int] = None,
):
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = read_pt_file(cfg["pt_csv"])
    common = expr.columns.intersection(pt_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping cells between expression and pseudotime")
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].values

    X = expr.T.values.astype(np.float32)
    if args.use_log1p and (not args.no_log1p):
        X = np.log1p(np.maximum(X, 0))

    lo, hi = np.quantile(pt, [args.pt_quantile, 1 - args.pt_quantile])
    early_mask = pt <= lo
    late_mask = pt >= hi
    if early_mask.sum() == 0 or late_mask.sum() == 0:
        raise ValueError("No early or late cells after quantile split")

    early_mean = X[early_mask].mean(axis=0)
    late_mean = X[late_mask].mean(axis=0)
    true_delta = late_mean - early_mean
    n_genes = len(true_delta)
    top_n = max(int(n_genes * args.top_percent / 100), 1)
    top_idx = np.argsort(-np.abs(true_delta))[:top_n]

    sym2ens = build_symbol_to_ensembl_map(gene_name_id)
    genes_original = expr.index.astype(str).tolist()

    def get_tid(g):
        g_norm = normalize_symbol(g)
        if g_norm in vocab:
            t = vocab[g_norm]
            if vocab_size_limit is not None and int(t) >= int(vocab_size_limit):
                return pad_id
            return t
        ens = sym2ens.get(g_norm)
        if ens and ens in vocab:
            t = vocab[ens]
            if vocab_size_limit is not None and int(t) >= int(vocab_size_limit):
                return pad_id
            return t
        return pad_id

    gene_tids = [get_tid(g) for g in genes_original]
    matched = sum(1 for t in gene_tids if t != pad_id)
    top_matched = sum(1 for i in top_idx if gene_tids[i] != pad_id)

    # Allowed token set for sampling: mapped gene tokens in this dataset (exclude pad/mask).
    allowed_token_ids = None
    if args.restrict_to_dataset_genes:
        allowed = sorted({int(t) for t in gene_tids if int(t) not in (pad_id, mask_id)})
        allowed_token_ids = torch.tensor(allowed, dtype=torch.long, device=device) if allowed else None

    effective_max_len = int(args.max_len)
    if max_position_embeddings is not None:
        effective_max_len = min(effective_max_len, int(max_position_embeddings))

    def cell_to_seq(x):
        order = np.argsort(-x)
        seq = []
        for j in order:
            t = gene_tids[j]
            if t == pad_id:
                continue
            seq.append(t)
            if len(seq) >= effective_max_len:
                break
        length = len(seq)
        if length < effective_max_len:
            seq += [pad_id] * (effective_max_len - length)
        return seq, length

    early_cells = []
    early_idx_all = np.where(early_mask)[0]
    if getattr(args, "max_early_cells", 0) and int(args.max_early_cells) > 0:
        early_idx_all = early_idx_all[: int(args.max_early_cells)]
    for i in early_idx_all:
        seq, length = cell_to_seq(X[i])
        if length > 10:
            early_cells.append((seq, length))
    if not early_cells:
        raise ValueError("No valid early cells after sequence building")

    # True iterative state across cells.
    init_cells = [np.array(seq, dtype=np.int64) for seq, _ in early_cells]
    curr_cells = [arr.copy() for arr in init_cells]
    cell_lengths = [int(L) for _, L in early_cells]

    # Example cell trace uses current state of first cell.
    example_len = cell_lengths[0]
    example_seqs_by_iter: List[np.ndarray] = [curr_cells[0].copy()]
    token_ids_by_iter: List[np.ndarray] = []  # per-iteration token ids for each early cell

    acc_curve = []
    acc_curve_inv_truth = []
    mean_delta_by_iter: List[np.ndarray] = []
    best_acc = -1.0

    for it in range(args.gen_iters):
        prev_state = [arr.copy() for arr in curr_cells]
        deltas_batch = []
        next_cells = []
        for ci, length in enumerate(cell_lengths):
            x0 = init_cells[ci]
            x_prev = curr_cells[ci]
            x1 = iterative_token_sampling(
                model=model,
                seq=torch.tensor(x_prev, dtype=torch.long),
                length=length,
                pad_id=pad_id,
                mask_id=mask_id,
                device=device,
                args=args,
                is_geneformer=is_geneformer,
                allowed_token_ids=allowed_token_ids,
                n_steps=1,
                vocab_size_limit=vocab_size_limit,
            )
            x1 = x1.astype(np.int64, copy=False)
            next_cells.append(x1)
            pos0 = positions_dict_from_seq(x0, length)
            pos1 = positions_dict_from_seq(x1, length)
            delta = np.zeros(n_genes, dtype=np.float32)
            for gi in range(n_genes):
                tid = gene_tids[gi]
                delta[gi] = pos1.get(tid, length) - pos0.get(tid, length)
            deltas_batch.append(delta)
        curr_cells = next_cells

        mean_delta = np.stack(deltas_batch).mean(axis=0)
        # 「」; log1p  true_delta :
        # rank  ≈ , -mean_delta  true_delta .
        pred_delta_expr_aligned = -mean_delta.astype(np.float32, copy=False)
        acc_now, inv_now = direction_accuracy_top_genes(
            pred_delta_expr_aligned,
            true_delta,
            top_idx,
            eps=float(getattr(args, "acc_eps", 0.0)),
        )

        if args.force_nondecreasing_acc and best_acc >= 0 and acc_now < best_acc:
            # Revert full state when this iteration hurts target metric.
            curr_cells = prev_state
            acc_now = best_acc
            inv_now = 1.0 - best_acc
            # Recompute mean_delta from reverted state for consistent artifacts.
            deltas_batch_revert = []
            for ci, length in enumerate(cell_lengths):
                x0 = init_cells[ci]
                x1 = curr_cells[ci]
                pos0 = positions_dict_from_seq(x0, length)
                pos1 = positions_dict_from_seq(x1, length)
                delta = np.zeros(n_genes, dtype=np.float32)
                for gi in range(n_genes):
                    tid = gene_tids[gi]
                    delta[gi] = pos1.get(tid, length) - pos0.get(tid, length)
                deltas_batch_revert.append(delta)
            mean_delta = np.stack(deltas_batch_revert).mean(axis=0)
        else:
            if acc_now > best_acc:
                best_acc = acc_now

        # Optional: save full per-cell token state for visualization.
        if args.save_cell_preds_by_iter:
            token_ids_by_iter.append(np.stack(curr_cells, axis=0).astype(np.int64, copy=False))

        mean_delta_by_iter.append(mean_delta.astype(np.float32, copy=False))
        acc_curve.append(acc_now)
        acc_curve_inv_truth.append(inv_now)

        # Record example sequence from iterative state.
        example_seq_np = curr_cells[0]
        example_seqs_by_iter.append(example_seq_np.copy())
        if args.print_every > 0 and ((it + 1) % args.print_every == 0):
            cov = float((example_seq_np[:example_len] != pad_id).mean())
            print(
                f"    Iter {it+1:>2}/{args.gen_iters} | acc={acc_curve[-1]:.2%} | "
                f"inv={acc_curve_inv_truth[-1]:.2%} | example_nonpad={cov:.2%}"
            )

    # --- token model: export "rank before vs rank after" ---
    # mean_delta_by_iter stores (pos_after - pos_before) averaged over early cells at each iter.
    # To export absolute positions, we need mean(pos_before) w.r.t. init_cells.
    mean_pos_before = np.zeros(n_genes, dtype=np.float32)
    for ci, length in enumerate(cell_lengths):
        x0 = init_cells[ci]
        pos0 = positions_dict_from_seq(x0, length)
        for gi in range(n_genes):
            tid = int(gene_tids[gi])
            mean_pos_before[gi] += float(pos0.get(tid, length))
    if len(cell_lengths) > 0:
        mean_pos_before /= float(len(cell_lengths))

    diagnostics = {
        "dataset": name,
        "model": args.model,
        "n_genes": n_genes,
        "n_cells": len(common),
        "n_early": int(early_mask.sum()),
        "n_early_cells_in_loop": len(early_cells),
        "n_late": int(late_mask.sum()),
        "vocab_match_rate": round(matched / max(1, n_genes) * 100, 2),
        "top_vocab_match_rate": round(top_matched / max(1, top_n) * 100, 2),
        "eval_percent": args.top_percent,
        "eval_genes_count": top_n,
        "final_acc_inv_truth": float(acc_curve_inv_truth[-1]) if acc_curve_inv_truth else None,
    }
    artifacts = {
        "genes": genes_original,
        "gene_tids": np.array(gene_tids, dtype=np.int64),
        "true_delta": true_delta.astype(np.float32, copy=False),
        "mean_pos_before": mean_pos_before.astype(np.float32, copy=False),
        "early_mean": early_mean.astype(np.float32, copy=False),
        "late_mean": late_mean.astype(np.float32, copy=False),
        "top_idx": top_idx.astype(np.int64, copy=False),
        "mean_delta_by_iter": np.stack(mean_delta_by_iter, axis=0) if mean_delta_by_iter else None,
        "example_seq_len": int(example_len),
        "example_seqs_by_iter": np.stack(example_seqs_by_iter, axis=0) if example_seqs_by_iter else None,
        "token_ids_by_iter": np.stack(token_ids_by_iter, axis=0) if args.save_cell_preds_by_iter and token_ids_by_iter else None,
    }
    return acc_curve, diagnostics, artifacts


def bin_expr_to_0_50(x: np.ndarray, do_log1p: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0, None)
    if do_log1p:
        x = np.log1p(x)
    vmax = max(np.percentile(x, 99.5), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def run_scgpt_dataset(name, cfg, model, vocab, device, args):
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    if args.scgpt_legacy_pt:
        pt_df = pd.read_csv(cfg["pt_csv"])
        pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    else:
        pt_df = read_pt_file(cfg["pt_csv"])
    common = expr.columns.intersection(pt_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping cells between expression and pseudotime")
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    genes_original = expr.index.astype(str).tolist()
    if cfg.get("species", "human") == "mouse":
        genes = [g.upper() for g in genes_original]
    else:
        genes = genes_original

    X = expr.T.to_numpy(dtype=np.float32)
    # Binning must follow scGPT training recipe: original script used NO_LOG1P -> no log1p here by default.
    X_bin = bin_expr_to_0_50(X, do_log1p=args.scgpt_bin_log1p)

    lo, hi = np.quantile(pt, [args.pt_quantile, 1 - args.pt_quantile])
    early = pt <= lo
    late = pt >= hi
    if early.sum() == 0 or late.sum() == 0:
        raise ValueError("No early or late cells after quantile split")

    early_mean = X_bin[early].mean(axis=0)
    late_mean = X_bin[late].mean(axis=0)
    true_delta = late_mean - early_mean

    n_genes = len(true_delta)
    top_n = max(int(n_genes * args.top_percent / 100), 1)
    top_idx = np.argsort(np.abs(true_delta))[::-1][:top_n]

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=np.int64)
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    values = np.concatenate([np.zeros((X_bin.shape[0], 1), dtype=np.float32), X_bin], axis=1)
    values_tensor = torch.tensor(values, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(values.shape[0], -1)
    update_mask = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask[1:] = True

    vals_all = values_tensor[early].clone()
    update_mask_t = torch.tensor(update_mask[None, :], device=device).bool()
    acc_curve = []
    acc_curve_inv_truth = []
    pred_delta_by_iter: List[np.ndarray] = []
    preds_full_by_iter: List[np.ndarray] = []  # [gen_iters, n_early, n_genes] (gene space)

    with torch.no_grad():
        for it in range(args.gen_iters):
            for start in range(0, vals_all.shape[0], args.batch_size):
                end = min(start + args.batch_size, vals_all.shape[0])
                bs = end - start

                vals = vals_all[start:end].to(device)
                src = gene_ids_tensor.expand(bs, -1).to(device)
                mask = pad_mask[early][start:end].to(device)
                freeze = mask | (~update_mask_t.expand(bs, -1))

                out = model(src=src, values=vals, src_key_padding_mask=mask)
                new_vals = out["mlm_output"]
                vals = torch.where(freeze, vals, args.ema_alpha * vals + (1 - args.ema_alpha) * new_vals)
                vals_all[start:end] = vals.detach().cpu()

            # Match pre_scgpt/run_multidataset_pseudotime.py: use tensor .numpy() as-is (fp16 on GPU path).
            pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
            pred_delta = pred_mean - early_mean
            pred_delta_by_iter.append(pred_delta.astype(np.float32, copy=False))

            if args.save_cell_preds_by_iter:
                # Save per-iteration per-cell predicted expression in gene space (drop CLS).
                preds_full_by_iter.append(vals_all[:, 1:].numpy().astype(np.float32, copy=False))

            acc_now, inv_now = direction_accuracy_top_genes(
                pred_delta,
                true_delta,
                top_idx,
                eps=float(getattr(args, "acc_eps", 0.0)),
            )
            acc_curve.append(acc_now)
            acc_curve_inv_truth.append(inv_now)
            if args.print_every > 0 and ((it + 1) % args.print_every == 0):
                print(
                    f"    Iter {it+1:>2}/{args.gen_iters} | acc={acc_curve[-1]:.2%} | "
                    f"inv={acc_curve_inv_truth[-1]:.2%}"
                )

    matched = sum(1 for g in genes if g in vocab)
    top_matched = sum(1 for i in top_idx if genes[i] in vocab)
    diagnostics = {
        "dataset": name,
        "model": args.model,
        "n_genes": n_genes,
        "n_cells": len(common),
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "vocab_match_rate": round(matched / max(1, n_genes) * 100, 2),
        "top_vocab_match_rate": round(top_matched / max(1, top_n) * 100, 2),
        "eval_percent": args.top_percent,
        "eval_genes_count": top_n,
        "final_acc_inv_truth": float(acc_curve_inv_truth[-1]) if acc_curve_inv_truth else None,
    }
    artifacts = {
        "genes": genes,
        "true_delta": true_delta.astype(np.float32, copy=False),
        "early_mean": early_mean.astype(np.float32, copy=False),
        "late_mean": late_mean.astype(np.float32, copy=False),
        "top_idx": top_idx.astype(np.int64, copy=False),
        "pred_delta_by_iter": np.stack(pred_delta_by_iter, axis=0) if pred_delta_by_iter else None,
        "preds_full_by_iter": np.stack(preds_full_by_iter, axis=0) if args.save_cell_preds_by_iter and preds_full_by_iter else None,
    }
    return acc_curve, diagnostics, artifacts


# ---------------------------------------------------------------------------
# scFoundation (from pre_scfoundation/final.py)
# ---------------------------------------------------------------------------
SCFOUNDATION_MODEL_CFG = {
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


def _scf_strip_prefix(sd: dict, prefix: str):
    out = {}
    for k, v in sd.items():
        out[k[len(prefix) :] if k.startswith(prefix) else k] = v
    return out


def scf_gather_data(data: torch.Tensor, labels: torch.Tensor, pad_token_id: int):
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


def scf_read_gene_index_tsv(path: str, G_model: int) -> Dict[str, int]:
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
    return dict(zip(df[gene_col].tolist(), df[idx_col].tolist()))


def scf_build_aligned_matrix(X: np.ndarray, genes: list, gene2idx_model: dict, G_model: int):
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


def scf_make_masks_present_only(
    raw_full: torch.Tensor,
    present_mask: torch.Tensor,
    mode: str,
    value_mask_prob: float,
    zero_mask_prob: float,
    seed: int,
):
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

    set_seed(seed)
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


def scf_build_io(vals: torch.Tensor, encoder_visible: torch.Tensor, config: dict):
    device = vals.device
    B, G = vals.shape
    decoder_data = vals
    decoder_data_padding = torch.full_like(decoder_data, False, dtype=torch.bool, device=device)
    encoder_data, encoder_data_padding = scf_gather_data(decoder_data, encoder_visible, config["pad_token_id"])
    gene_ids = torch.arange(G, device=device).unsqueeze(0).repeat(B, 1)
    encoder_pos, _ = scf_gather_data(gene_ids, encoder_visible, config["pad_token_id"])
    decoder_pos = gene_ids
    encoder_pos[encoder_data_padding] = config["seq_len"]
    decoder_pos[decoder_data_padding] = config["seq_len"]
    return encoder_data, encoder_data_padding, encoder_pos, encoder_visible, decoder_data, decoder_data_padding, decoder_pos


def scf_mean_match_calibration(vals, inp, present_mask, raw_full, mode: str = "present"):
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


def load_scfoundation_model(args, device: torch.device):
    root = Path(args.scf_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pretrainmodels import select_model

    ckpt = torch.load(args.scf_ckpt, map_location="cpu")
    if args.scf_mmf_key not in ckpt:
        raise RuntimeError(f"Checkpoint keys={list(ckpt.keys())}, missing key={args.scf_mmf_key!r}")
    gene_block = ckpt[args.scf_mmf_key]
    if not isinstance(gene_block, dict) or "state_dict" not in gene_block:
        raise RuntimeError(f"ckpt[{args.scf_mmf_key!r}] should contain 'state_dict'")
    sd = _scf_strip_prefix(gene_block["state_dict"], "model.")
    if "pos_emb.weight" in sd:
        seq_len = int(sd["pos_emb.weight"].shape[0] - 1)
    else:
        seq_len = int(SCFOUNDATION_MODEL_CFG["seq_len"])
    has_enc_performer = any(k.startswith("encoder.performer.") for k in sd.keys())
    has_dec_performer = any(k.startswith("decoder.performer.") for k in sd.keys())
    enc_type = "performer" if has_enc_performer else "transformer"
    dec_type = "performer" if has_dec_performer else "transformer"
    config = dict(SCFOUNDATION_MODEL_CFG)
    config["seq_len"] = seq_len
    config["encoder"] = dict(config["encoder"])
    config["decoder"] = dict(config["decoder"])
    config["encoder"]["module_type"] = enc_type
    config["decoder"]["module_type"] = dec_type
    print(f"[scFoundation] encoder={enc_type} decoder={dec_type} seq_len={seq_len}")
    model = select_model(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    if device.type == "cuda":
        model.half()
    if missing:
        print(f"[scFoundation][WARN] missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"[scFoundation][WARN] unexpected keys (first 10): {unexpected[:10]}")
    return model, config


@torch.no_grad()
def _scf_iterative_predict_curve(
    model,
    config: dict,
    values_full_init: torch.Tensor,
    raw_full: torch.Tensor,
    present_mask: torch.Tensor,
    update_scope: str,
    eval_model_idx: np.ndarray,
    early_mean_eval: np.ndarray,
    true_delta_eval: np.ndarray,
    n_iters: int,
    seed0: int,
    mode: str,
    value_mask_prob: float,
    zero_mask_prob: float,
    refresh_encoder: bool,
    resample_mask: bool,
    update_alpha: float,
    eps_dir: float,
    calibrate: bool,
    calibrate_on: str,
    save_full_cells_by_iter: bool = False,
):
    device = next(model.parameters()).device
    vals = values_full_init.to(device)
    inp = values_full_init.to(device)
    raw_full_d = raw_full.to(device)
    present_mask_d = present_mask.to(device)
    B, G = vals.shape
    acc_curve: List[float] = []
    pred_delta_eval_per_iter: List[np.ndarray] = []
    vals_by_iter: List[np.ndarray] = []  # only populated when save_full_cells_by_iter=True

    encoder_visible, update_pos = scf_make_masks_present_only(
        raw_full_d, present_mask_d, mode, value_mask_prob, zero_mask_prob, seed0,
    )

    for it in range(n_iters):
        if mode == "mae" and resample_mask:
            encoder_visible, update_pos = scf_make_masks_present_only(
                raw_full_d, present_mask_d, mode, value_mask_prob, zero_mask_prob, seed0 + it + 1,
            )
        if it == 0 or refresh_encoder:
            enc = scf_build_io(vals, encoder_visible, config)
            encoder_data, padding_label, enc_pos, enc_labels, dec_data, dec_pad, dec_pos = enc

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
            )
        present = present_mask_d.view(1, G).expand(B, G)
        if update_scope == "mask":
            pos = update_pos
        elif update_scope == "zero":
            pos = (raw_full_d <= 0) & present
        else:
            pos = present
        if update_alpha >= 1.0:
            vals[pos] = pred[pos]
        else:
            vals[pos] = (1.0 - update_alpha) * vals[pos] + update_alpha * pred[pos]
        if calibrate:
            vals = scf_mean_match_calibration(vals, inp, present_mask_d, raw_full_d, mode=calibrate_on)

        if save_full_cells_by_iter:
            # Snapshot full model-space predictions for each early cell in this batch.
            # Shape: [B, G_model]
            vals_by_iter.append(vals.detach().float().cpu().numpy())

        pred_mean_eval = vals[:, eval_model_idx].detach().float().mean(dim=0).cpu().numpy()
        pred_delta_eval = pred_mean_eval - early_mean_eval
        pred_delta_eval_per_iter.append(pred_delta_eval.astype(np.float32, copy=False))
        #  scGPT/ *_gene_result.csv :delta>0  Up, Down(delta==0  Down,)
        true_dir = np.where(true_delta_eval > 0, 1, -1)
        pred_dir = np.where(pred_delta_eval > 0, 1, -1)
        acc = float((pred_dir == true_dir).mean())
        acc_curve.append(acc)

    if save_full_cells_by_iter:
        return acc_curve, vals.detach().float().cpu().numpy(), pred_delta_eval_per_iter, vals_by_iter
    return acc_curve, vals.detach().float().cpu().numpy(), pred_delta_eval_per_iter


def run_scfoundation_dataset(name, cfg, model, config, gene2idx_model: Dict[str, int], device, args):
    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = read_pt_file(cfg["pt_csv"])
    common = expr.columns.intersection(pt_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping cells between expression and pseudotime")
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].to_numpy()

    genes_original = expr.index.astype(str).tolist()
    if cfg.get("species", "human") == "mouse":
        genes = [str(g).upper() for g in genes_original]
    else:
        genes = genes_original

    X = expr.T.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    X_proc = X.astype(np.float32) if not args.scf_no_identity_input else X.astype(np.float32)
    X_raw = X_proc.copy()

    lo, hi = np.quantile(pt, [args.pt_quantile, 1 - args.pt_quantile])
    early = pt <= lo
    late = pt >= hi
    if early.sum() == 0 or late.sum() == 0:
        raise ValueError("No early or late cells after quantile split")

    G_model = int(config["seq_len"])
    X_full, present_mask_np, map_idx, mapped = scf_build_aligned_matrix(X_proc, genes, gene2idx_model, G_model)
    X_full_raw, _, _, _ = scf_build_aligned_matrix(X_raw, genes, gene2idx_model, G_model)
    present_mask = torch.tensor(present_mask_np, dtype=torch.bool)

    early_mean_910 = X_proc[early].mean(axis=0)
    late_mean_910 = X_proc[late].mean(axis=0)
    true_delta_910 = late_mean_910 - early_mean_910
    n_genes = len(genes)

    mapped_mask_910 = map_idx >= 0
    idx_pool = np.where(mapped_mask_910)[0]
    if len(idx_pool) == 0:
        raise ValueError("No mapped genes for scFoundation")
    total_mapped = len(idx_pool)
    top_n = max(int(total_mapped * args.top_percent / 100), 1)
    order = np.argsort(np.abs(true_delta_910[idx_pool]))[::-1]
    eval_910 = idx_pool[order[:top_n]].copy()
    eval_model_idx = map_idx[eval_910]
    keep = eval_model_idx >= 0
    eval_910 = eval_910[keep]
    eval_model_idx = eval_model_idx[keep].astype(np.int64)
    if len(eval_model_idx) == 0:
        raise ValueError("Empty evaluation set after mapping")

    early_mean_eval = X_full[early][:, eval_model_idx].mean(axis=0)
    late_mean_eval = X_full[late][:, eval_model_idx].mean(axis=0)
    true_delta_eval = (late_mean_eval - early_mean_eval).astype(np.float32)

    early_idx = np.where(early)[0]
    refresh_enc = not args.scf_no_refresh_encoder
    resample_m = not args.scf_no_resample_mask

    acc_curve_accum: List[List[float]] = []
    pred_delta_batches: List[List[np.ndarray]] = []
    preds_910_batches: List[np.ndarray] = []
    preds_full_by_iter = None
    if args.save_cell_preds_by_iter:
        # 「Dataset」:[iter, n_early, n_genes], G_model .
        preds_full_by_iter = np.zeros((args.gen_iters, len(early_idx), n_genes), dtype=np.float32)

    for s in range(0, len(early_idx), args.batch_size):
        batch_sl = slice(s, min(s + args.batch_size, len(early_idx)))
        vals0 = torch.tensor(
            X_full[early_idx[batch_sl]],
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        rawb = torch.tensor(X_full_raw[early_idx[batch_sl]], dtype=torch.float32)
        if args.save_cell_preds_by_iter:
            curve, preds_full, pred_list, vals_by_iter_batch = _scf_iterative_predict_curve(
                model=model,
                config=config,
                values_full_init=vals0,
                raw_full=rawb,
                present_mask=present_mask,
                update_scope=args.scf_update_scope,
                eval_model_idx=eval_model_idx,
                early_mean_eval=early_mean_eval,
                true_delta_eval=true_delta_eval,
                n_iters=args.gen_iters,
                seed0=args.seed + s,
                mode=args.scf_mode,
                value_mask_prob=args.scf_value_mask_prob,
                zero_mask_prob=args.scf_zero_mask_prob,
                refresh_encoder=refresh_enc,
                resample_mask=resample_m,
                update_alpha=float(args.ema_alpha),
                eps_dir=float(args.scf_eps_dir),
                calibrate=args.scf_calibration,
                calibrate_on=args.scf_calibrate_on,
                save_full_cells_by_iter=True,
            )
            vals_arr = np.stack(vals_by_iter_batch, axis=0)  # [n_iters, B, G_model]
            # Model「Dataset」.
            #  j,Model index, vals_arr[..., midx]; X_proc.
            for it in range(args.gen_iters):
                for j in range(n_genes):
                    midx = int(map_idx[j])
                    if midx >= 0:
                        preds_full_by_iter[it, batch_sl, j] = vals_arr[it, :, midx]
                    else:
                        preds_full_by_iter[it, batch_sl, j] = X_proc[early_idx[batch_sl], j]
        else:
            curve, preds_full, pred_list = _scf_iterative_predict_curve(
                model=model,
                config=config,
                values_full_init=vals0,
                raw_full=rawb,
                present_mask=present_mask,
                update_scope=args.scf_update_scope,
                eval_model_idx=eval_model_idx,
                early_mean_eval=early_mean_eval,
                true_delta_eval=true_delta_eval,
                n_iters=args.gen_iters,
                seed0=args.seed + s,
                mode=args.scf_mode,
                value_mask_prob=args.scf_value_mask_prob,
                zero_mask_prob=args.scf_zero_mask_prob,
                refresh_encoder=refresh_enc,
                resample_mask=resample_m,
                update_alpha=float(args.ema_alpha),
                eps_dir=float(args.scf_eps_dir),
                calibrate=args.scf_calibration,
                calibrate_on=args.scf_calibrate_on,
                save_full_cells_by_iter=False,
            )
        acc_curve_accum.append(curve)
        pred_delta_batches.append(pred_list)
        n_early_batch = preds_full.shape[0]
        p910 = np.zeros((n_early_batch, n_genes), dtype=np.float32)
        for j in range(n_genes):
            midx = int(map_idx[j])
            if midx >= 0:
                p910[:, j] = preds_full[:, midx]
            else:
                p910[:, j] = X_proc[early_idx[batch_sl], j]
        preds_910_batches.append(p910)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    acc_curve = np.nanmean(np.array(acc_curve_accum, dtype=np.float64), axis=0).tolist()
    n_it = args.gen_iters
    pred_delta_eval_avg: List[np.ndarray] = []
    for it in range(n_it):
        stacks = np.stack([pred_delta_batches[b][it] for b in range(len(pred_delta_batches))], axis=0)
        pred_delta_eval_avg.append(stacks.mean(axis=0).astype(np.float32))

    pred_delta_by_iter = np.full((n_it, n_genes), np.nan, dtype=np.float32)
    for it in range(n_it):
        for k, gj in enumerate(eval_910):
            pred_delta_by_iter[it, gj] = pred_delta_eval_avg[it][k]

    preds_910_all = np.concatenate(preds_910_batches, axis=0)
    pred_mean_910 = preds_910_all.mean(axis=0)
    pred_delta_final_910 = pred_mean_910 - early_mean_910

    eps = float(args.scf_eps_dir)
    true_dir = np.where(true_delta_910 > 0, 1, -1)
    pred_dir = np.where(pred_delta_final_910 > 0, 1, -1)
    in_eval = np.zeros(n_genes, dtype=bool)
    in_eval[eval_910] = True
    last_pred = pred_delta_eval_avg[-1]
    last_true_d = np.where(true_delta_eval > 0, 1, -1)
    last_pred_d = np.where(last_pred > 0, 1, -1)
    inv_core = float((last_pred_d == (-last_true_d)).mean())

    matched_rate = mapped / max(1, n_genes)
    diagnostics = {
        "dataset": name,
        "model": args.model,
        "n_genes": n_genes,
        "n_cells": int(X.shape[0]),
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "vocab_match_rate": round(matched_rate * 100, 2),
        "mapped_genes": int(mapped),
        "top_vocab_match_rate": round(100.0 * sum(1 for i in eval_910 if map_idx[i] >= 0) / max(1, len(eval_910)), 2),
        "eval_percent": args.top_percent,
        "eval_genes_count": int(len(eval_910)),
        "scf_eps_dir": eps,
        "scf_eval_note": "top_percent of mapped genes; accuracy uses ±eps_dir and excludes true_dir==0",
        "final_acc_inv_truth": inv_core,
    }
    artifacts = {
        "genes": genes_original,
        "gene_tids": map_idx.astype(np.int64),
        "true_delta": true_delta_910.astype(np.float32, copy=False),
        "early_mean": early_mean_910.astype(np.float32, copy=False),
        "late_mean": late_mean_910.astype(np.float32, copy=False),
        "top_idx": eval_910.astype(np.int64, copy=False),
        "pred_delta_by_iter": pred_delta_by_iter,
        "pred_mean_910": pred_mean_910.astype(np.float32, copy=False),
        "pred_delta_final_910": pred_delta_final_910.astype(np.float32, copy=False),
        "map_idx": map_idx,
        "preds_full_by_iter": preds_full_by_iter,
    }
    return acc_curve, diagnostics, artifacts


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Return 2D PCA coordinates using numpy SVD."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Input must be 2D array for PCA.")
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x_centered = x - x.mean(axis=0, keepdims=True)
    # Handle degenerate case with one sample.
    if x_centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    comp = vt[:2].T if vt.shape[0] >= 2 else vt[:1].T
    z = x_centered @ comp
    if z.shape[1] == 1:
        z = np.concatenate([z, np.zeros((z.shape[0], 1), dtype=z.dtype)], axis=1)
    return z.astype(np.float32, copy=False)


def save_trajectory_plot_and_data(ds_dir: Path, name: str, args, artifacts: dict):
    if not args.save_trajectory_plot:
        return

    # Use per-iteration gene-level delta vectors as trajectory states.
    traj = artifacts.get("mean_delta_by_iter", None)
    traj_type = "mean_rank_delta"
    if traj is None:
        traj = artifacts.get("pred_delta_by_iter", None)
        traj_type = "pred_delta"
    if traj is None:
        return

    traj = np.asarray(traj, dtype=np.float32)
    if traj.size == 0:
        return

    np.save(ds_dir / "trajectory_raw_matrix.npy", traj)

    td = artifacts.get("true_delta")
    td = np.asarray(td, dtype=np.float32).ravel() if td is not None else None
    use_anchors = td is not None and td.shape[0] == traj.shape[1]

    if use_anchors:
        traj_c = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
        start_row = np.zeros((1, traj.shape[1]), dtype=np.float32)
        true_row = np.nan_to_num(td.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        all_X = np.vstack([start_row, traj_c, true_row])
        np.save(ds_dir / "trajectory_embed_input_matrix.npy", all_X)
        n_traj = int(traj.shape[0])
        n_all = int(all_X.shape[0])
    else:
        all_X = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
        n_traj = int(traj.shape[0])
        n_all = n_traj

    nn = min(15, max(2, n_all - 1))
    emb_name = args.trajectory_embed
    if args.trajectory_embed == "umap":
        try:
            import umap
        except Exception as e:
            raise RuntimeError(
                "UMAP requested but 'umap-learn' is not installed. "
                "Please run: pip install umap-learn"
            ) from e
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=0.15,
            metric="euclidean",
            random_state=args.seed,
        )
        coords = reducer.fit_transform(all_X).astype(np.float32, copy=False)
    else:
        coords = _pca_2d(all_X)

    # Table: every row in all_X with semantic label
    if use_anchors:
        point_labels = ["start_delta0"] + [f"iter_{k + 1}" for k in range(n_traj)] + ["true_delta_target"]
        iter_col = np.array([0] + list(range(1, n_traj + 1)) + [-1], dtype=np.int64)
    else:
        point_labels = [f"iter_{k + 1}" for k in range(n_traj)]
        iter_col = np.arange(1, n_traj + 1, dtype=np.int64)
    df = pd.DataFrame(
        {
            "point": point_labels,
            "iter": iter_col,
            f"{emb_name}1": coords[:, 0],
            f"{emb_name}2": coords[:, 1],
        }
    )
    df.to_csv(ds_dir / f"trajectory_{emb_name}2d.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=200)
    if use_anchors:
        # Indices: 0=start, 1..n_traj=iterations, n_traj+1=true target; pred end = n_traj
        sl = slice(1, n_traj + 1)
        c_traj = coords[sl]
        ax.plot(c_traj[:, 0], c_traj[:, 1], "-", color="0.7", lw=1.1, zorder=1)
        sc = ax.scatter(
            c_traj[:, 0],
            c_traj[:, 1],
            c=np.arange(1, n_traj + 1),
            cmap="viridis",
            s=45,
            zorder=2,
            edgecolors="white",
            linewidths=0.35,
        )
        fig.colorbar(sc, ax=ax, shrink=0.72, label="Iteration")
        i0, i_pred, i_true = 0, n_traj, n_traj + 1
        ax.scatter(
            coords[i0, 0],
            coords[i0, 1],
            s=160,
            c="#1b9e77",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label="Start (Δ=0)",
        )
        ax.scatter(
            coords[i_pred, 0],
            coords[i_pred, 1],
            s=160,
            c="#d95f02",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label="Pred end (last iter)",
        )
        ax.scatter(
            coords[i_true, 0],
            coords[i_true, 1],
            s=160,
            c="#7570b3",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label="True Δ (late−early)",
        )
        for a, b, col in [(i0, i_pred, "0.35"), (i_pred, i_true, "0.45")]:
            ax.annotate(
                "",
                xy=(coords[b, 0], coords[b, 1]),
                xytext=(coords[a, 0], coords[a, 1]),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.4, shrinkA=10, shrinkB=10),
                zorder=3,
            )
        subtitle = f"{traj_type} in gene space; 2D={emb_name.upper()}"
        ax.set_title(f"{name}\n{subtitle}", fontsize=10)
    else:
        ax.plot(coords[:, 0], coords[:, 1], "-o", linewidth=1.2, markersize=4.0, color="#2F6BFF")
        for i, (x, y) in enumerate(coords, start=1):
            if i == 1 or i == len(coords) or (i % max(1, args.print_every) == 0):
                ax.text(x, y, str(i), fontsize=7, ha="left", va="bottom")
        ax.scatter(coords[0, 0], coords[0, 1], color="#00A087", s=40, label="first iter", zorder=5)
        ax.scatter(coords[-1, 0], coords[-1, 1], color="#E64B35", s=40, label="last iter", zorder=5)
        ax.set_title(f"{name} trajectory ({traj_type}, {emb_name.upper()})")

    ax.set_xlabel(f"{emb_name.upper()}1")
    ax.set_ylabel(f"{emb_name.upper()}2")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(ds_dir / f"trajectory_{emb_name}2d.png", dpi=300)
    plt.close(fig)


def save_dataset_artifacts(outdir: Path, name: str, args, diag: dict, artifacts: dict):
    if not args.save_iterations:
        return

    ds_dir = outdir / "per_dataset" / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays as npy (compact + fast)
    if artifacts.get("mean_delta_by_iter") is not None:
        np.save(ds_dir / "mean_rank_delta_by_iter.npy", artifacts["mean_delta_by_iter"])
    if artifacts.get("pred_delta_by_iter") is not None:
        np.save(ds_dir / "pred_delta_by_iter.npy", artifacts["pred_delta_by_iter"])
    if artifacts.get("preds_full_by_iter") is not None:
        outp = ds_dir / "preds_full_by_iter.npy"
        np.save(outp, artifacts["preds_full_by_iter"])
        print(f"    [save-cell-preds-by-iter] {outp.name} -> {outp}")
    if artifacts.get("token_ids_by_iter") is not None:
        outp = ds_dir / "token_ids_by_iter.npy"
        np.save(outp, artifacts["token_ids_by_iter"])
        print(f"    [save-cell-preds-by-iter] {outp.name} -> {outp}")
    if artifacts.get("example_seqs_by_iter") is not None:
        np.save(ds_dir / "example_seqs_by_iter.npy", artifacts["example_seqs_by_iter"])

    genes = artifacts["genes"]
    true_delta = artifacts["true_delta"]
    top_idx = set(int(i) for i in artifacts["top_idx"].tolist())

    # Final per-gene table
    if "mean_delta_by_iter" in artifacts and artifacts["mean_delta_by_iter"] is not None:
        pred_rank_delta_final = artifacts["mean_delta_by_iter"][-1]
        pred_dir_from_rank_final = -pred_rank_delta_final
        token_ids = artifacts.get("gene_tids", np.full(len(genes), -1, dtype=np.int64))
        is_mapped = token_ids != 0
        df = pd.DataFrame(
            {
                "gene": genes,
                "token_id": token_ids,
                "is_mapped": is_mapped,
                "true_delta": true_delta,
                "pred_rank_delta": pred_rank_delta_final,
                "pred_dir_from_rank": pred_dir_from_rank_final,
                "true_sign": np.sign(true_delta),
                "pred_sign": np.sign(pred_dir_from_rank_final),
            }
        )
        # For unmapped genes (token_id==0), pred_rank_delta is not meaningful.
        df.loc[~df["is_mapped"], ["pred_rank_delta", "pred_dir_from_rank", "pred_sign"]] = np.nan
        df["correct_sign"] = df["true_sign"].values == df["pred_sign"].values
        df["in_top_eval"] = [i in top_idx for i in range(len(genes))]
        df.to_csv(ds_dir / "per_gene_final_changes.csv", index=False)

        # Export "rank before vs rank after" for manual inspection.
        # We report mean positions across early cells:
        #   pos_before_mean = mean(init_positions)
        #   rank_delta_mean = mean(pos_after - pos_before)
        #   pos_after_mean  = pos_before_mean + rank_delta_mean
        mean_pos_before = artifacts.get("mean_pos_before", None)
        if mean_pos_before is not None:
            pos_before_mean = mean_pos_before.astype(np.float32, copy=False)
            rank_delta_mean = pred_rank_delta_final.astype(np.float32, copy=False)
            pos_after_mean = pos_before_mean + rank_delta_mean

            true_delta_np = true_delta.astype(np.float32, copy=False)
            dir_true = np.where(true_delta_np > 0, "Up", "Down")
            dir_pred = np.where(pred_dir_from_rank_final > 0, "Up", "Down")
            dir_correct = (dir_true == dir_pred).astype(np.int64)

            rank_df = pd.DataFrame(
                {
                    "gene": genes,
                    "token_id": token_ids,
                    "is_mapped": is_mapped,
                    "pos_before_mean": pos_before_mean,
                    "pos_after_mean": pos_after_mean,
                    "rank_delta_mean": rank_delta_mean,
                    "dir_true": dir_true,
                    "dir_pred": dir_pred,
                    "dir_correct": dir_correct,
                }
            )
            rank_df.to_csv(
                ds_dir / f"{name}_rank_before_after_final.csv",
                index=False,
                float_format="%.15g",
            )

            top_idx_list = sorted(list(top_idx))
            if len(top_idx_list) > 0:
                rank_df.iloc[top_idx_list].to_csv(
                    ds_dir / f"{name}_rank_before_after_final_top.csv",
                    index=False,
                    float_format="%.15g",
                )

        # Export direction table matching scGPT's *_gene_result.csv columns:
        # gene,true_early_mean,true_late_mean,pred_late_like_mean,delta_true,delta_pred,dir_true,dir_pred,dir_correct
        true_early_mean = artifacts["early_mean"]
        true_late_mean = artifacts["late_mean"]
        delta_true = true_late_mean - true_early_mean

        # token Model"/",:
        #   rank_delta = pos_after - pos_before
        #   rank () => pred_dir_from_rank_final = -rank_delta > 0 => Up
        pred_late_like_mean = np.full_like(true_early_mean, np.nan, dtype=np.float32)
        #  pred_dir_from_rank_final  delta_pred: Up/Down( dir_pred )
        delta_pred = pred_dir_from_rank_final.astype(np.float32, copy=False)

        dir_true = np.where(delta_true > 0, "Up", "Down")
        dir_pred = np.where(pred_dir_from_rank_final > 0, "Up", "Down")
        dir_correct = (dir_true == dir_pred).astype(np.int64)
        gene_result_df = pd.DataFrame(
            {
                "gene": genes,
                "true_early_mean": true_early_mean,
                "true_late_mean": true_late_mean,
                "pred_late_like_mean": pred_late_like_mean,
                "delta_true": delta_true,
                "delta_pred": delta_pred,
                "dir_true": dir_true,
                "dir_pred": dir_pred,
                "dir_correct": dir_correct,
            }
        )
        #  per_dataset/<name>/..., outdir/<name>_gene_result.csv, scGPT .
        gene_result_df.to_csv(ds_dir / f"{name}_gene_result.csv", index=False, float_format="%.15g")
        gene_result_df.to_csv(outdir / f"{name}_gene_result.csv", index=False, float_format="%.15g")

        # Example-cell per-gene position deltas (token model only)
        if artifacts.get("example_seqs_by_iter") is not None:
            seq0 = artifacts["example_seqs_by_iter"][0]
            seqN = artifacts["example_seqs_by_iter"][-1]
            L = int(artifacts.get("example_seq_len", len(seq0)))
            pos0 = positions_dict_from_seq(seq0, L)
            posN = positions_dict_from_seq(seqN, L)
            token_ids = artifacts.get("gene_tids", np.full(len(genes), -1, dtype=np.int64))
            is_mapped = token_ids != 0

            pos_before = []
            pos_after = []
            present_before = []
            present_after = []
            for t in token_ids:
                t = int(t)
                pb = pos0.get(t, None)
                pa = posN.get(t, None)
                present_before.append(pb is not None)
                present_after.append(pa is not None)
                pos_before.append(pb)
                pos_after.append(pa)

            df2 = pd.DataFrame(
                {
                    "gene": genes,
                    "token_id": token_ids,
                    "is_mapped": is_mapped,
                    "present_before": present_before,
                    "present_after": present_after,
                    "pos_before": pos_before,
                    "pos_after": pos_after,
                }
            )
            df2["rank_delta_example_cell"] = np.where(
                df2["is_mapped"] & df2["present_before"] & df2["present_after"],
                df2["pos_after"].astype(float) - df2["pos_before"].astype(float),
                np.nan,
            )
            df2.to_csv(ds_dir / "example_cell_rank_changes.csv", index=False)

    elif "pred_delta_by_iter" in artifacts and artifacts["pred_delta_by_iter"] is not None:
        pred_delta_final = artifacts["pred_delta_by_iter"][-1]
        df = pd.DataFrame(
            {
                "gene": genes,
                "true_delta": true_delta,
                "pred_delta": pred_delta_final,
                "true_sign": np.sign(true_delta),
                "pred_sign": np.sign(pred_delta_final),
            }
        )
        # scFoundation / NaN unmapped: skip sign compare where pred is missing
        ok = ~np.isnan(pred_delta_final)
        df["correct_sign"] = np.where(ok, df["true_sign"].values == df["pred_sign"].values, np.nan)
        df["in_top_eval"] = [i in top_idx for i in range(len(genes))]
        if "map_idx" in artifacts:
            df["model_index"] = artifacts["map_idx"]
            df["mapped"] = artifacts["map_idx"] >= 0
        df.to_csv(ds_dir / "per_gene_final_changes.csv", index=False)

        # Export direction table matching scGPT's *_gene_result.csv columns:
        # gene,true_early_mean,true_late_mean,pred_late_like_mean,delta_true,delta_pred,dir_true,dir_pred,dir_correct
        true_early_mean = artifacts.get("early_mean")
        true_late_mean = artifacts.get("late_mean")
        if true_early_mean is not None and true_late_mean is not None:
            delta_true = true_late_mean - true_early_mean
            delta_pred = pred_delta_final
            pred_late_like_mean = true_early_mean + delta_pred
            dir_true = np.where(delta_true > 0, "Up", "Down")

            pred_nan = np.isnan(delta_pred)
            dir_pred = np.where(pred_nan, "", np.where(delta_pred > 0, "Up", "Down"))
            dir_correct = np.where(
                pred_nan,
                np.nan,
                (np.where(delta_pred > 0, 1, -1) == np.where(delta_true > 0, 1, -1)).astype(np.int64),
            )

            gene_result_df = pd.DataFrame(
                {
                    "gene": genes,
                    "true_early_mean": true_early_mean,
                    "true_late_mean": true_late_mean,
                    "pred_late_like_mean": pred_late_like_mean,
                    "delta_true": delta_true,
                    "delta_pred": delta_pred,
                    "dir_true": dir_true,
                    "dir_pred": dir_pred,
                    "dir_correct": dir_correct,
                }
            )
            gene_result_df.to_csv(ds_dir / f"{name}_gene_result.csv", index=False, float_format="%.15g")
            gene_result_df.to_csv(outdir / f"{name}_gene_result.csv", index=False, float_format="%.15g")

    # Optional trajectory visualization + plotting source data
    save_trajectory_plot_and_data(ds_dir, name, args, artifacts)


def load_scgpt_model(args, device):
    if args.scgpt_repo_dir not in sys.path:
        sys.path.insert(0, args.scgpt_repo_dir)
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    with open(Path(args.scgpt_model_dir) / "args.json") as f:
        cfg = json.load(f)
    vocab = GeneVocab.from_file(Path(args.scgpt_model_dir) / "vocab.json")
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
    ckpt = torch.load(Path(args.scgpt_model_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.half()
    return model, vocab


def main():
    args = parse_args()
    args.use_log1p = args.use_log1p and (not args.no_log1p)
    set_seed(args.seed)
    datasets_cfg = resolve_datasets(args)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    all_curves = {}
    all_diag = {}
    errors = {}

    if args.model == "geneformer":
        require_paths(args, ["geneformer_model_dir", "geneformer_dicts_dir"])
        model = BertForMaskedLM.from_pretrained(args.geneformer_model_dir).to(device).eval()
        vocab, gene_name_id, pad_id, mask_id = load_geneformer_dicts(Path(args.geneformer_dicts_dir))
        runner = lambda n, c: run_token_model_dataset(n, c, model, vocab, gene_name_id, pad_id, mask_id, device, args, True)
    elif args.model == "langcell":
        require_paths(args, ["langcell_model_dir", "geneformer_dicts_dir"])
        model = LangCellModel(args.langcell_model_dir).to(device).eval()
        vocab, gene_name_id, pad_id, mask_id = load_geneformer_dicts(Path(args.geneformer_dicts_dir))
        runner = lambda n, c: run_token_model_dataset(n, c, model, vocab, gene_name_id, pad_id, mask_id, device, args, False)
    elif args.model == "sccello":
        require_paths(args, ["sccello_model_dir", "sccello_repo_dir", "geneformer_dicts_dir"])
        # In this pseudotime loop, scCello frequently hits CUDA index/cublas runtime faults.
        # Default to CPU unless user explicitly opts in with --sccello-cuda.
        if device.type == "cuda" and (not args.sccello_cuda):
            print("  [WARN] sccello defaults to CPU in this runner; use --sccello-cuda to force GPU.")
            device = torch.device("cpu")

        # scCello provides a masked-LM head in sc_foundation_evals/sccello; wrap to return logits tensor.
        if args.sccello_repo_dir not in sys.path:
            sys.path.insert(0, args.sccello_repo_dir)
        from sccello.src.model_prototype_contrastive import PrototypeContrastiveForMaskedLM

        inner = PrototypeContrastiveForMaskedLM.from_pretrained(
            args.sccello_model_dir,
            ignore_mismatched_sizes=True,
        ).to(device).eval()
        model = ScCelloMaskedLMWrapper(inner).to(device).eval()

        # IMPORTANT: scCello expects pad_id=0 and mask_id=1 (see scCello collators).
        # Also, protect against any token-id mismatch by clamping to scCello vocab_size.
        vocab, gene_name_id, _pad, _mask = load_geneformer_dicts(Path(args.geneformer_dicts_dir))
        sc_ens2t = load_sccello_ensembl_to_token_id(Path(args.sccello_repo_dir))
        if sc_ens2t is not None:
            vocab = merge_vocab_with_sccello_ids(vocab, gene_name_id, sc_ens2t)
            print(
                f"  [sccello] Using token_dictionary.pkl from repo ({len(sc_ens2t)} Ensembl→id entries) for gene tokens."
            )
        else:
            print(
                "  [WARN] sccello: token_dictionary.pkl not found under "
                f"{Path(args.sccello_repo_dir) / 'sccello' / 'data' / 'token_vocabulary'} — "
                "gene ids fall back to Geneformer mapping; logits rows may not match scCello embeddings."
            )
        pad_id = int(getattr(inner.config, "pad_token_id", 0))
        mask_id = 1
        runner = lambda n, c: run_token_model_dataset(
            n,
            c,
            model,
            vocab,
            gene_name_id,
            pad_id,
            mask_id,
            device,
            args,
            False,
            vocab_size_limit=int(getattr(inner.config, "vocab_size", 0)) or None,
            max_position_embeddings=int(getattr(inner.config, "max_position_embeddings", 0)) or None,
        )
    elif args.model == "scgpt":
        require_paths(args, ["scgpt_model_dir"])
        model, vocab = load_scgpt_model(args, device)
        runner = lambda n, c: run_scgpt_dataset(n, c, model, vocab, device, args)
    else:
        require_paths(args, ["scf_root", "scf_ckpt", "scf_gene_index_tsv"])
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        model, scf_cfg = load_scfoundation_model(args, device)
        G_model = int(scf_cfg["seq_len"])
        scf_gene2idx = scf_read_gene_index_tsv(args.scf_gene_index_tsv, G_model)
        runner = lambda n, c: run_scfoundation_dataset(n, c, model, scf_cfg, scf_gene2idx, device, args)

    selected = None
    if args.datasets.strip():
        selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
        unknown = sorted(selected - set(datasets_cfg.keys()))
        if unknown:
            raise ValueError(f"Unknown dataset(s): {unknown}. Valid: {sorted(datasets_cfg.keys())}")

    for name, cfg in datasets_cfg.items():
        if selected is not None and name not in selected:
            continue
        print(f"\n===== {name} ({args.model}) =====")
        try:
            acc_curve, diag, artifacts = runner(name, cfg)
            all_curves[name] = acc_curve
            all_diag[name] = diag
            save_dataset_artifacts(outdir, name, args, diag, artifacts)
            inv = diag.get("final_acc_inv_truth", None)
            if inv is not None:
                print(f"  Final accuracy: {acc_curve[-1]:.2%} | inv-truth: {inv:.2%}")
            else:
                print(f"  Final accuracy: {acc_curve[-1]:.2%}")
        except Exception as e:
            errors[name] = str(e)
            print(f"  [ERROR] {name}: {e}")

    with open(outdir / "accuracy_curves.json", "w") as f:
        json.dump(all_curves, f, indent=2)
    with open(outdir / "diagnostics.json", "w") as f:
        json.dump(all_diag, f, indent=2)
    with open(outdir / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SUMMARY | model={args.model} | outdir={outdir}")
    print("=" * 70)
    for name, acc in all_curves.items():
        print(f"{name:<12} | Accuracy: {acc[-1]:.2%}")
    if errors:
        print("-" * 70)
        print("FAILED DATASETS")
        for k, v in errors.items():
            print(f"{k:<12} | {v}")


if __name__ == "__main__":
    main()
