import argparse
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel

import scanpy as sc

from utils_heads import parse_head_indices


warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export LangCell per-attention-head TSVs (Gene1/Gene2/EdgeWeight) for one dataset. "
            "This script does not use any ground-truth network filtering."
        )
    )
    p.add_argument("--data-type", required=True, choices=["CHIP", "Non_CHIP", "STRING"])
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument(
        "--parent-model-dir",
        required=True,
        type=str,
        help="Parent weights dir containing LangCell/ and Geneformer/dicts/.",
    )
    p.add_argument("--langcell-model-dir", default="", type=str, help="Optional override for LangCell model dir (must contain cell_bert).")
    p.add_argument("--dict-dir", default="", type=str, help="Optional override for Geneformer dict dir.")
    p.add_argument("--target-layer", default=-1, type=int, help="Attention layer index (-1 means last).")
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--head-indices",
        default="all",
        type=str,
        help="Comma-separated head indices, e.g. '0,2'. Default 'all'.",
    )
    return p.parse_args()


def get_expr_csv(input_root: Path, data_type: str, dataset: str) -> Path:
    if data_type == "CHIP":
        return input_root / "CHIP" / f"{dataset}_chip_matched-ExpressionData.csv"
    return input_root / data_type / f"{dataset}_processed-ExpressionData.csv"


def tokenize_cell(
    gene_ids: np.ndarray,
    gene_values: np.ndarray,
    pad_token_id: int,
) -> Tuple[List[int], List[int]]:
    sorted_indices = np.argsort(-gene_values)
    sorted_genes = gene_ids[sorted_indices]
    sorted_values = gene_values[sorted_indices]

    nonzero_mask = sorted_values > 0
    sorted_genes = sorted_genes[nonzero_mask]
    sorted_values = sorted_values[nonzero_mask]
    sorted_indices = sorted_indices[nonzero_mask]

    tokens: List[int] = []
    kept_indices: List[int] = []

    # No additional filtering: keep all non-zero expressed genes
    for idx, gene_id in enumerate(sorted_genes):
        tokens.append(int(gene_id))
        kept_indices.append(int(sorted_indices[idx]))

    if len(tokens) == 0:
        tokens = [pad_token_id]
        kept_indices = [0]
    return tokens, kept_indices


def rank_normalize_heads(attn_scores: torch.Tensor) -> torch.Tensor:
    """Rank-normalize attention while preserving head dimension. Shape: [b, h, M, M]."""
    b, h, M, _ = attn_scores.shape
    x = attn_scores.reshape((-1, M))
    order = torch.argsort(x, dim=1)
    rank = torch.argsort(order, dim=1)
    x = rank.reshape((-1, h, M, M)).float() / float(M)

    x = x.permute(0, 1, 3, 2).reshape((-1, M))
    order = torch.argsort(x, dim=1)
    rank = torch.argsort(order, dim=1)
    x = (rank.reshape((-1, h, M, M)).float() / float(M)).permute(0, 1, 3, 2)
    return x


def reverse_permute_4d(tensor4d: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Reverse permutation for [b, h, M, M] tensor; indices is [b, M]."""
    reverse_indices = torch.argsort(indices, dim=1)  # [b, M]
    b, h, M, _ = tensor4d.shape
    expanded_row = reverse_indices.unsqueeze(1).unsqueeze(-1).expand(-1, h, M, M)
    out = torch.gather(tensor4d, 2, expanded_row)
    expanded_col = reverse_indices.unsqueeze(1).unsqueeze(2).expand(-1, h, M, M)
    out = torch.gather(out, 3, expanded_col)
    return out


class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data: Dict[str, List[List[int]]]):
        self.data = tokenized_data

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "input_ids": self.data["input_ids"][idx],
            "sorted_indices": self.data["sorted_indices"][idx],
            "idx": idx,
        }


class Collator:
    def __init__(self, add_cls: bool = True):
        self.add_cls = add_cls

    def __call__(self, features: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        sorted_indices = [f["sorted_indices"] for f in features]
        idx = [int(f["idx"]) for f in features]

        max_len = max(len(ids) for ids in input_ids)
        padded_ids = []
        padded_sorted = []
        attention_mask = []
        for ids, inds in zip(input_ids, sorted_indices):
            pad = max_len - len(ids)
            padded_ids.append(ids + [0] * pad)
            padded_sorted.append(inds + list(range(len(inds), max_len)))
            attention_mask.append([1] * len(ids) + [0] * pad)

        if self.add_cls:
            padded_ids = [[0] + x for x in padded_ids]
            padded_sorted = [[0] + [i + 1 for i in x] for x in padded_sorted]
            attention_mask = [[1] + x for x in attention_mask]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sorted_indices": torch.tensor(padded_sorted, dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    parent_model_dir = Path(args.parent_model_dir)

    langcell_model_dir = Path(args.langcell_model_dir) if args.langcell_model_dir else parent_model_dir / "LangCell"
    dict_dir = Path(args.dict_dir) if args.dict_dir else parent_model_dir / "Geneformer" / "dicts"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    expr_csv = get_expr_csv(input_root, args.data_type, args.dataset)
    if not expr_csv.exists():
        raise FileNotFoundError(f"Expression CSV not found: {expr_csv}")

    token_pkl = dict_dir / "token_dictionary.pkl"
    gene_name_id_pkl = dict_dir / "gene_name_id_dict.pkl"
    for p in [token_pkl, gene_name_id_pkl]:
        if not p.exists():
            raise FileNotFoundError(f"Missing dict file: {p}")

    with open(token_pkl, "rb") as f:
        vocab = pickle.load(f)  # token -> id
    with open(gene_name_id_pkl, "rb") as f:
        gene_name_id = pickle.load(f)  # Symbol -> ENSG
    pad_token_id = int(vocab.get("<pad>", 0))

    # Load expression (genes x cells -> cells x genes)
    df = pd.read_csv(expr_csv, index_col=0).T
    adata = sc.AnnData(X=df.values)
    adata.obs_names = df.index
    adata.var_names = df.columns

    # Map genes to ENSG and ensure token exists
    gene_names = [str(x).strip() for x in adata.var_names.tolist()]
    ensg_list: List[str] = []
    keep_mask: List[bool] = []
    for g in gene_names:
        ensg = None
        if g in gene_name_id:
            ensg = gene_name_id[g]
        elif g.startswith("ENSG"):
            ensg = g
        else:
            g_up = g.upper()
            for sym, ens in gene_name_id.items():
                if isinstance(sym, str) and sym.upper() == g_up:
                    ensg = ens
                    break
        if ensg is None or ensg not in vocab:
            keep_mask.append(False)
            ensg_list.append("")
        else:
            keep_mask.append(True)
            ensg_list.append(ensg)

    if sum(keep_mask) == 0:
        raise RuntimeError("No genes matched the LangCell vocabulary.")

    adata = adata[:, keep_mask].copy()
    kept_genes = [str(x).strip() for x in adata.var_names.tolist()]
    kept_ensg = [ensg for ensg, keep in zip(ensg_list, keep_mask) if keep]

    # No extra filtering: keep all cells that appear in the input CSV
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    gene_ids = np.array([int(vocab[e]) for e in kept_ensg], dtype=int)

    # Tokenize each cell
    tokenized_cells = []
    for i in tqdm(range(X.shape[0]), desc="Tokenizing cells"):
        tokens, kept_indices = tokenize_cell(
            gene_ids=gene_ids,
            gene_values=X[i, :],
            pad_token_id=pad_token_id,
        )
        tokenized_cells.append({"input_ids": tokens, "sorted_indices": kept_indices})

    tokenized_data = {
        "input_ids": [c["input_ids"] for c in tokenized_cells],
        "sorted_indices": [c["sorted_indices"] for c in tokenized_cells],
    }

    ds = TokenizedDataset(tokenized_data)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=Collator(add_cls=True), num_workers=0)

    # Load model
    model = BertModel.from_pretrained(str(langcell_model_dir / "cell_bert"))
    model = model.to(device)
    model.eval()

    # Resolve layer index
    n_layers = int(model.config.num_hidden_layers)
    target_layer = n_layers - 1 if int(args.target_layer) < 0 else int(args.target_layer)
    if target_layer < 0 or target_layer >= n_layers:
        raise ValueError(f"Invalid target layer: {target_layer} for n_layers={n_layers}")

    attention_sum_heads = None  # [heads, G, G]
    num_cells = 0

    n_genes = len(kept_genes)
    add_cls = True

    for batch in tqdm(dl, desc="Running attention forward"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sorted_indices = batch["sorted_indices"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, output_attentions=True)
        attn_scores = outputs.attentions[target_layer]  # [b, heads, seq, seq]
        num_heads = attn_scores.size(1)

        if add_cls:
            attn_scores = attn_scores[..., 1:, 1:]
            sorted_indices = sorted_indices[:, 1:]
            mask_2d = attention_mask[:, 1:]
        else:
            mask_2d = attention_mask

        attn_scores = rank_normalize_heads(attn_scores)
        attn_scores = reverse_permute_4d(attn_scores, sorted_indices)
        # mask_matrix: [b, 1, M, M], broadcast on head dimension
        mask_matrix = mask_2d[:, None, :, None] * mask_2d[:, None, None, :]
        attn_scores = attn_scores * mask_matrix

        attn_np = attn_scores.detach().cpu().numpy()  # [b, h, M, M]
        sorted_np = sorted_indices.detach().cpu().numpy()  # [b, M]
        if attn_np.ndim != 4:
            raise RuntimeError(f"Expected attention ndim=4 ([b,h,M,M]), got shape={attn_np.shape}")
        bsz, _, M, _ = attn_np.shape

        # Fill full gene-space attention matrices. Use per-head np.ix_ to avoid numpy axis reordering.
        full_attn_heads = np.zeros((bsz, num_heads, n_genes, n_genes), dtype=np.float32)
        for b in range(bsz):
            idx = np.clip(sorted_np[b], 0, n_genes - 1).astype(np.int64)
            ii = np.ix_(idx, idx)
            for h in range(num_heads):
                full_attn_heads[b, h][ii] = attn_np[b, h]

        if attention_sum_heads is None:
            attention_sum_heads = full_attn_heads.sum(axis=0)  # [h, G, G]
        else:
            attention_sum_heads += full_attn_heads.sum(axis=0)
        num_cells += bsz

        del outputs, attn_scores, input_ids, attention_mask
        torch.cuda.empty_cache()

    avg_attention_heads = attention_sum_heads / float(num_cells)
    output_root.mkdir(parents=True, exist_ok=True)

    for h in parse_head_indices(args.head_indices, avg_attention_heads.shape[0]):
        mat = avg_attention_heads[h]
        edges = pd.DataFrame(mat, index=kept_genes, columns=kept_genes).stack().reset_index()
        edges.columns = ["Gene1", "Gene2", "EdgeWeight"]
        edges = edges[edges["Gene1"] != edges["Gene2"]]
        edges = edges.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
        out_path = output_root / f"langcell_{args.dataset}_head{h}.tsv"
        edges.to_csv(out_path, sep="\t", index=False)
        print(f"[INFO] Saved {out_path} (edges={len(edges)})")


if __name__ == "__main__":
    main()

