import argparse
import os
import pickle
import time
from pathlib import Path

import loompy
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM

from utils_heads import parse_head_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Geneformer per-head TSV files.")
    p.add_argument("--data-type", required=True, choices=["CHIP", "Non_CHIP", "STRING"])
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--weights-root", required=True, type=str, help="Root containing Geneformer/")
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--model-version", default="6L", type=str)
    p.add_argument("--target-layer", default=-1, type=int)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument(
        "--head-indices",
        default="all",
        type=str,
        help="Comma-separated head indices, e.g. '0,2'. Default 'all'.",
    )
    return p.parse_args()


def rank_normalize_heads(attn: torch.Tensor) -> torch.Tensor:
    b, h, m, _ = attn.shape
    out = torch.zeros_like(attn, dtype=torch.float32)
    for bi in range(b):
        for hi in range(h):
            row = attn[bi, hi]
            sidx = torch.argsort(row, dim=1, descending=True)
            rank = torch.argsort(sidx, dim=1)
            out[bi, hi] = rank.float() / (m - 1 if m > 1 else 1)
    for bi in range(b):
        for hi in range(h):
            col = out[bi, hi].T
            sidx = torch.argsort(col, dim=1, descending=True)
            rank = torch.argsort(sidx, dim=1)
            out[bi, hi] = (rank.float() / (m - 1 if m > 1 else 1)).T
    return out


def reverse_permute_4d(attn: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # attn: [b,h,m,m], indices: [b,m]
    b, h, m, _ = attn.shape
    out = torch.zeros_like(attn)
    inv = torch.argsort(indices, dim=1)
    for bi in range(b):
        p = inv[bi]
        out[bi] = attn[bi][:, p][:, :, p]
    return out


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    sorted_indices = []
    for x in batch:
        ids = x["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        sidx = x.get("sorted_indices", list(range(len(ids))))
        sorted_indices.append(sidx + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "sorted_indices": torch.tensor(sorted_indices),
    }


def main():
    args = parse_args()
    data_type = args.data_type
    dataset = args.dataset
    weights_root = Path(args.weights_root)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    out_dir = output_root / "geneformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    geneformer_base = weights_root / "Geneformer"
    dict_dir = geneformer_base / "dicts"
    model_dir = geneformer_base / "default" / args.model_version

    file_suffix = "_chip_matched" if data_type == "CHIP" else "_processed"
    input_subdir = "CHIP" if data_type == "CHIP" else data_type
    csv_path = input_root / input_subdir / f"{dataset}{file_suffix}-ExpressionData.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with open(dict_dir / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_dict = pickle.load(f)
    with open(dict_dir / "token_dictionary.pkl", "rb") as f:
        token_dict = pickle.load(f)

    adata = sc.read_csv(str(csv_path)).T
    gene_symbols = adata.var_names.tolist()
    matched_mask = [g in gene_name_dict for g in gene_symbols]
    adata = adata[:, matched_mask].copy()
    matched_gene_symbols = [g for g, m in zip(gene_symbols, matched_mask) if m]
    ensembl_ids = [gene_name_dict[g] for g in matched_gene_symbols]
    if len(matched_gene_symbols) == 0:
        raise ValueError("No matched genes in Geneformer dictionary.")

    adata.var["gene_name"] = matched_gene_symbols
    adata.var["ensembl_id"] = ensembl_ids
    adata.var_names = matched_gene_symbols
    sc.pp.filter_cells(adata, min_genes=200)
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1 if scipy.sparse.issparse(adata.X) else adata.X.sum(axis=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "unknown"

    temp_pre = out_dir / "temp_preprocessed" / f"{data_type}_{dataset}"
    temp_tok = out_dir / "temp_tokenized" / f"{data_type}_{dataset}"
    temp_pre.mkdir(parents=True, exist_ok=True)
    temp_tok.mkdir(parents=True, exist_ok=True)
    loom_path = temp_pre / f"{data_type}_{dataset}.loom"
    adata.write_loom(str(loom_path), write_obsm_varm=False)

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={"cell_type": "cell_type"},
        nproc=4,
        gene_median_file=str(dict_dir / "gene_median_dictionary.pkl"),
        token_dictionary_file=str(dict_dir / "token_dictionary.pkl"),
    )
    tokenizer.tokenize_data(
        data_directory=str(temp_pre),
        output_directory=str(temp_tok),
        output_prefix=f"{data_type}_{dataset}",
        file_format="loom",
        use_generator=False,
    )

    tokenized = load_from_disk(str(temp_tok / f"{data_type}_{dataset}.dataset"))
    dl = DataLoader(tokenized, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_fn)

    model = BertForMaskedLM.from_pretrained(str(model_dir), output_attentions=True, output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    target_layer = n_layers - 1 if int(args.target_layer) < 0 else int(args.target_layer)
    n_heads = int(model.config.num_attention_heads)

    id_to_ensembl = {v: k for k, v in token_dict.items()}
    ensembl_to_gene = {v: k for k, v in gene_name_dict.items()}
    final_gene_names = matched_gene_symbols
    gene_to_global_idx = {g: i for i, g in enumerate(final_gene_names)}
    n_genes = len(final_gene_names)

    attn_sum = np.zeros((n_heads, n_genes, n_genes), dtype=np.float64)
    attn_cnt = np.zeros((n_heads, n_genes, n_genes), dtype=np.int32)

    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting geneformer heads"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sorted_indices = batch["sorted_indices"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attn = outputs.attentions[target_layer]  # [b,h,m,m]
            attn = rank_normalize_heads(attn)
            attn = reverse_permute_4d(attn, sorted_indices)
            mask = attention_mask[:, None, :, None] * attention_mask[:, None, None, :]
            attn = attn * mask

            attn_np = attn.detach().cpu().numpy()
            input_ids_np = batch["input_ids"].cpu().numpy()
            mask_np = batch["attention_mask"].cpu().numpy()
            bsz = attn_np.shape[0]
            for bi in range(bsz):
                token_positions = np.where(mask_np[bi] > 0)[0]
                selected_pos = []
                selected_global = []
                seen = set()
                for pos in token_positions:
                    gid = int(input_ids_np[bi, pos])
                    gene = ensembl_to_gene.get(id_to_ensembl.get(gid, ""), "")
                    if not gene:
                        continue
                    gi = gene_to_global_idx.get(gene)
                    if gi is None or gi in seen:
                        continue
                    seen.add(gi)
                    selected_pos.append(int(pos))
                    selected_global.append(int(gi))
                if len(selected_global) < 2:
                    continue
                pos_arr = np.array(selected_pos, dtype=np.int32)
                gi_arr = np.array(selected_global, dtype=np.int32)
                for h in range(n_heads):
                    cell_h = attn_np[bi, h][np.ix_(pos_arr, pos_arr)]
                    attn_sum[h][np.ix_(gi_arr, gi_arr)] += cell_h
                    attn_cnt[h][np.ix_(gi_arr, gi_arr)] += 1

    attn_avg = np.divide(attn_sum, attn_cnt, out=np.zeros_like(attn_sum), where=attn_cnt > 0)

    for h in parse_head_indices(args.head_indices, n_heads):
        mat = attn_avg[h]
        edges = pd.DataFrame(mat, index=final_gene_names, columns=final_gene_names).stack().reset_index()
        edges.columns = ["Gene1", "Gene2", "EdgeWeight"]
        edges = edges[edges["Gene1"] != edges["Gene2"]]
        edges = edges.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
        out_path = out_dir / f"geneformer_{dataset}_head{h}.tsv"
        edges.to_csv(out_path, sep="\t", index=False)
        print(f"[INFO] Saved {out_path} edges={len(edges)}")

    try:
        if loom_path.exists():
            os.remove(loom_path)
        if (temp_tok / f"{data_type}_{dataset}.dataset").exists():
            import shutil
            shutil.rmtree(temp_tok / f"{data_type}_{dataset}.dataset", ignore_errors=True)
    except Exception as e:
        print(f"[WARN] cleanup failed: {e}")
    print(f"[INFO] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
