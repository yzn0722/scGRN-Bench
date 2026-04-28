
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangCell hidden-state gene embeddings -> cosine edges (Geneformer-aligned, batch)

 Geneformer :
   Symbol -> gene_name_id_dict.pkl -> ENSG -> token_dictionary.pkl

Dataset 1  TSV :
   {MODEL_NAME}_{DATASET}.tsv   (:Gene1, Gene2, EdgeWeight;;)

:
   run_params.json

 embedding (embedding )
"""

import os
import json
import pickle
import warnings
from pathlib import Path
import datetime
import traceback
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch
from transformers import BertModel


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    #  IO
    "INPUT_ROOT": "",
    "OUTPUT_ROOT": "",
    "TARGET_FOLDERS": ["CHIP"],   #  ["CHIP","Non_CHIP","STRING"]; None  INPUT_ROOT 

    # Model/
    "DICT_PATH": "",
    "LANGCELL_MODEL_PATH": "",
    "MODEL_NAME": "LangCell",     # (Model)

    # ( 512/512  ~1000 gene ; Geneformer  256/256)
    "SEQ_TOPK_GENES": 512,
    "MAX_SEQ_LEN": 512,
    "USE_LOG1P": False,

    # hidden extraction
    "HIDDEN_LAYER": -1,     # -1 = 
    "N_CELLS": None,        # None = 
    "BATCH_SIZE": 16,

    # 
    "SAVE_ALL_EDGES": True,   # True =>  i!=j (, NxN)
    "TOPK_PER_GENE": 1000,    # SAVE_ALL_EDGES=False : Gene1  topK

    # 
    "SEED": 42,
}


def parse_args():
    p = argparse.ArgumentParser(description="Extract LangCell hidden embeddings and export cosine-edge TSVs.")
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--dict-path", required=True, type=str)
    p.add_argument("--langcell-model-path", required=True, type=str)
    p.add_argument("--folders", nargs="+", default=["CHIP"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


ARGS = parse_args()
CONFIG["INPUT_ROOT"] = ARGS.input_root
CONFIG["OUTPUT_ROOT"] = ARGS.output_root
CONFIG["DICT_PATH"] = ARGS.dict_path
CONFIG["LANGCELL_MODEL_PATH"] = ARGS.langcell_model_path
CONFIG["TARGET_FOLDERS"] = ARGS.folders
CONFIG["BATCH_SIZE"] = int(ARGS.batch_size)
CONFIG["SEED"] = int(ARGS.seed)


# =============================================================================
# Utils
# =============================================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))


def extract_dataset_name(file_path: str) -> str:
    base = os.path.basename(file_path)
    if "_chip_matched-ExpressionData.csv" in base:
        return base.split("_chip_matched-ExpressionData.csv")[0]
    if "_processed-ExpressionData.csv" in base:
        return base.split("_processed-ExpressionData.csv")[0]
    return base.split("-ExpressionData.csv")[0]


def read_expression_matrix(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expression file not found: {path}")

    df = None
    try:
        df = pd.read_csv(path, sep="\t", header=0, index_col=0)
        if df.shape[1] == 0:
            df = None
    except Exception:
        df = None

    if df is None:
        df = pd.read_csv(path, sep=None, engine="python", header=0, index_col=0)

    df.index = df.index.astype(str).str.strip()
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")]

    if df.select_dtypes(include=[np.number]).shape[1] == 0:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.fillna(0.0)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty expression matrix after parsing: {df.shape}")

    print(f"Expression loaded: {df.shape[0]} genes x {df.shape[1]} cells")
    return df


def load_token_dictionary(dict_path: str) -> dict:
    p = os.path.join(dict_path, "token_dictionary.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}")
    with open(p, "rb") as f:
        token_to_id = pickle.load(f)
    print(f"Token dictionary loaded: {len(token_to_id)} tokens")
    return token_to_id


def load_gene_name_id_dict(dict_path: str) -> dict:
    p = os.path.join(dict_path, "gene_name_id_dict.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}")
    with open(p, "rb") as f:
        gene_name_id = pickle.load(f)  # Symbol -> ENSG ( str)
    print(f"Gene name dictionary loaded: {len(gene_name_id)} symbols")
    return gene_name_id


def get_pad_id(token_to_id: dict) -> int:
    for k in ["<pad>", "[PAD]", "pad", "PAD"]:
        if k in token_to_id:
            return int(token_to_id[k])
    return 0


def match_genes_to_tokens_like_geneformer(symbols, token_to_id, gene_name_id_dict):
    """
    Geneformer-aligned mapping:
      1)  gene  token key( ENSG),
      2)  Symbol -> ENSG via gene_name_id_dict, ENSG -> token_dictionary
    """
    gene_to_token_id = {}
    missing = 0
    for sym in symbols:
        if sym in token_to_id:
            gene_to_token_id[sym] = int(token_to_id[sym])
            continue
        ensg = gene_name_id_dict.get(sym, None)
        if ensg is not None and ensg in token_to_id:
            gene_to_token_id[sym] = int(token_to_id[ensg])
        else:
            missing += 1
    return gene_to_token_id, missing


def _get_hidden(outputs, layer_idx: int):
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[layer_idx]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    raise RuntimeError("Model outputs have no hidden states.")


def build_sequences(expr_df, gene_to_token_id, topk, max_len, n_cells, use_log1p, seed):
    set_seed(seed)

    X = expr_df.values.astype(np.float32)  # [G, C]
    if use_log1p:
        X = np.log1p(np.maximum(X, 0.0))

    genes = expr_df.index.astype(str).tolist()
    G, C = X.shape

    mappable_idx = [i for i, g in enumerate(genes) if g in gene_to_token_id]
    if len(mappable_idx) == 0:
        raise RuntimeError("No mappable genes after Geneformer-aligned mapping.")

    X_map = X[mappable_idx, :]
    genes_map = [genes[i] for i in mappable_idx]

    if n_cells is not None and n_cells > 0 and C > int(n_cells):
        col_idx = np.random.choice(C, size=int(n_cells), replace=False)
    else:
        col_idx = np.arange(C)

    seqs = []
    for ci in tqdm(col_idx, desc="Build sequences", leave=False):
        col = X_map[:, ci]
        k = min(int(topk), len(col))
        # fast top-k then sort descending
        top_i = np.argpartition(-col, kth=k - 1)[:k]
        top_i = top_i[np.argsort(-col[top_i])]
        ids = [gene_to_token_id[genes_map[j]] for j in top_i]
        ids = ids[:max_len]
        seqs.append(ids)

    return seqs, int(len(col_idx)), int(len(mappable_idx))


def compute_and_save_all_cosine_edges_stream(genes, emb, out_path):
    """
     i!=j (, NxN),:Gene1 Gene2 EdgeWeight
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = emb.shape[0]
    denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    E = emb / denom

    total = 0
    with open(out_path, "w") as f:
        f.write("Gene1\tGene2\tEdgeWeight\n")
        for i in tqdm(range(N), desc="All cosine edges (stream)"):
            sims = E[i] @ E.T
            sims[i] = -np.inf
            g1 = genes[i]
            for j in range(N):
                if j == i:
                    continue
                f.write(f"{g1}\t{genes[j]}\t{float(sims[j]):.8f}\n")
            total += (N - 1)

    return {"mode": "all_stream", "n_genes": int(N), "n_edges": int(total)}


def compute_and_save_topk_cosine_edges(genes, emb, out_path, topk: int):
    """
     Gene1  topK (),:Gene1 Gene2 EdgeWeight
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = emb.shape[0]
    denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    E = emb / denom

    total_edges = 0
    with open(out_path, "w") as f:
        f.write("Gene1\tGene2\tEdgeWeight\n")
        for i in tqdm(range(N), desc=f"Top{topk} cosine per Gene1"):
            sims = E[i] @ E.T
            sims[i] = -np.inf
            k = min(int(topk), N - 1)
            idx = np.argpartition(-sims, kth=k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]
            g1 = genes[i]
            for j in idx:
                f.write(f"{g1}\t{genes[j]}\t{float(sims[j]):.8f}\n")
            total_edges += k

    return {"mode": "topk", "n_genes": int(N), "n_edges": int(total_edges), "topk_per_gene": int(topk)}


# =============================================================================
# Single dataset processing
# =============================================================================
def process_single_dataset(expr_file_path: str, config: dict):
    start_time = datetime.datetime.now()
    dataset_name = extract_dataset_name(expr_file_path)
    folder_name = Path(expr_file_path).parent.name

    out_dir = Path(config["OUTPUT_ROOT"]) / folder_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Dataset_Name": dataset_name,
        "Folder_Name": folder_name,
        "Input_File": str(expr_file_path),
        "Output_Dir": str(out_dir),
        "Process_Status": "Success",
        "Process_Time_Seconds": 0.0,
        "Input_Genes_Count": 0,
        "Mappable_Genes_Count": 0,
        "Final_Genes_Count": 0,
        "Embedding_Dim": 0,
        "Total_Edges_Generated": 0,
        "Error_Message": "",
    }

    try:
        print(f"\n{'='*60}")
        print(f"[INFO] Processing: {dataset_name} (folder: {folder_name})")
        print(f"[INFO] Input: {expr_file_path}")

        # 1) load expression
        expr_df = read_expression_matrix(expr_file_path)
        symbols = expr_df.index.astype(str).tolist()
        record["Input_Genes_Count"] = len(symbols)

        # 2) load dicts
        token_to_id = load_token_dictionary(config["DICT_PATH"])
        gene_name_id_dict = load_gene_name_id_dict(config["DICT_PATH"])
        pad_id = get_pad_id(token_to_id)

        # 3) map genes -> token_id (Geneformer-aligned)
        gene_to_token_id, missing_map = match_genes_to_tokens_like_geneformer(
            symbols, token_to_id, gene_name_id_dict
        )
        record["Mappable_Genes_Count"] = len(gene_to_token_id)
        print(f"Mappable genes: {len(gene_to_token_id)}/{len(symbols)} (missing={missing_map})")
        if len(gene_to_token_id) == 0:
            raise RuntimeError("0 mappable genes after Geneformer-aligned mapping.")

        # reverse map token_id -> symbol (first wins)
        token_id_to_symbol = {}
        for sym, tid in gene_to_token_id.items():
            if tid not in token_id_to_symbol:
                token_id_to_symbol[tid] = sym

        # 4) build sequences
        seqs, used_cells, used_genes_for_seq = build_sequences(
            expr_df=expr_df,
            gene_to_token_id=gene_to_token_id,
            topk=config["SEQ_TOPK_GENES"],
            max_len=config["MAX_SEQ_LEN"],
            n_cells=config["N_CELLS"],
            use_log1p=config["USE_LOG1P"],
            seed=config["SEED"],
        )

        # pad sequences to MAX_SEQ_LEN
        L = int(config["MAX_SEQ_LEN"])
        input_ids = np.full((len(seqs), L), pad_id, dtype=np.int64)
        for i, ids in enumerate(seqs):
            n = min(len(ids), L)
            if n > 0:
                input_ids[i, :n] = np.array(ids[:n], dtype=np.int64)

        # 5) load LangCell model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertModel.from_pretrained(
            config["LANGCELL_MODEL_PATH"],
            ignore_mismatched_sizes=True,
            output_hidden_states=True
        ).to(device)
        model.eval()

        # 6) forward & aggregate hidden states per token id
        sum_vec, cnt = {}, {}
        bs = int(config["BATCH_SIZE"])
        layer_idx = int(config["HIDDEN_LAYER"])

        with torch.no_grad():
            for start in tqdm(range(0, input_ids.shape[0], bs), desc="Forward LangCell"):
                end = min(input_ids.shape[0], start + bs)
                ids = torch.from_numpy(input_ids[start:end]).to(device)
                attn = (ids != pad_id).long()

                out = model(input_ids=ids, attention_mask=attn)
                h = _get_hidden(out, layer_idx)  # [B, L, H]
                h = h.detach().cpu().numpy()
                ids_np = ids.detach().cpu().numpy()

                B, L2, H = h.shape
                for bi in range(B):
                    for li in range(L2):
                        tid = int(ids_np[bi, li])
                        if tid == pad_id:
                            continue
                        if tid not in sum_vec:
                            sum_vec[tid] = h[bi, li].astype(np.float64)
                            cnt[tid] = 1
                        else:
                            sum_vec[tid] += h[bi, li].astype(np.float64)
                            cnt[tid] += 1

        # 7) gene embedding in memory (no saving)
        gene_emb = {}
        for tid, s in sum_vec.items():
            sym = token_id_to_symbol.get(tid, None)
            if sym is None:
                continue
            gene_emb[sym] = (s / max(cnt.get(tid, 1), 1)).astype(np.float32)

        genes = sorted(gene_emb.keys())
        emb = np.stack([gene_emb[g] for g in genes], axis=0)
        dim = int(emb.shape[1])

        record["Final_Genes_Count"] = len(genes)
        record["Embedding_Dim"] = dim

        # safety check
        try:
            assert dim == int(model.config.hidden_size), \
                f"Embedding dim {dim} != hidden_size {model.config.hidden_size}"
        except Exception:
            print("[WARN] Embedding dim != model hidden_size (check model/config)")

        print(f"Final (in-memory) embeddings: {len(genes)} genes x {dim} dim")
        print(f"Used cells: {used_cells}")

        # 8) edges TSV output name: MODEL_NAME_DATASET.tsv
        model_name = config.get("MODEL_NAME") or Path(config["LANGCELL_MODEL_PATH"]).name
        prefix = f"{_safe(model_name)}_{_safe(dataset_name)}"
        edge_tsv = out_dir / f"{prefix}.tsv"

        if bool(config["SAVE_ALL_EDGES"]):
            cosine_info = compute_and_save_all_cosine_edges_stream(genes, emb, edge_tsv)
        else:
            cosine_info = compute_and_save_topk_cosine_edges(
                genes, emb, edge_tsv, topk=int(config["TOPK_PER_GENE"])
            )

        record["Total_Edges_Generated"] = int(cosine_info["n_edges"])
        print(f"[INFO] Saved edges TSV: {edge_tsv}")

        # 9) save run params
        run_params = {
            "dataset": dataset_name,
            "folder": folder_name,
            "input_file": str(expr_file_path),
            "output_dir": str(out_dir),
            "outputs": {
                "edges_tsv": str(edge_tsv),
            },
            "model": {
                "model_name": str(model_name),
                "langcell_model_path": config["LANGCELL_MODEL_PATH"],
                "hidden_layer": int(config["HIDDEN_LAYER"]),
                "hidden_size": int(getattr(model.config, "hidden_size", dim)),
                "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", -1)),
                "num_attention_heads": int(getattr(model.config, "num_attention_heads", -1)),
                "max_position_embeddings": int(getattr(model.config, "max_position_embeddings", -1)),
                "vocab_size": int(getattr(model.config, "vocab_size", -1)),
            },
            "dicts": {
                "dict_path": config["DICT_PATH"],
                "mapping_mode": "gene_name_id_dict.pkl + token_dictionary.pkl (Geneformer-aligned)",
            },
            "sequence": {
                "seq_topk_genes": int(config["SEQ_TOPK_GENES"]),
                "max_seq_len": int(config["MAX_SEQ_LEN"]),
                "use_log1p": bool(config["USE_LOG1P"]),
                "n_cells": config["N_CELLS"],
            },
            "forward": {
                "batch_size": int(config["BATCH_SIZE"]),
                "device": str(device),
                "seed": int(config["SEED"]),
                "pad_id": int(pad_id),
            },
            "stats": {
                "n_input_genes": int(len(symbols)),
                "n_mappable_genes": int(len(gene_to_token_id)),
                "n_genes_used_for_seq": int(used_genes_for_seq),
                "n_cells_used": int(used_cells),
                "n_final_genes_with_embedding": int(len(genes)),
                "embedding_dim": int(dim),
                "missing_mapping_count": int(missing_map),
            },
            "cosine_export": cosine_info,
            "processing_time_seconds": round((datetime.datetime.now() - start_time).total_seconds(), 2),
        }
        with open(out_dir / "run_params.json", "w") as f:
            json.dump(run_params, f, indent=2, ensure_ascii=False)

        record["Process_Time_Seconds"] = run_params["processing_time_seconds"]
        print(f"[INFO] Done | time: {record['Process_Time_Seconds']}s")

    except Exception as e:
        traceback.print_exc()
        record["Process_Status"] = "Failed"
        record["Error_Message"] = str(e)[:300]
        print(f"[ERROR] Failed: {dataset_name} | {record['Error_Message']}")

    return record


# =============================================================================
# Batch main
# =============================================================================
def main():
    set_seed(CONFIG["SEED"])
    input_root = Path(CONFIG["INPUT_ROOT"])
    output_root = Path(CONFIG["OUTPUT_ROOT"])
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] LangCell batch (Geneformer-aligned) start")
    print(f"[INFO] INPUT_ROOT : {input_root}")
    print(f"[INFO] OUTPUT_ROOT: {output_root}")

    # folders
    if CONFIG["TARGET_FOLDERS"] is None:
        folders = sorted([p.name for p in input_root.iterdir() if p.is_dir()])
    else:
        folders = CONFIG["TARGET_FOLDERS"]
    print(f"[INFO] Folders: {folders}")

    ok, fail = 0, 0

    for folder in folders:
        folder_path = input_root / folder
        if not folder_path.exists():
            print(f"\n[WARN] Missing folder, skip: {folder_path}")
            continue

        print(f"\n{'='*60}\n[INFO] Folder: {folder}")

        # keep your original filename patterns
        if folder == "CHIP":
            expr_files = list(folder_path.glob("*_chip_matched-ExpressionData.csv"))
        else:
            expr_files = list(folder_path.glob("*_processed-ExpressionData.csv"))

        print(f"[INFO] Found {len(expr_files)} expression files")

        for expr_file in expr_files:
            rec = process_single_dataset(str(expr_file), CONFIG)
            if rec["Process_Status"] == "Success":
                ok += 1
            else:
                fail += 1

    print(f"\n{'='*60}")
    print(f"[INFO] Done. success={ok}, fail={fail}")
    print(f"[INFO] Output at: {output_root}")


if __name__ == "__main__":
    main()
