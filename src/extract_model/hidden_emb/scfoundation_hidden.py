#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scFoundation hidden -> cosine edges (/)

""( 910)Generated:
  -  19264 Model
  -  19264 embedding  embedding
  - 

"""

import os
import sys
import json
import random
import warnings
from pathlib import Path
import traceback
import datetime
import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ======================  ======================
CONFIG = {
    "INPUT_ROOT": "",
    "OUTPUT_ROOT": "",

    "MODEL_NAME": "scFoundation",
    "SCFOUNDATION_ROOT": "",
    "MODEL_PATH": "",
    "VOCAB_PATH": "",

    "TGT_HIGHRES": "t4",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 8,

    "SAVE_ALL_EDGES": True,      # True: all i!=j edges (stream)
    "TOPK_PER_GENE": 1000,       # ()

    "SEED": 42,
}


def parse_args():
    p = argparse.ArgumentParser(description="Extract scFoundation hidden embeddings and export cosine-edge TSVs.")
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--scfoundation-root", required=True, type=str)
    p.add_argument("--model-path", required=True, type=str)
    p.add_argument("--vocab-path", required=True, type=str)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return p.parse_args()


ARGS = parse_args()
CONFIG["INPUT_ROOT"] = ARGS.input_root
CONFIG["OUTPUT_ROOT"] = ARGS.output_root
CONFIG["SCFOUNDATION_ROOT"] = ARGS.scfoundation_root
CONFIG["MODEL_PATH"] = ARGS.model_path
CONFIG["VOCAB_PATH"] = ARGS.vocab_path
CONFIG["BATCH_SIZE"] = int(ARGS.batch_size)
CONFIG["SEED"] = int(ARGS.seed)
CONFIG["DEVICE"] = ARGS.device

INPUT_ROOT = Path(CONFIG["INPUT_ROOT"])
OUTPUT_ROOT = Path(CONFIG["OUTPUT_ROOT"])
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ======================  import: load.py  ======================
SCFOUNDATION_ROOT = Path(CONFIG["SCFOUNDATION_ROOT"])
CANDIDATE_PATHS = [
    SCFOUNDATION_ROOT / "model",                 # /.../scFoundation-main/model/load.py
    SCFOUNDATION_ROOT / "model" / "pretrainmodels",
]

found = False
for p in CANDIDATE_PATHS:
    if (p / "load.py").exists():
        sys.path.insert(0, str(p))
        found = True
        break

if not found:
    raise FileNotFoundError(
        " load.py, scFoundation .:\n" +
        "\n".join([str(x) for x in CANDIDATE_PATHS])
    )

from load import load_model_frommmf, getEncoerDecoderData


# ====================== 1. () ======================
def main_gene_selection(X_df: pd.DataFrame, gene_list: list):
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(
        np.zeros((X_df.shape[0], len(to_fill_columns)), dtype=np.float32),
        columns=to_fill_columns,
        index=X_df.index
    )
    X_df = pd.concat([X_df, padding_df], axis=1)
    X_df = X_df[gene_list]
    return X_df, to_fill_columns


def load_official_gene_list():
    gene_list_df = pd.read_csv(CONFIG["VOCAB_PATH"], header=0, delimiter="\t")
    return list(gene_list_df["gene_name"])


# ====================== 2. Model ======================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_official_model():
    set_seed(CONFIG["SEED"])
    ckpt_path = CONFIG["MODEL_PATH"]
    key = "gene"

    pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
    pretrainmodel.to(CONFIG["DEVICE"])
    pretrainmodel.eval()
    pretrainmodel.to_final = None

    print(f"[INFO] Model loaded | device: {CONFIG['DEVICE']}")
    return pretrainmodel, pretrainconfig


# ====================== 3.  + dataset name ======================
def extract_dataset_name(file_path: str) -> str:
    base = os.path.basename(file_path)
    if "_chip_matched-ExpressionData.csv" in base:
        return base.split("_chip_matched-ExpressionData.csv")[0]
    if "_processed-ExpressionData.csv" in base:
        return base.split("_processed-ExpressionData.csv")[0]
    return base.split("-ExpressionData.csv")[0]


def read_expression_matrix_gene_by_cell(expr_path: str) -> pd.DataFrame:
    """
     ExpressionData.csv(gene x cell), cell x gene
     gene  upper+strip()
    """
    df = None
    try:
        df = pd.read_csv(expr_path, sep="\t", header=0, index_col=0)
        if df.shape[1] == 0:
            df = None
    except Exception:
        df = None

    if df is None:
        df = pd.read_csv(expr_path, sep=None, engine="python", header=0, index_col=0)

    df.index = df.index.astype(str).str.strip()
    df = df[~df.index.duplicated(keep="first")]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    expr_df = df.T
    expr_df.columns = [str(c).upper().strip() for c in expr_df.columns]
    return expr_df


# ====================== 4.  embedding mean(:) ======================
def extract_gene_embedding_mean(expr_path: str, model, pretrainconfig):
    print(f"\n[INFO] Processing file: {expr_path}")
    official_gene_list = load_official_gene_list()

    expr_df = read_expression_matrix_gene_by_cell(expr_path)
    original_genes_upper = [str(c).upper().strip() for c in expr_df.columns]

    expr_df, to_fill_cols = main_gene_selection(expr_df, official_gene_list)
    n_cells_total = expr_df.shape[0]
    print(f"📏 : cells={expr_df.shape[0]} | genes={expr_df.shape[1]} (filled={len(to_fill_cols)})")
    assert expr_df.shape[1] == 19264, "19264!"

    batch_size = int(CONFIG["BATCH_SIZE"])
    device = CONFIG["DEVICE"]
    tgt_res = float(CONFIG["TGT_HIGHRES"][1:])  # "t4" -> 4.0

    sum_emb = None
    count = 0
    batchcontainer = []

    with torch.no_grad():
        for i in tqdm(range(n_cells_total), desc="Forward scFoundation"):
            cell_expr = expr_df.iloc[i, :].to_numpy(dtype=np.float32).tolist()
            totalcount = float(np.sum(cell_expr)) + 1e-6
            input_19266 = cell_expr + [tgt_res, np.log10(totalcount)]

            batchcontainer.append(torch.tensor(input_19266, dtype=torch.float32, device=device).unsqueeze(0))

            flush = (len(batchcontainer) == batch_size) or (i == n_cells_total - 1)
            if not flush:
                continue

            batch_tensor = torch.cat(batchcontainer, dim=0)  # [B, 19266]
            batchcontainer = []

            encoder_data, encoder_pos_ids, encoder_pad, encoder_labels, \
            decoder_data, decoder_pad, new_data_raw, mask_labels, decoder_pos_ids = getEncoerDecoderData(
                batch_tensor, batch_tensor, pretrainconfig
            )

            out = model.forward(
                x=encoder_data,
                padding_label=encoder_pad,
                encoder_position_gene_ids=encoder_pos_ids,
                encoder_labels=encoder_labels,
                decoder_data=decoder_data,
                mask_gene_name=False,
                mask_labels=None,
                decoder_position_gene_ids=decoder_pos_ids,
                decoder_data_padding_labels=decoder_pad,
            )

            gene_emb_batch = out[:, :19264, :].contiguous().detach().cpu().numpy().astype(np.float32)  # [B, 19264, D]

            if sum_emb is None:
                sum_emb = np.sum(gene_emb_batch, axis=0)
            else:
                sum_emb += np.sum(gene_emb_batch, axis=0)

            count += gene_emb_batch.shape[0]

    gene_embeddings_mean = sum_emb / max(count, 1)
    print(f"[INFO] Embedding extraction complete | cells={count} | shape={gene_embeddings_mean.shape}")

    return gene_embeddings_mean, official_gene_list, original_genes_upper, int(n_cells_total)


# ====================== 5. : ======================
def compute_and_save_all_cosine_edges_stream(genes: list, emb: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    emb = np.asarray(emb, dtype=np.float32)
    N = emb.shape[0]
    denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    E = emb / denom

    total_edges = 0
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
            total_edges += (N - 1)

    return {"mode": "all_stream", "n_genes": int(N), "n_edges": int(total_edges)}


# ====================== 6. Dataset() ======================
def process_single_dataset(expr_path: str, folder_name: str, model, pretrainconfig):
    start_time = datetime.datetime.now()
    dataset_name = extract_dataset_name(expr_path)

    out_dir = OUTPUT_ROOT / folder_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model_Name": CONFIG["MODEL_NAME"],
        "Folder_Name": folder_name,
        "Dataset_Name": dataset_name,
        "Input_File": str(expr_path),
        "Process_Status": "Success",
        "Error_Message": "",
        "Process_Time_Seconds": 0.0,
        "Input_Genes_Count": 0,
        "Matched_Genes_Count": 0,
        "Unmatched_Genes_Count": 0,
        "Final_Genes_Count": 0,
        "Embedding_Dim": 0,
        "Total_Edges_Generated": 0,
        "Output_TSV_Path": "",
    }

    try:
        gene_emb_19264, official_gene_list, original_genes_upper, n_cells_total = extract_gene_embedding_mean(
            expr_path, model, pretrainconfig
        )
        record["Input_Genes_Count"] = int(len(original_genes_upper))

        official_upper = [str(g).upper().strip() for g in official_gene_list]
        official_index = {g: i for i, g in enumerate(official_upper)}  # O(1)

        matched_genes = []
        matched_idx = []
        unmatched = []

        for g in original_genes_upper:
            if g in official_index:
                matched_genes.append(g)
                matched_idx.append(official_index[g])
            else:
                unmatched.append(g)

        record["Matched_Genes_Count"] = int(len(matched_genes))
        record["Unmatched_Genes_Count"] = int(len(unmatched))

        if len(matched_genes) < 2:
            raise RuntimeError(f"Matched genes < 2 (matched={len(matched_genes)}),Generated.")

        emb_filtered = gene_emb_19264[np.array(matched_idx, dtype=int)]
        record["Final_Genes_Count"] = int(emb_filtered.shape[0])
        record["Embedding_Dim"] = int(emb_filtered.shape[1])

        edge_tsv = out_dir / f"{CONFIG['MODEL_NAME']}_{dataset_name}.tsv"
        cosine_info = compute_and_save_all_cosine_edges_stream(matched_genes, emb_filtered, edge_tsv)

        record["Total_Edges_Generated"] = int(cosine_info["n_edges"])
        record["Output_TSV_Path"] = str(edge_tsv)

        run_params = {
            "dataset": dataset_name,
            "folder": folder_name,
            "input_file": str(expr_path),
            "output_dir": str(out_dir),
            "outputs": {"edges_tsv": str(edge_tsv)},
            "model": {
                "model_name": CONFIG["MODEL_NAME"],
                "model_path": CONFIG["MODEL_PATH"],
                "vocab_path": CONFIG["VOCAB_PATH"],
                "device": CONFIG["DEVICE"],
                "tgt_highres": CONFIG["TGT_HIGHRES"],
            },
            "forward": {
                "batch_size": int(CONFIG["BATCH_SIZE"]),
            },
            "gene_filtering": {
                "original_gene_count": int(len(original_genes_upper)),
                "matched_gene_count": int(len(matched_genes)),
                "unmatched_gene_count": int(len(unmatched)),
                "unmatched_gene_examples": unmatched[:20],
                "official_gene_count": int(len(official_gene_list)),
            },
            "stats": {
                "n_cells_total": int(n_cells_total),
                "n_cells_used_for_mean": int(n_cells_total),
                "embedding_dim": int(emb_filtered.shape[1]),
            },
            "cosine_export": cosine_info,
            "processing_time_seconds": round((datetime.datetime.now() - start_time).total_seconds(), 2),
        }

        with open(out_dir / "run_params.json", "w") as f:
            json.dump(run_params, f, indent=2, ensure_ascii=False)

        record["Process_Time_Seconds"] = run_params["processing_time_seconds"]
        print(f"[INFO] Completed: {dataset_name} | edges={edge_tsv} | time={record['Process_Time_Seconds']}s")

    except Exception as e:
        traceback.print_exc()
        record["Process_Status"] = "Failed"
        record["Error_Message"] = str(e)[:300]
        print(f"[ERROR] Failed: {dataset_name} | {record['Error_Message']}")

    return record


# ====================== 7. () ======================
def process_all_datasets():
    print("[INFO] Start scFoundation batch processing (unified input/output).")
    print(f"[INFO] INPUT_ROOT : {INPUT_ROOT}")
    print(f"[INFO] OUTPUT_ROOT: {OUTPUT_ROOT}\n")

    model, pretrainconfig = load_official_model()

    target_folders = ["CHIP", "Non_CHIP", "STRING"]
    all_records = []

    for folder in target_folders:
        folder_path = INPUT_ROOT / folder
        if not folder_path.exists():
            print(f"[WARN] Folder not found, skip: {folder_path}")
            continue

        if folder == "CHIP":
            expr_files = list(folder_path.glob("*_chip_matched-ExpressionData.csv"))
        else:
            expr_files = list(folder_path.glob("*_processed-ExpressionData.csv"))

        print(f"\n{'='*60}")
        print(f"[INFO] Folder: {folder} | found {len(expr_files)} files")

        for expr_file in expr_files:
            rec = process_single_dataset(str(expr_file), folder, model, pretrainconfig)
            all_records.append(rec)

    if all_records:
        summary_df = pd.DataFrame(all_records)
        summary_path = OUTPUT_ROOT / f"{CONFIG['MODEL_NAME']}_processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        ok = sum(r["Process_Status"] == "Success" for r in all_records)
        fail = len(all_records) - ok
        print(f"\n[INFO] Summary file: {summary_path}")
        print(f"[INFO] Statistics: success {ok} / fail {fail}")

    print(f"\n{'='*60}")
    print(f"[INFO] All done. Outputs saved in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_all_datasets()
