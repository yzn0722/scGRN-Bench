



import os 
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime  
from transformers import BertModel

import argparse
from pathlib import Path

# -------------------------- Runtime-injected values (no hardcoded paths) --------------------------
ARGS = None
input_root = None
output_root = None
gene_emb_df = None

# -------------------------- Preload model and dictionaries --------------------------
def load_langcell_model_and_dict(parent_model_dir: str):
    """Preload the LangCell model and gene dictionaries."""
    # 1. Load the LangCell model
    saved_model_path = f"{parent_model_dir}/LangCell/cell_bert"
    model = BertModel.from_pretrained(saved_model_path, ignore_mismatched_sizes=True)
    token_emb = model.state_dict()['embeddings.word_embeddings.weight']
    
    # 2. Load Geneformer dictionaries (reused)
    dict_paths = f"{parent_model_dir}/Geneformer/dicts"  
    with open(os.path.join(dict_paths, "token_dictionary.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(dict_paths, "gene_name_id_dict.pkl"), "rb") as f:
        gene_name_id = pickle.load(f)
    gene_id_name = {v: k for k, v in gene_name_id.items()}
    
    # 3. LangCell(CLS token)
    token_emb_filtered = token_emb[:-1, :]  
    vocab_keys = list(vocab.keys())[:len(token_emb_filtered)]
    
    gene_emb_df = pd.DataFrame(token_emb_filtered.numpy())
    gene_emb_df["ENSG_ID"] = vocab_keys
    gene_emb_df["Symbol"] = gene_emb_df["ENSG_ID"].apply(lambda x: gene_id_name.get(x, None))
    gene_emb_df = gene_emb_df.dropna(subset=["Symbol"]).reset_index(drop=True)
    
    print(f"[INFO] LangCell loaded. Embedded genes: {len(gene_emb_df)}")
    return model, gene_emb_df

def parse_args():
    p = argparse.ArgumentParser(description="Export LangCell embedding-weight cosine edges as TSVs.")
    p.add_argument("--input-root", required=True, type=str, help="Input root with CHIP/Non_CHIP/STRING folders.")
    p.add_argument("--output-root", required=True, type=str, help="Output directory for TSV + run records.")
    p.add_argument("--parent-model-dir", required=True, type=str, help="Parent weights directory containing LangCell/ and Geneformer/dicts/.")
    p.add_argument("--folders", nargs="+", default=["CHIP"], help='Folders to process. Default: ["CHIP"].')
    return p.parse_args()

# --------------------------  --------------------------
def extract_dataset_name(file_path):
    """Extract the normalized dataset name"""
    file_basename = os.path.basename(file_path)
    if "_chip_matched-ExpressionData.csv" in file_basename:
        dataset_name = file_basename.split('_chip_matched-ExpressionData.csv')[0]
    elif "_processed-ExpressionData.csv" in file_basename:
        dataset_name = file_basename.split('_processed-ExpressionData.csv')[0]
    else:
        dataset_name = file_basename.split('-ExpressionData.csv')[0]
    return dataset_name

# -------------------------- Single-file processing function (export all directed edges) --------------------------
def process_single_pair(expr_path):
    # 1. Extract the normalized dataset name
    dataset_name = extract_dataset_name(expr_path)
    print(f"\n=====================================")
    print(f"[INFO] Processing dataset: {dataset_name}")
    print(f"   Expr:{expr_path}")

    # 2. Read the input gene set
    try:
        data_df = pd.read_csv(expr_path)
        gene_symbols = data_df.iloc[1:, 0].tolist()
        gene_set_df = pd.DataFrame({"Symbol": gene_symbols}).dropna()
        input_genes_count = len(gene_set_df)
        print(f"[INFO] Input genes: {input_genes_count}")
    except Exception as e:
        print(f"[WARN] Failed to read expression file: {e}; skip this dataset.")
        return

    # 3. Match LangCell embeddings
    gene_emb_filtered = pd.merge(gene_set_df, gene_emb_df, on="Symbol", how="inner")
    matched_genes_count = len(gene_emb_filtered)
    if matched_genes_count == 0:
        print("[WARN] No matched genes; skip.")
        return
    print(f"[INFO] Matched genes: {matched_genes_count}")

    # 4. Compute cosine similarity and export all directed edges
    gene_names = gene_emb_filtered["Symbol"].tolist()
    embedding_cols = [col for col in gene_emb_filtered.columns if col not in ["Symbol", "ENSG_ID"]]
    embedding_matrix = gene_emb_filtered[embedding_cols].values

    # Compute the symmetric similarity matrix
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Generate directed edges for every i!=j pair
    edge_weights = []
    n_genes = len(gene_names)
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:  # 
                continue
            edge_weights.append({
                'Gene1': gene_names[i],
                'Gene2': gene_names[j],
                'EdgeWeight': round(float(similarity_matrix[i, j]), 15)
            })
    
    total_edges = len(edge_weights)
    if total_edges == 0:
        print("[WARN] No valid gene pairs; skip saving.")
        return

    # 5. Save the TSV as <model>_<dataset>.tsv
    tsv_filename = f"LangCell_{dataset_name}.tsv"
    tsv_path = os.path.join(output_root, tsv_filename)
    
    # Sort by EdgeWeight in descending order while keeping all edges
    edges_df = pd.DataFrame(edge_weights)
    edges_df = edges_df.sort_values(by="EdgeWeight", ascending=False)
    edges_df.to_csv(tsv_path, sep='\t', index=False)
    
    # 
    avg_weight = np.mean([ew['EdgeWeight'] for ew in edge_weights])
    print("\n[INFO] Result summary")
    print(f"   Total directed edges: {total_edges} | Mean weight: {avg_weight:.4f}")
    print(f"[INFO] TSV saved: {tsv_path}")

    # 6. 
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": "LangCell",
        "Dataset": dataset_name,
        "Input gene count": input_genes_count,
        "Matched gene count": matched_genes_count,
        "Total directed edge count": total_edges,
        "TSV path": tsv_path
    }
    record_csv = os.path.join(output_root, "LangCellrun_records.csv")
    pd.DataFrame([record]).to_csv(record_csv, mode='a', header=not os.path.exists(record_csv), index=False)

# -------------------------- Main batch traversal function --------------------------
def main():
    global ARGS, input_root, output_root, gene_emb_df
    ARGS = parse_args()
    input_root = ARGS.input_root
    output_root = ARGS.output_root
    os.makedirs(output_root, exist_ok=True)

    _, gene_emb_df = load_langcell_model_and_dict(parent_model_dir=ARGS.parent_model_dir)

    print("\n[INFO] Start LangCell batch processing (directed non-self-loop edges).")
    print(f"Input root:{input_root}")
    print(f"Output root:{output_root}")
    
    target_folders = ARGS.folders
    for folder in target_folders:
        folder_path = os.path.join(input_root, folder)
        if not os.path.exists(folder_path):
            print(f"\n[WARN] Folder not found: {folder_path}; skip.")
            continue
        
        print(f"\n=====================================")
        print(f"[INFO] Scanning folder: {folder}")
        
        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
        if not expr_files:
            print("[WARN] No ExpressionData.csv files found; skip.")
            continue
        
        for expr_file in expr_files:
            expr_path = os.path.join(folder_path, expr_file)
            process_single_pair(expr_path)

if __name__ == "__main__":
    main()
    print("\n=====================================")
    print(f"[INFO] All files processed. Outputs saved in: {output_root}")