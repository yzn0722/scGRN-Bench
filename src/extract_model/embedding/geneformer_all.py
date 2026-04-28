




import os 
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForMaskedLM
from datetime import datetime  

import argparse
from pathlib import Path

# -------------------------- Runtime-injected values (no hardcoded paths) --------------------------
ARGS = None
input_root = None
output_root = None
gene_emb_df = None


def parse_args():
    p = argparse.ArgumentParser(description="Export Geneformer embedding-weight cosine edges as TSVs.")
    p.add_argument("--input-root", required=True, type=str, help="Input root with CHIP/Non_CHIP/STRING folders.")
    p.add_argument("--output-root", required=True, type=str, help="Output directory for TSV + run records.")
    p.add_argument("--model-dir", required=True, type=str, help="Geneformer model directory (e.g. .../Geneformer/default/12L).")
    p.add_argument("--dict-dir", required=True, type=str, help="Geneformer dict directory containing token_dictionary.pkl etc.")
    p.add_argument("--folders", nargs="+", default=["CHIP"], help='Folders to process. Default: ["CHIP"].')
    return p.parse_args()


def load_geneformer_embedding_df(model_dir: Path, dict_dir: Path) -> pd.DataFrame:
    model = BertForMaskedLM.from_pretrained(
        str(model_dir),
        output_attentions=False,
        output_hidden_states=True,
    )

    with open(dict_dir / "token_dictionary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(dict_dir / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_id = pickle.load(f)
    gene_id_name = {v: k for k, v in gene_name_id.items()}  # ENSG -> Symbol

    token_emb = model.state_dict()["bert.embeddings.word_embeddings.weight"]
    gene_emb_df_local = pd.DataFrame(token_emb.detach().cpu().numpy())
    gene_emb_df_local["ENSG_ID"] = list(vocab.keys())[: len(gene_emb_df_local)]
    gene_emb_df_local["Symbol"] = gene_emb_df_local["ENSG_ID"].apply(lambda x: gene_id_name.get(x, None))
    gene_emb_df_local = gene_emb_df_local.dropna(subset=["Symbol"]).reset_index(drop=True)
    return gene_emb_df_local

# --------------------------  --------------------------
def extract_dataset_name(file_path):
    """Extract the normalized dataset name by removing the CHIP/Non_CHIP/STRING suffix."""
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

    # 2. ExpressionData
    try:
        data_df = pd.read_csv(expr_path)
        gene_symbols = data_df.iloc[1:, 0].tolist()
        gene_set_df = pd.DataFrame({"Symbol": gene_symbols}).dropna()
        input_genes_count = len(gene_set_df)
        print(f"[INFO] Input genes: {input_genes_count}")
    except Exception as e:
        print(f"[WARN] Failed to read expression file: {e}; skip this dataset.")
        return

    # 3. Geneformer
    gene_emb_filtered = pd.merge(gene_set_df, gene_emb_df, on="Symbol", how="inner")
    matched_genes_count = len(gene_emb_filtered)
    if matched_genes_count == 0:
        print("[WARN] No matched genes; skip.")
        return
    print(f"[INFO] Matched genes: {matched_genes_count}")

    # 4. Compute cosine similarity and export all directed edges
    def save_similarity_tsv():
        # 
        embedding_cols = [col for col in gene_emb_filtered.columns if col not in ["Symbol", "ENSG_ID"]]
        embedding_matrix = gene_emb_filtered[embedding_cols].values
        emb_dim = len(embedding_cols)

        if embedding_matrix.size == 0:
            print("[WARN] Empty embedding matrix; skip saving.")
            return None
        
        # Compute the symmetric similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)
        gene_names = gene_emb_filtered["Symbol"].tolist()
        
        # :i≠j,Generated(A→B + B→A)
        edge_weights = []
        n_genes = len(gene_names)
        for i in range(n_genes):
            for j in range(n_genes):
                if i == j:  # Remove self-loops only and keep all directed edges
                    continue
                edge_weights.append({
                    'Gene1': gene_names[i],
                    'Gene2': gene_names[j],
                    'EdgeWeight': round(float(similarity_matrix[i, j]), 15)
                })
        
        total_edges = len(edge_weights)
        if total_edges == 0:
            print("[WARN] No valid gene pairs; skip saving.")
            return None

        # Save the TSV as <model>_<dataset>.tsv
        tsv_filename = f"Geneformer_{dataset_name}.tsv"
        tsv_path = os.path.join(output_root, tsv_filename)
        
        # Sort by EdgeWeight in descending order while keeping all edges
        edges_df = pd.DataFrame(edge_weights)
        edges_df = edges_df.sort_values(by="EdgeWeight", ascending=False)
        edges_df.to_csv(tsv_path, sep='\t', index=False)

        # 
        avg_weight = np.mean([ew['EdgeWeight'] for ew in edge_weights]) if total_edges > 0 else 0.0
        print("\n[INFO] Result summary")
        print(f"   Total directed edges: {total_edges} | Mean weight: {avg_weight:.4f}")
        print(f"[INFO] TSV saved: {tsv_path}")
        return tsv_path

    # 
    save_result = save_similarity_tsv()
    if save_result is None:
        return

    # 5. 
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": "Geneformer",
        "Dataset": dataset_name,
        "Input gene count": input_genes_count,
        "Matched gene count": matched_genes_count,
        "Total directed edge count": len(pd.read_csv(save_result)),
        "TSV path": save_result
    }
    record_csv = os.path.join(output_root, "Geneformerrun_records.csv")
    pd.DataFrame([record]).to_csv(record_csv, mode='a', header=not os.path.exists(record_csv), index=False)

# --------------------------  --------------------------
def main():
    global ARGS, input_root, output_root, gene_emb_df
    ARGS = parse_args()
    input_root = ARGS.input_root
    output_root = ARGS.output_root
    os.makedirs(output_root, exist_ok=True)

    gene_emb_df = load_geneformer_embedding_df(model_dir=Path(ARGS.model_dir), dict_dir=Path(ARGS.dict_dir))
    print("[INFO] Initialization complete: model, dicts, and embeddings loaded.")

    target_folders = ARGS.folders
    for folder in target_folders:
        folder_path = os.path.join(input_root, folder)
        if not os.path.exists(folder_path):
            print(f"[WARN] Folder not found: {folder_path}; skip.")
            continue

        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
        for expr_file in expr_files:
            expr_path = os.path.join(folder_path, expr_file)
            process_single_pair(expr_path)

if __name__ == "__main__":
    main()
    print("\n=====================================")
    print(f"[INFO] All files processed. Outputs saved in: {output_root}")