import os 
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime  

import argparse
from pathlib import Path

# -------------------------- Runtime-injected values (no hardcoded paths) --------------------------
ARGS = None
INPUT_ROOT = None
OUTPUT_ROOT = None
MODEL_NAME = "scCello"
PARENT_MODEL_DIR = None
gene_emb_df = None


def parse_args():
    p = argparse.ArgumentParser(description="Export scCello embedding-weight cosine edges as TSVs.")
    p.add_argument("--input-root", required=True, type=str, help="Input root with CHIP/Non_CHIP/STRING folders.")
    p.add_argument("--output-root", required=True, type=str, help="Output directory for TSV + run records.")
    p.add_argument("--parent-model-dir", required=True, type=str, help="Parent weights dir containing scCello/ and Geneformer/dicts/.")
    p.add_argument("--folders", nargs="+", default=["CHIP"], help='Folders to process. Default: ["CHIP"].')
    return p.parse_args()


def load_sccello_embedding_df(parent_model_dir: Path) -> pd.DataFrame:
    # local import so file can be imported without the repo on sys.path
    sys.path.insert(0, str(parent_model_dir.parent))  # best-effort; user may not need this if package is installed
    try:
        from sc_foundation_evals.sccello.src.model_prototype_contrastive import PrototypeContrastiveModel
    except Exception as e:
        raise RuntimeError(
            "Cannot import sc_foundation_evals.sccello... "
            "Provide a python package or add the correct repo path to PYTHONPATH.\n"
            f"import error: {e}"
        )

    saved_model_path = parent_model_dir / "scCello"
    model = PrototypeContrastiveModel.from_pretrained(str(saved_model_path), ignore_mismatched_sizes=True)
    model_state_dict = model.state_dict()
    if "embeddings.word_embeddings.weight" not in model_state_dict:
        raise KeyError("Missing embeddings.word_embeddings.weight in scCello state_dict")

    token_emb = model_state_dict["embeddings.word_embeddings.weight"]
    token_emb_filtered = token_emb[:-1, :]  # drop CLS

    dict_paths = parent_model_dir / "Geneformer" / "dicts"
    with open(dict_paths / "token_dictionary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(dict_paths / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_id = pickle.load(f)
    gene_id_name = {v: k for k, v in gene_name_id.items()}

    vocab_keys = list(vocab.keys())[: len(token_emb_filtered)]
    gene_emb_df_local = pd.DataFrame(token_emb_filtered.detach().cpu().numpy())
    gene_emb_df_local["ENSG_ID"] = vocab_keys
    gene_emb_df_local["Symbol"] = gene_emb_df_local["ENSG_ID"].apply(lambda x: gene_id_name.get(x, None))
    gene_emb_df_local = gene_emb_df_local.dropna(subset=["Symbol"]).reset_index(drop=True)
    return gene_emb_df_local

# -------------------------- Preload model and dictionaries (simplified version) --------------------------
def load_sccello_model_and_dict():
    """Preload the scCello model and gene dictionaries (simplified version)."""
    # 1. Load the scCello model
    saved_model_path = f"{PARENT_MODEL_DIR}/scCello"
    model = PrototypeContrastiveModel.from_pretrained(saved_model_path, ignore_mismatched_sizes=True)
    model_state_dict = model.state_dict()
    
    # ()
    if 'embeddings.word_embeddings.weight' not in model_state_dict:
        raise KeyError("Model 'embeddings.word_embeddings.weight'")
    token_emb = model_state_dict['embeddings.word_embeddings.weight']
    token_emb_filtered = token_emb[:-1, :]  # CLS token
    
    # 2. Load shared dictionaries
    dict_paths = f"{PARENT_MODEL_DIR}/Geneformer/dicts"  
    with open(os.path.join(dict_paths, "token_dictionary.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(dict_paths, "gene_name_id_dict.pkl"), "rb") as f:
        gene_name_id = pickle.load(f)
    gene_id_name = {v: k for k, v in gene_name_id.items()}
    
    # Prepare the embedding dataframe
    vocab_keys = list(vocab.keys())[:len(token_emb_filtered)]
    gene_emb_df = pd.DataFrame(token_emb_filtered.numpy())
    gene_emb_df["ENSG_ID"] = vocab_keys
    gene_emb_df["Symbol"] = gene_emb_df["ENSG_ID"].apply(lambda x: gene_id_name.get(x, None))
    gene_emb_df = gene_emb_df.dropna(subset=["Symbol"]).reset_index(drop=True)
    
    print(f"[INFO] scCello loaded. Embedded genes: {len(gene_emb_df)}")
    return model, gene_emb_df

# 
model = None
gene_emb_df = None

# -------------------------- Utility function for extracting the normalized dataset name --------------------------
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

# -------------------------- Single-file processing function (export all directed edges without filtering) --------------------------
def process_single_pair(expr_path):
    """Process one input file and export all directed edges without filtering."""
    start_time = datetime.now()
    # 1. Extract the normalized dataset name()
    dataset_name = extract_dataset_name(expr_path)
    print(f"\n=====================================")
    print(f"[INFO] Processing dataset: {dataset_name}")
    print(f"   Expr:{expr_path}")

    # Initialize the run record
    run_record = {
        "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model_Name": MODEL_NAME,
        "Dataset_Name": dataset_name,
        "Input_File": expr_path,
        "Input_Genes_Count": 0,
        "Matched_Genes_Count": 0,
        "Match_Rate(%)": 0.0,
        "Total_Edges_Generated": 0,
        "Output_TSV_Path": "",
        "Process_Status": "Success",
        "Process_Time_Seconds": 0.0
    }

    try:
        # 2. Read the input gene set
        data_df = pd.read_csv(expr_path)
        gene_symbols = data_df.iloc[1:, 0].tolist()
        gene_set_df = pd.DataFrame({"Symbol": gene_symbols}).dropna()
        input_genes_count = len(gene_set_df)
        run_record["Input_Genes_Count"] = input_genes_count
        print(f"[INFO] Input genes: {input_genes_count}")

        # 3. Match scCello embeddings
        gene_emb_filtered = pd.merge(gene_set_df, gene_emb_df, on="Symbol", how="inner")
        matched_genes_count = len(gene_emb_filtered)
        match_rate = round(matched_genes_count / input_genes_count * 100, 2) if input_genes_count > 0 else 0
        run_record["Matched_Genes_Count"] = matched_genes_count
        run_record["Match_Rate(%)"] = match_rate
        print(f"[INFO] Matched genes: {matched_genes_count} | match rate: {match_rate}%")

        if matched_genes_count == 0:
            raise ValueError("The input gene set does not overlap with scCello gene symbols.")

        # 4. Compute cosine similarity and export all directed edges
        gene_names = gene_emb_filtered["Symbol"].tolist()
        embedding_cols = [col for col in gene_emb_filtered.columns if col not in ["Symbol", "ENSG_ID"]]
        embedding_matrix = gene_emb_filtered[embedding_cols].values

        # Compute the symmetric similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)

        # Generate all directed edges for every i!=j pair while removing self-loops only.
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
        run_record["Total_Edges_Generated"] = total_edges
        print(f"[INFO] Generated directed edges: {total_edges}")

        if total_edges == 0:
            raise ValueError("No valid gene pairs are available for edge generation.")

        # 5. Save the TSV using the unified naming scheme scCello_<dataset>.tsv
        tsv_filename = f"{MODEL_NAME}_{dataset_name}.tsv"
        tsv_path = os.path.join(OUTPUT_ROOT, tsv_filename)
        
        # Sort by EdgeWeight in descending order while keeping all edges
        edges_df = pd.DataFrame(edge_weights)
        edges_df = edges_df.sort_values(by="EdgeWeight", ascending=False)
        edges_df.to_csv(tsv_path, sep='\t', index=False)
        run_record["Output_TSV_Path"] = tsv_path
        print(f"[INFO] TSV saved: {tsv_path}")

        # 6. Record elapsed time
        run_record["Process_Time_Seconds"] = round((datetime.now() - start_time).total_seconds(), 2)

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        run_record["Process_Status"] = f"Failed: {str(e)[:100]}"
        import traceback
        traceback.print_exc()

    # 7. Save the simplified run record
    record_filename = f"{MODEL_NAME}_run_records.csv"
    record_csv_path = os.path.join(OUTPUT_ROOT, record_filename)
    record_df = pd.DataFrame([run_record])
    
    if os.path.exists(record_csv_path):
        record_df.to_csv(record_csv_path, mode='a', header=False, index=False)
    else:
        record_df.to_csv(record_csv_path, mode='w', header=True, index=False)
    
    print(f"\n[INFO] Done | elapsed: {run_record['Process_Time_Seconds']}s")

# -------------------------- Main batch traversal function (simplified version) --------------------------
def main():
    global ARGS, INPUT_ROOT, OUTPUT_ROOT, PARENT_MODEL_DIR, gene_emb_df
    ARGS = parse_args()
    INPUT_ROOT = ARGS.input_root
    OUTPUT_ROOT = ARGS.output_root
    PARENT_MODEL_DIR = ARGS.parent_model_dir
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"[INFO] Initialization complete | output root: {OUTPUT_ROOT}")

    gene_emb_df = load_sccello_embedding_df(parent_model_dir=Path(PARENT_MODEL_DIR))

    print("\n[INFO] Start scCello batch processing (directed non-self-loop edges).")
    print(f"Input root:{INPUT_ROOT}")
    print(f"Output root:{OUTPUT_ROOT}")
    
    # Target folder list
    target_folders = ARGS.folders
    
    for folder in target_folders:
        folder_path = os.path.join(INPUT_ROOT, folder)
        if not os.path.exists(folder_path):
            print(f"\n[WARN] Folder not found: {folder_path}; skip.")
            continue
        
        print(f"\n=====================================")
        print(f"[INFO] Scanning folder: {folder}")
        
        # Find all ExpressionData files
        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
        if not expr_files:
            print("[WARN] No ExpressionData.csv files found; skip.")
            continue
        
        # Process each input file
        for expr_file in expr_files:
            expr_path = os.path.join(folder_path, expr_file)
            process_single_pair(expr_path)

if __name__ == "__main__":
    main()
    print("\n=====================================")
    print(f"[INFO] All files processed. Outputs saved in: {OUTPUT_ROOT}")