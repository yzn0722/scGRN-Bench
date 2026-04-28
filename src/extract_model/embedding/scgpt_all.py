import copy
import json
import os
from pathlib import Path
import sys
import warnings
import pickle as pkl
import time
from datetime import datetime

import torch
import scanpy as sc
import numpy as np
import pandas as pd
import tqdm
import yaml
from sklearn.metrics.pairwise import cosine_similarity

import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="Export scGPT embedding-weight cosine edges as TSVs (Gene1/Gene2/EdgeWeight)."
    )
    p.add_argument("--input-root", required=True, type=str, help="Input root with CHIP/Non_CHIP/STRING subfolders.")
    p.add_argument("--output-root", required=True, type=str, help="Output directory for TSV + run logs.")
    p.add_argument("--model-dir", required=True, type=str, help="scGPT model dir containing vocab.json and best_model.pt.")
    p.add_argument("--folders", nargs="+", default=["CHIP"], help='Folders to process. Default: ["CHIP"].')
    return p.parse_args()


ARGS = None

# -------------------------- (;) --------------------------
INPUT_ROOT = None
OUTPUT_ROOT = None
MODEL_DIR = None
MODEL_FILE = None
VOCAB_FILE = None
RESULTS_LOG_FILE = None

# Model
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
MODEL_NAME = "scGPT"  # Model

# -------------------------- :Vocabulary --------------------------
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# --------------------------  --------------------------
def save_run_statistics(stats_dict):
    """CSV"""
    stats_df = pd.DataFrame([stats_dict])
    if not RESULTS_LOG_FILE.exists():
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='w')
        print(f"\n[INFO] Stats log created: {RESULTS_LOG_FILE}")
    else:
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='a', header=False)
        print(f"\n[INFO] Stats appended: {RESULTS_LOG_FILE}")

def extract_dataset_name(file_path):
    """Extract the normalized dataset name()"""
    file_name = Path(file_path).name
    # Dataset
    if "_chip_matched-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_chip_matched-ExpressionData.csv')[0]
    elif "_processed-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_processed-ExpressionData.csv')[0]
    else:
        dataset_name = file_name.split('-ExpressionData.csv')[0]
    return dataset_name

# -------------------------- Gene embedding processing class --------------------------
class GeneEmbeddingProcessor:
    """Processing gene embeddings,Generated"""
    def __init__(self, gene_emb_dict):
        self.gene_emb_dict = gene_emb_dict
        self.gene_embeddings = None
        self.gene_names = []
        self.similarity_matrix = None
        self.process_embeddings()
    
    def process_embeddings(self):
        """Process embedding data"""
        print("\nProcessing gene embeddings...")
        self.gene_names = list(self.gene_emb_dict.keys())
        self.gene_embeddings = np.array(list(self.gene_emb_dict.values()))
        print(f"Processed successfully: {len(self.gene_names)} ")
        if len(self.gene_embeddings) > 0:
            print(f"Embedding dimension: {self.gene_embeddings.shape[1]}")
    
    def compute_similarity_matrix(self):
        """()"""
        if self.gene_embeddings is None or len(self.gene_embeddings) == 0:
            raise ValueError("")
        print(f"\nComputing cosine similarity between genes...")
        self.similarity_matrix = cosine_similarity(self.gene_embeddings)
        np.fill_diagonal(self.similarity_matrix, 0)  # 
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def generate_all_interactions(self):
        """Generated(i≠j,)"""
        if self.similarity_matrix is None:
            raise ValueError(" compute_similarity_matrix ")
        edge_weights = []
        n_genes = len(self.gene_names)
        for i in range(n_genes):
            gene1 = self.gene_names[i]
            for j in range(n_genes):
                if i != j:  # Remove self-loops only and keep all directed edges
                    gene2 = self.gene_names[j]
                    weight = self.similarity_matrix[i, j]
                    edge_weights.append({
                        'Gene1': gene1,
                        'Gene2': gene2,
                        'EdgeWeight': round(weight, 15)
                    })
        print(f"\nGenerated {len(edge_weights)} all non-self-loop directed gene pairs")
        return edge_weights

# -------------------------- scGPTVocabulary --------------------------
def preload_scgpt_embeddings(model_file: Path, vocab_file: Path):
    """scGPTVocabulary()"""
    # 1. Vocabulary
    print("[INFO] Loading scGPT vocabulary...")
    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    gene2idx = vocab.get_stoi()
    idx2gene = {v: k for k, v in gene2idx.items()}
    print(f"[INFO] Vocabulary loaded. Total tokens: {len(gene2idx)}")
    
    # 2. Model,
    print("[INFO] Loading scGPT embedding weights...")
    state_dict = torch.load(model_file, map_location='cpu')
    
    # :(encoder.embedding.weight)
    emb_weight_candidates = [
        "encoder.embedding.weight",          # 
        "encoder.embeddings.weight",         # scGPT
        "embeddings.word_embeddings.weight", # Transformer
        "word_embeddings.weight",            # 
        "embedding.weight"                   # 
    ]
    
    emb_weight_key = None
    for candidate in emb_weight_candidates:
        if candidate in state_dict:
            emb_weight_key = candidate
            break
    
    if emb_weight_key is None:
        # 20,
        top_keys = list(state_dict.keys())[:20]
        raise KeyError(f"!20:{top_keys}")
    
    print(f"[INFO] Found embedding key: {emb_weight_key}")
    token_emb = state_dict[emb_weight_key]
    print(f"[INFO] Embedding weights loaded. Shape: {token_emb.shape}")
    
    # 3. →(token)
    gene_emb_dict = {}
    special_token_ids = [gene2idx[s] for s in SPECIAL_TOKENS if s in gene2idx]
    print(f"[INFO] Special token IDs: {special_token_ids}")
    
    # 
    max_idx = min(len(token_emb), len(idx2gene))
    for idx in range(max_idx):
        if idx in idx2gene and idx not in special_token_ids:
            gene_name = idx2gene[idx]
            gene_emb = token_emb[idx].detach().cpu().numpy()
            gene_emb_dict[gene_name] = gene_emb
    
    print(f"[INFO] Gene embedding map built. Valid genes: {len(gene_emb_dict)}")
    print(f"5:{list(gene_emb_dict.keys())[:5]}")
    
    return gene2idx, gene_emb_dict

gene2idx = None
preloaded_emb_dict = None

# --------------------------  --------------------------
def process_single_pair(expr_path):
    """Process one input file and export all directed edges with consistent naming."""
    start_time = time.time()
    run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Extract the normalized dataset name()
    dataset_name = extract_dataset_name(expr_path)
    # :scGPT_Dataset.tsv()
    output_tsv_name = f"{MODEL_NAME}_{dataset_name}.tsv"
    output_tsv_path = OUTPUT_ROOT / output_tsv_name
    
    print(f"\n=====================================")
    print(f"[INFO] Processing dataset: {dataset_name}")
    print(f"   Expr:{expr_path}")
    print(f"   :{output_tsv_path}")
    print(f"=====================================")

    # 
    stats = {
        'Run_Datetime': run_datetime,
        'Dataset_Name': dataset_name,
        'Input_File': str(expr_path),
        'Output_File': str(output_tsv_path),
        'Input_Genes_Count': 0,
        'Matched_Genes_Count': 0,
        'Matched_Ratio(%)': 0.0,
        'Total_Edges_Generated': 0,
        'Model_Name': MODEL_NAME,
        'Embedding_Size': preloaded_emb_dict[list(preloaded_emb_dict.keys())[0]].shape[0] if preloaded_emb_dict else 0,
        'Run_Time_Seconds': 0.0,
        'Process_Status': "Success"
    }

    try:
        # -------------------------- Step1:read the input file and extract gene names --------------------------
        print("\n[INFO] Step 1: read input expression file")
        expr_df = pd.read_csv(expr_path, header=0, index_col=0)
        input_genes = expr_df.index.tolist()
        stats['Input_Genes_Count'] = len(input_genes)
        print(f"File contains {len(input_genes)} ")

        # -------------------------- Step2: --------------------------
        print("\n[INFO] Step 2: match scGPT gene embeddings")
        # (+)
        input_genes_clean = {str(gene).strip().upper(): gene for gene in input_genes}
        preloaded_genes_clean = {str(gene).strip().upper(): gene for gene in preloaded_emb_dict.keys()}
        
        # 
        matched_genes = []
        matched_emb = []
        for gene_clean, gene_original in input_genes_clean.items():
            if gene_clean in preloaded_genes_clean:
                matched_genes.append(gene_original)
                matched_emb.append(preloaded_emb_dict[preloaded_genes_clean[gene_clean]])
        
        stats['Matched_Genes_Count'] = len(matched_genes)
        stats['Matched_Ratio(%)'] = round(len(matched_genes) / len(input_genes) * 100, 2) if len(input_genes) > 0 else 0.0
        
        print(f"Genes matched to scGPT embeddings:{len(matched_genes)}  ({stats['Matched_Ratio(%)']}%)")
        print(f"First 5 matched genes: {matched_genes[:5]}")
        
        if len(matched_genes) == 0:
            print("[WARN] No matched genes; skip this dataset.")
            stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
            save_run_statistics(stats)
            return
        
        # 
        gene_emb_dict = {gene: emb for gene, emb in zip(matched_genes, matched_emb)}

        # -------------------------- Step3:Generated --------------------------
        print("\n[INFO] Step 3: generate directed non-self-loop edges")
        processor = GeneEmbeddingProcessor(gene_emb_dict)
        processor.compute_similarity_matrix()
        
        # Generated
        all_interactions = processor.generate_all_interactions()
        stats['Total_Edges_Generated'] = len(all_interactions)
        
        # Sort by weight in descending order
        final_interactions = sorted(all_interactions, key=lambda x: x['EdgeWeight'], reverse=True)

        # -------------------------- Step4:save the final TSV file --------------------------
        print("\n[INFO] Step 4: save final TSV")
        interactions_df = pd.DataFrame(final_interactions)
        interactions_df.to_csv(output_tsv_path, sep='\t', index=False)
        
        print(f"\n[INFO] Final TSV saved: {output_tsv_path}")
        print(f"File contains {len(interactions_df)} ")
        if len(interactions_df) > 0:
            print(f"Weight range:{interactions_df['EdgeWeight'].min():.6f} ~ {interactions_df['EdgeWeight'].max():.6f}")
    
    except Exception as e:
        print(f"[ERROR] Dataset processing failed: {e}")
        stats['Process_Status'] = f"Failed: {str(e)[:100]}"
        import traceback
        traceback.print_exc()
    
    # -------------------------- Step5:save run statistics --------------------------
    stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
    print(f"\n[INFO] Runtime summary | elapsed: {stats['Run_Time_Seconds']} seconds")
    print(f"   Input gene count:{stats['Input_Genes_Count']} | Matched gene count:{stats['Matched_Genes_Count']}")
    print(f"   Generated edge count:{stats['Total_Edges_Generated']}")
    
    save_run_statistics(stats)

# -------------------------- Main batch traversal function --------------------------
def main():
    global ARGS, INPUT_ROOT, OUTPUT_ROOT, MODEL_DIR, MODEL_FILE, VOCAB_FILE, RESULTS_LOG_FILE
    global gene2idx, preloaded_emb_dict

    ARGS = parse_args()
    INPUT_ROOT = Path(ARGS.input_root)
    OUTPUT_ROOT = Path(ARGS.output_root)
    MODEL_DIR = Path(ARGS.model_dir)
    MODEL_FILE = MODEL_DIR / "best_model.pt"
    VOCAB_FILE = MODEL_DIR / "vocab.json"

    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    RESULTS_LOG_FILE = OUTPUT_ROOT / "scGPT_run_statistics.csv"

    # preload once (moved from import-time to runtime)
    try:
        gene2idx, preloaded_emb_dict = preload_scgpt_embeddings(MODEL_FILE, VOCAB_FILE)
    except Exception as e:
        print(f"[ERROR] Preload failed: {e}")
        sys.exit(1)

    print("[INFO] Start scGPT batch processing (directed non-self-loop edges).")
    print(f"Input root:{INPUT_ROOT}")
    print(f"Output root:{OUTPUT_ROOT}")
    
    # 
    warnings.filterwarnings('ignore', category=ResourceWarning)
    
    # Target folder list
    target_folders = ARGS.folders
    
    for folder in target_folders:
        folder_path = INPUT_ROOT / folder
        if not folder_path.exists():
            print(f"\n[WARN] Folder not found: {folder_path}; skip.")
            continue
        
        print(f"\n=====================================")
        print(f"[INFO] Scanning folder: {folder}")
        print(f"=====================================")
        
        # Find all ExpressionData files
        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
        if not expr_files:
            print(f"[WARN] No ExpressionData.csv files in {folder}; skip.")
            continue
        
        # ExpressionData
        for expr_file in expr_files:
            expr_path = folder_path / expr_file
            #  ExpressionData 
            process_single_pair(expr_path)
    
    print(f"\n[INFO] All files processed. Outputs saved in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()