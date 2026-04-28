


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
import numpy as np
import pandas as pd
import tqdm
import yaml
from sklearn.metrics.pairwise import cosine_similarity

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Export scFoundation pos-embedding cosine edges as TSVs.")
    p.add_argument("--input-root", required=True, type=str, help="Input root with CHIP/Non_CHIP/STRING subfolders.")
    p.add_argument("--output-root", required=True, type=str, help="Output directory for TSV + run logs.")
    p.add_argument("--ckpt-path", required=True, type=str, help="scFoundation models.ckpt path.")
    p.add_argument("--vocab-path", required=True, type=str, help="OS_scRNA_gene_index.19264.tsv path.")
    p.add_argument("--folders", nargs="+", default=["CHIP"], help='Folders to process. Default: ["CHIP"].')
    return p.parse_args()


ARGS = None
INPUT_ROOT = None
OUTPUT_ROOT = None
CKPT_PATH = None
VOCAB_PATH = None
RESULTS_LOG_FILE = None

# Model
MODEL_NAME = "scFoundation"  # Model
EMBEDDING_SIZE = 768  # scFoundation pos_emb

# --------------------------  --------------------------
def extract_dataset_name(file_path):
    """Extract the normalized dataset name(,Model)"""
    file_name = Path(file_path).name
    # Dataset
    if "_chip_matched-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_chip_matched-ExpressionData.csv')[0]
    elif "_processed-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_processed-ExpressionData.csv')[0]
    else:
        dataset_name = file_name.split('-ExpressionData.csv')[0]
    return dataset_name

def save_run_statistics(stats_dict):
    """CSV(Model)"""
    stats_df = pd.DataFrame([stats_dict])
    if not RESULTS_LOG_FILE.exists():
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='w')
        print(f"\n[INFO] Stats log created: {RESULTS_LOG_FILE}")
    else:
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='a', header=False)
        print(f"\n[INFO] Stats appended: {RESULTS_LOG_FILE}")

# -------------------------- Gene embedding processing class --------------------------
class GeneEmbeddingProcessor:
    """scFoundation,Generated(Model)"""
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
        print(f"First 5 gene names: {self.gene_names[:5]}")
        if len(self.gene_embeddings) > 0:
            print(f"Embedding dimension: {self.gene_embeddings.shape[1]}")
    
    def compute_similarity_matrix(self):
        """Compute the cosine similarity matrix and remove self-loops."""
        if self.gene_embeddings is None or len(self.gene_embeddings) == 0:
            raise ValueError("")
        print(f"\nComputing cosine similarity between genes...")
        self.similarity_matrix = cosine_similarity(self.gene_embeddings)
        np.fill_diagonal(self.similarity_matrix, 0)  # 
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def generate_all_interactions(self):
        """Generated(i≠j,Model)"""
        if self.similarity_matrix is None:
            raise ValueError(" compute_similarity_matrix ")
        edge_weights = []
        n_genes = len(self.gene_names)
        for i in range(n_genes):
            gene1 = self.gene_names[i]
            for j in range(n_genes):
                if i != j:  # ,
                    gene2 = self.gene_names[j]
                    weight = self.similarity_matrix[i, j]
                    edge_weights.append({
                        'Gene1': gene1,
                        'Gene2': gene2,
                        'EdgeWeight': round(weight, 15)
                    })
        print(f"\nGenerated {len(edge_weights)} all non-self-loop directed gene pairs")
        return edge_weights

# --------------------------  --------------------------
def process_single_pair(expr_path):
    """(,)"""
    start_time = time.time()
    run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Extract the normalized dataset name()
    dataset_name = extract_dataset_name(expr_path)
    # :scFoundation_Dataset.tsv(Model)
    output_tsv_name = f"{MODEL_NAME}_{dataset_name}.tsv"
    output_tsv_path = OUTPUT_ROOT / output_tsv_name
    
    print(f"\n=====================================")
    print(f"[INFO] Processing dataset: {dataset_name}")
    print(f"   Expr:{expr_path}")
    print(f"   :{output_tsv_path}")
    print(f"=====================================")

    # (Model)
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
        'Embedding_Size': EMBEDDING_SIZE,
        'Run_Time_Seconds': 0.0,
        'Process_Status': "Success"
    }

    try:
        # -------------------------- Step1:load scFoundation weights and extract pos_emb --------------------------
        print("\n[INFO] Step 1: load scFoundation weights and extract pos_emb.weight")
        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"Modeldoes not exist:{CKPT_PATH}")
        
        # ckptpos_emb.weight
        ckpt_data = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        pos_emb_weight = ckpt_data["gene"]["state_dict"]["model.pos_emb.weight"]  # [19267, 768]
        pos_emb_np = pos_emb_weight.cpu().numpy()
        print(f"[INFO] pos_emb.weight loaded: shape={pos_emb_np.shape}")

        # -------------------------- Step2:load the vocabulary and build the gene-id mapping --------------------------
        print("\n[INFO] Step 2: load vocab and build gene-id mapping")
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(f"Vocabularydoes not exist:{VOCAB_PATH}")
        
        vocab_df = pd.read_csv(VOCAB_PATH, sep="\t")
        vocab_df["gene_name"] = vocab_df["gene_name"].str.upper().str.strip()
        gene2id = dict(zip(vocab_df["gene_name"], vocab_df["index"].astype(int)))
        print(f"[INFO] Vocab loaded. Total genes: {len(gene2id)}")

        # -------------------------- Step3:read the input file and extract gene names --------------------------
        print("\n[INFO] Step 3: read expression file and gene names")
        expr_df = pd.read_csv(expr_path, header=0, index_col=0)
        input_genes = expr_df.index.tolist()
        # 
        input_genes = [str(g).upper().strip() for g in input_genes if pd.notna(g) and g != ""]
        stats['Input_Genes_Count'] = len(input_genes)
        print(f"File contains {len(input_genes)} ")
        print(f"First 5 gene names: {input_genes[:5]}")

        # -------------------------- Step4:match embeddings for input genes --------------------------
        print("\n[INFO] Step 4: match input genes with scFoundation embeddings")
        target_gene_emb_dict = {}
        for gene in input_genes:
            if gene in gene2id:
                gene_id = gene2id[gene]
                # ID
                if gene_id < pos_emb_np.shape[0]:
                    target_gene_emb_dict[gene] = pos_emb_np[gene_id]
        
        # 
        stats['Matched_Genes_Count'] = len(target_gene_emb_dict)
        stats['Matched_Ratio(%)'] = round(len(target_gene_emb_dict) / len(input_genes) * 100, 2) if len(input_genes) > 0 else 0.0
        
        print(f"scFoundation:{len(target_gene_emb_dict)}  ({stats['Matched_Ratio(%)']}%)")
        print(f"First 5 matched genes: {list(target_gene_emb_dict.keys())[:5]}")
        
        if len(target_gene_emb_dict) == 0:
            print("[WARN] No matched genes; skip this dataset.")
            stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
            save_run_statistics(stats)
            return

        # -------------------------- Step5:Generated --------------------------
        print("\n[INFO] Step 5: compute cosine similarity and directed edges")
        processor = GeneEmbeddingProcessor(target_gene_emb_dict)
        processor.compute_similarity_matrix()
        
        # Generated
        all_interactions = processor.generate_all_interactions()
        stats['Total_Edges_Generated'] = len(all_interactions)
        
        # Sort by weight in descending order
        final_interactions = sorted(all_interactions, key=lambda x: x['EdgeWeight'], reverse=True)

        # -------------------------- Step6:save the final TSV file --------------------------
        print("\n[INFO] Step 6: save final TSV")
        interactions_df = pd.DataFrame(final_interactions)
        interactions_df.to_csv(output_tsv_path, sep='\t', index=False, encoding="utf-8")
        
        print(f"\n[INFO] Final TSV saved: {output_tsv_path}")
        print(f"File contains {len(interactions_df)} ")
        if len(interactions_df) > 0:
            print(f"Weight range:{interactions_df['EdgeWeight'].min():.6f} ~ {interactions_df['EdgeWeight'].max():.6f}")
            print(f":{interactions_df['EdgeWeight'].mean():.6f}")
    
    except Exception as e:
        print(f"[ERROR] Dataset processing failed: {e}")
        stats['Process_Status'] = f"Failed: {str(e)[:100]}"
        import traceback
        traceback.print_exc()
    
    # -------------------------- Step7:save run statistics --------------------------
    stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
    print(f"\n[INFO] Runtime summary | elapsed: {stats['Run_Time_Seconds']} seconds")
    print(f"   Input gene count:{stats['Input_Genes_Count']} | Matched gene count:{stats['Matched_Genes_Count']}")
    print(f"   Generated edge count:{stats['Total_Edges_Generated']}")
    
    save_run_statistics(stats)

# -------------------------- Main batch traversal function --------------------------
def main():
    global ARGS, INPUT_ROOT, OUTPUT_ROOT, CKPT_PATH, VOCAB_PATH, RESULTS_LOG_FILE
    ARGS = parse_args()
    INPUT_ROOT = Path(ARGS.input_root)
    OUTPUT_ROOT = Path(ARGS.output_root)
    CKPT_PATH = Path(ARGS.ckpt_path)
    VOCAB_PATH = Path(ARGS.vocab_path)

    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    RESULTS_LOG_FILE = OUTPUT_ROOT / "scFoundation_run_statistics.csv"

    print("[INFO] Start scFoundation batch processing (directed non-self-loop edges).")
    print(f"Input root:{INPUT_ROOT}")
    print(f"Output root:{OUTPUT_ROOT}")
    print(f"Model:{CKPT_PATH}")
    print(f"Vocabulary:{VOCAB_PATH}")
    
    # Ignore noisy warnings
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





