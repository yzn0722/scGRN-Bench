


"""
Geneformer attention extraction - batch mode (supports CHIP/Non_CHIP/STRING)
Unified output format with batch-level parameter logging.
"""

import scanpy as sc
import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import BertForMaskedLM
from torch.utils.data import DataLoader
from datasets import load_from_disk
import scipy
import loompy
import datetime
import sys
from tqdm import tqdm

# ==================== Command-line arguments ====================
if len(sys.argv) != 3:
    print("Usage: python geneformer_attention_batch.py <data_type> <dataset_name>")
    print("Example: python geneformer_attention_batch.py CHIP hESC")
    print("      python geneformer_attention_batch.py Non_CHIP hHep")
    print("      python geneformer_attention_batch.py STRING mHSC-E")
    sys.exit(1)

data_type = sys.argv[1]  # Data type: CHIP / Non_CHIP / STRING
dataset = sys.argv[2]    # Dataset name, such as hESC or hHep
MODEL_VERSION = "6L"     # Fixed model version
MODEL_NAME = f"geneformer_{MODEL_VERSION}"



# Validate data type
valid_data_types = ["CHIP", "Non_CHIP", "STRING"]
if data_type not in valid_data_types:
    raise ValueError(f"Unsupported data type: {data_type}(supported values{valid_data_types})")

# ==================== Path configuration ====================
# Base paths
GENEFORMER_BASE = "/mnt/md0/yzn/scFM-Bench-main/data/weights/Geneformer"
DICT_DIR = f"{GENEFORMER_BASE}/dicts"
MODEL_DIR = f"{GENEFORMER_BASE}/default/{MODEL_VERSION}"
INPUT_ROOT = "/mnt/md0/yzn/Beeline-master/benchmark_SF/input_process1000"
OUTPUT_ROOT = Path("/mnt/md0/yzn/Beeline-master/benchmark_SF/model/output_att1000/geneformer")

# Resolve input subdirectory and filename suffix from data type
if data_type == "CHIP":
    file_suffix = "_chip_matched"
    input_subdir = "CHIP"
elif data_type == "Non_CHIP" or data_type == "STRING":
    file_suffix = "_processed"
    input_subdir = data_type

# Input file paths
csv_path = Path(f"{INPUT_ROOT}/{input_subdir}/{dataset}{file_suffix}-ExpressionData.csv")
network_path = f"{INPUT_ROOT}/{input_subdir}/{dataset}{file_suffix}-network.csv"

# Output directory grouped by data type
TYPE_OUTPUT_DIR = OUTPUT_ROOT / data_type
TYPE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Temporary directories scoped by data type and dataset
TEMP_PREPROCESSED_DIR = OUTPUT_ROOT / "temp_preprocessed" / f"{data_type}_{dataset}"
TEMP_TOKENIZED_DIR = OUTPUT_ROOT / "temp_tokenized" / f"{data_type}_{dataset}"
TEMP_PREPROCESSED_DIR.mkdir(exist_ok=True, parents=True)
TEMP_TOKENIZED_DIR.mkdir(exist_ok=True, parents=True)

# Output naming convention: model-version-type-dataset-file
file_prefix = f"geneformer-{MODEL_VERSION}-{data_type}-{dataset}-att"
# Only keep the filtered TSV path
filtered_tsv_path = TYPE_OUTPUT_DIR / f"{file_prefix}_filtered.tsv"

# Shared parameter summary CSV
ALL_PARAMS_CSV = OUTPUT_ROOT / "geneformer-all-params.csv"

print("="*80)
print(f"[INFO] Task config: {MODEL_NAME} | {data_type} | {dataset}")
print("="*80)
print(f"Model path: {MODEL_DIR}")
print(f"Input expression file: {csv_path}")
print(f"Input network file: {network_path}")
print(f"Output directory: {TYPE_OUTPUT_DIR}")
print(f"Parameter summary file: {ALL_PARAMS_CSV}")
print("="*80)

# ========== Load data ==========
print("\n" + "="*60)
print("[STEP 1] Load raw data")
print("="*60)

if not csv_path.exists():
    raise FileNotFoundError(f"Expression file not found: {csv_path}")

adata = sc.read_csv(str(csv_path))
adata = adata.T
adata_original_shape = adata.shape  # Keep the original shape for logging
print(f"Raw data shape (cells x genes): {adata.shape}")
print(f"Raw cell count: {adata.n_obs}")
print(f"Raw gene count: {adata.n_vars}")

# ========== Load Geneformer dictionaries ==========
print("\n" + "="*60)
print("[STEP 2] Load Geneformer dictionary")
print("="*60)

with open(f"{DICT_DIR}/gene_name_id_dict.pkl", "rb") as f:
    gene_name_dict = pickle.load(f)

with open(f"{DICT_DIR}/token_dictionary.pkl", "rb") as f:
    token_dict = pickle.load(f)

print(f"Geneformer dictionary contains {len(gene_name_dict)} genes")

# ========== Match genes and keep matched entries only ==========
print("\n" + "="*60)
print("[STEP 3] Match genes (keep matched only)")
print("="*60)

gene_symbol_col = None
for col in ['gene_name', 'ensembl_id', 'gene_symbols', 'symbol']:
    if col in adata.var.columns:
        gene_symbol_col = col
        break

if gene_symbol_col:
    gene_symbols = adata.var[gene_symbol_col].tolist()
else:
    gene_symbols = adata.var_names.tolist()

matched_mask = [gene in gene_name_dict for gene in gene_symbols]
matched_count = sum(matched_mask)
unmatched_count = len(matched_mask) - matched_count

print(f"\nMatch summary:")
print(f"  Total genes: {len(gene_symbols)}")
print(f"  Matched genes: {matched_count} ({100*matched_count/len(gene_symbols):.1f}%)")
print(f"  Unmatched genes: {unmatched_count} ({100*unmatched_count/len(gene_symbols):.1f}%)")

if unmatched_count > 0:
    unmatched_genes = [g for g, m in zip(gene_symbols, matched_mask) if not m]
    print(f"\n[WARN] Dropped {unmatched_count} unmatched genes (first 10): {unmatched_genes[:10]}")

if matched_count == 0:
    print("\n[ERROR] No genes matched Geneformer dictionary.")
    raise ValueError("No genes matched the Geneformer dictionary")

adata = adata[:, matched_mask].copy()
matched_gene_symbols = [g for g, m in zip(gene_symbols, matched_mask) if m]
ensembl_ids = [gene_name_dict[gene] for gene in matched_gene_symbols]

adata.var['gene_name'] = matched_gene_symbols
adata.var['ensembl_id'] = ensembl_ids
adata.var_names = matched_gene_symbols

print(f"\n[INFO] Kept {adata.n_vars} matched genes for analysis")
print(f"First 10 genes: {matched_gene_symbols[:10]}")

# ========== QC and normalization ==========
print("\n" + "="*60)
print("[STEP 4] QC and normalization")
print("="*60)

adata_pre_qc_n_obs = adata.n_obs
adata_pre_qc_n_vars = adata.n_vars
print(f"Before QC: {adata.shape}")

sc.pp.filter_cells(adata, min_genes=200)
print(f"After QC: {adata.shape}")

print("\nNormalizing...")
adata.obs['n_counts'] = (adata.X.sum(axis=1).A1 
                         if scipy.sparse.issparse(adata.X) 
                         else adata.X.sum(axis=1))
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

if 'cell_type' not in adata.obs.columns:
    adata.obs['cell_type'] = 'unknown'

print(f"[INFO] Preprocessing complete")

# ========== Save as loom and tokenize ==========
print("\n" + "="*60)
print("[STEP 5] Tokenization")
print("="*60)

loom_path = TEMP_PREPROCESSED_DIR / f"{data_type}_{dataset}.loom"
adata.write_loom(str(loom_path), write_obsm_varm=False)
print(f"[INFO] Loom file saved: {loom_path}")

from geneformer import TranscriptomeTokenizer
tokenizer = TranscriptomeTokenizer(
    custom_attr_name_dict={"cell_type": "cell_type"},
    nproc=4,
    gene_median_file=f"{DICT_DIR}/gene_median_dictionary.pkl",
    token_dictionary_file=f"{DICT_DIR}/token_dictionary.pkl"
)

tokenizer.tokenize_data(
    data_directory=str(TEMP_PREPROCESSED_DIR),
    output_directory=str(TEMP_TOKENIZED_DIR),
    output_prefix=f"{data_type}_{dataset}",
    file_format="loom",
    use_generator=False
)

print("[INFO] Tokenization complete")

# ========== Core helper functions ==========
def rank_normalize_fixed(attn_scores):
    """Fixed rank normalization while preserving the original logic."""
    batch_size, num_heads, M, _ = attn_scores.shape
    attn_normed = torch.zeros_like(attn_scores, dtype=torch.float32)
    
    # Row-wise rank normalization (descending)
    for b in range(batch_size):
        for h in range(num_heads):
            row_scores = attn_scores[b, h]
            sorted_indices = torch.argsort(row_scores, dim=1, descending=True)
            rank = torch.argsort(sorted_indices, dim=1)
            row_normed = rank.float() / (M - 1) if M > 1 else rank.float()
            attn_normed[b, h] = row_normed
    
    # Column-wise rank normalization (descending)
    for b in range(batch_size):
        for h in range(num_heads):
            col_scores = attn_normed[b, h].T
            sorted_indices = torch.argsort(col_scores, dim=1, descending=True)
            rank = torch.argsort(sorted_indices, dim=1)
            col_normed = rank.float() / (M - 1) if M > 1 else rank.float()
            attn_normed[b, h] = col_normed.T
    
    return attn_normed

def reverse_permute_fixed(tensor, indices):
    """Restore gene order while preserving the original logic."""
    batch_size = tensor.size(0)
    device = tensor.device
    
    if len(tensor.shape) == 3:  # [batch, M, M]
        M = tensor.size(1)
        inverse_indices = torch.zeros(batch_size, M, dtype=torch.long, device=device)
        for i in range(batch_size):
            valid_len = min(len(indices[i]), M)
            if valid_len > 0:
                inverse_indices[i, indices[i][:valid_len]] = torch.arange(valid_len, device=device)
        
        result = torch.zeros_like(tensor)
        for i in range(batch_size):
            temp = tensor[i][inverse_indices[i]]
            result[i] = temp[:, inverse_indices[i]]
        return result
    
    elif len(tensor.shape) == 2:  # [batch, M]
        result = torch.zeros_like(tensor)
        for i in range(batch_size):
            result[i, indices[i]] = tensor[i]
        return result

# ========== Extract Geneformer attention ==========
print("\n" + "="*60)
print("[STEP 6] Extract attention")
print("="*60)

# Load model
print("\nLoading model...")
model = BertForMaskedLM.from_pretrained(
    MODEL_DIR,
    output_attentions=True,
    output_hidden_states=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
print(f"[INFO] Model loaded on {device}")
print(f"  Total layers: {num_layers}")
print(f"  Attention heads: {num_heads}")

# Load tokenized data
print("\nLoading tokenized data...")
tokenized_dataset_path = TEMP_TOKENIZED_DIR / f"{data_type}_{dataset}.dataset"
tokenized_dataset = load_from_disk(str(tokenized_dataset_path))
tokenized_size = len(tokenized_dataset)
print(f"[INFO] Dataset size: {tokenized_size} cells")

# Build gene-name mapping
print("\nBuilding gene-name mapping...")
id_to_ensembl = {v: k for k, v in token_dict.items()}
ensembl_to_gene = {v: k for k, v in gene_name_dict.items()}

first_cell = tokenized_dataset[0]
gene_ids_from_tokens = np.array(first_cell['input_ids'])
gene_names_from_tokens = [
    ensembl_to_gene.get(id_to_ensembl.get(gid, ""), "") 
    for gid in gene_ids_from_tokens
]

valid_mask = [name != "" for name in gene_names_from_tokens]
valid_indices = [i for i, v in enumerate(valid_mask) if v]
final_gene_names = [gene_names_from_tokens[i] for i in valid_indices]
n_final_genes = len(final_gene_names)

print(f"[INFO] Valid genes: {n_final_genes}")
print(f"First 10 genes: {final_gene_names[:10]}")

# Prepare dataloader
batch_size = 8
max_seq_len = max(len(item['input_ids']) for item in tokenized_dataset)

def collate_fn(batch):
    input_ids = []
    attention_masks = []
    sorted_indices = []
    
    for item in batch:
        ids = item['input_ids']
        padding_length = max_seq_len - len(ids)
        
        padded_ids = ids + [0] * padding_length
        attention_mask = [1] * len(ids) + [0] * padding_length
        
        if 'sorted_indices' in item:
            sorted_idx = item['sorted_indices']
        else:
            sorted_idx = list(range(len(ids)))
        
        sorted_idx = sorted_idx + [0] * padding_length
        
        input_ids.append(padded_ids)
        attention_masks.append(attention_mask)
        sorted_indices.append(sorted_idx)
    
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_masks),
        'sorted_indices': torch.tensor(sorted_indices)
    }

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

print(f"[INFO] Dataloader ready: {len(dataloader)} batches")

# Extract attention
target_layer = -1
print(f"\nStart attention extraction (Layer {target_layer})...")

attention_sum = None
num_cells_processed = 0
batch_debug = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sorted_indices = batch['sorted_indices'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # Extract target-layer attention
        attn_scores = outputs.attentions[target_layer]
        
        # Normalize
        attn_scores = rank_normalize_fixed(attn_scores)
        
        # Average over attention heads
        attn_scores = attn_scores.mean(dim=1)
        
        # Restore gene order
        attn_scores = reverse_permute_fixed(attn_scores, sorted_indices)
        
        attn_numpy = attn_scores.cpu().numpy()
        
        # Apply mask and remove padding
        mask_2d = attention_mask.cpu().numpy()
        mask_matrix = mask_2d[:, :, None] * mask_2d[:, None, :]
        attn_numpy = attn_numpy * mask_matrix
        
        # Debug information
        if batch_debug < 2:
            print(f"\nBatch {batch_idx+1} statistics:")
            print(f"  Attention mean: {attn_numpy.mean():.6f}")
            print(f"  Attention max: {attn_numpy.max():.6f}")
            print(f"  Non-zero ratio: {(attn_numpy != 0).mean():.6f}")
            batch_debug += 1
        
        # Accumulate attention
        if attention_sum is None:
            attention_sum = attn_numpy.sum(axis=0)
        else:
            attention_sum += attn_numpy.sum(axis=0)
        
        num_cells_processed += input_ids.shape[0]
        
        # Clear GPU cache
        del outputs, attn_scores, input_ids, attention_mask
        torch.cuda.empty_cache()

print(f"[INFO] Attention extraction complete. Processed {num_cells_processed} cells")

# Compute mean attention
avg_attention = attention_sum / num_cells_processed

# Keep the valid-gene region
gene_attention = avg_attention[np.ix_(valid_indices, valid_indices)]

print(f"\nGene-gene attention matrix shape: {gene_attention.shape}")
print(f"Attentionstatistics:")
print(f"  Mean: {gene_attention.mean():.6f}")
print(f"  Max: {gene_attention.max():.6f}")
print(f"  Min: {gene_attention.min():.6f}")

# ========== Save outputs ==========
print("\n" + "="*60)
print("[STEP 7] Save results")
print("="*60)

# Build a 3-column TSV table for downstream filtering
gene_attention_df = pd.DataFrame(
    gene_attention,
    index=final_gene_names,
    columns=final_gene_names
)

gene_interactions_df = gene_attention_df.stack().reset_index()
gene_interactions_df.columns = ["Gene1", "Gene2", "EdgeWeight"]
gene_interactions_df = gene_interactions_df[gene_interactions_df["Gene1"] != gene_interactions_df["Gene2"]]
gene_interactions_df = gene_interactions_df.sort_values(by="EdgeWeight", ascending=False).reset_index(drop=True)

# Raw TSV saving is intentionally disabled
# Raw TSV saving code removed
# gene_interactions_df.to_csv(interactions_tsv_path, sep="\t", index=False)
# print(f"[INFO] Raw interaction TSV: {interactions_tsv_path}")

# Filter with the network file
print("\nFiltering interaction TSV...")
if not os.path.exists(network_path):
    raise FileNotFoundError(f"Network file not found: {network_path}")

network_df = pd.read_csv(network_path)
target_gene1_list = network_df["Gene1"].unique().tolist()
filtered_interactions = gene_interactions_df[gene_interactions_df["Gene1"].isin(target_gene1_list)]
filtered_interactions = filtered_interactions.sort_values(by="EdgeWeight", ascending=False).reset_index(drop=True)

# Save the filtered TSV
filtered_interactions.to_csv(filtered_tsv_path, sep="\t", index=False)
print(f"[INFO] Filtered interaction TSV: {filtered_tsv_path}")

# ========== Record run parameters ==========
print("\n" + "="*60)
print("[STEP 8] Save run parameters")
print("="*60)

# Compute runtime from script start
run_time = (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getctime(__file__))).total_seconds() / 60

# Collect run parameters
params = {
    "Model name": "geneformer",
    "Model version": MODEL_VERSION,
    "Data type": data_type,
    "Dataset name": dataset,
    "Raw data shape (cells x genes)": str(adata_original_shape),
    "Total genes after transpose": len(gene_symbols),
    "Matched genes in dictionary": matched_count,
    "Gene dictionary match rate (%)": f"{matched_count/len(gene_symbols)*100:.1f}",
    "Cell count before QC": adata_pre_qc_n_obs,
    "Gene count before QC": adata_pre_qc_n_vars,
    "Cell count after QC": adata.n_obs,
    "Gene count after QC": adata.n_vars,
    "Dataset size after tokenization": tokenized_size,
    "Final genes used by the model": n_final_genes,
    "Model device": str(device),
    "Total extracted gene pairs": len(gene_interactions_df),
    "Total extracted edges": len(gene_interactions_df),
    "Gene1 count in labels": len(target_gene1_list),
    "Filtered edges with Gene1 in labels": len(filtered_interactions),
    "Filtering rate (%)": f"{len(filtered_interactions)/len(gene_interactions_df)*100:.1f}" if len(gene_interactions_df) > 0 else "0",
    "Min": f"{gene_attention.min():.6f}",
    "Max": f"{gene_attention.max():.6f}",
    "Mean": f"{gene_attention.mean():.6f}",
    "Median weight": f"{np.median(gene_attention):.6f}",
    "Runtime (minutes)": f"{run_time:.2f}",
    "Run timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    # TSV path
    "Filtered TSV path": str(filtered_tsv_path)
}

# Append parameters to the summary CSV
params_df = pd.DataFrame([params])
if ALL_PARAMS_CSV.exists():
    params_df.to_csv(ALL_PARAMS_CSV, mode='a', header=False, index=False, encoding="utf-8")
else:
    params_df.to_csv(ALL_PARAMS_CSV, mode='w', header=True, index=False, encoding="utf-8")

print(f"[INFO] Parameters appended to: {ALL_PARAMS_CSV}")

# ========== Clean temporary files ==========
print("\n" + "="*60)
print("[STEP 9] Clean temporary files")
print("="*60)

import shutil
# Remove temporary loom file
if loom_path.exists():
    os.remove(loom_path)
    print(f"[INFO] Removed temporary loom file: {loom_path}")

# Remove temporary tokenized directory
if TEMP_TOKENIZED_DIR.exists():
    shutil.rmtree(TEMP_TOKENIZED_DIR)
    print(f"[INFO] Removed temporary tokenized directory: {TEMP_TOKENIZED_DIR}")

# Remove the temporary preprocessing directory if empty
if TEMP_PREPROCESSED_DIR.exists() and not any(TEMP_PREPROCESSED_DIR.iterdir()):
    os.rmdir(TEMP_PREPROCESSED_DIR)
    # Try removing the parent temporary directory if empty
    temp_parent = TEMP_PREPROCESSED_DIR.parent
    if temp_parent.exists() and not any(temp_parent.iterdir()):
        os.rmdir(temp_parent)
    print(f"[INFO] Removed temporary preprocessing directory: {TEMP_PREPROCESSED_DIR}")

# ========== Final summary ==========
print("\n" + "="*80)
print("[INFO] Final statistics")
print("="*80)
print(f"Task: {MODEL_NAME} | {data_type} | {dataset}")
print(f"\nstatistics:")
print(f"  Valid genes: {n_final_genes}")
print(f"  : {num_cells_processed}")
print(f"  : {len(gene_interactions_df)} ()")
print(f"  : {len(filtered_interactions)}")
print(f"  : {len(filtered_interactions)/len(gene_interactions_df)*100:.2f}%" if len(gene_interactions_df) > 0 else "0%")
print(f"  Timestamp: {run_time:.2f}")
print(f"\n:")
print(f"  - TSV: {filtered_tsv_path.name}")
print(f"  - CSV: {ALL_PARAMS_CSV.name}")
print("\n[INFO] All processing completed.")
print("="*80)