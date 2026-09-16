

"""
scCello Attention - (CHIP/Non_CHIP+CSV)
,DatasetCSV
"""

import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
import time
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
import argparse

# ==================== CLI  ====================
def parse_args():
    p = argparse.ArgumentParser(description="scCello attention extraction for one dataset.")
    p.add_argument("data_type", choices=["CHIP", "Non_CHIP", "STRING"])
    p.add_argument("dataset", type=str)
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--model-path", required=True, type=str)
    p.add_argument("--dict-dir", required=True, type=str)
    p.add_argument("--sccello-repo-dir", required=True, type=str, help="Path containing sccello/src.")
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--target-layer", default=-1, type=int)
    return p.parse_args()


ARGS = parse_args()
data_type = ARGS.data_type
dataset = ARGS.dataset
MODEL_NAME = "scCello"   # Model,


# ==================== ()====================
# 
INPUT_ROOT = ARGS.input_root
# 
OUTPUT_ROOT = ARGS.output_root
# ()
TYPE_OUTPUT_DIR = f"{OUTPUT_ROOT}/{data_type}"

# Create the output directory
os.makedirs(TYPE_OUTPUT_DIR, exist_ok=True)

# (,)
TEMP_LOOM_PATH = f"{OUTPUT_ROOT}/temp_{data_type}_{dataset}.loom"
TEMP_DATASET_DIR = f"{OUTPUT_ROOT}/temp_{data_type}_{dataset}.dataset"

# Model/()
SCCELLO_BASE = ARGS.model_path
MODEL_PATH = SCCELLO_BASE
DICT_DIR = ARGS.dict_dir

# CSV(Dataset)
ALL_PARAMS_CSV = f"{OUTPUT_ROOT}/{MODEL_NAME}-all-params.csv"

# 
if data_type == "CHIP":
    file_suffix = "_chip_matched"
    input_subdir = "CHIP"
elif data_type == "Non_CHIP":
    file_suffix = "_processed"
    input_subdir = "Non_CHIP"
elif data_type == "STRING":
    file_suffix = "_processed"
    input_subdir = "STRING"
else:
    raise ValueError(f": {data_type}(CHIP/Non_CHIP/STRING)")

# ()
INPUT_CSV = f"{INPUT_ROOT}/{input_subdir}/{dataset}{file_suffix}-ExpressionData.csv"

# (+Dataset)
TSV_PATH = f"{TYPE_OUTPUT_DIR}/{MODEL_NAME}-{data_type}-{dataset}-gene_attention_edges.tsv"


# ====================  ====================
BATCH_SIZE = int(ARGS.batch_size)
NUM_WORKERS = int(ARGS.num_workers)
TARGET_LAYER = int(ARGS.target_layer)
ADD_CLS = True


# ====================  ====================
SCCELLO_BASE_SRC = str(Path(ARGS.sccello_repo_dir) / "sccello")
SCCELLO_SRC = os.path.join(SCCELLO_BASE_SRC, "src")
SC_FOUNDATION_BASE = ARGS.sccello_repo_dir

for path in [SCCELLO_BASE_SRC, SCCELLO_SRC, SC_FOUNDATION_BASE]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model_prototype_contrastive import PrototypeContrastiveForMaskedLM as scCelloModel


# ==================== DataCollator ====================
class SimpleCollator:
    def __init__(self, add_cls=True):
        self.add_cls = add_cls
    
    def __call__(self, features):
        batch = {}
        max_length = max(len(f['input_ids']) for f in features)
        if self.add_cls:
            max_length += 1
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_idx = []
        batch_sorted_indices = []
        
        for f in features:
            input_ids = f['input_ids']
            if not isinstance(input_ids, list):
                input_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
            
            if self.add_cls:
                input_ids = [0] + input_ids
            
            seq_len = len(input_ids)
            attention_mask = [1] * seq_len + [0] * (max_length - seq_len)
            input_ids = input_ids + [0] * (max_length - seq_len)
            
            sorted_indices = f.get('sorted_indices', list(range(len(f['input_ids']))))
            if not isinstance(sorted_indices, list):
                sorted_indices = sorted_indices.tolist() if hasattr(sorted_indices, 'tolist') else list(sorted_indices)
            
            if self.add_cls:
                sorted_indices = [0] + [i+1 for i in sorted_indices]
            sorted_indices = sorted_indices + [0] * (max_length - len(sorted_indices))
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(int(f.get('label', 0)))
            batch_idx.append(int(f.get('idx', 0)))
            batch_sorted_indices.append(sorted_indices)
        
        batch['input_ids'] = torch.tensor(batch_input_ids, dtype=torch.long)
        batch['attention_mask'] = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        batch['idx'] = torch.tensor(batch_idx, dtype=torch.long)
        batch['sorted_indices'] = torch.tensor(batch_sorted_indices, dtype=torch.long)
        
        return batch


# ==================== reverse_permute ====================
def reverse_permute(tensor, indices):
    device = tensor.device
    if indices.device != device:
        indices = indices.to(device)
    
    batch_size, seq_len = indices.shape
    indices = torch.clamp(indices, 0, seq_len - 1)
    
    inverse_indices = torch.zeros_like(indices, device=device)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(indices)
    inverse_indices.scatter_(1, indices, torch.arange(seq_len, device=device).unsqueeze(0).expand_as(indices))
    
    if tensor.dim() == 2:
        return torch.gather(tensor, 1, inverse_indices)
    elif tensor.dim() == 3:
        inverse_indices_expanded = inverse_indices.unsqueeze(1).expand(-1, tensor.size(1), -1)
        result = torch.gather(tensor, 2, inverse_indices_expanded)
        inverse_indices_expanded = inverse_indices.unsqueeze(2).expand(-1, -1, result.size(2))
        return torch.gather(result, 1, inverse_indices_expanded)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


# ==================== Timestamp()====================
start_time = time.time()


# ==================== Label()====================
print("="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] Init")
print("="*80)
print(f"✓ : {TYPE_OUTPUT_DIR}")
print(f"✓ : {ALL_PARAMS_CSV}")


# ==================== (+Dataset)====================
print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] ")
print("="*80)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"does not exist: {INPUT_CSV}")

# 
adata = sc.read_csv(INPUT_CSV)
adata_original_shape = adata.shape  # 
print(f"✓ : {INPUT_CSV}")
print(f"✓ (): {adata_original_shape}")

# ×
adata = adata.T
adata_transposed_shape = adata.shape
print(f"✓ (×): {adata_transposed_shape}")

# 
gene_symbols = adata.var_names.tolist()
print(f"✓ : {len(gene_symbols)}")

# 
with open(f"{DICT_DIR}/gene_name_id_dict.pkl", "rb") as f:
    gene_name_dict = pickle.load(f)
valid_mask = [gene in gene_name_dict for gene in gene_symbols]
valid_gene_count = sum(valid_mask)
print(f"✓ : {valid_gene_count}/{len(gene_symbols)} ({valid_gene_count/len(gene_symbols)*100:.1f}%)")

# 
adata = adata[:, valid_mask]
final_gene_symbols = [g for g, v in zip(gene_symbols, valid_mask) if v]
ensembl_ids = [gene_name_dict[gene] for gene in final_gene_symbols]
adata.var['gene_name'] = final_gene_symbols
adata.var['ensembl_id'] = ensembl_ids
adata.var_names = final_gene_symbols

# 
adata_pre_qc_n_obs = adata.n_obs
adata_pre_qc_n_vars = adata.n_vars
print(f"\n✓ : {adata_pre_qc_n_obs}, : {adata_pre_qc_n_vars}")

# 
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print(f"✓ : {adata.n_obs}, : {adata.n_vars}")

# 
adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if scipy.sparse.issparse(adata.X) else adata.X.sum(axis=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 
if 'cell_type' not in adata.obs.columns:
    adata.obs['cell_type'] = 'unknown'
adata.obs['adata_order'] = range(adata.n_obs)


# ==================== Tokenization ====================
print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] Tokenization")
print("="*80)

# loom
adata.write_loom(TEMP_LOOM_PATH, write_obsm_varm=False)
print(f"✓ loom: {TEMP_LOOM_PATH}")

tokenizer = TranscriptomeTokenizer(
    custom_attr_name_dict={"cell_type": "cell_type", "adata_order": "adata_order"},
    nproc=NUM_WORKERS,
    gene_median_file=f"{DICT_DIR}/gene_median_dictionary.pkl",
    token_dictionary_file=f"{DICT_DIR}/token_dictionary.pkl"
)

tokenizer.tokenize_data(
    data_directory=OUTPUT_ROOT,
    output_directory=OUTPUT_ROOT,
    output_prefix=f"temp_{data_type}_{dataset}",
    file_format="loom",
    use_generator=False
)

tokenized_dataset = load_from_disk(TEMP_DATASET_DIR)
print(f"✓ Dataset: {TEMP_DATASET_DIR}")
print(f"✓ TokenizationDataset: {len(tokenized_dataset)}")

# Dataset
types = list(set(tokenized_dataset['cell_type']))
type2num = dict([(type_name, i) for i, type_name in enumerate(types)])
def add_labels_and_indices(example, idx):
    example["label"] = type2num[example['cell_type']]
    example["idx"] = idx
    if 'sorted_indices' not in example:
        example['sorted_indices'] = list(range(len(example['input_ids'])))
    return example
tokenized_dataset = tokenized_dataset.map(add_labels_and_indices, with_indices=True, num_proc=NUM_WORKERS)

required_columns = ['input_ids', 'label', 'idx', 'sorted_indices']
remove_columns = [col for col in tokenized_dataset.column_names if col not in required_columns]
if remove_columns:
    tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)

def convert_to_lists(example):
    if not isinstance(example['input_ids'], list):
        example['input_ids'] = example['input_ids'].tolist()
    if not isinstance(example['sorted_indices'], list):
        example['sorted_indices'] = example['sorted_indices'].tolist()
    return example
tokenized_dataset = tokenized_dataset.map(convert_to_lists, num_proc=1)


# ====================  ====================
with open(f"{DICT_DIR}/token_dictionary.pkl", "rb") as f:
    vocab = pickle.load(f)
with open(f"{DICT_DIR}/gene_name_id_dict.pkl", "rb") as f:
    gene_name_id = pickle.load(f)
with open(f"{DICT_DIR}/gene_median_dictionary.pkl", "rb") as f:
    gene_median_dict = pickle.load(f)

id2name = {v: k for k, v in gene_name_id.items()}
gene_keys = list(gene_median_dict.keys())
genelist_dict = dict(zip(gene_keys, [True] * len(gene_keys)))

coding_miRNA_loc = np.where([genelist_dict.get(i, False) for i in adata.var["ensembl_id"]])[0]
ori_gene_ids = np.array([vocab[g] for g in adata.var["ensembl_id"][coding_miRNA_loc]])
ori_gene_names = [id2name.get(i, "") for i in adata.var["ensembl_id"][coding_miRNA_loc]]
n_genes = len(ori_gene_names)
print(f"\n✓ Model: {n_genes}")

token_id_to_gene_name = {}
for ensembl_id, gene_name in zip(adata.var["ensembl_id"][coding_miRNA_loc], ori_gene_names):
    token_id = vocab.get(ensembl_id)
    if token_id is not None:
        token_id_to_gene_name[token_id] = gene_name
print(f"✓ token-: {len(token_id_to_gene_name)}")


# ==================== Model ====================
print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] Model")
print("="*80)

model = scCelloModel.from_pretrained(MODEL_PATH, output_hidden_states=True)
we_before = model.bert.embeddings.word_embeddings.weight
print(f"✓ embeddingWeight range: [{we_before.min().item():.4f}, {we_before.max().item():.4f}]")

# ()
if we_before.abs().max().item() == 0:
    weight_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Modeldoes not exist: {weight_path}")
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    we_after = model.bert.embeddings.word_embeddings.weight
    print(f"✓ embeddingWeight range: [{we_after.min().item():.4f}, {we_after.max().item():.4f}]")

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"✓ Model: {device}")


# ==================== DataLoader ====================
tokenized_dataset.set_format(type=None)
collator = SimpleCollator(add_cls=ADD_CLS)
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collator,
    shuffle=False,
    num_workers=0
)
print(f"\n✓ DataLoader: {len(dataloader)}")


# ==================== Attention ====================
print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] Attention")
print("="*80)

gene_pair_attention = defaultdict(lambda: {'sum': 0.0, 'count': 0})
attention_debug_shown = False

with torch.no_grad():
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing", leave=False)):
        cell_input_ids = batch_data['input_ids'].to(device)
        cell_atts = batch_data['attention_mask'].to(device)
        
        # 
        cell_output = model.bert(
            input_ids=cell_input_ids,
            attention_mask=cell_atts,
            output_attentions=True
        )
        attn_scores = cell_output.attentions[TARGET_LAYER]
        num_heads = attn_scores.size(1)
        
        # CLS
        sorted_indices = batch_data["sorted_indices"].to(device)
        input_ids = batch_data["input_ids"].to(device)
        if ADD_CLS:
            attn_scores = attn_scores[..., 1:, 1:]
            sorted_indices = sorted_indices[:, 1:] - 1
            input_ids = input_ids[:, 1:]
        
        # Rank Normalization
        M = attn_scores.shape[-1]
        attn_scores = attn_scores.reshape((-1, M))
        order = torch.argsort(attn_scores, dim=1)
        rank = torch.argsort(order, dim=1)
        attn_scores = rank.reshape((-1, num_heads, M, M)) / M
        
        attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
        order = torch.argsort(attn_scores, dim=1)
        rank = torch.argsort(order, dim=1)
        attn_scores = (rank.reshape((-1, num_heads, M, M)) / M).permute(0, 1, 3, 2)
        
        # heads
        attn_scores = attn_scores.mean(1)
        
        # Attention
        if not attention_debug_shown and batch_idx == 0:
            print(f"\n✓ Attention(11):")
            print(f"  : {attn_scores.shape}, : [{attn_scores.min().item():.6f}, {attn_scores.max().item():.6f}]")
            print(f"  5×5:\n{attn_scores[0].cpu().numpy()[:5, :5]}")
            attention_debug_shown = True
        
        # 
        attn_scores = reverse_permute(attn_scores, sorted_indices)
        gene_ids = reverse_permute(input_ids, sorted_indices)
        
        # attention
        outputs = attn_scores.cpu().numpy()
        gene_ids_np = gene_ids.cpu().numpy()
        batch_size_curr = outputs.shape[0]
        for b in range(batch_size_curr):
            cell_attn = outputs[b]
            cell_gene_ids = gene_ids_np[b]
            cell_gene_names = [token_id_to_gene_name.get(int(gene_id)) for gene_id in cell_gene_ids]
            
            for i in range(len(cell_gene_names)):
                if cell_gene_names[i] is None:
                    continue
                gene_i = cell_gene_names[i]
                for j in range(len(cell_gene_names)):
                    if cell_gene_names[j] is None or i == j:
                        continue
                    gene_j = cell_gene_names[j]
                    gene_pair_attention[(gene_i, gene_j)]['sum'] += cell_attn[i, j]
                    gene_pair_attention[(gene_i, gene_j)]['count'] += 1
        
        # 
        del cell_output, attn_scores
        torch.cuda.empty_cache()

print(f"\n✓ : {len(gene_pair_attention)}")


# ====================  ====================
print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] ")
print("="*80)

# 
all_edges = []
for (gene_i, gene_j), stats in gene_pair_attention.items():
    if stats['count'] > 0:
        all_edges.append({
            'Gene1': gene_i,
            'Gene2': gene_j,
            'EdgeWeight': stats['sum'] / stats['count']
        })
all_edges_df = pd.DataFrame(all_edges)
print(f"✓ : {len(all_edges_df)}")

# no extra filtering: keep all directed edges (self-loop already removed)
all_edges_df = all_edges_df.sort_values('EdgeWeight', ascending=False).reset_index(drop=True)
all_edges_df.to_csv(TSV_PATH, sep='\t', index=False)
print(f"✓ : {TSV_PATH}")


# ==================== CSV ====================
print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] (CSV)")
print("="*80)

# Timestamp
end_time = time.time()
run_time = (end_time - start_time) / 60  # 

# 
params = {
    "Model": MODEL_NAME,
    "": data_type,
    "Dataset": dataset,
    "()": str(adata_original_shape),
    "(×)": str(adata_transposed_shape),
    "": len(gene_symbols),
    "": valid_gene_count,
    "(%)": f"{valid_gene_count/len(gene_symbols)*100:.1f}",
    "": adata_pre_qc_n_obs,
    "": adata_pre_qc_n_vars,
    "": adata.n_obs,
    "": adata.n_vars,
    "TokenizationDataset": len(tokenized_dataset),
    "Model": n_genes,
    "Model": str(device),
    "": len(gene_pair_attention),
    "": len(all_edges_df),
    "LabelGene1": 0,
    "(Gene1Label)": len(all_edges_df),
    "(%)": "100.0",
    "": f"{all_edges_df['EdgeWeight'].min():.6f}" if not all_edges_df.empty else "0",
    "": f"{all_edges_df['EdgeWeight'].max():.6f}" if not all_edges_df.empty else "0",
    "Timestamp()": f"{run_time:.2f}",
    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "": TSV_PATH
}

# DataFrame
params_df = pd.DataFrame([params])

# CSV
if os.path.exists(ALL_PARAMS_CSV):
    # ,()
    params_df.to_csv(ALL_PARAMS_CSV, mode='a', header=False, index=False, encoding="utf-8")
    print(f"✓ : {ALL_PARAMS_CSV}")
else:
    # does not exist,
    params_df.to_csv(ALL_PARAMS_CSV, mode='w', header=True, index=False, encoding="utf-8")
    print(f"✓ : {ALL_PARAMS_CSV}")

# 
print(f"\n✓ Dataset:")
print(f"  - -Dataset: {data_type}-{dataset}")
print(f"  - : {adata.n_obs}, : {adata.n_vars}")
print(f"  - : {len(all_edges_df)}, Weight range: [{all_edges_df['EdgeWeight'].min():.6f}, {all_edges_df['EdgeWeight'].max():.6f}]")
print(f"  - Timestamp: {run_time:.2f}")


# ====================  ====================
print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] ")
print("="*80)

# loom
if os.path.exists(TEMP_LOOM_PATH):
    os.remove(TEMP_LOOM_PATH)
    print(f"✓ : {TEMP_LOOM_PATH}")

# Dataset
if os.path.exists(TEMP_DATASET_DIR):
    import shutil
    shutil.rmtree(TEMP_DATASET_DIR)
    print(f"✓ : {TEMP_DATASET_DIR}")

print("\n" + "="*80)
print(f"[{MODEL_NAME}-{data_type}-{dataset}] !")
print(f":")
print(f"  - : {ALL_PARAMS_CSV}")
print(f"  - : {TSV_PATH}")
print("="*80)