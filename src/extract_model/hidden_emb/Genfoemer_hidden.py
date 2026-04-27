# #!/usr/bin/env python3
# """
# Geneformer Hidden State Embedding 提取脚本（完整版）
# - 提取基于 hidden states 的基因 embedding
# - 保存所有基因之间的余弦相似度边（无限制）
# - 使用所有细胞和所有基因
# """

# import os
# import sys
# import json
# import pickle
# from pathlib import Path
# from typing import List, Tuple, Dict, Optional
# from collections import defaultdict
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from transformers import BertForMaskedLM


# # ===========================
# # 配置参数
# # ===========================
# CONFIG = {
#     # 模型相关
#     "MODEL_DIR": "/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/default/12L",
#     "DICT_DIR": "/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts",  # 字典文件路径
#     "HIDDEN_LAYER": -1,  # 提取哪一层的 hidden states（-1 = 最后一层）
    
#     # 数据相关
#     "DATA_ROOT": "/mnt/10T/yzn/benchmark_GRN/input_process",
#     "OUTPUT_ROOT": "/mnt/10T/yzn/benchmark_GRN/Geneformer_hidden_embeddings",
    
#     # 序列构建参数
#     "SEQ_TOPK_GENES": 256,      # 每个细胞选择多少高表达基因构建序列
#     "MAX_SEQ_LEN": 256,         # 序列最大长度
#     "USE_LOG1P": False,          # 是否对表达量做 log1p 归一化
    
#     # ⭐ 关键修改：不限制细胞和基因数量
#     "N_CELLS": None,            # None = 使用所有细胞
#     "SAVE_ALL_EDGES": True,     # 保存所有边，不做任何限制
    
#     # 其他
#     "BATCH_SIZE": 16,
#     "NUM_WORKERS": 4,
#     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
# }


# # ===========================
# # 辅助函数
# # ===========================

# def load_token_dictionary(dict_dir: str) -> Dict[str, int]:
#     """加载 Geneformer 的 token 字典"""
#     token_dict_path = Path(dict_dir) / "token_dictionary.pkl"
#     with open(token_dict_path, "rb") as f:
#         token_dict = pickle.load(f)
#     print(f"Token dictionary loaded: {len(token_dict)} tokens")
#     return token_dict


# def load_gene_name_id_dict(dict_dir: str) -> Dict[str, int]:
#     """加载基因名到 Ensembl ID 的映射"""
#     gene_dict_path = Path(dict_dir) / "gene_name_id_dict.pkl"
#     with open(gene_dict_path, "rb") as f:
#         gene_dict = pickle.load(f)
#     print(f"Gene name dictionary loaded: {len(gene_dict)} symbols")
#     return gene_dict


# def match_genes_to_tokens(
#     genes: List[str],
#     token_dict: Dict[str, int],
#     gene_dict: Dict[str, int]
# ) -> Tuple[List[str], List[int]]:
#     """
#     将基因符号映射到 token ID
#     返回: (matched_genes, token_ids)
#     """
#     matched_genes = []
#     token_ids = []
    
#     for gene in genes:
#         # 尝试直接匹配
#         if gene in token_dict:
#             matched_genes.append(gene)
#             token_ids.append(token_dict[gene])
#             continue
        
#         # 尝试通过 gene_dict 转换为 Ensembl ID
#         if gene in gene_dict:
#             ensembl_id = gene_dict[gene]
#             if ensembl_id in token_dict:
#                 matched_genes.append(gene)
#                 token_ids.append(token_dict[ensembl_id])
    
#     return matched_genes, token_ids


# def build_cell_sequences(
#     expr_matrix: np.ndarray,
#     genes: List[str],
#     token_ids: List[int],
#     topk: int = 256,
#     max_len: int = 256,
#     use_log1p: bool = True
# ) -> List[List[int]]:
#     """
#     为每个细胞构建基因序列（按表达量排序）
    
#     Args:
#         expr_matrix: [n_genes, n_cells]
#         genes: 基因列表
#         token_ids: 对应的 token ID
#         topk: 每个细胞选择多少高表达基因
#         max_len: 序列最大长度
#         use_log1p: 是否 log1p 归一化
    
#     Returns:
#         sequences: List of token ID sequences
#     """
#     n_genes, n_cells = expr_matrix.shape
    
#     if use_log1p:
#         expr_matrix = np.log1p(expr_matrix)
    
#     sequences = []
#     for cell_idx in range(n_cells):
#         cell_expr = expr_matrix[:, cell_idx]
        
#         # 按表达量排序，选择 top-K 基因
#         top_indices = np.argsort(cell_expr)[::-1][:topk]
        
#         # 构建序列（保持排序顺序）
#         seq = [token_ids[i] for i in top_indices]
        
#         # 截断
#         if len(seq) > max_len:
#             seq = seq[:max_len]
        
#         sequences.append(seq)
    
#     return sequences


# # ===========================
# # Dataset 类
# # ===========================

# class GeneSequenceDataset(Dataset):
#     """基因序列数据集"""
    
#     def __init__(self, sequences: List[List[int]], max_len: int = 256):
#         self.sequences = sequences
#         self.max_len = max_len
    
#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         seq = self.sequences[idx]
#         return {
#             "input_ids": seq,
#             "length": len(seq)
#         }


# def collate_fn(batch):
#     """动态padding"""
#     max_len = max(item["length"] for item in batch)
    
#     input_ids = []
#     attention_mask = []
    
#     for item in batch:
#         seq = item["input_ids"]
#         pad_len = max_len - len(seq)
        
#         # Padding
#         padded_seq = seq + [0] * pad_len
#         mask = [1] * len(seq) + [0] * pad_len
        
#         input_ids.append(padded_seq)
#         attention_mask.append(mask)
    
#     return {
#         "input_ids": torch.tensor(input_ids, dtype=torch.long),
#         "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
#     }


# # ===========================
# # Hidden States 提取
# # ===========================

# def extract_hidden_embeddings(
#     model: BertForMaskedLM,
#     sequences: List[List[int]],
#     genes: List[str],
#     token_ids: List[int],
#     config: dict
# ) -> Tuple[np.ndarray, List[str]]:
#     """
#     通过前向传播提取 hidden states，聚合为基因 embedding
    
#     Returns:
#         embeddings: [n_genes, hidden_dim]
#         final_genes: 实际有 embedding 的基因列表
#     """
#     device = config["DEVICE"]
#     model = model.to(device)
#     model.eval()
    
#     # 创建 DataLoader
#     dataset = GeneSequenceDataset(sequences, config["MAX_SEQ_LEN"])
#     loader = DataLoader(
#         dataset,
#         batch_size=config["BATCH_SIZE"],
#         shuffle=False,
#         num_workers=config["NUM_WORKERS"],
#         collate_fn=collate_fn
#     )
    
#     # 累积每个 token 的 hidden states
#     token_to_hidden = defaultdict(list)
    
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Forward Geneformer"):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
            
#             # 前向传播，获取 hidden states
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 output_hidden_states=True
#             )
            
#             # 提取指定层的 hidden states
#             # outputs.hidden_states: tuple of (batch_size, seq_len, hidden_dim)
#             hidden_states = outputs.hidden_states[config["HIDDEN_LAYER"]]
#             # hidden_states: [batch_size, seq_len, hidden_dim]
            
#             # 遍历 batch 中的每个序列
#             for seq_idx in range(hidden_states.size(0)):
#                 seq_tokens = input_ids[seq_idx].cpu().numpy()
#                 seq_mask = attention_mask[seq_idx].cpu().numpy()
#                 seq_hidden = hidden_states[seq_idx].cpu().numpy()  # [seq_len, hidden_dim]
                
#                 # 只处理非 padding 的 token
#                 for pos_idx in range(len(seq_tokens)):
#                     if seq_mask[pos_idx] == 0:
#                         break
                    
#                     token_id = int(seq_tokens[pos_idx])
#                     token_hidden = seq_hidden[pos_idx]
                    
#                     token_to_hidden[token_id].append(token_hidden)
    
#     # 为每个基因聚合 hidden states（取平均）
#     gene_to_idx = {g: i for i, g in enumerate(genes)}
#     token_to_gene = {tid: genes[i] for i, tid in enumerate(token_ids)}
    
#     final_genes = []
#     embeddings_list = []
    
#     for token_id, gene in token_to_gene.items():
#         if token_id in token_to_hidden:
#             hidden_list = token_to_hidden[token_id]
#             avg_hidden = np.mean(hidden_list, axis=0)
            
#             final_genes.append(gene)
#             embeddings_list.append(avg_hidden)
    
#     embeddings = np.array(embeddings_list)  # [n_genes, hidden_dim]
    
#     return embeddings, final_genes


# # ===========================
# # ⭐ 修改：保存所有余弦相似度边
# # ===========================

# def compute_and_save_all_cosine_edges(
#     genes: List[str],
#     emb: np.ndarray,
#     output_path: str
# ) -> Dict:
#     """
#     计算所有基因对之间的余弦相似度，保存所有边（去除自环）
    
#     Args:
#         genes: 基因列表 [n_genes]
#         emb: embedding矩阵 [n_genes, hidden_dim]
#         output_path: 输出TSV路径
    
#     Returns:
#         info: 统计信息
#     """
#     n_genes = len(genes)
    
#     # 归一化
#     norms = np.linalg.norm(emb, axis=1, keepdims=True)
#     emb_norm = emb / (norms + 1e-10)
    
#     # 计算余弦相似度矩阵
#     print(f"📊 计算 {n_genes} x {n_genes} 余弦相似度矩阵...")
#     cosine_sim = emb_norm @ emb_norm.T  # [n_genes, n_genes]
    
#     # 保存所有边（去除自环：i != j）
#     print(f"💾 保存所有边到 {output_path} (去除自环)...")
#     edges = []
    
#     for i in range(n_genes):
#         for j in range(n_genes):
#             if i != j:  # ⭐ 去除自环
#                 edges.append({
#                     "gene1": genes[i],
#                     "gene2": genes[j],
#                     "cosine": cosine_sim[i, j]
#                 })
    
#     df = pd.DataFrame(edges)
#     df.to_csv(output_path, sep="\t", index=False)
    
#     info = {
#         "n_genes": n_genes,
#         "n_edges": len(edges),
#         "description": "All pairwise cosine similarities (excluding self-loops)"
#     }
    
#     print(f"✅ 保存了 {len(edges):,} 条边 (去除 {n_genes} 个自环)")
    
#     return info


# # ===========================
# # 主处理流程
# # ===========================

# def process_single_dataset(expr_path: str, config: dict) -> Dict:
#     """处理单个数据集"""
    
#     # 1. 加载表达矩阵
#     df_expr = pd.read_csv(expr_path, index_col=0)
#     genes = df_expr.index.tolist()
#     expr_matrix = df_expr.values  # [n_genes, n_cells]
    
#     n_genes, n_cells = expr_matrix.shape
#     print(f"Expression loaded: {n_genes} genes x {n_cells} cells")
#     print(f"✅ 输入基因数：{n_genes}")
    
#     # 2. 加载字典
#     token_dict = load_token_dictionary(config["DICT_DIR"])
#     gene_dict = load_gene_name_id_dict(config["DICT_DIR"])
    
#     # 3. 匹配基因到 token
#     matched_genes, token_ids = match_genes_to_tokens(genes, token_dict, gene_dict)
#     print(f"✅ 匹配到的基因：{len(matched_genes)}/{n_genes}")
    
#     if len(matched_genes) == 0:
#         raise ValueError("没有匹配到任何基因！")
    
#     # 过滤表达矩阵
#     gene_to_idx = {g: i for i, g in enumerate(genes)}
#     matched_indices = [gene_to_idx[g] for g in matched_genes]
#     expr_matrix_matched = expr_matrix[matched_indices, :]
    
#     # 4. 使用所有细胞（不限制数量）
#     n_cells_use = n_cells
#     expr_matrix_use = expr_matrix_matched
    
#     # 5. 构建序列
#     sequences = build_cell_sequences(
#         expr_matrix_use,
#         matched_genes,
#         token_ids,
#         topk=config["SEQ_TOPK_GENES"],
#         max_len=config["MAX_SEQ_LEN"],
#         use_log1p=config["USE_LOG1P"]
#     )
    
#     # 6. 加载模型
#     print(f"📦 加载模型: {config['MODEL_DIR']}")
#     model = BertForMaskedLM.from_pretrained(
#         config["MODEL_DIR"],
#         output_hidden_states=True
#     )
    
#     # 7. 提取 hidden embeddings
#     embeddings, final_genes = extract_hidden_embeddings(
#         model, sequences, matched_genes, token_ids, config
#     )
    
#     print(f"✅ 最终hidden embeddings: {len(final_genes)} genes x {embeddings.shape[1]} dim")
#     print(f"   使用细胞数: {n_cells_use}")
    
#     # 8. 准备输出目录
#     dataset_name = Path(expr_path).stem.replace("-ExpressionData", "")
#     output_dir = Path(config["OUTPUT_ROOT"]) / Path(expr_path).parent.name / dataset_name
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # 9. 保存 embedding
#     emb_tsv = output_dir / "Geneformer_hidden_gene_embedding.tsv"
#     emb_npz = output_dir / "Geneformer_hidden_gene_embedding.npz"
    
#     df_emb = pd.DataFrame(embeddings, index=final_genes)
#     df_emb.to_csv(emb_tsv, sep="\t")
#     np.savez_compressed(emb_npz, embeddings=embeddings, genes=final_genes)
    
#     print(f"💾 Embedding 已保存:")
#     print(f"   TSV: {emb_tsv}")
#     print(f"   NPZ: {emb_npz}")
    
#     # 10. ⭐ 保存所有余弦相似度边
#     cosine_tsv = output_dir / f"Geneformer_{dataset_name}_all_edges.tsv"
#     cosine_info = compute_and_save_all_cosine_edges(final_genes, embeddings, cosine_tsv)
    
#     # 11. 保存元数据
#     meta = {
#         "dataset": dataset_name,
#         "n_input_genes": n_genes,
#         "n_matched_genes": len(matched_genes),
#         "n_final_genes": len(final_genes),
#         "n_cells_total": n_cells,
#         "n_cells_used": n_cells_use,
#         "embedding_dim": embeddings.shape[1],
#         "hidden_layer": config["HIDDEN_LAYER"],
#         "seq_topk_genes": config["SEQ_TOPK_GENES"],
#         "max_seq_len": config["MAX_SEQ_LEN"],
#         "use_log1p": config["USE_LOG1P"],
#         "cosine_edges": cosine_info
#     }
    
#     meta_path = output_dir / "embedding_meta.json"
#     with open(meta_path, "w") as f:
#         json.dump(meta, f, indent=2)
    
#     print(f"📄 元数据: {meta_path}")
#     print()
    
#     return meta


# # ===========================
# # 批量处理
# # ===========================

# def main():
#     """批量处理所有数据集"""
    
#     # 创建输出根目录
#     output_root = Path(CONFIG["OUTPUT_ROOT"])
#     output_root.mkdir(parents=True, exist_ok=True)
    
#     data_root = Path(CONFIG["DATA_ROOT"])
    
#     # 遍历所有文件夹
#     folders = sorted([f for f in data_root.iterdir() if f.is_dir()])
    
#     all_records = []
    
#     for folder in folders:
#         expr_files = sorted(folder.glob("*-ExpressionData.csv"))
        
#         if not expr_files:
#             continue
        
#         print(f"\n{'='*60}")
#         print(f"📂 处理文件夹: {folder.name} ({len(expr_files)} 个文件)")
#         print(f"{'='*60}")
        
#         for expr_path in expr_files:
#             dataset_name = expr_path.stem.replace("-ExpressionData", "")
#             print(f"🔍 处理数据集：{dataset_name}")
#             print(f"   Expr文件：{expr_path}")
            
#             try:
#                 record = process_single_dataset(str(expr_path), CONFIG)
#                 all_records.append(record)
#             except Exception as e:
#                 print(f"❌ 处理失败 {expr_path.name}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
    
#     # 保存总结
#     summary_path = Path(CONFIG["OUTPUT_ROOT"]) / "processing_summary.json"
#     with open(summary_path, "w") as f:
#         json.dump(all_records, f, indent=2)
    
#     print(f"\n{'='*60}")
#     print(f"✅ 全部完成！共处理 {len(all_records)} 个数据集")
#     print(f"📊 总结文件: {summary_path}")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     main()











#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geneformer hidden-state -> cosine edges (LangCell-aligned IO rules)

✅ 输入遍历：按 CHIP/Non_CHIP/STRING 等文件夹，自动找 ExpressionData.csv
   - CHIP: *_chip_matched-ExpressionData.csv
   - others: *_processed-ExpressionData.csv

✅ 输出：只输出一个边 TSV
   {MODEL_NAME}_{DATASET}.tsv  (columns: Gene1, Gene2, EdgeWeight; directed; no self-loop)

✅ 保存关键参数：run_params.json

❌ 不输出 embedding 文件（仅在内存中生成用于算边）
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import datetime
import traceback
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM

warnings.filterwarnings("ignore")


# ===========================
# CONFIG (aligned)
# ===========================
CONFIG = {
    # Model
    "MODEL_DIR": "",
    "DICT_DIR": "",
    "MODEL_NAME": "Geneformer",
    "HIDDEN_LAYER": -1,

    # Batch IO (aligned with LangCell)
    "INPUT_ROOT": "",
    "OUTPUT_ROOT": "",
    "TARGET_FOLDERS": ["CHIP"],  # e.g. ["CHIP","Non_CHIP","STRING"] or None for all subfolders

    # Sequence
    "SEQ_TOPK_GENES": 512,
    "MAX_SEQ_LEN": 512,
    "USE_LOG1P": False,
    "N_CELLS": None,  # None = all cells

    # Forward
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,

    # Cosine edges
    "SAVE_ALL_EDGES": True,  # True => all i!=j edges (stream)
    "TOPK_PER_GENE": 1000,   # used when SAVE_ALL_EDGES=False
}


def parse_args():
    p = argparse.ArgumentParser(description="Extract Geneformer hidden embeddings and export cosine-edge TSVs.")
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--dict-dir", required=True, type=str)
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--folders", nargs="+", default=["CHIP"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


ARGS = parse_args()
CONFIG["MODEL_DIR"] = ARGS.model_dir
CONFIG["DICT_DIR"] = ARGS.dict_dir
CONFIG["INPUT_ROOT"] = ARGS.input_root
CONFIG["OUTPUT_ROOT"] = ARGS.output_root
CONFIG["TARGET_FOLDERS"] = ARGS.folders
CONFIG["BATCH_SIZE"] = int(ARGS.batch_size)
CONFIG["SEED"] = int(ARGS.seed)


# ===========================
# Utils
# ===========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))


def extract_dataset_name(file_path: str) -> str:
    base = os.path.basename(file_path)
    if "_chip_matched-ExpressionData.csv" in base:
        return base.split("_chip_matched-ExpressionData.csv")[0]
    if "_processed-ExpressionData.csv" in base:
        return base.split("_processed-ExpressionData.csv")[0]
    return base.split("-ExpressionData.csv")[0]


def load_token_dictionary(dict_dir: str) -> Dict[str, int]:
    token_dict_path = Path(dict_dir) / "token_dictionary.pkl"
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    print(f"Token dictionary loaded: {len(token_dict)} tokens")
    return token_dict


def load_gene_name_id_dict(dict_dir: str) -> Dict[str, str]:
    gene_dict_path = Path(dict_dir) / "gene_name_id_dict.pkl"
    with open(gene_dict_path, "rb") as f:
        gene_dict = pickle.load(f)  # symbol -> ensg
    print(f"Gene name dictionary loaded: {len(gene_dict)} symbols")
    return gene_dict


def match_genes_to_tokens(
    genes: List[str],
    token_dict: Dict[str, int],
    gene_dict: Dict[str, str]
) -> Tuple[List[str], List[int], int]:
    """
    返回: matched_genes, token_ids, missing_count
    """
    matched_genes = []
    token_ids = []
    missing = 0

    for gene in genes:
        # direct match (sometimes gene index is already ENSG)
        if gene in token_dict:
            matched_genes.append(gene)
            token_ids.append(int(token_dict[gene]))
            continue

        if gene in gene_dict:
            ensg = gene_dict[gene]
            if ensg in token_dict:
                matched_genes.append(gene)
                token_ids.append(int(token_dict[ensg]))
            else:
                missing += 1
        else:
            missing += 1

    return matched_genes, token_ids, missing


def build_cell_sequences(
    expr_matrix: np.ndarray,
    token_ids: List[int],
    topk: int,
    max_len: int,
    use_log1p: bool
) -> List[List[int]]:
    """
    expr_matrix: [n_genes, n_cells] (already filtered to matched genes)
    token_ids: aligned with expr_matrix gene axis
    """
    n_genes, n_cells = expr_matrix.shape
    X = expr_matrix.astype(np.float32)

    if use_log1p:
        X = np.log1p(np.maximum(X, 0.0))

    seqs = []
    for cell_idx in range(n_cells):
        col = X[:, cell_idx]
        k = min(int(topk), len(col))
        top_i = np.argpartition(-col, kth=k - 1)[:k]
        top_i = top_i[np.argsort(-col[top_i])]
        seq = [token_ids[i] for i in top_i]
        seq = seq[:max_len]
        seqs.append(seq)

    return seqs


class GeneSequenceDataset(Dataset):
    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"input_ids": seq, "length": len(seq)}


def collate_fn(batch):
    max_len = max(item["length"] for item in batch)
    input_ids, attention_mask = [], []

    for item in batch:
        seq = item["input_ids"]
        pad_len = max_len - len(seq)
        input_ids.append(seq + [0] * pad_len)           # pad_id = 0 in config/token dict usually
        attention_mask.append([1] * len(seq) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }


def extract_hidden_embeddings(
    model: BertForMaskedLM,
    sequences: List[List[int]],
    genes: List[str],
    token_ids: List[int],
    config: dict
) -> Tuple[np.ndarray, List[str]]:
    """
    返回 embeddings [n_final_genes, hidden_dim], final_genes
    """
    device = config["DEVICE"]
    model = model.to(device)
    model.eval()

    dataset = GeneSequenceDataset(sequences)
    loader = DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["NUM_WORKERS"],
        collate_fn=collate_fn
    )

    token_to_sum = {}
    token_to_cnt = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Forward Geneformer"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states[config["HIDDEN_LAYER"]]  # [B, L, H]
            h = hidden_states.detach().cpu().numpy()
            ids_np = input_ids.detach().cpu().numpy()
            mask_np = attention_mask.detach().cpu().numpy()

            B, L, H = h.shape
            for bi in range(B):
                for li in range(L):
                    if mask_np[bi, li] == 0:
                        break
                    tid = int(ids_np[bi, li])
                    vec = h[bi, li].astype(np.float64)
                    if tid not in token_to_sum:
                        token_to_sum[tid] = vec
                        token_to_cnt[tid] = 1
                    else:
                        token_to_sum[tid] += vec
                        token_to_cnt[tid] += 1

    # token_id -> gene symbol
    token_to_gene = {int(tid): genes[i] for i, tid in enumerate(token_ids)}

    final_genes = []
    emb_list = []
    for tid, gene in token_to_gene.items():
        if tid in token_to_sum:
            emb = (token_to_sum[tid] / max(token_to_cnt.get(tid, 1), 1)).astype(np.float32)
            final_genes.append(gene)
            emb_list.append(emb)

    embeddings = np.array(emb_list, dtype=np.float32)
    return embeddings, final_genes


def compute_and_save_all_cosine_edges_stream(genes, emb, out_path):
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


# ===========================
# Single dataset
# ===========================
def process_single_dataset(expr_path: str, config: dict) -> dict:
    start_time = datetime.datetime.now()
    dataset_name = extract_dataset_name(expr_path)
    folder_name = Path(expr_path).parent.name

    out_dir = Path(config["OUTPUT_ROOT"]) / folder_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "Dataset_Name": dataset_name,
        "Folder_Name": folder_name,
        "Input_File": str(expr_path),
        "Output_Dir": str(out_dir),
        "Process_Status": "Success",
        "Error_Message": "",
    }

    try:
        # 1) load expr
        df_expr = pd.read_csv(expr_path, index_col=0)
        df_expr.index = df_expr.index.astype(str).str.strip()
        df_expr = df_expr[~df_expr.index.duplicated(keep="first")]
        df_expr = df_expr.fillna(0.0)

        genes = df_expr.index.tolist()
        expr_matrix = df_expr.values  # [n_genes, n_cells]
        n_genes, n_cells = expr_matrix.shape
        print(f"Expression loaded: {n_genes} genes x {n_cells} cells")

        # 2) dicts
        token_dict = load_token_dictionary(config["DICT_DIR"])
        gene_dict = load_gene_name_id_dict(config["DICT_DIR"])

        # 3) match genes
        matched_genes, token_ids, missing_map = match_genes_to_tokens(genes, token_dict, gene_dict)
        print(f"Matched genes: {len(matched_genes)}/{n_genes} (missing={missing_map})")
        if len(matched_genes) == 0:
            raise ValueError("No matched genes after mapping!")

        # filter matrix
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        matched_indices = [gene_to_idx[g] for g in matched_genes]
        expr_matched = expr_matrix[matched_indices, :]

        # 4) cell sampling (optional)
        if config["N_CELLS"] is not None and int(config["N_CELLS"]) > 0 and n_cells > int(config["N_CELLS"]):
            rng = np.random.default_rng(config["SEED"])
            col_idx = rng.choice(n_cells, size=int(config["N_CELLS"]), replace=False)
            expr_use = expr_matched[:, col_idx]
        else:
            col_idx = np.arange(n_cells)
            expr_use = expr_matched

        n_cells_use = expr_use.shape[1]

        # 5) sequences
        sequences = build_cell_sequences(
            expr_use,
            token_ids=token_ids,
            topk=config["SEQ_TOPK_GENES"],
            max_len=config["MAX_SEQ_LEN"],
            use_log1p=config["USE_LOG1P"]
        )

        # 6) model
        print(f"📦 Loading model: {config['MODEL_DIR']}")
        model = BertForMaskedLM.from_pretrained(config["MODEL_DIR"], output_hidden_states=True)

        # 7) extract embeddings (in memory)
        embeddings, final_genes = extract_hidden_embeddings(
            model, sequences, matched_genes, token_ids, config
        )
        print(f"Final (in-memory) embeddings: {len(final_genes)} genes x {embeddings.shape[1]} dim")

        # 8) edges output: MODEL_NAME_DATASET.tsv
        model_name = config.get("MODEL_NAME") or Path(config["MODEL_DIR"]).name
        prefix = f"{_safe(model_name)}_{_safe(dataset_name)}"
        edge_tsv = out_dir / f"{prefix}.tsv"

        if bool(config["SAVE_ALL_EDGES"]):
            cosine_info = compute_and_save_all_cosine_edges_stream(final_genes, embeddings, edge_tsv)
        else:
            cosine_info = compute_and_save_topk_cosine_edges(
                final_genes, embeddings, edge_tsv, topk=int(config["TOPK_PER_GENE"])
            )

        # 9) run params
        run_params = {
            "dataset": dataset_name,
            "folder": folder_name,
            "input_file": str(expr_path),
            "output_dir": str(out_dir),
            "outputs": {"edges_tsv": str(edge_tsv)},
            "model": {
                "model_name": str(model_name),
                "model_dir": config["MODEL_DIR"],
                "hidden_layer": int(config["HIDDEN_LAYER"]),
                "device": str(config["DEVICE"]),
            },
            "dicts": {
                "dict_dir": config["DICT_DIR"],
                "mapping_mode": "gene_name_id_dict.pkl + token_dictionary.pkl",
            },
            "sequence": {
                "seq_topk_genes": int(config["SEQ_TOPK_GENES"]),
                "max_seq_len": int(config["MAX_SEQ_LEN"]),
                "use_log1p": bool(config["USE_LOG1P"]),
                "n_cells": config["N_CELLS"],
            },
            "stats": {
                "n_input_genes": int(n_genes),
                "n_matched_genes": int(len(matched_genes)),
                "n_final_genes_with_embedding": int(len(final_genes)),
                "n_cells_total": int(n_cells),
                "n_cells_used": int(n_cells_use),
                "embedding_dim": int(embeddings.shape[1]),
                "missing_mapping_count": int(missing_map),
            },
            "cosine_export": cosine_info,
            "processing_time_seconds": round((datetime.datetime.now() - start_time).total_seconds(), 2),
        }

        with open(out_dir / "run_params.json", "w") as f:
            json.dump(run_params, f, indent=2, ensure_ascii=False)

        record.update({
            "Edges_TSV": str(edge_tsv),
            "n_edges": int(cosine_info["n_edges"]),
            "Process_Time_Seconds": run_params["processing_time_seconds"],
        })

        print(f"✅ Saved edges TSV: {edge_tsv}")

    except Exception as e:
        traceback.print_exc()
        record["Process_Status"] = "Failed"
        record["Error_Message"] = str(e)[:300]

    return record


# ===========================
# Batch main (aligned folder rules)
# ===========================
def main():
    set_seed(CONFIG["SEED"])

    input_root = Path(CONFIG["INPUT_ROOT"])
    output_root = Path(CONFIG["OUTPUT_ROOT"])
    output_root.mkdir(parents=True, exist_ok=True)

    # folders
    if CONFIG["TARGET_FOLDERS"] is None:
        folders = sorted([p.name for p in input_root.iterdir() if p.is_dir()])
    else:
        folders = CONFIG["TARGET_FOLDERS"]

    print("\n🚀 Geneformer batch (LangCell-aligned IO) start")
    print(f"📥 INPUT_ROOT : {input_root}")
    print(f"📤 OUTPUT_ROOT: {output_root}")
    print(f"⚙️  Folders: {folders}")

    all_records = []

    for folder in folders:
        folder_path = input_root / folder
        if not folder_path.exists():
            print(f"⚠️ Missing folder, skip: {folder_path}")
            continue

        print(f"\n{'='*60}\n📂 Folder: {folder}")

        if folder == "CHIP":
            expr_files = list(folder_path.glob("*_chip_matched-ExpressionData.csv"))
        else:
            expr_files = list(folder_path.glob("*_processed-ExpressionData.csv"))

        print(f"🔍 Found {len(expr_files)} expression files")
        for expr_file in expr_files:
            dataset_name = extract_dataset_name(str(expr_file))
            print(f"🔍 Dataset: {dataset_name}")
            rec = process_single_dataset(str(expr_file), CONFIG)
            all_records.append(rec)

    # summary
    summary_path = output_root / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    ok = sum(r.get("Process_Status") == "Success" for r in all_records)
    fail = len(all_records) - ok

    print(f"\n{'='*60}")
    print(f"✅ Finished. success={ok}, fail={fail}")
    print(f"📊 Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
