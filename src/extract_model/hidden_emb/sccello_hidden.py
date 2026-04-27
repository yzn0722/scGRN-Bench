# import os 
# import sys
# import numpy as np
# import pandas as pd
# import torch
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# from tqdm import tqdm
# import json

# # -------------------------- 全局配置 --------------------------
# INPUT_ROOT = "/mnt/10T/yzn/benchmark_GRN/input_process"
# OUTPUT_ROOT = "/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/sccello_hidden"

# MODEL_NAME = "scCello"
# MODEL_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/scCello"
# DICT_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts"

# # Hidden state 配置
# CONFIG = {
#     "HIDDEN_LAYER": -1,        # 提取最后一层 hidden states
#     "SEQ_TOPK_GENES": 256,     # 每个细胞选择 top-K 高表达基因
#     "MAX_SEQ_LEN": 256,        # 最大序列长度
#     "N_CELLS": None,           # None = 使用所有细胞
#     "USE_LOG1P": False,         # 是否 log1p 归一化
#     "BATCH_SIZE": 16,          # 批处理大小
#     "SAVE_ALL_EDGES": True,    # 保存所有边
# }

# # 添加依赖路径
# sys.path.insert(0, "/mnt/10T/yzn/benchmark_GRN")
# sys.path.insert(0, "/mnt/10T/yzn/benchmark_GRN/sc_foundation_evals")

# from sc_foundation_evals.sccello.src.model_prototype_contrastive import PrototypeContrastiveModel

# # 创建输出目录
# os.makedirs(OUTPUT_ROOT, exist_ok=True)
# print(f"✅ 初始化完成 | 输出根目录：{OUTPUT_ROOT}")

# # -------------------------- 加载模型和字典 --------------------------
# def load_sccello_resources():
#     """加载 scCello 模型和基因字典"""
#     # 1. 加载模型
#     print(f"📦 加载 scCello 模型: {MODEL_DIR}")
#     model = PrototypeContrastiveModel.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
#     model.eval()  # 设置为评估模式
    
#     # 2. 加载字典
#     with open(os.path.join(DICT_DIR, "token_dictionary.pkl"), "rb") as f:
#         token_dict = pickle.load(f)
#     with open(os.path.join(DICT_DIR, "gene_name_id_dict.pkl"), "rb") as f:
#         gene_name_id = pickle.load(f)
    
#     gene_id_name = {v: k for k, v in gene_name_id.items()}
    
#     print(f"✅ Token dictionary: {len(token_dict)} tokens")
#     print(f"✅ Gene name dictionary: {len(gene_name_id)} symbols")
    
#     return model, token_dict, gene_name_id, gene_id_name

# # 全局加载
# MODEL, TOKEN_DICT, GENE_NAME_ID, GENE_ID_NAME = load_sccello_resources()

# # -------------------------- 工具函数 --------------------------
# def extract_dataset_name(file_path):
#     """提取纯数据集名（移除后缀）"""
#     file_basename = os.path.basename(file_path)
#     if "_chip_matched-ExpressionData.csv" in file_basename:
#         dataset_name = file_basename.split('_chip_matched-ExpressionData.csv')[0]
#     elif "_processed-ExpressionData.csv" in file_basename:
#         dataset_name = file_basename.split('_processed-ExpressionData.csv')[0]
#     else:
#         dataset_name = file_basename.split('-ExpressionData.csv')[0]
#     return dataset_name

# # -------------------------- 序列构建 --------------------------
# def build_gene_sequences(expr_df, token_dict, gene_name_id, config):
#     """
#     根据表达矩阵构建基因序列（类似 Geneformer）
    
#     Returns:
#         sequences: List[List[int]] - 每个细胞的 token id 序列
#         gene_symbols: List[str] - 所有出现的基因 symbol
#         gene_to_token: Dict[str, int] - 基因 symbol 到 token id 的映射
#     """
#     # 1. 提取基因和细胞
#     gene_symbols = expr_df.iloc[1:, 0].tolist()  # 第一列是基因名
#     expr_matrix = expr_df.iloc[1:, 1:].values.astype(float)  # 表达矩阵
#     n_genes, n_cells = expr_matrix.shape
    
#     print(f"Expression matrix: {n_genes} genes x {n_cells} cells")
    
#     # 2. 匹配基因到 token
#     gene_to_token = {}
#     valid_gene_idx = []
    
#     for i, symbol in enumerate(gene_symbols):
#         if symbol in gene_name_id:
#             ensg_id = gene_name_id[symbol]
#             if ensg_id in token_dict:
#                 gene_to_token[symbol] = token_dict[ensg_id]
#                 valid_gene_idx.append(i)
    
#     print(f"✅ 匹配到的基因: {len(valid_gene_idx)}/{n_genes}")
    
#     # 3. 过滤表达矩阵
#     expr_matrix_filtered = expr_matrix[valid_gene_idx, :]
#     valid_gene_symbols = [gene_symbols[i] for i in valid_gene_idx]
    
#     # 4. 归一化（可选）
#     if config["USE_LOG1P"]:
#         expr_matrix_filtered = np.log1p(expr_matrix_filtered)
    
#     # 5. 限制细胞数
#     if config["N_CELLS"] is not None and config["N_CELLS"] < n_cells:
#         cell_indices = np.random.choice(n_cells, config["N_CELLS"], replace=False)
#         expr_matrix_filtered = expr_matrix_filtered[:, cell_indices]
#         n_cells_use = config["N_CELLS"]
#     else:
#         n_cells_use = n_cells
    
#     print(f"使用细胞数: {n_cells_use}")
    
#     # 6. 构建序列（每个细胞选 top-K 高表达基因）
#     sequences = []
#     topk = min(config["SEQ_TOPK_GENES"], len(valid_gene_symbols))
    
#     for cell_idx in range(n_cells_use):
#         expr_values = expr_matrix_filtered[:, cell_idx]
#         topk_indices = np.argsort(expr_values)[-topk:][::-1]  # 降序
        
#         # 构建 token id 序列
#         seq = [gene_to_token[valid_gene_symbols[i]] for i in topk_indices]
#         sequences.append(seq[:config["MAX_SEQ_LEN"]])
    
#     return sequences, valid_gene_symbols, gene_to_token

# # -------------------------- 提取 Hidden States --------------------------
# def extract_hidden_embeddings(model, sequences, gene_symbols, gene_to_token, config):
#     """
#     通过前向传播提取 hidden state embeddings
    
#     Returns:
#         gene_embeddings: np.ndarray [n_genes, hidden_dim]
#         final_gene_list: List[str]
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     hidden_dim = model.config.hidden_size
#     batch_size = config["BATCH_SIZE"]
#     n_batches = (len(sequences) + batch_size - 1) // batch_size
    
#     # 累积每个基因的 hidden states
#     gene_hidden_accumulator = {symbol: [] for symbol in gene_symbols}
    
#     print(f"Forward scCello (batch_size={batch_size}):")
    
#     with torch.no_grad():
#         for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
#             # 准备批次数据
#             start_idx = batch_idx * batch_size
#             end_idx = min(start_idx + batch_size, len(sequences))
#             batch_seqs = sequences[start_idx:end_idx]
            
#             # Padding
#             max_len = max(len(seq) for seq in batch_seqs)
#             input_ids = []
#             attention_mask = []
            
#             for seq in batch_seqs:
#                 padded_seq = seq + [0] * (max_len - len(seq))
#                 mask = [1] * len(seq) + [0] * (max_len - len(seq))
#                 input_ids.append(padded_seq)
#                 attention_mask.append(mask)
            
#             input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
#             attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
            
#             # 前向传播
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 output_hidden_states=True
#             )
            
#             # 提取指定层的 hidden states
#             hidden_states = outputs.hidden_states[config["HIDDEN_LAYER"]]
#             # shape: [batch_size, seq_len, hidden_dim]
            
#             # 累积每个基因的 hidden state
#             for i, seq in enumerate(batch_seqs):
#                 for j, token_id in enumerate(seq):
#                     # 找到对应的基因 symbol
#                     symbol = None
#                     for s, tid in gene_to_token.items():
#                         if tid == token_id:
#                             symbol = s
#                             break
                    
#                     if symbol:
#                         hidden_vec = hidden_states[i, j, :].cpu().numpy()
#                         gene_hidden_accumulator[symbol].append(hidden_vec)
    
#     # 对每个基因求平均
#     gene_embeddings = []
#     final_gene_list = []
    
#     for symbol in gene_symbols:
#         if len(gene_hidden_accumulator[symbol]) > 0:
#             avg_hidden = np.mean(gene_hidden_accumulator[symbol], axis=0)
#             gene_embeddings.append(avg_hidden)
#             final_gene_list.append(symbol)
    
#     gene_embeddings = np.array(gene_embeddings)
    
#     print(f"✅ 最终 hidden embeddings: {len(final_gene_list)} genes x {hidden_dim} dim")
    
#     return gene_embeddings, final_gene_list

# # -------------------------- 计算并保存所有余弦边 --------------------------
# def compute_and_save_all_cosine_edges(gene_list, embeddings, output_tsv):
#     """
#     计算所有基因对的余弦相似度并保存（去除自环）
    
#     Returns:
#         dict: 边的统计信息
#     """
#     n_genes = len(gene_list)
    
#     # 计算余弦相似度矩阵
#     print(f"计算 {n_genes}x{n_genes} 余弦相似度矩阵...")
#     cosine_matrix = cosine_similarity(embeddings)
    
#     # 构建边列表（去除自环）
#     edges = []
#     for i in range(n_genes):
#         for j in range(n_genes):
#             if i != j:  # ⭐ 去除自环
#                 edges.append({
#                     'Gene1': gene_list[i],
#                     'Gene2': gene_list[j],
#                     'EdgeWeight': round(float(cosine_matrix[i, j]), 15)
#                 })
    
#     # 保存为 TSV
#     edges_df = pd.DataFrame(edges)
#     edges_df = edges_df.sort_values(by='EdgeWeight', ascending=False)
#     edges_df.to_csv(output_tsv, sep='\t', index=False)
    
#     print(f"✅ 保存所有边: {len(edges)} 条 (去除 {n_genes} 个自环)")
#     print(f"   文件: {output_tsv}")
    
#     return {
#         "total_edges": len(edges),
#         "n_genes": n_genes,
#         "self_loops_removed": n_genes
#     }

# # -------------------------- 主处理函数 --------------------------
# def process_single_dataset(expr_path, config):
#     """处理单个数据集"""
#     start_time = datetime.now()
#     dataset_name = extract_dataset_name(expr_path)
    
#     print(f"\n{'='*60}")
#     print(f"🔍 处理数据集：{dataset_name}")
#     print(f"   Expr文件：{expr_path}")
    
#     # 初始化记录
#     record = {
#         "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
#         "Model_Name": MODEL_NAME,
#         "Dataset_Name": dataset_name,
#         "Input_File": expr_path,
#         "Input_Genes_Count": 0,
#         "Matched_Genes_Count": 0,
#         "Final_Genes_Count": 0,
#         "Total_Edges_Generated": 0,
#         "Output_TSV_Path": "",
#         "Process_Status": "Success",
#         "Process_Time_Seconds": 0.0
#     }
    
#     try:
#         # 1. 读取表达矩阵
#         expr_df = pd.read_csv(expr_path)
#         n_input_genes = len(expr_df) - 1  # 第一行是 header
#         record["Input_Genes_Count"] = n_input_genes
        
#         # 2. 构建序列
#         sequences, gene_symbols, gene_to_token = build_gene_sequences(
#             expr_df, TOKEN_DICT, GENE_NAME_ID, config
#         )
#         record["Matched_Genes_Count"] = len(gene_symbols)
        
#         # 3. 提取 hidden embeddings
#         gene_embeddings, final_gene_list = extract_hidden_embeddings(
#             MODEL, sequences, gene_symbols, gene_to_token, config
#         )
#         record["Final_Genes_Count"] = len(final_gene_list)
        
#         # 4. 保存 embeddings
#         dataset_output_dir = os.path.join(OUTPUT_ROOT, dataset_name)
#         os.makedirs(dataset_output_dir, exist_ok=True)
        
#         emb_tsv = os.path.join(dataset_output_dir, f"{MODEL_NAME}_hidden_gene_embedding.tsv")
#         emb_df = pd.DataFrame(gene_embeddings, index=final_gene_list)
#         emb_df.index.name = "Gene"
#         emb_df.to_csv(emb_tsv, sep='\t')
#         print(f"✅ Embeddings 保存: {emb_tsv}")
        
#         # 5. 计算并保存所有余弦边
#         cosine_tsv = os.path.join(dataset_output_dir, f"{MODEL_NAME}_{dataset_name}_all_edges.tsv")
#         cosine_info = compute_and_save_all_cosine_edges(final_gene_list, gene_embeddings, cosine_tsv)
#         record["Total_Edges_Generated"] = cosine_info["total_edges"]
#         record["Output_TSV_Path"] = cosine_tsv
        
#         # 6. 保存元数据
#         meta = {
#             "dataset": dataset_name,
#             "model": MODEL_NAME,
#             "n_genes": len(final_gene_list),
#             "embedding_dim": gene_embeddings.shape[1],
#             "hidden_layer": config["HIDDEN_LAYER"],
#             "seq_topk": config["SEQ_TOPK_GENES"],
#             "n_cells_used": len(sequences),
#             "config": config
#         }
#         meta_path = os.path.join(dataset_output_dir, "embedding_meta.json")
#         with open(meta_path, "w") as f:
#             json.dump(meta, f, indent=2)
        
#         # 7. 更新耗时
#         record["Process_Time_Seconds"] = round((datetime.now() - start_time).total_seconds(), 2)
#         print(f"✅ 处理完成 | 耗时: {record['Process_Time_Seconds']}秒")
        
#     except Exception as e:
#         print(f"❌ 处理失败: {e}")
#         record["Process_Status"] = f"Failed: {str(e)[:100]}"
#         import traceback
#         traceback.print_exc()
    
#     return record

# # -------------------------- 批量处理 --------------------------
# def main():
#     print(f"\n🚀 开始 scCello Hidden States 批量处理流程")
#     print(f"输入根目录: {INPUT_ROOT}")
#     print(f"输出根目录: {OUTPUT_ROOT}")
    
#     # 创建输出根目录
#     os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
#     target_folders = ["CHIP", "Non_CHIP", "STRING"]
#     all_records = []
    
#     for folder in target_folders:
#         folder_path = os.path.join(INPUT_ROOT, folder)
#         if not os.path.exists(folder_path):
#             print(f"\n⚠️ 文件夹不存在，跳过: {folder_path}")
#             continue
        
#         print(f"\n{'='*60}")
#         print(f"📂 处理文件夹: {folder}")
        
#         # 找到所有 ExpressionData 文件
#         expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
#         print(f"找到 {len(expr_files)} 个文件")
        
#         for expr_file in expr_files:
#             expr_path = os.path.join(folder_path, expr_file)
#             record = process_single_dataset(expr_path, CONFIG)
#             all_records.append(record)
    
#     # 保存汇总记录
#     if all_records:
#         summary_df = pd.DataFrame(all_records)
#         summary_path = os.path.join(OUTPUT_ROOT, f"{MODEL_NAME}_hidden_processing_summary.csv")
#         summary_df.to_csv(summary_path, index=False)
#         print(f"\n📊 汇总记录保存: {summary_path}")
    
#     print(f"\n{'='*60}")
#     print(f"🎉 所有处理完成！结果保存在: {OUTPUT_ROOT}")

# if __name__ == "__main__":
#     main()









#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import traceback
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -------------------------- 全局配置 --------------------------
INPUT_ROOT = ""
OUTPUT_ROOT = ""

MODEL_NAME = "scCello"
MODEL_DIR = ""
DICT_DIR = ""

CONFIG = {
    "HIDDEN_LAYER": -1,        # 提取最后一层 hidden states
    "SEQ_TOPK_GENES": 512,     # 每个细胞选择 top-K 高表达基因
    "MAX_SEQ_LEN": 512,        # 最大序列长度
    "N_CELLS": None,           # None = 使用所有细胞
    "USE_LOG1P": False,        # 是否 log1p 归一化
    "BATCH_SIZE": 16,          # 批处理大小
    "SAVE_ALL_EDGES": True,    # True: 输出所有 i!=j 边（流式写）
    "TOPK_PER_GENE": 1000,     # SAVE_ALL_EDGES=False 时启用
    "SEED": 42,
}

def parse_args():
    p = argparse.ArgumentParser(description="Extract scCello hidden embeddings and export cosine-edge TSVs.")
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--dict-dir", required=True, type=str)
    p.add_argument("--repo-root", default="", type=str, help="Optional repo root to append to PYTHONPATH for sc_foundation_evals.")
    p.add_argument("--folders", nargs="+", default=["CHIP", "Non_CHIP", "STRING"])
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


ARGS = parse_args()
INPUT_ROOT = ARGS.input_root
OUTPUT_ROOT = ARGS.output_root
MODEL_DIR = ARGS.model_dir
DICT_DIR = ARGS.dict_dir
CONFIG["BATCH_SIZE"] = int(ARGS.batch_size)
CONFIG["SEED"] = int(ARGS.seed)

if ARGS.repo_root:
    sys.path.insert(0, ARGS.repo_root)
    sys.path.insert(0, str(Path(ARGS.repo_root) / "sc_foundation_evals"))

from sc_foundation_evals.sccello.src.model_prototype_contrastive import PrototypeContrastiveModel

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"✅ 初始化完成 | 输出根目录：{OUTPUT_ROOT}")


# -------------------------- 工具函数 --------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))


def extract_dataset_name(file_path: str) -> str:
    file_basename = os.path.basename(file_path)
    if "_chip_matched-ExpressionData.csv" in file_basename:
        return file_basename.split("_chip_matched-ExpressionData.csv")[0]
    if "_processed-ExpressionData.csv" in file_basename:
        return file_basename.split("_processed-ExpressionData.csv")[0]
    return file_basename.split("-ExpressionData.csv")[0]


def read_expression_matrix(path: str) -> pd.DataFrame:
    """
    统一读取：行=gene symbol，列=cell/sample
    兼容 tsv/csv/不确定分隔符
    """
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

    # 转数值
    if df.select_dtypes(include=[np.number]).shape[1] == 0:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty expression matrix after parsing: {df.shape}")

    print(f"Expression loaded: {df.shape[0]} genes x {df.shape[1]} cells")
    return df


def get_pad_id(token_dict: dict) -> int:
    for k in ["<pad>", "[PAD]", "pad", "PAD"]:
        if k in token_dict:
            return int(token_dict[k])
    # scCello/Geneformer 通常 pad_id=0
    return 0


def _get_hidden(outputs, layer_idx: int):
    """
    兼容 outputs.hidden_states / outputs.last_hidden_state
    """
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[layer_idx]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    raise RuntimeError("Model outputs have no hidden states.")


# -------------------------- 加载模型和字典（全局一次） --------------------------
def load_sccello_resources():
    print(f"📦 加载 scCello 模型: {MODEL_DIR}")
    model = PrototypeContrastiveModel.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
    model.eval()

    with open(os.path.join(DICT_DIR, "token_dictionary.pkl"), "rb") as f:
        token_dict = pickle.load(f)
    with open(os.path.join(DICT_DIR, "gene_name_id_dict.pkl"), "rb") as f:
        gene_name_id = pickle.load(f)

    print(f"✅ Token dictionary: {len(token_dict)} tokens")
    print(f"✅ Gene name dictionary: {len(gene_name_id)} symbols")
    return model, token_dict, gene_name_id


MODEL, TOKEN_DICT, GENE_NAME_ID = load_sccello_resources()


# -------------------------- 序列构建（Geneformer-aligned mapping） --------------------------
def build_gene_sequences(expr_df: pd.DataFrame, token_dict: dict, gene_name_id: dict, config: dict):
    """
    expr_df: index=Symbol, columns=cells
    返回：
      sequences: List[List[int]]
      token_id_to_symbol: Dict[int, str]  (用于快速反查 token->symbol)
      gene_to_token_id: Dict[str, int]
      used_cells: int
      used_genes_for_seq: int (可映射基因数)
    """
    X = expr_df.values.astype(np.float32)  # [G, C]
    genes = expr_df.index.astype(str).tolist()
    G, C = X.shape

    if config["USE_LOG1P"]:
        X = np.log1p(np.maximum(X, 0.0))

    # Symbol -> token_id（和 Geneformer 逻辑一致：Symbol->ENSG->token）
    gene_to_token_id = {}
    token_id_to_symbol = {}
    mappable_idx = []

    for i, sym in enumerate(genes):
        # direct match (如果 sym 本身就是 token key，比如 ENSG)
        if sym in token_dict:
            tid = int(token_dict[sym])
            gene_to_token_id[sym] = tid
            token_id_to_symbol.setdefault(tid, sym)
            mappable_idx.append(i)
            continue

        ensg = gene_name_id.get(sym, None)
        if ensg is not None and ensg in token_dict:
            tid = int(token_dict[ensg])
            gene_to_token_id[sym] = tid
            token_id_to_symbol.setdefault(tid, sym)
            mappable_idx.append(i)

    if len(mappable_idx) == 0:
        raise RuntimeError("0 mappable genes after mapping (gene_name_id_dict + token_dictionary).")

    X_map = X[mappable_idx, :]
    genes_map = [genes[i] for i in mappable_idx]

    # cell sampling
    if config["N_CELLS"] is not None and int(config["N_CELLS"]) > 0 and C > int(config["N_CELLS"]):
        col_idx = np.random.choice(C, size=int(config["N_CELLS"]), replace=False)
    else:
        col_idx = np.arange(C)

    topk = min(int(config["SEQ_TOPK_GENES"]), len(genes_map))
    max_len = int(config["MAX_SEQ_LEN"])
    pad_id = get_pad_id(token_dict)

    sequences = []
    for ci in tqdm(col_idx, desc="Build sequences", leave=False):
        col = X_map[:, ci]
        k = min(topk, len(col))
        top_i = np.argpartition(-col, kth=k - 1)[:k]
        top_i = top_i[np.argsort(-col[top_i])]  # desc
        seq = [gene_to_token_id[genes_map[j]] for j in top_i]
        seq = seq[:max_len]
        sequences.append(seq)

    return sequences, token_id_to_symbol, gene_to_token_id, int(len(col_idx)), int(len(mappable_idx)), pad_id


# -------------------------- Hidden 提取（高效聚合） --------------------------
def extract_hidden_embeddings(model, sequences, token_id_to_symbol, pad_id: int, config: dict):
    """
    返回：
      emb: np.ndarray [n_final_genes, hidden_dim]
      final_genes: List[str]  (与 emb 行对应)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    bs = int(config["BATCH_SIZE"])
    layer_idx = int(config["HIDDEN_LAYER"])

    sum_vec = {}
    cnt = {}

    n_batches = (len(sequences) + bs - 1) // bs
    print(f"Forward scCello (batch_size={bs}, n_batches={n_batches})")

    with torch.no_grad():
        for b in tqdm(range(n_batches), desc="Forward scCello"):
            s = b * bs
            e = min(s + bs, len(sequences))
            batch_seqs = sequences[s:e]

            max_len = max(len(x) for x in batch_seqs)
            input_ids = []
            attention_mask = []
            for seq in batch_seqs:
                pad_len = max_len - len(seq)
                input_ids.append(seq + [pad_id] * pad_len)
                attention_mask.append([1] * len(seq) + [0] * pad_len)

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            h = _get_hidden(outputs, layer_idx)  # [B, L, H]
            h = h.detach().cpu().numpy()
            ids_np = input_ids.detach().cpu().numpy()
            mask_np = attention_mask.detach().cpu().numpy()

            B, L, Hdim = h.shape
            for bi in range(B):
                for li in range(L):
                    if mask_np[bi, li] == 0:
                        break
                    tid = int(ids_np[bi, li])
                    if tid == pad_id:
                        continue
                    if tid not in sum_vec:
                        sum_vec[tid] = h[bi, li].astype(np.float64)
                        cnt[tid] = 1
                    else:
                        sum_vec[tid] += h[bi, li].astype(np.float64)
                        cnt[tid] += 1

    # token_id -> symbol -> embedding
    gene_emb = {}
    for tid, vec in sum_vec.items():
        sym = token_id_to_symbol.get(tid, None)
        if sym is None:
            continue
        gene_emb[sym] = (vec / max(cnt.get(tid, 1), 1)).astype(np.float32)

    final_genes = sorted(gene_emb.keys())
    emb = np.stack([gene_emb[g] for g in final_genes], axis=0)
    return emb, final_genes


# -------------------------- 边输出：流式写（不爆内存） --------------------------
def compute_and_save_all_cosine_edges_stream(genes, emb, out_path: str):
    """
    输出所有 i!=j 的有向边（流式写），列：Gene1 Gene2 EdgeWeight
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


def compute_and_save_topk_cosine_edges(genes, emb, out_path: str, topk: int):
    """
    输出每个 Gene1 的 topK（有向）
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


# -------------------------- 单数据集流程（对齐 IO） --------------------------
def process_single_dataset(expr_path: str, folder_name: str, config: dict):
    start_time = datetime.now()
    dataset_name = extract_dataset_name(expr_path)

    out_dir = Path(OUTPUT_ROOT) / folder_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model_Name": MODEL_NAME,
        "Folder_Name": folder_name,
        "Dataset_Name": dataset_name,
        "Input_File": str(expr_path),
        "Process_Status": "Success",
        "Error_Message": "",
        "Process_Time_Seconds": 0.0,
        "Input_Genes_Count": 0,
        "Mappable_Genes_Count": 0,
        "Final_Genes_Count": 0,
        "Embedding_Dim": 0,
        "Total_Edges_Generated": 0,
        "Output_TSV_Path": "",
    }

    try:
        print(f"\n{'='*60}")
        print(f"🔍 处理数据集：{dataset_name} | Folder: {folder_name}")
        print(f"   Expr文件：{expr_path}")

        # 1) 读取表达矩阵（统一：index=gene, columns=cells）
        expr_df = read_expression_matrix(expr_path)
        record["Input_Genes_Count"] = int(expr_df.shape[0])

        # 2) 构建序列（Geneformer-aligned mapping）
        sequences, token_id_to_symbol, gene_to_token_id, used_cells, used_genes, pad_id = build_gene_sequences(
            expr_df, TOKEN_DICT, GENE_NAME_ID, config
        )
        record["Mappable_Genes_Count"] = int(used_genes)

        # 3) hidden embeddings（内存）
        emb, final_genes = extract_hidden_embeddings(
            MODEL, sequences, token_id_to_symbol, pad_id, config
        )
        record["Final_Genes_Count"] = int(len(final_genes))
        record["Embedding_Dim"] = int(emb.shape[1])

        # 4) 只输出边文件：{MODEL_NAME}_{DATASET}.tsv
        prefix = f"{_safe(MODEL_NAME)}_{_safe(dataset_name)}"
        edge_tsv = out_dir / f"{prefix}.tsv"

        if bool(config["SAVE_ALL_EDGES"]):
            cosine_info = compute_and_save_all_cosine_edges_stream(final_genes, emb, str(edge_tsv))
        else:
            cosine_info = compute_and_save_topk_cosine_edges(
                final_genes, emb, str(edge_tsv), topk=int(config["TOPK_PER_GENE"])
            )

        record["Total_Edges_Generated"] = int(cosine_info["n_edges"])
        record["Output_TSV_Path"] = str(edge_tsv)

        # 5) 保存关键参数 run_params.json
        model_hidden = int(getattr(getattr(MODEL, "config", None), "hidden_size", emb.shape[1]))
        run_params = {
            "dataset": dataset_name,
            "folder": folder_name,
            "input_file": str(expr_path),
            "output_dir": str(out_dir),
            "outputs": {"edges_tsv": str(edge_tsv)},
            "model": {
                "model_name": MODEL_NAME,
                "model_dir": MODEL_DIR,
                "hidden_layer": int(config["HIDDEN_LAYER"]),
                "hidden_size": model_hidden,
            },
            "dicts": {
                "dict_dir": DICT_DIR,
                "mapping_mode": "gene_name_id_dict.pkl + token_dictionary.pkl",
                "pad_id": int(pad_id),
            },
            "sequence": {
                "seq_topk_genes": int(config["SEQ_TOPK_GENES"]),
                "max_seq_len": int(config["MAX_SEQ_LEN"]),
                "use_log1p": bool(config["USE_LOG1P"]),
                "n_cells": config["N_CELLS"],
                "n_cells_used": int(used_cells),
            },
            "stats": {
                "n_input_genes": int(expr_df.shape[0]),
                "n_mappable_genes": int(used_genes),
                "n_final_genes_with_embedding": int(len(final_genes)),
                "embedding_dim": int(emb.shape[1]),
            },
            "cosine_export": cosine_info,
            "processing_time_seconds": round((datetime.now() - start_time).total_seconds(), 2),
        }

        with open(out_dir / "run_params.json", "w") as f:
            json.dump(run_params, f, indent=2, ensure_ascii=False)

        record["Process_Time_Seconds"] = run_params["processing_time_seconds"]
        print(f"✅ 完成 | edges: {edge_tsv} | time: {record['Process_Time_Seconds']}s")

    except Exception as e:
        traceback.print_exc()
        record["Process_Status"] = "Failed"
        record["Error_Message"] = str(e)[:300]
        print(f"❌ 处理失败: {dataset_name} | {record['Error_Message']}")

    return record


# -------------------------- 批量处理（对齐 IO） --------------------------
def main():
    set_seed(CONFIG["SEED"])

    print(f"\n🚀 开始 {MODEL_NAME} Hidden->Edges 批量处理流程")
    print(f"📥 输入根目录: {INPUT_ROOT}")
    print(f"📤 输出根目录: {OUTPUT_ROOT}")

    input_root = Path(INPUT_ROOT)
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    target_folders = ARGS.folders
    all_records = []

    for folder in target_folders:
        folder_path = input_root / folder
        if not folder_path.exists():
            print(f"\n⚠️ 文件夹不存在，跳过: {folder_path}")
            continue

        print(f"\n{'='*60}")
        print(f"📂 处理文件夹: {folder}")

        # 对齐你前面规则：CHIP / others 的文件名模式不同
        if folder == "CHIP":
            expr_files = list(folder_path.glob("*_chip_matched-ExpressionData.csv"))
        else:
            expr_files = list(folder_path.glob("*_processed-ExpressionData.csv"))

        print(f"🔍 找到 {len(expr_files)} 个表达矩阵文件")

        for expr_file in expr_files:
            rec = process_single_dataset(str(expr_file), folder, CONFIG)
            all_records.append(rec)

    # 汇总
    if all_records:
        summary_df = pd.DataFrame(all_records)
        summary_path = out_root / f"{MODEL_NAME}_processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        ok = sum(r["Process_Status"] == "Success" for r in all_records)
        fail = len(all_records) - ok
        print(f"\n📊 汇总记录保存: {summary_path}")
        print(f"📈 处理统计: 成功 {ok} / 失败 {fail}")

    print(f"\n{'='*60}")
    print(f"🎉 所有处理完成！结果保存在: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
