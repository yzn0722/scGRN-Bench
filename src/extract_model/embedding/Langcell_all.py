



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

# -------------------------- 运行时注入（禁止硬编码） --------------------------
ARGS = None
input_root = None
output_root = None
gene_emb_df = None

# -------------------------- 预加载模型/字典 --------------------------
def load_langcell_model_and_dict(parent_model_dir: str):
    """预加载LangCell模型和基因字典"""
    # 1. 加载LangCell模型
    saved_model_path = f"{parent_model_dir}/LangCell/cell_bert"
    model = BertModel.from_pretrained(saved_model_path, ignore_mismatched_sizes=True)
    token_emb = model.state_dict()['embeddings.word_embeddings.weight']
    
    # 2. 加载Geneformer字典（复用）
    dict_paths = f"{parent_model_dir}/Geneformer/dicts"  
    with open(os.path.join(dict_paths, "token_dictionary.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(dict_paths, "gene_name_id_dict.pkl"), "rb") as f:
        gene_name_id = pickle.load(f)
    gene_id_name = {v: k for k, v in gene_name_id.items()}
    
    # 3. 预处理LangCell嵌入（移除CLS token）
    token_emb_filtered = token_emb[:-1, :]  
    vocab_keys = list(vocab.keys())[:len(token_emb_filtered)]
    
    gene_emb_df = pd.DataFrame(token_emb_filtered.numpy())
    gene_emb_df["ENSG_ID"] = vocab_keys
    gene_emb_df["Symbol"] = gene_emb_df["ENSG_ID"].apply(lambda x: gene_id_name.get(x, None))
    gene_emb_df = gene_emb_df.dropna(subset=["Symbol"]).reset_index(drop=True)
    
    print(f"✅ LangCell模型加载完成 | 嵌入基因数：{len(gene_emb_df)}")
    return model, gene_emb_df

def parse_args():
    p = argparse.ArgumentParser(description="Export LangCell embedding-weight cosine edges as TSVs.")
    p.add_argument("--input-root", required=True, type=str, help="Input root with CHIP/Non_CHIP/STRING folders.")
    p.add_argument("--output-root", required=True, type=str, help="Output directory for TSV + run records.")
    p.add_argument("--parent-model-dir", required=True, type=str, help="Parent weights directory containing LangCell/ and Geneformer/dicts/.")
    p.add_argument("--folders", nargs="+", default=["CHIP"], help='Folders to process. Default: ["CHIP"].')
    return p.parse_args()

# -------------------------- 工具函数 --------------------------
def extract_dataset_name(file_path):
    """提取纯数据集名"""
    file_basename = os.path.basename(file_path)
    if "_chip_matched-ExpressionData.csv" in file_basename:
        dataset_name = file_basename.split('_chip_matched-ExpressionData.csv')[0]
    elif "_processed-ExpressionData.csv" in file_basename:
        dataset_name = file_basename.split('_processed-ExpressionData.csv')[0]
    else:
        dataset_name = file_basename.split('-ExpressionData.csv')[0]
    return dataset_name

# -------------------------- 单文件对处理函数（输出全量双向边） --------------------------
def process_single_pair(expr_path, network_path):
    # 1. 提取纯数据集名称
    dataset_name = extract_dataset_name(expr_path)
    print(f"\n=====================================")
    print(f"🔍 处理数据集：{dataset_name}")
    print(f"   Expr文件：{expr_path}")

    # 2. 读取参考基因集
    try:
        data_df = pd.read_csv(expr_path)
        gene_symbols = data_df.iloc[1:, 0].tolist()
        gene_set_df = pd.DataFrame({"Symbol": gene_symbols}).dropna()
        input_genes_count = len(gene_set_df)
        print(f"✅ 参考基因集：{input_genes_count}个基因")
    except Exception as e:
        print(f"⚠️ 读取Expr文件失败：{e} → 跳过该文件对")
        return

    # 3. 匹配LangCell嵌入
    gene_emb_filtered = pd.merge(gene_set_df, gene_emb_df, on="Symbol", how="inner")
    matched_genes_count = len(gene_emb_filtered)
    if matched_genes_count == 0:
        print(f"⚠️ 基因无匹配 → 跳过")
        return
    print(f"✅ 匹配到的基因：{matched_genes_count}个")

    # 4. 计算余弦相似度（输出全量双向边）
    gene_names = gene_emb_filtered["Symbol"].tolist()
    embedding_cols = [col for col in gene_emb_filtered.columns if col not in ["Symbol", "ENSG_ID"]]
    embedding_matrix = gene_emb_filtered[embedding_cols].values

    # 计算相似度矩阵（无向，对称）
    similarity_matrix = cosine_similarity(embedding_matrix)

    # 核心修改：遍历所有i≠j，生成双向边
    edge_weights = []
    n_genes = len(gene_names)
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:  # 仅过滤自环
                continue
            edge_weights.append({
                'Gene1': gene_names[i],
                'Gene2': gene_names[j],
                'EdgeWeight': round(float(similarity_matrix[i, j]), 15)
            })
    
    total_edges = len(edge_weights)
    if total_edges == 0:
        print(f"⚠️ 无有效基因对 → 跳过保存")
        return

    # 5. 保存TSV：模型_数据集.tsv
    tsv_filename = f"LangCell_{dataset_name}.tsv"
    tsv_path = os.path.join(output_root, tsv_filename)
    
    # 按EdgeWeight降序排序（保留所有边）
    edges_df = pd.DataFrame(edge_weights)
    edges_df = edges_df.sort_values(by="EdgeWeight", ascending=False)
    edges_df.to_csv(tsv_path, sep='\t', index=False)
    
    # 统计信息
    avg_weight = np.mean([ew['EdgeWeight'] for ew in edge_weights])
    print(f"\n📈 结果统计：")
    print(f"   总生成边（双向）：{total_edges} | 平均权重：{avg_weight:.4f}")
    print(f"✅ TSV已保存：{tsv_path}")

    # 6. 记录运行信息
    record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型": "LangCell",
        "数据集": dataset_name,
        "输入基因数": input_genes_count,
        "匹配基因数": matched_genes_count,
        "总双向边数": total_edges,
        "TSV路径": tsv_path
    }
    record_csv = os.path.join(output_root, "LangCell运行记录.csv")
    pd.DataFrame([record]).to_csv(record_csv, mode='a', header=not os.path.exists(record_csv), index=False)

# -------------------------- 批量遍历主函数 --------------------------
def main():
    global ARGS, input_root, output_root, gene_emb_df
    ARGS = parse_args()
    input_root = ARGS.input_root
    output_root = ARGS.output_root
    os.makedirs(output_root, exist_ok=True)

    _, gene_emb_df = load_langcell_model_and_dict(parent_model_dir=ARGS.parent_model_dir)

    print(f"\n🚀 开始LangCell批量处理流程（输出全量双向边）")
    print(f"输入根目录：{input_root}")
    print(f"输出根目录：{output_root}")
    
    target_folders = ARGS.folders
    for folder in target_folders:
        folder_path = os.path.join(input_root, folder)
        if not os.path.exists(folder_path):
            print(f"\n⚠️ 文件夹不存在，跳过：{folder_path}")
            continue
        
        print(f"\n=====================================")
        print(f"📂 扫描文件夹：{folder}")
        
        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
        if not expr_files:
            print(f"⚠️ 未找到ExpressionData.csv文件，跳过")
            continue
        
        for expr_file in expr_files:
            expr_path = os.path.join(folder_path, expr_file)
            network_file = expr_file.replace('-ExpressionData.csv', '-network.csv')
            network_path = os.path.join(folder_path, network_file)
            
            if not os.path.exists(network_path):
                print(f"⚠️ 缺失Network文件：{network_path} → 跳过{expr_file}")
                continue
            
            process_single_pair(expr_path, network_path)

if __name__ == "__main__":
    main()
    print("\n=====================================")
    print(f"🎉 所有文件处理完成！结果保存在：{output_root}")