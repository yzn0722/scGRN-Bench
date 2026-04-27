




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

# -------------------------- 运行时注入（禁止硬编码） --------------------------
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

# -------------------------- 工具函数 --------------------------
def extract_dataset_name(file_path):
    """提取纯数据集名（移除CHIP/Non_CHIP/STRING后缀）"""
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

    # 2. 读取ExpressionData
    try:
        data_df = pd.read_csv(expr_path)
        gene_symbols = data_df.iloc[1:, 0].tolist()
        gene_set_df = pd.DataFrame({"Symbol": gene_symbols}).dropna()
        input_genes_count = len(gene_set_df)
        print(f"✅ 参考基因集：{input_genes_count}个基因")
    except Exception as e:
        print(f"⚠️ 读取Expr文件失败：{e} → 跳过该文件对")
        return

    # 3. 匹配Geneformer嵌入
    gene_emb_filtered = pd.merge(gene_set_df, gene_emb_df, on="Symbol", how="inner")
    matched_genes_count = len(gene_emb_filtered)
    if matched_genes_count == 0:
        print(f"⚠️ 基因无匹配 → 跳过")
        return
    print(f"✅ 匹配到的基因：{matched_genes_count}个")

    # 4. 计算余弦相似度（输出全量双向边）
    def save_similarity_tsv():
        # 提取嵌入矩阵
        embedding_cols = [col for col in gene_emb_filtered.columns if col not in ["Symbol", "ENSG_ID"]]
        embedding_matrix = gene_emb_filtered[embedding_cols].values
        emb_dim = len(embedding_cols)

        if embedding_matrix.size == 0:
            print(f"⚠️ 嵌入矩阵为空 → 跳过保存")
            return None
        
        # 计算相似度矩阵（无向，对称）
        similarity_matrix = cosine_similarity(embedding_matrix)
        gene_names = gene_emb_filtered["Symbol"].tolist()
        
        # 核心修改：遍历所有i≠j的组合，生成双向边（A→B + B→A）
        edge_weights = []
        n_genes = len(gene_names)
        for i in range(n_genes):
            for j in range(n_genes):
                if i == j:  # 仅过滤自环，保留所有双向边
                    continue
                edge_weights.append({
                    'Gene1': gene_names[i],
                    'Gene2': gene_names[j],
                    'EdgeWeight': round(float(similarity_matrix[i, j]), 15)
                })
        
        total_edges = len(edge_weights)
        if total_edges == 0:
            print(f"⚠️ 无有效基因对 → 跳过保存")
            return None

        # 保存TSV：模型_数据集.tsv
        tsv_filename = f"Geneformer_{dataset_name}.tsv"
        tsv_path = os.path.join(output_root, tsv_filename)
        
        # 按EdgeWeight降序排序（保留所有边）
        edges_df = pd.DataFrame(edge_weights)
        edges_df = edges_df.sort_values(by="EdgeWeight", ascending=False)
        edges_df.to_csv(tsv_path, sep='\t', index=False)

        # 统计信息
        avg_weight = np.mean([ew['EdgeWeight'] for ew in edge_weights]) if total_edges > 0 else 0.0
        print(f"\n📈 结果统计：")
        print(f"   总生成边（双向）：{total_edges} | 平均权重：{avg_weight:.4f}")
        print(f"✅ TSV已保存：{tsv_path}")
        return tsv_path

    # 执行保存
    save_result = save_similarity_tsv()
    if save_result is None:
        return

    # 5. 记录运行信息
    record = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型": "Geneformer",
        "数据集": dataset_name,
        "输入基因数": input_genes_count,
        "匹配基因数": matched_genes_count,
        "总双向边数": len(pd.read_csv(save_result)),
        "TSV路径": save_result
    }
    record_csv = os.path.join(output_root, "Geneformer运行记录.csv")
    pd.DataFrame([record]).to_csv(record_csv, mode='a', header=not os.path.exists(record_csv), index=False)

# -------------------------- 批量遍历 --------------------------
def main():
    global ARGS, input_root, output_root, gene_emb_df
    ARGS = parse_args()
    input_root = ARGS.input_root
    output_root = ARGS.output_root
    os.makedirs(output_root, exist_ok=True)

    gene_emb_df = load_geneformer_embedding_df(model_dir=Path(ARGS.model_dir), dict_dir=Path(ARGS.dict_dir))
    print("✅ 初始化完成：模型、字典、嵌入已加载")

    target_folders = ARGS.folders
    for folder in target_folders:
        folder_path = os.path.join(input_root, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ 文件夹不存在：{folder_path} → 跳过")
            continue

        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
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