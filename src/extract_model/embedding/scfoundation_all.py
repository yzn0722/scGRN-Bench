


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

# 模型固定参数
MODEL_NAME = "scFoundation"  # 模型名称
EMBEDDING_SIZE = 768  # scFoundation pos_emb维度

# -------------------------- 工具函数 --------------------------
def extract_dataset_name(file_path):
    """提取纯数据集名（移除所有后缀，与其他模型统一）"""
    file_name = Path(file_path).name
    # 统一提取数据集核心名称
    if "_chip_matched-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_chip_matched-ExpressionData.csv')[0]
    elif "_processed-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_processed-ExpressionData.csv')[0]
    else:
        dataset_name = file_name.split('-ExpressionData.csv')[0]
    return dataset_name

def save_run_statistics(stats_dict):
    """保存运行统计信息到CSV文件（与其他模型统一格式）"""
    stats_df = pd.DataFrame([stats_dict])
    if not RESULTS_LOG_FILE.exists():
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='w')
        print(f"\n✅ 统计日志文件已创建：{RESULTS_LOG_FILE}")
    else:
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='a', header=False)
        print(f"\n✅ 统计信息已追加到日志文件：{RESULTS_LOG_FILE}")

# -------------------------- 基因嵌入处理类 --------------------------
class GeneEmbeddingProcessor:
    """处理scFoundation基因嵌入数据，生成全量双向边（与其他模型统一）"""
    def __init__(self, gene_emb_dict):
        self.gene_emb_dict = gene_emb_dict
        self.gene_embeddings = None
        self.gene_names = []
        self.similarity_matrix = None
        self.process_embeddings()
    
    def process_embeddings(self):
        """处理嵌入数据"""
        print("\n处理基因嵌入数据...")
        self.gene_names = list(self.gene_emb_dict.keys())
        self.gene_embeddings = np.array(list(self.gene_emb_dict.values()))
        print(f"成功处理 {len(self.gene_names)} 个基因")
        print(f"前5个基因名称: {self.gene_names[:5]}")
        if len(self.gene_embeddings) > 0:
            print(f"嵌入维度: {self.gene_embeddings.shape[1]}")
    
    def compute_similarity_matrix(self):
        """计算余弦相似度矩阵（固定方法，移除自环）"""
        if self.gene_embeddings is None or len(self.gene_embeddings) == 0:
            raise ValueError("没有有效的基因嵌入数据")
        print(f"\n计算基因间余弦相似度...")
        self.similarity_matrix = cosine_similarity(self.gene_embeddings)
        np.fill_diagonal(self.similarity_matrix, 0)  # 移除自环
        print(f"相似度矩阵形状: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def generate_all_interactions(self):
        """生成全量双向边（i≠j，与其他模型统一规则）"""
        if self.similarity_matrix is None:
            raise ValueError("请先调用 compute_similarity_matrix 计算相似度矩阵")
        edge_weights = []
        n_genes = len(self.gene_names)
        for i in range(n_genes):
            gene1 = self.gene_names[i]
            for j in range(n_genes):
                if i != j:  # 全量双向边，仅过滤自环
                    gene2 = self.gene_names[j]
                    weight = self.similarity_matrix[i, j]
                    edge_weights.append({
                        'Gene1': gene1,
                        'Gene2': gene2,
                        'EdgeWeight': round(weight, 15)
                    })
        print(f"\n生成 {len(edge_weights)} 个全量非自环双向基因对")
        return edge_weights

# -------------------------- 单文件对处理函数 --------------------------
def process_single_pair(expr_path, network_path):
    """处理单个文件对（统一格式，移除冗余逻辑）"""
    start_time = time.time()
    run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. 提取纯数据集名称（无文件夹后缀）
    dataset_name = extract_dataset_name(expr_path)
    # 输出文件命名：scFoundation_数据集.tsv（与其他模型统一）
    output_tsv_name = f"{MODEL_NAME}_{dataset_name}.tsv"
    output_tsv_path = OUTPUT_ROOT / output_tsv_name
    
    print(f"\n=====================================")
    print(f"🔍 处理数据集：{dataset_name}")
    print(f"   Expr文件：{expr_path}")
    print(f"   输出文件：{output_tsv_path}")
    print(f"=====================================")

    # 初始化核心统计字典（与其他模型统一字段）
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
        # -------------------------- 步骤1：加载scFoundation权重并提取pos_emb --------------------------
        print("\n📌 步骤1：加载scFoundation权重并提取pos_emb.weight")
        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"模型权重文件不存在：{CKPT_PATH}")
        
        # 加载ckpt并提取pos_emb.weight
        ckpt_data = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        pos_emb_weight = ckpt_data["gene"]["state_dict"]["model.pos_emb.weight"]  # [19267, 768]
        pos_emb_np = pos_emb_weight.cpu().numpy()
        print(f"✅ 提取pos_emb.weight完成：形状={pos_emb_np.shape}")

        # -------------------------- 步骤2：读取词汇表，构建基因-ID映射 --------------------------
        print("\n📌 步骤2：读取词汇表，构建基因-ID映射")
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(f"词汇表文件不存在：{VOCAB_PATH}")
        
        vocab_df = pd.read_csv(VOCAB_PATH, sep="\t")
        vocab_df["gene_name"] = vocab_df["gene_name"].str.upper().str.strip()
        gene2id = dict(zip(vocab_df["gene_name"], vocab_df["index"].astype(int)))
        print(f"✅ 词汇表加载完成 | 总基因数：{len(gene2id)}")

        # -------------------------- 步骤3：读取输入文件，提取基因名 --------------------------
        print("\n📌 步骤3：读取输入表达文件并提取基因名")
        expr_df = pd.read_csv(expr_path, header=0, index_col=0)
        input_genes = expr_df.index.tolist()
        # 清洗输入基因名
        input_genes = [str(g).upper().strip() for g in input_genes if pd.notna(g) and g != ""]
        stats['Input_Genes_Count'] = len(input_genes)
        print(f"输入文件包含 {len(input_genes)} 个有效基因")
        print(f"前5个基因名称示例: {input_genes[:5]}")

        # -------------------------- 步骤4：匹配输入基因的embedding --------------------------
        print("\n📌 步骤4：匹配输入基因与scFoundation的embedding")
        target_gene_emb_dict = {}
        for gene in input_genes:
            if gene in gene2id:
                gene_id = gene2id[gene]
                # 确保ID不越界
                if gene_id < pos_emb_np.shape[0]:
                    target_gene_emb_dict[gene] = pos_emb_np[gene_id]
        
        # 更新统计信息
        stats['Matched_Genes_Count'] = len(target_gene_emb_dict)
        stats['Matched_Ratio(%)'] = round(len(target_gene_emb_dict) / len(input_genes) * 100, 2) if len(input_genes) > 0 else 0.0
        
        print(f"匹配到scFoundation嵌入的基因：{len(target_gene_emb_dict)} 个 ({stats['Matched_Ratio(%)']}%)")
        print(f"匹配的前5个基因: {list(target_gene_emb_dict.keys())[:5]}")
        
        if len(target_gene_emb_dict) == 0:
            print("⚠️ 没有匹配到任何基因，跳过该文件！")
            stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
            save_run_statistics(stats)
            return

        # -------------------------- 步骤5：生成全量双向边 --------------------------
        print("\n📌 步骤5：计算余弦相似度并生成全量双向边")
        processor = GeneEmbeddingProcessor(target_gene_emb_dict)
        processor.compute_similarity_matrix()
        
        # 生成全量双向边
        all_interactions = processor.generate_all_interactions()
        stats['Total_Edges_Generated'] = len(all_interactions)
        
        # 按权重降序排序
        final_interactions = sorted(all_interactions, key=lambda x: x['EdgeWeight'], reverse=True)

        # -------------------------- 步骤6：保存最终TSV文件 --------------------------
        print("\n📌 步骤6：保存最终TSV文件")
        interactions_df = pd.DataFrame(final_interactions)
        interactions_df.to_csv(output_tsv_path, sep='\t', index=False, encoding="utf-8")
        
        print(f"\n✅ 最终TSV文件已保存至：{output_tsv_path}")
        print(f"文件包含 {len(interactions_df)} 个全量双向边")
        if len(interactions_df) > 0:
            print(f"权重范围：{interactions_df['EdgeWeight'].min():.6f} ~ {interactions_df['EdgeWeight'].max():.6f}")
            print(f"权重平均值：{interactions_df['EdgeWeight'].mean():.6f}")
    
    except Exception as e:
        print(f"❌ 处理文件失败：{e}")
        stats['Process_Status'] = f"Failed: {str(e)[:100]}"
        import traceback
        traceback.print_exc()
    
    # -------------------------- 步骤7：保存统计信息 --------------------------
    stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
    print(f"\n📊 运行统计摘要 | 耗时：{stats['Run_Time_Seconds']} 秒")
    print(f"   输入基因数：{stats['Input_Genes_Count']} | 匹配基因数：{stats['Matched_Genes_Count']}")
    print(f"   生成边数：{stats['Total_Edges_Generated']}")
    
    save_run_statistics(stats)

# -------------------------- 批量遍历主函数 --------------------------
def main():
    global ARGS, INPUT_ROOT, OUTPUT_ROOT, CKPT_PATH, VOCAB_PATH, RESULTS_LOG_FILE
    ARGS = parse_args()
    INPUT_ROOT = Path(ARGS.input_root)
    OUTPUT_ROOT = Path(ARGS.output_root)
    CKPT_PATH = Path(ARGS.ckpt_path)
    VOCAB_PATH = Path(ARGS.vocab_path)

    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    RESULTS_LOG_FILE = OUTPUT_ROOT / "scFoundation_run_statistics.csv"

    print("🚀 开始scFoundation批量处理流程（输出全量双向边）")
    print(f"输入根目录：{INPUT_ROOT}")
    print(f"输出根目录：{OUTPUT_ROOT}")
    print(f"模型权重：{CKPT_PATH}")
    print(f"词汇表：{VOCAB_PATH}")
    
    # 忽略冗余警告
    warnings.filterwarnings('ignore', category=ResourceWarning)
    
    # 目标文件夹列表
    target_folders = ARGS.folders
    
    for folder in target_folders:
        folder_path = INPUT_ROOT / folder
        if not folder_path.exists():
            print(f"\n⚠️ 文件夹不存在，跳过：{folder_path}")
            continue
        
        print(f"\n=====================================")
        print(f"📂 扫描文件夹：{folder}")
        print(f"=====================================")
        
        # 找到所有ExpressionData文件
        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
        if not expr_files:
            print(f"⚠️ 文件夹 {folder} 中未找到ExpressionData.csv文件，跳过")
            continue
        
        # 处理每个ExpressionData文件
        for expr_file in expr_files:
            expr_path = folder_path / expr_file
            # 匹配对应的network文件（仅检查存在性）
            network_file = expr_file.replace('-ExpressionData.csv', '-network.csv')
            network_path = folder_path / network_file
            
            if not network_path.exists():
                print(f"⚠️ 缺失对应的network文件：{network_path} → 跳过 {expr_file}")
                continue
            
            # 处理当前文件对
            process_single_pair(expr_path, network_path)
    
    print(f"\n🎉 所有文件处理完成！结果保存在：{OUTPUT_ROOT}")

if __name__ == "__main__":
    main()





