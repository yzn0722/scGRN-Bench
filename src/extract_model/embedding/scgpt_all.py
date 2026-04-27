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

# -------------------------- 全局路径（运行时注入；禁止硬编码） --------------------------
INPUT_ROOT = None
OUTPUT_ROOT = None
MODEL_DIR = None
MODEL_FILE = None
VOCAB_FILE = None
RESULTS_LOG_FILE = None

# 模型固定参数
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
MODEL_NAME = "scGPT"  # 模型名称

# -------------------------- 核心修复：直接加载词汇表和嵌入权重 --------------------------
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# -------------------------- 工具函数 --------------------------
def save_run_statistics(stats_dict):
    """保存运行统计信息到CSV文件"""
    stats_df = pd.DataFrame([stats_dict])
    if not RESULTS_LOG_FILE.exists():
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='w')
        print(f"\n✅ 统计日志文件已创建：{RESULTS_LOG_FILE}")
    else:
        stats_df.to_csv(RESULTS_LOG_FILE, index=False, mode='a', header=False)
        print(f"\n✅ 统计信息已追加到日志文件：{RESULTS_LOG_FILE}")

def extract_dataset_name(file_path):
    """提取纯数据集名（移除所有后缀）"""
    file_name = Path(file_path).name
    # 统一提取数据集核心名称
    if "_chip_matched-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_chip_matched-ExpressionData.csv')[0]
    elif "_processed-ExpressionData.csv" in file_name:
        dataset_name = file_name.split('_processed-ExpressionData.csv')[0]
    else:
        dataset_name = file_name.split('-ExpressionData.csv')[0]
    return dataset_name

# -------------------------- 基因嵌入处理类 --------------------------
class GeneEmbeddingProcessor:
    """处理基因嵌入数据，生成全量双向边"""
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
        if len(self.gene_embeddings) > 0:
            print(f"嵌入维度: {self.gene_embeddings.shape[1]}")
    
    def compute_similarity_matrix(self):
        """计算余弦相似度矩阵（固定方法）"""
        if self.gene_embeddings is None or len(self.gene_embeddings) == 0:
            raise ValueError("没有有效的基因嵌入数据")
        print(f"\n计算基因间余弦相似度...")
        self.similarity_matrix = cosine_similarity(self.gene_embeddings)
        np.fill_diagonal(self.similarity_matrix, 0)  # 移除自环
        print(f"相似度矩阵形状: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def generate_all_interactions(self):
        """生成全量双向边（i≠j，无筛选）"""
        if self.similarity_matrix is None:
            raise ValueError("请先调用 compute_similarity_matrix 计算相似度矩阵")
        edge_weights = []
        n_genes = len(self.gene_names)
        for i in range(n_genes):
            gene1 = self.gene_names[i]
            for j in range(n_genes):
                if i != j:  # 仅过滤自环，保留所有双向边
                    gene2 = self.gene_names[j]
                    weight = self.similarity_matrix[i, j]
                    edge_weights.append({
                        'Gene1': gene1,
                        'Gene2': gene2,
                        'EdgeWeight': round(weight, 15)
                    })
        print(f"\n生成 {len(edge_weights)} 个全量非自环基因对")
        return edge_weights

# -------------------------- 预加载scGPT嵌入权重和词汇表 --------------------------
def preload_scgpt_embeddings(model_file: Path, vocab_file: Path):
    """预加载scGPT的基因嵌入权重和词汇表（仅加载一次）"""
    # 1. 加载词汇表
    print("📌 加载scGPT词汇表...")
    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    gene2idx = vocab.get_stoi()
    idx2gene = {v: k for k, v in gene2idx.items()}
    print(f"✅ 词汇表加载完成 | 总词汇数：{len(gene2idx)}")
    
    # 2. 直接加载模型权重文件，提取嵌入层
    print("📌 加载scGPT嵌入层权重...")
    state_dict = torch.load(model_file, map_location='cpu')
    
    # 核心修复：适配更多嵌入层参数名（包含encoder.embedding.weight）
    emb_weight_candidates = [
        "encoder.embedding.weight",          # 当前报错中出现的参数名
        "encoder.embeddings.weight",         # 旧版scGPT参数名
        "embeddings.word_embeddings.weight", # 标准Transformer参数名
        "word_embeddings.weight",            # 极简版参数名
        "embedding.weight"                   # 最简化参数名
    ]
    
    emb_weight_key = None
    for candidate in emb_weight_candidates:
        if candidate in state_dict:
            emb_weight_key = candidate
            break
    
    if emb_weight_key is None:
        # 打印前20个参数名，便于排查
        top_keys = list(state_dict.keys())[:20]
        raise KeyError(f"未找到嵌入层权重！权重文件中的参数名前20个：{top_keys}")
    
    print(f"✅ 找到嵌入层参数：{emb_weight_key}")
    token_emb = state_dict[emb_weight_key]
    print(f"✅ 嵌入层权重加载完成 | 形状：{token_emb.shape}")
    
    # 3. 构建基因→嵌入的映射（排除特殊token）
    gene_emb_dict = {}
    special_token_ids = [gene2idx[s] for s in SPECIAL_TOKENS if s in gene2idx]
    print(f"🔍 特殊token ID：{special_token_ids}")
    
    # 确保索引不越界
    max_idx = min(len(token_emb), len(idx2gene))
    for idx in range(max_idx):
        if idx in idx2gene and idx not in special_token_ids:
            gene_name = idx2gene[idx]
            gene_emb = token_emb[idx].detach().cpu().numpy()
            gene_emb_dict[gene_name] = gene_emb
    
    print(f"✅ 基因嵌入映射构建完成 | 有效基因数：{len(gene_emb_dict)}")
    print(f"前5个基因：{list(gene_emb_dict.keys())[:5]}")
    
    return gene2idx, gene_emb_dict

gene2idx = None
preloaded_emb_dict = None

# -------------------------- 单文件对处理函数 --------------------------
def process_single_pair(expr_path, network_path):
    """处理单个文件对，输出全量双向边，统一命名风格"""
    start_time = time.time()
    run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. 提取纯数据集名称（无文件夹后缀）
    dataset_name = extract_dataset_name(expr_path)
    # 输出文件命名：scGPT_数据集.tsv（统一风格）
    output_tsv_name = f"{MODEL_NAME}_{dataset_name}.tsv"
    output_tsv_path = OUTPUT_ROOT / output_tsv_name
    
    print(f"\n=====================================")
    print(f"🔍 处理数据集：{dataset_name}")
    print(f"   Expr文件：{expr_path}")
    print(f"   输出文件：{output_tsv_path}")
    print(f"=====================================")

    # 初始化核心统计字典
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
        # -------------------------- 步骤1：读取输入文件，提取基因名 --------------------------
        print("\n📌 步骤1：读取输入表达文件")
        expr_df = pd.read_csv(expr_path, header=0, index_col=0)
        input_genes = expr_df.index.tolist()
        stats['Input_Genes_Count'] = len(input_genes)
        print(f"输入文件包含 {len(input_genes)} 个基因")

        # -------------------------- 步骤2：匹配预加载的基因嵌入 --------------------------
        print("\n📌 步骤2：匹配scGPT基因嵌入")
        # 统一基因名格式（大小写+去空格）
        input_genes_clean = {str(gene).strip().upper(): gene for gene in input_genes}
        preloaded_genes_clean = {str(gene).strip().upper(): gene for gene in preloaded_emb_dict.keys()}
        
        # 匹配基因
        matched_genes = []
        matched_emb = []
        for gene_clean, gene_original in input_genes_clean.items():
            if gene_clean in preloaded_genes_clean:
                matched_genes.append(gene_original)
                matched_emb.append(preloaded_emb_dict[preloaded_genes_clean[gene_clean]])
        
        stats['Matched_Genes_Count'] = len(matched_genes)
        stats['Matched_Ratio(%)'] = round(len(matched_genes) / len(input_genes) * 100, 2) if len(input_genes) > 0 else 0.0
        
        print(f"匹配到scGPT嵌入的基因：{len(matched_genes)} 个 ({stats['Matched_Ratio(%)']}%)")
        print(f"匹配的前5个基因: {matched_genes[:5]}")
        
        if len(matched_genes) == 0:
            print("⚠️ 没有匹配到任何基因，跳过该文件！")
            stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
            save_run_statistics(stats)
            return
        
        # 构建当前文件的基因嵌入字典
        gene_emb_dict = {gene: emb for gene, emb in zip(matched_genes, matched_emb)}

        # -------------------------- 步骤3：生成全量双向边 --------------------------
        print("\n📌 步骤3：生成全量双向边权重")
        processor = GeneEmbeddingProcessor(gene_emb_dict)
        processor.compute_similarity_matrix()
        
        # 生成所有非自环双向边
        all_interactions = processor.generate_all_interactions()
        stats['Total_Edges_Generated'] = len(all_interactions)
        
        # 按权重降序排序
        final_interactions = sorted(all_interactions, key=lambda x: x['EdgeWeight'], reverse=True)

        # -------------------------- 步骤4：保存最终TSV文件 --------------------------
        print("\n📌 步骤4：保存最终TSV文件")
        interactions_df = pd.DataFrame(final_interactions)
        interactions_df.to_csv(output_tsv_path, sep='\t', index=False)
        
        print(f"\n✅ 最终TSV文件已保存至：{output_tsv_path}")
        print(f"文件包含 {len(interactions_df)} 个全量双向边")
        if len(interactions_df) > 0:
            print(f"权重范围：{interactions_df['EdgeWeight'].min():.6f} ~ {interactions_df['EdgeWeight'].max():.6f}")
    
    except Exception as e:
        print(f"❌ 处理文件失败：{e}")
        stats['Process_Status'] = f"Failed: {str(e)[:100]}"
        import traceback
        traceback.print_exc()
    
    # -------------------------- 步骤5：保存统计信息 --------------------------
    stats['Run_Time_Seconds'] = round(time.time() - start_time, 2)
    print(f"\n📊 运行统计摘要 | 耗时：{stats['Run_Time_Seconds']} 秒")
    print(f"   输入基因数：{stats['Input_Genes_Count']} | 匹配基因数：{stats['Matched_Genes_Count']}")
    print(f"   生成边数：{stats['Total_Edges_Generated']}")
    
    save_run_statistics(stats)

# -------------------------- 批量遍历主函数 --------------------------
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
        print(f"❌ 预加载失败：{e}")
        sys.exit(1)

    print("🚀 开始scGPT批量处理流程（输出全量双向边）")
    print(f"输入根目录：{INPUT_ROOT}")
    print(f"输出根目录：{OUTPUT_ROOT}")
    
    # 忽略临时文件清理警告
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