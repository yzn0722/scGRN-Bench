import os 
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime  

import argparse
from pathlib import Path

# -------------------------- 运行时注入（禁止硬编码） --------------------------
ARGS = None
INPUT_ROOT = None
OUTPUT_ROOT = None
MODEL_NAME = "scCello"
PARENT_MODEL_DIR = None
gene_emb_df = None


def parse_args():
    p = argparse.ArgumentParser(description="Export scCello embedding-weight cosine edges as TSVs.")
    p.add_argument("--input-root", required=True, type=str, help="Input root with CHIP/Non_CHIP/STRING folders.")
    p.add_argument("--output-root", required=True, type=str, help="Output directory for TSV + run records.")
    p.add_argument("--parent-model-dir", required=True, type=str, help="Parent weights dir containing scCello/ and Geneformer/dicts/.")
    p.add_argument("--folders", nargs="+", default=["CHIP"], help='Folders to process. Default: ["CHIP"].')
    return p.parse_args()


def load_sccello_embedding_df(parent_model_dir: Path) -> pd.DataFrame:
    # local import so file can be imported without the repo on sys.path
    sys.path.insert(0, str(parent_model_dir.parent))  # best-effort; user may not need this if package is installed
    try:
        from sc_foundation_evals.sccello.src.model_prototype_contrastive import PrototypeContrastiveModel
    except Exception as e:
        raise RuntimeError(
            "Cannot import sc_foundation_evals.sccello... "
            "Provide a python package or add the correct repo path to PYTHONPATH.\n"
            f"import error: {e}"
        )

    saved_model_path = parent_model_dir / "scCello"
    model = PrototypeContrastiveModel.from_pretrained(str(saved_model_path), ignore_mismatched_sizes=True)
    model_state_dict = model.state_dict()
    if "embeddings.word_embeddings.weight" not in model_state_dict:
        raise KeyError("Missing embeddings.word_embeddings.weight in scCello state_dict")

    token_emb = model_state_dict["embeddings.word_embeddings.weight"]
    token_emb_filtered = token_emb[:-1, :]  # drop CLS

    dict_paths = parent_model_dir / "Geneformer" / "dicts"
    with open(dict_paths / "token_dictionary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(dict_paths / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_id = pickle.load(f)
    gene_id_name = {v: k for k, v in gene_name_id.items()}

    vocab_keys = list(vocab.keys())[: len(token_emb_filtered)]
    gene_emb_df_local = pd.DataFrame(token_emb_filtered.detach().cpu().numpy())
    gene_emb_df_local["ENSG_ID"] = vocab_keys
    gene_emb_df_local["Symbol"] = gene_emb_df_local["ENSG_ID"].apply(lambda x: gene_id_name.get(x, None))
    gene_emb_df_local = gene_emb_df_local.dropna(subset=["Symbol"]).reset_index(drop=True)
    return gene_emb_df_local

# -------------------------- 预加载模型/字典（简化版，保留核心） --------------------------
def load_sccello_model_and_dict():
    """预加载scCello模型和基因字典（简化版）"""
    # 1. 加载scCello模型
    saved_model_path = f"{PARENT_MODEL_DIR}/scCello"
    model = PrototypeContrastiveModel.from_pretrained(saved_model_path, ignore_mismatched_sizes=True)
    model_state_dict = model.state_dict()
    
    # 提取嵌入层权重（核心）
    if 'embeddings.word_embeddings.weight' not in model_state_dict:
        raise KeyError("模型中未找到嵌入层权重 'embeddings.word_embeddings.weight'")
    token_emb = model_state_dict['embeddings.word_embeddings.weight']
    token_emb_filtered = token_emb[:-1, :]  # 移除CLS token
    
    # 2. 加载共用字典
    dict_paths = f"{PARENT_MODEL_DIR}/Geneformer/dicts"  
    with open(os.path.join(dict_paths, "token_dictionary.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(dict_paths, "gene_name_id_dict.pkl"), "rb") as f:
        gene_name_id = pickle.load(f)
    gene_id_name = {v: k for k, v in gene_name_id.items()}
    
    # 预处理嵌入DataFrame
    vocab_keys = list(vocab.keys())[:len(token_emb_filtered)]
    gene_emb_df = pd.DataFrame(token_emb_filtered.numpy())
    gene_emb_df["ENSG_ID"] = vocab_keys
    gene_emb_df["Symbol"] = gene_emb_df["ENSG_ID"].apply(lambda x: gene_id_name.get(x, None))
    gene_emb_df = gene_emb_df.dropna(subset=["Symbol"]).reset_index(drop=True)
    
    print(f"✅ scCello模型加载完成 | 嵌入基因数：{len(gene_emb_df)}")
    return model, gene_emb_df

# 执行预加载
model = None
gene_emb_df = None

# -------------------------- 工具函数（统一提取纯数据集名） --------------------------
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

# -------------------------- 单文件对处理函数（输出全量双向边+移除筛选） --------------------------
def process_single_pair(expr_path, network_path):
    """处理单个文件对，输出全量双向边，无筛选"""
    start_time = datetime.now()
    # 1. 提取纯数据集名称（无数据类型）
    dataset_name = extract_dataset_name(expr_path)
    print(f"\n=====================================")
    print(f"🔍 处理数据集：{dataset_name}")
    print(f"   Expr文件：{expr_path}")

    # 初始化运行记录
    run_record = {
        "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model_Name": MODEL_NAME,
        "Dataset_Name": dataset_name,
        "Input_File": expr_path,
        "Input_Genes_Count": 0,
        "Matched_Genes_Count": 0,
        "Match_Rate(%)": 0.0,
        "Total_Edges_Generated": 0,
        "Output_TSV_Path": "",
        "Process_Status": "Success",
        "Process_Time_Seconds": 0.0
    }

    try:
        # 2. 读取参考基因集
        data_df = pd.read_csv(expr_path)
        gene_symbols = data_df.iloc[1:, 0].tolist()
        gene_set_df = pd.DataFrame({"Symbol": gene_symbols}).dropna()
        input_genes_count = len(gene_set_df)
        run_record["Input_Genes_Count"] = input_genes_count
        print(f"✅ 参考基因集：{input_genes_count}个基因")

        # 3. 匹配scCello嵌入
        gene_emb_filtered = pd.merge(gene_set_df, gene_emb_df, on="Symbol", how="inner")
        matched_genes_count = len(gene_emb_filtered)
        match_rate = round(matched_genes_count / input_genes_count * 100, 2) if input_genes_count > 0 else 0
        run_record["Matched_Genes_Count"] = matched_genes_count
        run_record["Match_Rate(%)"] = match_rate
        print(f"✅ 匹配基因数：{matched_genes_count} | 匹配率：{match_rate}%")

        if matched_genes_count == 0:
            raise ValueError("参考基因集与scCello的基因Symbol完全不匹配！")

        # 4. 计算余弦相似度（输出全量双向边）
        gene_names = gene_emb_filtered["Symbol"].tolist()
        embedding_cols = [col for col in gene_emb_filtered.columns if col not in ["Symbol", "ENSG_ID"]]
        embedding_matrix = gene_emb_filtered[embedding_cols].values

        # 计算相似度矩阵（无向，对称）
        similarity_matrix = cosine_similarity(embedding_matrix)

        # 核心修改：遍历所有i≠j，生成全量双向边（仅过滤自环）
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
        run_record["Total_Edges_Generated"] = total_edges
        print(f"✅ 生成全量双向边：{total_edges}条")

        if total_edges == 0:
            raise ValueError("无有效基因对可生成边！")

        # 5. 保存TSV：统一命名为 scCello_数据集.tsv
        tsv_filename = f"{MODEL_NAME}_{dataset_name}.tsv"
        tsv_path = os.path.join(OUTPUT_ROOT, tsv_filename)
        
        # 按EdgeWeight降序排序（保留所有边）
        edges_df = pd.DataFrame(edge_weights)
        edges_df = edges_df.sort_values(by="EdgeWeight", ascending=False)
        edges_df.to_csv(tsv_path, sep='\t', index=False)
        run_record["Output_TSV_Path"] = tsv_path
        print(f"✅ TSV文件已保存：{tsv_path}")

        # 6. 补充耗时信息
        run_record["Process_Time_Seconds"] = round((datetime.now() - start_time).total_seconds(), 2)

    except Exception as e:
        print(f"❌ 处理失败：{e}")
        run_record["Process_Status"] = f"Failed: {str(e)[:100]}"
        import traceback
        traceback.print_exc()

    # 7. 保存简化版运行记录
    record_filename = f"{MODEL_NAME}_run_records.csv"
    record_csv_path = os.path.join(OUTPUT_ROOT, record_filename)
    record_df = pd.DataFrame([run_record])
    
    if os.path.exists(record_csv_path):
        record_df.to_csv(record_csv_path, mode='a', header=False, index=False)
    else:
        record_df.to_csv(record_csv_path, mode='w', header=True, index=False)
    
    print(f"\n📊 处理完成 | 耗时：{run_record['Process_Time_Seconds']}秒")

# -------------------------- 批量遍历主函数（简化版） --------------------------
def main():
    global ARGS, INPUT_ROOT, OUTPUT_ROOT, PARENT_MODEL_DIR, gene_emb_df
    ARGS = parse_args()
    INPUT_ROOT = ARGS.input_root
    OUTPUT_ROOT = ARGS.output_root
    PARENT_MODEL_DIR = ARGS.parent_model_dir
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"✅ 初始化完成 | 输出根目录：{OUTPUT_ROOT}")

    gene_emb_df = load_sccello_embedding_df(parent_model_dir=Path(PARENT_MODEL_DIR))

    print(f"\n🚀 开始scCello批量处理流程（输出全量双向边）")
    print(f"输入根目录：{INPUT_ROOT}")
    print(f"输出根目录：{OUTPUT_ROOT}")
    
    # 目标文件夹列表
    target_folders = ARGS.folders
    
    for folder in target_folders:
        folder_path = os.path.join(INPUT_ROOT, folder)
        if not os.path.exists(folder_path):
            print(f"\n⚠️ 文件夹不存在，跳过：{folder_path}")
            continue
        
        print(f"\n=====================================")
        print(f"📂 扫描文件夹：{folder}")
        
        # 找到所有ExpressionData文件
        expr_files = [f for f in os.listdir(folder_path) if f.endswith('-ExpressionData.csv')]
        if not expr_files:
            print(f"⚠️ 未找到ExpressionData.csv文件，跳过")
            continue
        
        # 处理每个文件对
        for expr_file in expr_files:
            expr_path = os.path.join(folder_path, expr_file)
            network_file = expr_file.replace('-ExpressionData.csv', '-network.csv')
            network_path = os.path.join(folder_path, network_file)
            
            # 仅检查network文件存在性，不读取/筛选
            if not os.path.exists(network_path):
                print(f"⚠️ 缺失Network文件：{network_path} → 跳过{expr_file}")
                continue
            
            process_single_pair(expr_path, network_path)

if __name__ == "__main__":
    main()
    print("\n=====================================")
    print(f"🎉 所有文件处理完成！结果保存在：{OUTPUT_ROOT}")