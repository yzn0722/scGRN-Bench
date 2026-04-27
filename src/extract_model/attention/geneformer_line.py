


"""
Geneformer Attention提取 - 批量处理版（支持CHIP/Non_CHIP/STRING）
统一输出格式，参数汇总到单个CSV，支持批量运行
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

# ==================== 接收命令行参数 ====================
if len(sys.argv) != 3:
    print("用法: python geneformer_attention_batch.py <数据类型> <数据集名称>")
    print("例如: python geneformer_attention_batch.py CHIP hESC")
    print("      python geneformer_attention_batch.py Non_CHIP hHep")
    print("      python geneformer_attention_batch.py STRING mHSC-E")
    sys.exit(1)

data_type = sys.argv[1]  # 数据类型：CHIP / Non_CHIP / STRING
dataset = sys.argv[2]    # 数据集名称：hESC、hHep等
MODEL_VERSION = "6L"     # 固定模型版本
MODEL_NAME = f"geneformer_{MODEL_VERSION}"



# 校验数据类型
valid_data_types = ["CHIP", "Non_CHIP", "STRING"]
if data_type not in valid_data_types:
    raise ValueError(f"不支持的数据类型: {data_type}（仅支持{valid_data_types}）")

# ==================== 路径配置（动态适配类型+数据集）====================
# 根路径配置
GENEFORMER_BASE = "/mnt/md0/yzn/scFM-Bench-main/data/weights/Geneformer"
DICT_DIR = f"{GENEFORMER_BASE}/dicts"
MODEL_DIR = f"{GENEFORMER_BASE}/default/{MODEL_VERSION}"
INPUT_ROOT = "/mnt/md0/yzn/Beeline-master/benchmark_SF/input_process1000"
OUTPUT_ROOT = Path("/mnt/md0/yzn/Beeline-master/benchmark_SF/model/output_att1000/geneformer")

# 根据类型确定输入路径和文件后缀
if data_type == "CHIP":
    file_suffix = "_chip_matched"
    input_subdir = "CHIP"
elif data_type == "Non_CHIP" or data_type == "STRING":
    file_suffix = "_processed"
    input_subdir = data_type

# 输入文件路径（动态拼接）
csv_path = Path(f"{INPUT_ROOT}/{input_subdir}/{dataset}{file_suffix}-ExpressionData.csv")
network_path = f"{INPUT_ROOT}/{input_subdir}/{dataset}{file_suffix}-network.csv"

# 输出路径（按类型分目录）
TYPE_OUTPUT_DIR = OUTPUT_ROOT / data_type
TYPE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 临时文件路径（含类型+数据集标识，避免冲突）
TEMP_PREPROCESSED_DIR = OUTPUT_ROOT / "temp_preprocessed" / f"{data_type}_{dataset}"
TEMP_TOKENIZED_DIR = OUTPUT_ROOT / "temp_tokenized" / f"{data_type}_{dataset}"
TEMP_PREPROCESSED_DIR.mkdir(exist_ok=True, parents=True)
TEMP_TOKENIZED_DIR.mkdir(exist_ok=True, parents=True)

# 输出文件命名（统一规范：模型-版本-类型-数据集-文件名）
file_prefix = f"geneformer-{MODEL_VERSION}-{data_type}-{dataset}-att"
# 移除原始TSV路径定义
filtered_tsv_path = TYPE_OUTPUT_DIR / f"{file_prefix}_filtered.tsv"

# 总参数CSV文件（所有任务共用）
ALL_PARAMS_CSV = OUTPUT_ROOT / "geneformer-all-params.csv"

print("="*80)
print(f"📌 任务配置：{MODEL_NAME} | {data_type} | {dataset}")
print("="*80)
print(f"模型路径: {MODEL_DIR}")
print(f"输入表达数据: {csv_path}")
print(f"输入网络数据: {network_path}")
print(f"输出目录: {TYPE_OUTPUT_DIR}")
print(f"总参数文件: {ALL_PARAMS_CSV}")
print("="*80)

# ========== 加载数据 ==========
print("\n" + "="*60)
print("1️⃣ 加载原始数据")
print("="*60)

if not csv_path.exists():
    raise FileNotFoundError(f"表达数据文件不存在: {csv_path}")

adata = sc.read_csv(str(csv_path))
adata = adata.T
adata_original_shape = adata.shape  # 记录原始形状
print(f"原始数据形状 (细胞×基因): {adata.shape}")
print(f"原始细胞数: {adata.n_obs}")
print(f"原始基因数: {adata.n_vars}")

# ========== 加载Geneformer字典 ==========
print("\n" + "="*60)
print("2️⃣ 加载Geneformer字典")
print("="*60)

with open(f"{DICT_DIR}/gene_name_id_dict.pkl", "rb") as f:
    gene_name_dict = pickle.load(f)

with open(f"{DICT_DIR}/token_dictionary.pkl", "rb") as f:
    token_dict = pickle.load(f)

print(f"Geneformer字典包含 {len(gene_name_dict)} 个基因")

# ========== 匹配基因并只保留匹配的 ==========
print("\n" + "="*60)
print("3️⃣ 匹配基因（只保留匹配的）")
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

print(f"\n匹配统计:")
print(f"  总基因数: {len(gene_symbols)}")
print(f"  匹配的基因: {matched_count} ({100*matched_count/len(gene_symbols):.1f}%)")
print(f"  未匹配的基因: {unmatched_count} ({100*unmatched_count/len(gene_symbols):.1f}%)")

if unmatched_count > 0:
    unmatched_genes = [g for g, m in zip(gene_symbols, matched_mask) if not m]
    print(f"\n⚠️  丢弃 {unmatched_count} 个未匹配的基因（示例前10个）: {unmatched_genes[:10]}")

if matched_count == 0:
    print("\n❌ 错误: 没有基因匹配Geneformer字典！")
    raise ValueError("没有基因匹配Geneformer字典")

adata = adata[:, matched_mask].copy()
matched_gene_symbols = [g for g, m in zip(gene_symbols, matched_mask) if m]
ensembl_ids = [gene_name_dict[gene] for gene in matched_gene_symbols]

adata.var['gene_name'] = matched_gene_symbols
adata.var['ensembl_id'] = ensembl_ids
adata.var_names = matched_gene_symbols

print(f"\n✓ 保留 {adata.n_vars} 个匹配的基因用于分析")
print(f"前10个基因: {matched_gene_symbols[:10]}")

# ========== 质控和归一化 ==========
print("\n" + "="*60)
print("4️⃣ 质控和归一化")
print("="*60)

adata_pre_qc_n_obs = adata.n_obs
adata_pre_qc_n_vars = adata.n_vars
print(f"质控前: {adata.shape}")

sc.pp.filter_cells(adata, min_genes=200)
print(f"质控后: {adata.shape}")

print("\n归一化...")
adata.obs['n_counts'] = (adata.X.sum(axis=1).A1 
                         if scipy.sparse.issparse(adata.X) 
                         else adata.X.sum(axis=1))
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

if 'cell_type' not in adata.obs.columns:
    adata.obs['cell_type'] = 'unknown'

print(f"✓ 预处理完成")

# ========== 保存为Loom并Tokenize ==========
print("\n" + "="*60)
print("5️⃣ Tokenization")
print("="*60)

loom_path = TEMP_PREPROCESSED_DIR / f"{data_type}_{dataset}.loom"
adata.write_loom(str(loom_path), write_obsm_varm=False)
print(f"✓ Loom文件已保存: {loom_path}")

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

print("✓ Tokenization完成")

# ========== 核心工具函数（保留原修复逻辑） ==========
def rank_normalize_fixed(attn_scores):
    """修复后的Rank Normalization（保持原逻辑，修正排序方向和维度）"""
    batch_size, num_heads, M, _ = attn_scores.shape
    attn_normed = torch.zeros_like(attn_scores, dtype=torch.float32)
    
    # 行Rank Normalization（降序）
    for b in range(batch_size):
        for h in range(num_heads):
            row_scores = attn_scores[b, h]
            sorted_indices = torch.argsort(row_scores, dim=1, descending=True)
            rank = torch.argsort(sorted_indices, dim=1)
            row_normed = rank.float() / (M - 1) if M > 1 else rank.float()
            attn_normed[b, h] = row_normed
    
    # 列Rank Normalization（降序）
    for b in range(batch_size):
        for h in range(num_heads):
            col_scores = attn_normed[b, h].T
            sorted_indices = torch.argsort(col_scores, dim=1, descending=True)
            rank = torch.argsort(sorted_indices, dim=1)
            col_normed = rank.float() / (M - 1) if M > 1 else rank.float()
            attn_normed[b, h] = col_normed.T
    
    return attn_normed

def reverse_permute_fixed(tensor, indices):
    """修复后的基因顺序恢复（保持原逻辑，优化实现）"""
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

# ========== Geneformer Attention提取 ==========
print("\n" + "="*60)
print("6️⃣ 提取Attention")
print("="*60)

# 加载模型
print("\n加载模型...")
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
print(f"✓ 模型已加载到 {device}")
print(f"  总层数: {num_layers}")
print(f"  注意力头数: {num_heads}")

# 加载tokenized数据
print("\n加载tokenized数据...")
tokenized_dataset_path = TEMP_TOKENIZED_DIR / f"{data_type}_{dataset}.dataset"
tokenized_dataset = load_from_disk(str(tokenized_dataset_path))
tokenized_size = len(tokenized_dataset)
print(f"✓ 数据集大小: {tokenized_size} 个细胞")

# 获取基因名列表
print("\n获取基因名映射...")
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

print(f"✓ 有效基因数: {n_final_genes}")
print(f"前10个基因: {final_gene_names[:10]}")

# 准备DataLoader
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

print(f"✓ DataLoader准备完成: {len(dataloader)} 个batch")

# 提取attention
target_layer = -1
print(f"\n开始提取attention (Layer {target_layer})...")

attention_sum = None
num_cells_processed = 0
batch_debug = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sorted_indices = batch['sorted_indices'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # 提取目标层的attention
        attn_scores = outputs.attentions[target_layer]
        
        # 归一化
        attn_scores = rank_normalize_fixed(attn_scores)
        
        # 平均所有attention heads
        attn_scores = attn_scores.mean(dim=1)
        
        # 恢复基因顺序
        attn_scores = reverse_permute_fixed(attn_scores, sorted_indices)
        
        attn_numpy = attn_scores.cpu().numpy()
        
        # 应用mask（去除padding）
        mask_2d = attention_mask.cpu().numpy()
        mask_matrix = mask_2d[:, :, None] * mask_2d[:, None, :]
        attn_numpy = attn_numpy * mask_matrix
        
        # 调试信息
        if batch_debug < 2:
            print(f"\nBatch {batch_idx+1} 统计:")
            print(f"  Attention均值: {attn_numpy.mean():.6f}")
            print(f"  Attention最大值: {attn_numpy.max():.6f}")
            print(f"  非零值比例: {(attn_numpy != 0).mean():.6f}")
            batch_debug += 1
        
        # 累加attention
        if attention_sum is None:
            attention_sum = attn_numpy.sum(axis=0)
        else:
            attention_sum += attn_numpy.sum(axis=0)
        
        num_cells_processed += input_ids.shape[0]
        
        # 清理显存
        del outputs, attn_scores, input_ids, attention_mask
        torch.cuda.empty_cache()

print(f"✓ Attention提取完成，共处理 {num_cells_processed} 个细胞")

# 计算平均attention
avg_attention = attention_sum / num_cells_processed

# 提取有效基因区域
gene_attention = avg_attention[np.ix_(valid_indices, valid_indices)]

print(f"\n基因间attention矩阵形状: {gene_attention.shape}")
print(f"Attention统计:")
print(f"  均值: {gene_attention.mean():.6f}")
print(f"  最大值: {gene_attention.max():.6f}")
print(f"  最小值: {gene_attention.min():.6f}")

# ========== 保存结果 ==========
print("\n" + "="*60)
print("7️⃣ 保存结果")
print("="*60)

# 转换为三列TSV格式（仅用于后续筛选，不保存原始文件）
gene_attention_df = pd.DataFrame(
    gene_attention,
    index=final_gene_names,
    columns=final_gene_names
)

gene_interactions_df = gene_attention_df.stack().reset_index()
gene_interactions_df.columns = ["Gene1", "Gene2", "EdgeWeight"]
gene_interactions_df = gene_interactions_df[gene_interactions_df["Gene1"] != gene_interactions_df["Gene2"]]
gene_interactions_df = gene_interactions_df.sort_values(by="EdgeWeight", ascending=False).reset_index(drop=True)

# 移除原始TSV的保存逻辑
# ↓↓↓ 注释/删除原始TSV保存代码 ↓↓↓
# gene_interactions_df.to_csv(interactions_tsv_path, sep="\t", index=False)
# print(f"✓ 原始基因交互TSV: {interactions_tsv_path}")

# 筛选Network文件
print("\n筛选基因交互TSV...")
if not os.path.exists(network_path):
    raise FileNotFoundError(f"Network文件不存在: {network_path}")

network_df = pd.read_csv(network_path)
target_gene1_list = network_df["Gene1"].unique().tolist()
filtered_interactions = gene_interactions_df[gene_interactions_df["Gene1"].isin(target_gene1_list)]
filtered_interactions = filtered_interactions.sort_values(by="EdgeWeight", ascending=False).reset_index(drop=True)

# 保存筛选后TSV
filtered_interactions.to_csv(filtered_tsv_path, sep="\t", index=False)
print(f"✓ 筛选后基因交互TSV: {filtered_tsv_path}")

# ========== 记录关键参数（追加到总CSV） ==========
print("\n" + "="*60)
print("8️⃣ 记录运行参数")
print("="*60)

# 计算运行时间（从脚本开始到现在）
run_time = (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getctime(__file__))).total_seconds() / 60

# 整理参数（移除原始TSV路径相关字段）
params = {
    "模型名称": "geneformer",
    "模型版本": MODEL_VERSION,
    "数据类型": data_type,
    "数据集名称": dataset,
    "原始数据形状（细胞×基因）": str(adata_original_shape),
    "转置后总基因数": len(gene_symbols),
    "基因字典匹配数": matched_count,
    "基因字典匹配率(%)": f"{matched_count/len(gene_symbols)*100:.1f}",
    "质控前细胞数": adata_pre_qc_n_obs,
    "质控前基因数": adata_pre_qc_n_vars,
    "质控后细胞数": adata.n_obs,
    "质控后基因数": adata.n_vars,
    "Tokenization后数据集大小": tokenized_size,
    "最终用于模型的基因数": n_final_genes,
    "模型加载设备": str(device),
    "提取的总基因对数量": len(gene_interactions_df),
    "提取的总边数": len(gene_interactions_df),
    "Label中Gene1数量": len(target_gene1_list),
    "筛选后（Gene1在Label中）的边数": len(filtered_interactions),
    "筛选率(%)": f"{len(filtered_interactions)/len(gene_interactions_df)*100:.1f}" if len(gene_interactions_df) > 0 else "0",
    "权重最小值": f"{gene_attention.min():.6f}",
    "权重最大值": f"{gene_attention.max():.6f}",
    "权重均值": f"{gene_attention.mean():.6f}",
    "权重中位数": f"{np.median(gene_attention):.6f}",
    "运行时间(分钟)": f"{run_time:.2f}",
    "运行时间戳": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    # 移除原始TSV路径字段
    "筛选TSV路径": str(filtered_tsv_path)
}

# 转换为DataFrame并追加到总CSV
params_df = pd.DataFrame([params])
if ALL_PARAMS_CSV.exists():
    params_df.to_csv(ALL_PARAMS_CSV, mode='a', header=False, index=False, encoding="utf-8")
else:
    params_df.to_csv(ALL_PARAMS_CSV, mode='w', header=True, index=False, encoding="utf-8")

print(f"✓ 参数已追加到总文件: {ALL_PARAMS_CSV}")

# ========== 清理临时文件 ==========
print("\n" + "="*60)
print("9️⃣ 清理临时文件")
print("="*60)

import shutil
# 删除临时Loom文件
if loom_path.exists():
    os.remove(loom_path)
    print(f"✓ 删除临时Loom文件: {loom_path}")

# 删除临时tokenized目录
if TEMP_TOKENIZED_DIR.exists():
    shutil.rmtree(TEMP_TOKENIZED_DIR)
    print(f"✓ 删除临时Tokenized目录: {TEMP_TOKENIZED_DIR}")

# 删除临时预处理目录（如果为空）
if TEMP_PREPROCESSED_DIR.exists() and not any(TEMP_PREPROCESSED_DIR.iterdir()):
    os.rmdir(TEMP_PREPROCESSED_DIR)
    # 尝试删除上级临时目录（如果为空）
    temp_parent = TEMP_PREPROCESSED_DIR.parent
    if temp_parent.exists() and not any(temp_parent.iterdir()):
        os.rmdir(temp_parent)
    print(f"✓ 删除临时预处理目录: {TEMP_PREPROCESSED_DIR}")

# ========== 最终统计报告 ==========
print("\n" + "="*80)
print("📊 最终统计报告")
print("="*80)
print(f"任务: {MODEL_NAME} | {data_type} | {dataset}")
print(f"\n核心统计:")
print(f"  有效基因数: {n_final_genes}")
print(f"  处理细胞数: {num_cells_processed}")
print(f"  原始交互记录数: {len(gene_interactions_df)} (未保存)")
print(f"  筛选后交互记录数: {len(filtered_interactions)}")
print(f"  筛选保留比例: {len(filtered_interactions)/len(gene_interactions_df)*100:.2f}%" if len(gene_interactions_df) > 0 else "0%")
print(f"  运行时间: {run_time:.2f}分钟")
print(f"\n输出文件:")
print(f"  - 筛选TSV: {filtered_tsv_path.name}")
print(f"  - 总参数CSV: {ALL_PARAMS_CSV.name}")
print("\n✅ 所有处理完成！")
print("="*80)