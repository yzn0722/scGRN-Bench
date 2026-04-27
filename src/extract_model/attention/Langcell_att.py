"""
LangCell Attention提取 - 批量处理版（支持CHIP/Non_CHIP/STRING）
统一输出格式，参数汇总到单个CSV，支持批量运行
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import scanpy as sc
from scipy import sparse
from datetime import datetime
import json
from pathlib import Path
import sys
import shutil

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

# ==================== 接收命令行参数 ====================
if len(sys.argv) != 3:
    print("用法: python langcell_attention_batch.py <数据类型> <数据集名称>")
    print("例如: python langcell_attention_batch.py CHIP hESC")
    print("      python langcell_attention_batch.py Non_CHIP hHep")
    print("      python langcell_attention_batch.py STRING mHSC-E")
    sys.exit(1)

data_type = sys.argv[1]  # 数据类型：CHIP / Non_CHIP / STRING
dataset = sys.argv[2]    # 数据集名称：hESC、hHep等
MODEL_NAME = "LangCell"
USE_MEDIAN_FILTER = False  # 固定禁用median过滤（保持原配置）
BATCH_SIZE = 8
DEVICE = "cuda"
TARGET_LAYER = -1

# 校验数据类型
valid_data_types = ["CHIP", "Non_CHIP", "STRING"]
if data_type not in valid_data_types:
    raise ValueError(f"不支持的数据类型: {data_type}（仅支持{valid_data_types}）")

# ==================== 路径配置（动态适配类型+数据集）====================
# 根路径配置
MODEL_PATH = "/mnt/md0/yzn/scFM-Bench-main/data/weights/LangCell"
VOCAB_PATH = "/mnt/md0/yzn/scFM-Bench-main/data/weights/Geneformer/dicts"
INPUT_ROOT = "/mnt/md0/yzn/Beeline-master/benchmark_SF/input_process1000"
OUTPUT_ROOT = Path("/mnt/md0/yzn/Beeline-master/benchmark_SF/model/output_att1000/langcell")

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
TEMP_DIR = OUTPUT_ROOT / "temp" / f"{data_type}_{dataset}"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# 输出文件命名（统一规范：模型-类型-数据集-文件名）
file_prefix = f"langcell-{data_type}-{dataset}-att"
interactions_tsv_path = TYPE_OUTPUT_DIR / f"{file_prefix}.tsv"

# 总参数CSV文件（所有任务共用）
ALL_PARAMS_CSV = OUTPUT_ROOT / "langcell-all-params.csv"

print("="*80)
print(f"📌 任务配置：{MODEL_NAME} | {data_type} | {dataset}")
print("="*80)
print(f"模型路径: {MODEL_PATH}")
print(f"词汇表路径: {VOCAB_PATH}")
print(f"输入表达数据: {csv_path}")
print(f"输入网络数据: {network_path}")
print(f"输出目录: {TYPE_OUTPUT_DIR}")
print(f"总参数文件: {ALL_PARAMS_CSV}")
print(f"临时目录: {TEMP_DIR}")
print("="*80)

# ==================== Model Components ====================
class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
        
    def forward(self, hidden_states):
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output

# ==================== Dataset and Collator ====================
class GeneExpressionDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data['input_ids'][idx],
            'sorted_indices': self.data['sorted_indices'][idx],
            'idx': idx
        }

class DataCollatorForCellClassification:
    def __init__(self, add_cls=True):
        self.add_cls = add_cls
    
    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]
        sorted_indices = [f['sorted_indices'] for f in features]
        idx = [f['idx'] for f in features]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_sorted_indices = []
        attention_mask = []
        
        for ids, indices in zip(input_ids, sorted_indices):
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [0] * padding_length)
            padded_sorted_indices.append(indices + list(range(len(indices), max_len)))
            attention_mask.append([1] * len(ids) + [0] * padding_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'sorted_indices': torch.tensor(padded_sorted_indices, dtype=torch.long),
            'idx': torch.tensor(idx, dtype=torch.long)
        }

# ==================== Utility Functions ====================
def reverse_permute(tensor, indices):
    """Reverse the permutation applied by indices"""
    batch_size = tensor.shape[0]
    reverse_indices = torch.argsort(indices, dim=1)
    
    if len(tensor.shape) == 2:
        # For 1D case (gene_ids)
        return torch.gather(tensor, 1, reverse_indices)
    elif len(tensor.shape) == 3:
        # For 2D case (attention scores)
        seq_len = tensor.shape[1]
        expanded_indices = reverse_indices.unsqueeze(1).expand(-1, seq_len, -1)
        tensor = torch.gather(tensor, 2, expanded_indices)
        expanded_indices = reverse_indices.unsqueeze(2).expand(-1, -1, seq_len)
        tensor = torch.gather(tensor, 1, expanded_indices)
        return tensor
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

def tokenize_cell(gene_ids, gene_values, gene_median_dict, pad_token_id=0, use_median_filter=True):
    """Tokenize a single cell's gene expression"""
    # Rank genes by expression
    sorted_indices = np.argsort(-gene_values)
    sorted_genes = gene_ids[sorted_indices]
    sorted_values = gene_values[sorted_indices]
    
    # Filter out zero expression genes
    nonzero_mask = sorted_values > 0
    sorted_genes = sorted_genes[nonzero_mask]
    sorted_values = sorted_values[nonzero_mask]
    sorted_indices = sorted_indices[nonzero_mask]
    
    # Create token list and corresponding indices
    tokens = []
    kept_indices = []
    
    if use_median_filter:
        for idx, (gene_id, value) in enumerate(zip(sorted_genes, sorted_values)):
            median_val = gene_median_dict.get(gene_id, 0)
            if median_val > 0 and value >= median_val:
                tokens.append(int(gene_id))
                kept_indices.append(sorted_indices[idx])
    else:
        for idx, gene_id in enumerate(sorted_genes):
            tokens.append(int(gene_id))
            kept_indices.append(sorted_indices[idx])
    
    if len(tokens) == 0:
        tokens = [pad_token_id]
        kept_indices = [0]
    
    return tokens, kept_indices

# ==================== Main Extraction Class ====================
class LangCellAttentionExtractor:
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self.add_cls = True
        self.reference_genes = None
        self.reference_gene1_count = 0
        
        # 统计参数存储字典（最终写入总CSV）
        self.stats = {
            "模型名称": MODEL_NAME,
            "数据类型": data_type,
            "数据集名称": dataset,
            "模型路径": MODEL_PATH,
            "词汇表路径": VOCAB_PATH,
            "输入表达数据路径": str(csv_path),
            "输入网络数据路径": network_path,
            "使用Median过滤": USE_MEDIAN_FILTER,
            "目标注意力层": TARGET_LAYER,
            "Batch Size": BATCH_SIZE,
            "模型加载设备": str(self.device),
            "运行时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 加载参考网络
        self._load_reference_network()
        print(f"Using device: {self.device}")
        if not USE_MEDIAN_FILTER:
            print("⚠️  Median filtering disabled - will keep all non-zero expressed genes")
        if self.reference_genes:
            print(f"参考网络Gene1唯一基因数: {self.reference_gene1_count}")
        self.stats["Label中Gene1数量"] = self.reference_gene1_count

    def _load_reference_network(self):
        """加载对照网络，只获取Gene1列的唯一基因"""
        try:
            if not os.path.exists(network_path):
                print(f"⚠️  参考网络文件不存在: {network_path}")
                return
            
            df_ref = pd.read_csv(network_path)
            print(f"参考网络形状: {df_ref.shape}")
            
            if 'Gene1' not in df_ref.columns:
                raise ValueError("参考网络缺少Gene1列")
            
            self.reference_genes = set(df_ref['Gene1'].unique())
            self.reference_genes = {gene.strip() for gene in self.reference_genes if pd.notna(gene)}
            self.reference_gene1_count = len(self.reference_genes)
            print(f"参考网络Gene1列提取到 {self.reference_gene1_count} 个唯一基因")
        
        except Exception as e:
            print(f"⚠️  加载参考网络失败: {str(e)}")
            print("   将输出所有基因对（不按参考网络筛选）")
            self.reference_genes = None
            self.reference_gene1_count = 0

    def load_vocab(self):
        """Load vocabulary and gene mappings"""
        print("\n" + "="*60)
        print("1️⃣ 加载词汇表和基因映射")
        print("="*60)
        
        # Load token dictionary - 修复：使用全局VOCAB_PATH
        with open(os.path.join(VOCAB_PATH, "token_dictionary.pkl"), "rb") as f:
            self.vocab = pickle.load(f)
        
        self.pad_token_id = self.vocab.get("<pad>")
        self.vocab_size = len(self.vocab)
        
        # Load gene name to ID mapping - 修复：使用全局VOCAB_PATH
        with open(os.path.join(VOCAB_PATH, "gene_name_id_dict.pkl"), "rb") as f:
            self.gene_name_id = pickle.load(f)
        
        self.id2name = {v: k for k, v in self.gene_name_id.items()}
        
        # Load gene median dictionary - 修复：使用全局VOCAB_PATH
        with open(os.path.join(VOCAB_PATH, "gene_median_dictionary.pkl"), "rb") as f:
            self.gene_median_dict = pickle.load(f)
        
        self.gene_keys = list(self.gene_median_dict.keys())
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))
        
        print(f"词汇表大小: {self.vocab_size}")
        print(f"基因名称-ID映射数: {len(self.gene_name_id)}")
        self.stats["基因字典总数量"] = len(self.gene_name_id)

    def load_model(self):
        """Load pretrained LangCell model"""
        print("\n" + "="*60)
        print("2️⃣ 加载LangCell模型")
        print("="*60)
        
        # Load cell encoder (BERT) - 修复：使用全局MODEL_PATH
        self.model = BertModel.from_pretrained(
            os.path.join(MODEL_PATH, "cell_bert")
        )
        
        # Load cell pooler - 修复：使用全局MODEL_PATH
        self.cell_pooler = Pooler(
            self.model.config,
            proj_dim=256,
            pretrained_proj=os.path.join(MODEL_PATH, "cell_proj.bin")
        )
        
        self.model.to(self.device)
        self.cell_pooler.to(self.device)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型加载完成，总参数量: {total_params/1e6:.1f}M")
        self.stats["模型总参数量(M)"] = f"{total_params/1e6:.1f}"

    def load_csv_data(self):
        """Load gene expression data from CSV"""
        print("\n" + "="*60)
        print("3️⃣ 加载表达数据")
        print("="*60)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"表达数据文件不存在: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path, index_col=0)
        df = df.T
        original_gene_count = df.shape[1]
        original_cell_count = df.shape[0]
        original_possible_gene_pairs = original_gene_count * (original_gene_count - 1)
        
        print(f"数据形状 (细胞×基因): {df.shape}")
        print(f"原始细胞数: {original_cell_count}")
        print(f"原始基因数: {original_gene_count}")
        print(f"原始可能基因对（排除自环）: {original_possible_gene_pairs:,}")
        
        # Create AnnData object
        adata = sc.AnnData(X=df.values)
        adata.obs_names = df.index
        adata.var_names = df.columns
        adata.obs['condition'] = 'condition_1'
        
        self.adata = adata
        self.stats["原始数据形状（细胞×基因）"] = str(df.shape)
        self.stats["原始细胞数"] = original_cell_count
        self.stats["原始基因数"] = original_gene_count
        self.stats["原始可能基因对"] = original_possible_gene_pairs

    def map_genes_to_vocab(self):
        """Map gene names to vocabulary IDs"""
        print("\n" + "="*60)
        print("4️⃣ 基因与词汇表匹配")
        print("="*60)
        
        gene_names = self.adata.var_names.tolist()
        original_gene_count = len(gene_names)
        
        # 匹配基因到Ensembl ID
        ensembl_ids = []
        valid_genes = []
        
        for gene_name in gene_names:
            if gene_name in self.gene_name_id:
                ensembl_ids.append(self.gene_name_id[gene_name])
                valid_genes.append(True)
            elif gene_name.startswith('ENSG'):
                ensembl_ids.append(gene_name)
                valid_genes.append(True)
            else:
                # 不区分大小写匹配
                found = False
                for symbol, ens_id in self.gene_name_id.items():
                    if symbol.upper() == gene_name.upper():
                        ensembl_ids.append(ens_id)
                        valid_genes.append(True)
                        found = True
                        break
                if not found:
                    ensembl_ids.append(None)
                    valid_genes.append(False)
        
        self.adata.var['ensembl_id'] = ensembl_ids
        self.adata.var['valid_gene'] = valid_genes
        
        n_valid = sum(valid_genes)
        mapped_percentage = (n_valid / original_gene_count) * 100 if original_gene_count > 0 else 0
        
        print(f"匹配到词汇表的基因数: {n_valid}/{original_gene_count} ({mapped_percentage:.1f}%)")
        
        # 显示未匹配基因示例
        unmapped = [gene_names[i] for i, v in enumerate(valid_genes) if not v]
        if unmapped:
            print(f"未匹配基因示例（前5个）: {unmapped[:5]}")
        
        if n_valid == 0:
            raise ValueError("没有基因匹配到词汇表！请检查基因名称格式")
        
        # 过滤有效基因
        self.adata = self.adata[:, self.adata.var['valid_gene']]
        print(f"匹配后数据形状: {self.adata.shape}")
        
        self.stats["基因字典匹配数"] = n_valid
        self.stats["基因字典匹配率(%)"] = f"{mapped_percentage:.1f}"
        self.stats["匹配后基因数"] = self.adata.shape[1]
        self.stats["匹配后细胞数"] = self.adata.shape[0]

    def tokenize_data(self):
        """Tokenize gene expression data"""
        print("\n" + "="*60)
        print("5️⃣ 数据Tokenization")
        print("="*60)
        
        # 进一步过滤不在词汇表中的基因
        valid_genes_mask = []
        for eid in self.adata.var['ensembl_id']:
            valid_genes_mask.append(eid in self.vocab)
        
        valid_genes_mask = np.array(valid_genes_mask)
        n_invalid = (~valid_genes_mask).sum()
        
        if n_invalid > 0:
            print(f"过滤掉 {n_invalid} 个不在词汇表中的基因")
        
        # 最终过滤
        self.adata = self.adata[:, valid_genes_mask]
        filtered_gene_count = self.adata.shape[1]
        filtered_percentage = (filtered_gene_count / self.stats["原始基因数"]) * 100 if self.stats["原始基因数"] > 0 else 0
        
        print(f"最终数据形状: {self.adata.shape}")
        print(f"最终基因数: {filtered_gene_count} ({filtered_percentage:.1f}% of original)")
        
        if self.adata.shape[1] == 0:
            raise ValueError("没有基因保留在词汇表中！")
        
        # 准备表达数据
        X = self.adata.X
        if sparse.issparse(X):
            X = X.toarray()
        
        # 获取基因ID
        gene_ids = np.array([self.vocab[eid] for eid in self.adata.var['ensembl_id']])
        
        # 统计在median字典中的基因数
        genes_in_median_dict = sum([1 for gid in gene_ids if gid in self.gene_median_dict])
        print(f"在median字典中的基因数: {genes_in_median_dict}/{len(gene_ids)}")
        
        # Tokenize每个细胞
        tokenized_cells = []
        token_counts = []
        for i in tqdm(range(X.shape[0]), desc="Tokenizing cells"):
            cell_expr = X[i, :]
            tokens, sorted_indices = tokenize_cell(
                gene_ids, cell_expr, self.gene_median_dict, self.pad_token_id,
                use_median_filter=USE_MEDIAN_FILTER
            )
            tokenized_cells.append({
                'input_ids': tokens,
                'sorted_indices': sorted_indices
            })
            token_counts.append(len(tokens))
        
        # 整理tokenized数据
        self.tokenized_data = {
            'input_ids': [c['input_ids'] for c in tokenized_cells],
            'sorted_indices': [c['sorted_indices'] for c in tokenized_cells]
        }
        
        # Token统计
        token_counts = np.array(token_counts)
        print(f"Tokenization完成，共处理 {len(self.tokenized_data['input_ids'])} 个细胞")
        print(f"Token统计:")
        print(f"  平均每个细胞Token数: {token_counts.mean():.1f}")
        print(f"  中位数Token数: {np.median(token_counts):.1f}")
        print(f"  最小Token数: {token_counts.min()}")
        print(f"  最大Token数: {token_counts.max()}")
        
        self.stats["Tokenization后细胞数"] = len(self.tokenized_data['input_ids'])
        self.stats["最终用于模型的基因数"] = filtered_gene_count
        self.stats["平均每个细胞Token数"] = f"{token_counts.mean():.1f}"

    def create_dataloader(self):
        """Create DataLoader for batched inference"""
        print("\n" + "="*60)
        print("6️⃣ 创建DataLoader")
        print("="*60)
        
        dataset = GeneExpressionDataset(self.tokenized_data)
        collator = DataCollatorForCellClassification(add_cls=self.add_cls)
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collator,
            shuffle=False,
            num_workers=0
        )
        
        print(f"DataLoader创建完成，共 {len(self.dataloader)} 个batch")
        self.stats["DataLoader批次数量"] = len(self.dataloader)

    def extract_attention_weights(self):
        """Extract attention weights from specified layer and save as TSV"""
        print("\n" + "="*60)
        print(f"7️⃣ 提取注意力权重（Layer {TARGET_LAYER}）")
        print("="*60)
        
        # 基因信息
        ori_gene_names = [name.strip() for name in self.adata.var_names.tolist()]
        filtered_gene_count = len(ori_gene_names)
        print(f"处理基因数: {filtered_gene_count}")
        
        # 初始化统计
        dict_sum_condition = {}
        processed_cell_count = 0
        condition_ids = np.array(self.adata.obs["condition"].tolist())
        
        # 记录开始时间
        start_time = datetime.now()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.dataloader, desc="Processing batches")):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids,
                    attention_mask,
                    output_attentions=True
                )
                
                attn_scores = outputs.attentions[TARGET_LAYER]  # [batch, heads, seq, seq]
                num_heads = attn_scores.size(1)
                
                # 移除CLS token
                if self.add_cls:
                    attn_scores = attn_scores[..., 1:, 1:]
                    batch_data["sorted_indices"] = batch_data["sorted_indices"][:, 1:]
                    batch_data["input_ids"] = batch_data["input_ids"][:, 1:]
                
                M = attn_scores.shape[-1]
                if M == 0:
                    print(f"⚠️  空注意力矩阵，跳过该batch")
                    continue
                
                # 检查NaN/Inf
                if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                    print(f"⚠️  注意力分数包含NaN/Inf，跳过该batch")
                    continue
                
                # 行Rank归一化
                attn_scores = attn_scores.reshape((-1, M))
                order = torch.argsort(attn_scores, dim=1)
                rank = torch.argsort(order, dim=1)
                attn_scores = rank.reshape((-1, num_heads, M, M)).float() / M
                
                # 列Rank归一化
                attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
                order = torch.argsort(attn_scores, dim=1)
                rank = torch.argsort(order, dim=1)
                attn_scores = (rank.reshape((-1, num_heads, M, M)).float() / M).permute(0, 1, 3, 2)
                
                # 平均注意力头
                attn_scores = attn_scores.mean(1)
                
                # 恢复基因顺序
                sorted_indices = batch_data["sorted_indices"].to(attn_scores.device)
                attn_scores = reverse_permute(attn_scores, sorted_indices)
                
                # 转换为numpy
                attn_scores_np = attn_scores.cpu().numpy()
                sorted_indices_np = sorted_indices.cpu().numpy()
                
                # 应用mask
                mask_2d = attention_mask.cpu().numpy()[:, 1:] if self.add_cls else attention_mask.cpu().numpy()
                mask_matrix = mask_2d[:, :, None] * mask_2d[:, None, :]
                attn_scores_np = attn_scores_np * mask_matrix
                
                # 映射到完整基因空间
                batch_size_curr = attn_scores_np.shape[0]
                n_genes = filtered_gene_count
                full_attn = np.zeros((batch_size_curr, n_genes, n_genes), dtype=np.float32)
                
                for b in range(batch_size_curr):
                    indices = sorted_indices_np[b]
                    indices = np.clip(indices, 0, n_genes - 1)  # 防止索引越界
                    
                    for i, idx_i in enumerate(indices):
                        for j, idx_j in enumerate(indices):
                            if 0 <= idx_i < n_genes and 0 <= idx_j < n_genes:
                                full_attn[b, idx_i, idx_j] = attn_scores_np[b, i, j]
                
                # 累加
                batch_idx_np = batch_data["idx"].numpy()
                batch_conditions = condition_ids[batch_idx_np]
                
                for index, c in enumerate(batch_conditions):
                    if c not in dict_sum_condition:
                        dict_sum_condition[c] = full_attn[index, :, :].copy()
                    else:
                        dict_sum_condition[c] += full_attn[index, :, :]
                
                processed_cell_count += batch_size_curr
                
                # 清理显存
                del outputs, attn_scores, input_ids, attention_mask
                torch.cuda.empty_cache()
        
        # 计算运行时间
        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds() / 60
        print(f"\n注意力提取完成，共处理 {processed_cell_count} 个细胞")
        print(f"运行时间: {run_time:.2f} 分钟")
        
        self.stats["处理细胞数"] = processed_cell_count
        self.stats["运行时间(分钟)"] = f"{run_time:.2f}"
        
        # 平均注意力权重
        if not dict_sum_condition:
            raise ValueError("没有提取到有效注意力权重")
        
        # 处理每个condition（默认只有一个condition）
        tsv_gene_pair_count = 0
        covered_reference_gene1_count = 0
        attention_matrix = None
        
        for condition, attn_sum in dict_sum_condition.items():
            n_cells_processed = processed_cell_count
            if n_cells_processed == 0:
                print(f"⚠️  条件 {condition} 没有处理成功的细胞")
                continue
            
            # 平均
            attention_matrix = attn_sum / n_cells_processed
            print(f"\n条件 {condition}:")
            print(f"  注意力矩阵形状: {attention_matrix.shape}")
            print(f"  数值范围: [{np.nanmin(attention_matrix):.6f}, {np.nanmax(attention_matrix):.6f}]")
            print(f"  平均值: {np.nanmean(attention_matrix):.6f}")
            print(f"  中位数: {np.nanmedian(attention_matrix):.6f}")
            
            # 保存为TSV
            tsv_info = self._save_attention_as_tsv(attention_matrix, ori_gene_names, condition)
            tsv_gene_pair_count = tsv_info["gene_pair_count"]
            covered_reference_gene1_count = tsv_info["covered_reference_gene1_count"]
        
        # 更新统计
        self.stats["提取的总边数"] = tsv_gene_pair_count
        self.stats["筛选后（Gene1在Label中）的边数"] = tsv_gene_pair_count  # LangCell已在保存时筛选
        self.stats["筛选率(%)"] = f"100.0" if self.reference_genes else f"{(tsv_gene_pair_count/(filtered_gene_count*(filtered_gene_count-1)))*100:.1f}"
        self.stats["覆盖参考Gene1基因数"] = covered_reference_gene1_count
        self.stats["参考Gene1覆盖率(%)"] = f"{(covered_reference_gene1_count/self.reference_gene1_count)*100:.1f}" if self.reference_gene1_count > 0 else "0.0"
        self.stats["权重最小值"] = f"{np.nanmin(attention_matrix):.6f}" if attention_matrix is not None else "0.0"
        self.stats["权重最大值"] = f"{np.nanmax(attention_matrix):.6f}" if attention_matrix is not None else "0.0"
        self.stats["权重均值"] = f"{np.nanmean(attention_matrix):.6f}" if attention_matrix is not None else "0.0"
        self.stats["权重中位数"] = f"{np.nanmedian(attention_matrix):.6f}" if attention_matrix is not None else "0.0"
        self.stats["基因关系文件路径"] = str(interactions_tsv_path)
        
        return attention_matrix

    def _save_attention_as_tsv(self, attention_matrix, gene_names, condition):
        """将注意力权重保存为TSV文件（按参考网络筛选）"""
        print(f"\n8️⃣ 保存TSV文件（条件: {condition}）")
        
        tsv_data = []
        n_genes = len(gene_names)
        covered_reference_genes = set()
        
        # 遍历基因对
        for i in tqdm(range(n_genes), desc="Preparing TSV data"):
            gene1 = gene_names[i]
            
            # 按参考网络筛选Gene1
            if self.reference_genes and gene1 not in self.reference_genes:
                continue
            if self.reference_genes:
                covered_reference_genes.add(gene1)
            
            for j in range(n_genes):
                gene2 = gene_names[j]
                if gene1 == gene2:  # 跳过自环
                    continue
                
                score = attention_matrix[i, j]
                if score > 0:  # 只保留非零分数
                    tsv_data.append({
                        'Gene1': gene1,
                        'Gene2': gene2,
                        'Attention_score': round(score, 6)
                    })
        
        # 创建DataFrame并排序
        df_tsv = pd.DataFrame(tsv_data)
        df_tsv = df_tsv.sort_values('Attention_score', ascending=False).reset_index(drop=True)
        
        # 保存
        df_tsv.to_csv(interactions_tsv_path, sep='\t', index=False, header=True)
        print(f"✅ TSV文件保存至: {interactions_tsv_path}")
        
        # 统计
        gene_pair_count = len(df_tsv)
        covered_reference_gene1_count = len(covered_reference_genes)
        reference_coverage_percentage = (covered_reference_gene1_count / self.reference_gene1_count) * 100 if self.reference_gene1_count > 0 else 0.0
        
        print(f"  基因对数量: {gene_pair_count:,}")
        print(f"  Top 5基因对:")
        for _, row in df_tsv.head().iterrows():
            print(f"    {row['Gene1']} - {row['Gene2']}: {row['Attention_score']:.6f}")
        
        if self.reference_genes:
            print(f"  参考Gene1覆盖数: {covered_reference_gene1_count}/{self.reference_gene1_count} ({reference_coverage_percentage:.1f}%)")
        
        return {
            "tsv_path": str(interactions_tsv_path),
            "gene_pair_count": gene_pair_count,
            "covered_reference_gene1_count": covered_reference_gene1_count,
            "reference_coverage_percentage": reference_coverage_percentage
        }

    def save_stats_to_csv(self):
        """将统计参数追加到总CSV文件"""
        print("\n" + "="*60)
        print("9️⃣ 记录运行参数")
        print("="*60)
        
        # 转换为DataFrame
        stats_df = pd.DataFrame([self.stats])
        
        # 追加到总CSV
        if ALL_PARAMS_CSV.exists():
            stats_df.to_csv(ALL_PARAMS_CSV, mode='a', header=False, index=False, encoding="utf-8")
            print(f"✓ 参数已追加到总文件: {ALL_PARAMS_CSV}")
        else:
            stats_df.to_csv(ALL_PARAMS_CSV, mode='w', header=True, index=False, encoding="utf-8")
            print(f"✓ 已创建总参数文件并写入: {ALL_PARAMS_CSV}")
        
        # 打印关键统计
        print(f"\n关键参数摘要:")
        print(f"  - 数据集: {data_type}-{dataset}")
        print(f"  - 处理细胞数: {self.stats['处理细胞数']}")
        print(f"  - 最终基因数: {self.stats['最终用于模型的基因数']}")
        print(f"  - 输出基因对数量: {self.stats['提取的总边数']}")
        print(f"  - 运行时间: {self.stats['运行时间(分钟)']}分钟")

    def clean_temp_files(self):
        """清理临时文件"""
        print("\n" + "="*60)
        print("🔟 清理临时文件")
        print("="*60)
        
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print(f"✓ 删除临时目录: {TEMP_DIR}")
        
        # 尝试删除上级临时目录（如果为空）
        temp_parent = TEMP_DIR.parent
        if temp_parent.exists() and not any(temp_parent.iterdir()):
            os.rmdir(temp_parent)
            print(f"✓ 删除空临时父目录: {temp_parent}")

# ==================== Main Function ====================
def main():
    try:
        # 初始化提取器
        extractor = LangCellAttentionExtractor()
        
        # 执行完整流程
        extractor.load_vocab()
        extractor.load_model()
        extractor.load_csv_data()
        extractor.map_genes_to_vocab()
        extractor.tokenize_data()
        extractor.create_dataloader()
        extractor.extract_attention_weights()
        extractor.save_stats_to_csv()
        extractor.clean_temp_files()
        
        print("\n" + "="*80)
        print(f"✅ {MODEL_NAME}-{data_type}-{dataset} 所有处理完成！")
        print(f"输出文件汇总:")
        print(f"  - 总参数文件: {ALL_PARAMS_CSV}")
        print(f"  - 基因关系文件: {interactions_tsv_path}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        # 清理临时文件
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print(f"✓ 已清理临时文件: {TEMP_DIR}")
        sys.exit(1)

if __name__ == "__main__":
    main()

