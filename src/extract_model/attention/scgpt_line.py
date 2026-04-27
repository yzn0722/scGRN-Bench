import copy
import json
import os
from pathlib import Path
import sys
import warnings
import datetime
import glob

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, "/mnt/md0/yzn/DeepSEM-master/scGPT")

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed 
from scgpt.tokenizer import tokenize_and_pad_batch
from scipy.sparse import issparse

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# ==================== 固定配置（无需修改） ====================
# 输入根目录（ExpressionData和Network文件所在）
INPUT_ROOT = Path("/mnt/md0/yzn/Beeline-master/benchmark_SF/input_process1000")
# 输出根目录（固定）
OUTPUT_ROOT = Path("/mnt/md0/yzn/Beeline-master/benchmark_SF/model/output_att1000/scgpt")
# 模型目录（固定）
MODEL_DIR = Path("/mnt/checkpoints/scgpt_all_human_Oct15-09-07-2024_53m")
# 模型名称（固定为scgpt_att）
MODEL_NAME = "scgpt_att"
# 运行参数（可微调）
SEED = 42
BATCH_SIZE = 8
ATTN_LAYERS = 11
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
N_BINS = 51
PAD_VALUE = -2

# ==================== 1. 初始化模型（只加载一次，批量复用） ====================
def init_scgpt_model():
    """初始化scGPT模型（批量处理时只加载一次）"""
    set_seed(SEED)
    
    # 加载词汇表
    vocab_file = MODEL_DIR / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    
    # 加载模型配置
    model_config_file = MODEL_DIR / "args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=PAD_VALUE,
        n_input_bins=N_BINS,
        use_fast_transformer=True,
    )
    
    # 加载模型权重
    model_file = MODEL_DIR / "best_model.pt"
    try:
        model.load_state_dict(torch.load(model_file))
        print(f"✅ 完整加载模型权重：{model_file}")
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✅ 加载匹配的模型参数（{len(pretrained_dict)}个）")
    
    model.to(device)
    model.eval()
    return model, vocab, device, model_configs

# ==================== 2. 批量获取待处理文件 ====================
def get_all_expression_files():
    """遍历INPUT_ROOT，获取所有ExpressionData文件及对应Network文件"""
    task_list = []
    # 遍历CHIP/Non_CHIP/STRING子目录
    for data_type in ["CHIP", "Non_CHIP", "STRING"]:
        data_type_dir = INPUT_ROOT / data_type
        if not data_type_dir.exists():
            print(f"⚠️ 跳过不存在的目录：{data_type_dir}")
            continue
        
        # 匹配不同数据类型的ExpressionData文件
        if data_type == "CHIP":
            # CHIP格式：*_chip_matched-ExpressionData.csv
            expr_files = glob.glob(str(data_type_dir / "*_chip_matched-ExpressionData.csv"))
        else:
            # Non_CHIP/STRING格式：*_processed-ExpressionData.csv
            expr_files = glob.glob(str(data_type_dir / "*_processed-ExpressionData.csv"))
        
        for expr_file in expr_files:
            expr_file = Path(expr_file)
            # 提取数据集名
            if data_type == "CHIP":
                dataset_name = expr_file.stem.replace("_chip_matched-ExpressionData", "")
            else:
                dataset_name = expr_file.stem.replace("_processed-ExpressionData", "")
            
            # 匹配对应的Network文件
            if data_type == "CHIP":
                network_file = data_type_dir / f"{dataset_name}_chip_matched-network.csv"
            else:
                network_file = data_type_dir / f"{dataset_name}_processed-network.csv"
            
            if not network_file.exists():
                print(f"❌ 跳过{dataset_name}：Network文件不存在 {network_file}")
                continue
            
            task_list.append({
                "data_type": data_type,
                "dataset_name": dataset_name,
                "expr_file": expr_file,
                "network_file": network_file
            })
    
    print(f"\n📋 批量任务列表：共{len(task_list)}个数据集待处理")
    for idx, task in enumerate(task_list):
        print(f"  {idx+1}. {task['data_type']}/{task['dataset_name']}")
    return task_list

# ==================== 3. 处理单个数据集 ====================
def process_single_dataset(model, vocab, device, model_configs, task):
    """处理单个数据集的注意力提取"""
    data_type = task["data_type"]
    dataset_name = task["dataset_name"]
    expr_file = task["expr_file"]
    network_file = task["network_file"]
    
    # 输出路径（固定结构：OUTPUT_ROOT/data_type/）
    output_dir = OUTPUT_ROOT / data_type
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 文件前缀（固定模型名+数据集名：scgpt_att_数据集名）
    file_prefix = f"{MODEL_NAME}_{dataset_name}"
    
    try:
        # ==================== 数据预处理 ====================
        print(f"\n🔍 开始处理：{data_type}/{dataset_name}")
        print(f"   表达矩阵：{expr_file}")
        print(f"   Network文件：{network_file}")
        
        # 读取并转置表达矩阵
        df = pd.read_csv(expr_file, index_col=0)
        df = df.T  # 行=细胞，列=基因
        genes = df.columns.tolist()
        all_counts = df.values
        
        # 转换为AnnData
        adata = AnnData(
            X=all_counts,
            obs=pd.DataFrame(index=df.index),
            var=pd.DataFrame(index=genes)
        )
        print(f"   数据规模：{adata.n_obs}细胞 | {adata.n_vars}基因")
        
        # ==================== Tokenization ====================
        all_counts = adata.X.A if issparse(adata.X) else adata.X
        gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)
        
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=len(genes) + 1,
            vocab=vocab,
            pad_token=PAD_TOKEN,
            pad_value=PAD_VALUE,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[PAD_TOKEN])
        
        # ==================== 提取注意力权重 ====================
        torch.cuda.empty_cache()
        nhead = model_configs["nheads"]
        attention_sum = None
        num_cells = 0
        M = all_gene_ids.size(1)  # 序列长度
        N = all_gene_ids.size(0)  # 细胞数
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for i in tqdm(range(0, N, BATCH_SIZE), desc=f"   Batch处理"):
                current_batch_size = min(BATCH_SIZE, N - i)
                
                # 当前batch数据
                batch_gene_ids = all_gene_ids[i:i+current_batch_size].to(device)
                batch_values = all_values[i:i+current_batch_size].to(device)
                batch_mask = src_key_padding_mask[i:i+current_batch_size].to(device)
                
                # 初始embedding
                src_embs = model.encoder(batch_gene_ids)
                val_embs = model.value_encoder(batch_values)
                total_embs = src_embs + val_embs
                
                # BatchNorm（如有）
                if hasattr(model, 'bn'):
                    total_embs = model.bn(total_embs.permute(0,2,1)).permute(0,2,1)
                
                # Transformer层前向
                for layer_idx in range(ATTN_LAYERS):
                    layer = model.transformer_encoder.layers[layer_idx]
                    
                    # Self-Attention
                    qkv = layer.self_attn.Wqkv(total_embs)
                    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=nhead)
                    q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
                    
                    q = q.permute(0,2,1,3)
                    k = k.permute(0,2,1,3)
                    v = v.permute(0,2,1,3)
                    
                    # 注意力分数
                    scale = 1.0 / np.sqrt(q.shape[-1])
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                    if batch_mask is not None:
                        mask_expanded = batch_mask.unsqueeze(1).unsqueeze(2)
                        attn_weights = attn_weights.masked_fill(mask_expanded, -65504.0)
                    
                    attn_probs = torch.softmax(attn_weights, dim=-1)
                    
                    # 注意力输出
                    attn_output = torch.matmul(attn_probs, v)
                    attn_output = attn_output.permute(0,2,1,3)
                    attn_output = attn_output.reshape(current_batch_size, M, -1)
                    attn_output = layer.self_attn.out_proj(attn_output)
                    
                    # 残差+LayerNorm
                    total_embs = total_embs + attn_output
                    total_embs = layer.norm1(total_embs)
                    
                    # Feed-Forward
                    ffn_output = layer.linear2(layer.activation(layer.linear1(total_embs)))
                    total_embs = total_embs + ffn_output
                    total_embs = layer.norm2(total_embs)
                
                # 提取最后一层注意力
                final_layer = model.transformer_encoder.layers[ATTN_LAYERS]
                qkv = final_layer.self_attn.Wqkv(total_embs)
                qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=nhead)
                q, k = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :]
                
                # 最终注意力分数
                attn_scores = q.permute(0,2,1,3) @ k.permute(0,2,3,1)
                if batch_mask is not None:
                    mask_expanded = batch_mask.unsqueeze(1).unsqueeze(2)
                    attn_scores = attn_scores.masked_fill(mask_expanded, -65504.0)
                
                scale = 1.0 / np.sqrt(q.shape[-1])
                attn_probs = torch.softmax(attn_scores * scale, dim=-1)
                attn_scores_final = attn_probs.mean(1).detach().cpu().numpy()
                
                # 累加
                if attention_sum is None:
                    attention_sum = attn_scores_final.sum(axis=0)
                else:
                    attention_sum += attn_scores_final.sum(axis=0)
                num_cells += current_batch_size
                
                # 清理缓存
                if i % (BATCH_SIZE * 10) == 0:
                    torch.cuda.empty_cache()
        
        # 平均注意力
        avg_attention = attention_sum / num_cells
        gene_attention = avg_attention[1:, 1:]  # 去掉<cls> token
        
        # ==================== 仅保存筛选后TSV ====================
        # 原始交互（仅中间使用，不保存）
        interactions_df = pd.DataFrame(
            gene_attention, index=genes, columns=genes
        ).stack().reset_index()
        interactions_df.columns = ["Gene1", "Gene2", "EdgeWeight"]
        interactions_df = interactions_df[interactions_df["Gene1"] != interactions_df["Gene2"]]
        interactions_df = interactions_df.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
        
        # 按Network筛选
        network_df = pd.read_csv(network_file)
        target_gene1 = network_df["Gene1"].unique().tolist()
        filtered_df = interactions_df[interactions_df["Gene1"].isin(target_gene1)]
        filtered_df = filtered_df.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
        
        # 仅保存筛选后TSV
        filtered_tsv = output_dir / f"{file_prefix}_filtered.tsv"
        filtered_df.to_csv(filtered_tsv, sep="\t", index=False)
        
        # ==================== 记录参数（移除原始文件相关） ====================
        params = {
            "运行时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "模型名称": MODEL_NAME,
            "数据类型": data_type,
            "数据集名": dataset_name,
            "表达矩阵路径": str(expr_file),
            "Network路径": str(network_file),
            "输出筛选文件": str(filtered_tsv),
            "基因数": len(genes),
            "细胞数": adata.n_obs,
            "原始交互数": len(interactions_df),
            "筛选后交互数": len(filtered_df),
            "筛选保留比例": f"{len(filtered_df)/len(interactions_df)*100:.2f}%"
        }
        
        # 保存参数
        params_csv = output_dir / f"{MODEL_NAME}_run_parameters.csv"
        if not params_csv.exists():
            pd.DataFrame([params]).to_csv(params_csv, index=False, encoding="utf-8")
        else:
            pd.DataFrame([params]).to_csv(params_csv, mode="a", header=False, index=False, encoding="utf-8")
        
        print(f"✅ 处理完成：")
        print(f"   筛选文件：{filtered_tsv}")
        return True
    
    except Exception as e:
        print(f"❌ 处理失败 {data_type}/{dataset_name}：{str(e)[:200]}")
        return False

# ==================== 4. 主函数（批量处理） ====================
def main():
    print("="*80)
    print("📦 scGPT批量注意力提取脚本（仅输出筛选后TSV）")
    print(f"模型名称：{MODEL_NAME}")
    print(f"输入根目录：{INPUT_ROOT}")
    print(f"输出根目录：{OUTPUT_ROOT}")
    print("="*80)
    
    # 1. 初始化模型（只加载一次）
    print("\n🔧 初始化scGPT模型...")
    model, vocab, device, model_configs = init_scgpt_model()
    
    # 2. 获取批量任务列表
    task_list = get_all_expression_files()
    if not task_list:
        print("❌ 无有效任务，退出")
        return
    
    # 3. 批量处理
    success_count = 0
    fail_count = 0
    print("\n🚀 开始批量处理...")
    for task in task_list:
        if process_single_dataset(model, vocab, device, model_configs, task):
            success_count += 1
        else:
            fail_count += 1
    
    # 4. 汇总结果
    print("\n" + "="*80)
    print("📊 批量处理汇总")
    print("="*80)
    print(f"总任务数：{len(task_list)}")
    print(f"成功数：{success_count}")
    print(f"失败数：{fail_count}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print("⚠️ 仅生成_filtered.tsv文件，未生成原始.tsv文件")
    print("="*80)

if __name__ == "__main__":
    main()