
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import os
from pathlib import Path
import sys
import warnings
import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed

# ==================== 环境与警告配置 ====================
os.environ["KMP_WARNINGS"] = "off"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

# ==================== CLI / 全局默认配置（禁止硬编码绝对路径） ====================
import argparse
from typing import Optional


def _env_or(argval: Optional[str], env_key: str) -> Optional[str]:
    return argval if argval not in (None, "") else os.environ.get(env_key)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract scGPT hidden-state gene embeddings and export cosine-edge TSVs (Gene1/Gene2/EdgeWeight)."
    )
    p.add_argument(
        "--input-root",
        type=str,
        default=os.environ.get("FOUNDBENCH_INPUT_ROOT"),
        help="Input root directory containing CHIP/Non_CHIP/STRING subfolders with *ExpressionData.csv files.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=os.environ.get("FOUNDBENCH_OUTPUT_ROOT"),
        help="Output root directory. Each dataset writes one {MODEL_NAME}_{DATASET}.tsv plus run_params.json.",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("FOUNDBENCH_SCGPT_MODEL_DIR"),
        help="scGPT model directory containing vocab.json, args.json, best_model.pt.",
    )
    p.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=["CHIP"],
        help='Subfolders under input-root to process (e.g. CHIP Non_CHIP STRING). Default: ["CHIP"].',
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--batch-size", type=int, default=8, help="Forward batch size.")
    p.add_argument("--seq-topk-genes", type=int, default=512, help="Top-K expressed genes per cell to build sequences.")
    p.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length.")
    p.add_argument("--n-cells", type=int, default=None, help="Use only N sampled cells (default: all).")
    p.add_argument("--use-log1p", action="store_true", help="Apply log1p to expression values before ranking.")
    p.add_argument("--no-save-all-edges", action="store_true", help="Disable all-edges export (use topK per gene).")
    p.add_argument("--topk-per-gene", type=int, default=1000, help="When not saving all edges, topK edges per Gene1.")
    p.add_argument("--clear-cache-every", type=int, default=10, help="Clear CUDA cache every N batches.")
    args = p.parse_args()

    if not args.input_root or not args.output_root or not args.model_dir:
        p.error(
            "Missing required paths. Provide --input-root/--output-root/--model-dir "
            "or set env FOUNDBENCH_INPUT_ROOT/FOUNDBENCH_OUTPUT_ROOT/FOUNDBENCH_SCGPT_MODEL_DIR."
        )
    return args


MODEL_NAME = "scGPT"

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
N_BINS = 51
PAD_VALUE = -2

CONFIG = {}

# ==================== 1. 初始化 scGPT 模型（只加载一次） ====================
def init_scgpt_model(model_dir: Path, seed: int):
    set_seed(seed)

    vocab_file = model_dir / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_file}")
    vocab = GeneVocab.from_file(vocab_file)

    # 补特殊 token
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
            print(f"⚠️ 补充缺失的特殊Token: {s}")

    model_config_file = model_dir / "args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用计算设备: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()

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

    model_file = model_dir / "best_model.pt"
    try:
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ 完整加载模型权重: {model_file}")
    except Exception as e:
        print(f"⚠️ 完整加载失败，尝试部分加载: {str(e)[:80]}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"✅ 成功加载 {len(pretrained_dict)}/{len(model_dict)} 个匹配参数")

    model = model.to(device)
    model.eval()

    # transformer encoder half（GPU 优化）
    if device.type == "cuda":
        model.transformer_encoder = model.transformer_encoder.half()
        print("✅ Transformer Encoder 转换为 float16（GPU优化）")

    return model, vocab, device, model_configs


MODEL = None
VOCAB = None
DEVICE = None
MODEL_CONFIGS = None

# ==================== 2. 工具函数 ====================
def extract_dataset_name(file_path: str) -> str:
    base = os.path.basename(file_path)
    for suffix in ["_chip_matched-ExpressionData.csv", "_processed-ExpressionData.csv", "-ExpressionData.csv"]:
        if suffix in base:
            return base.split(suffix)[0]
    return base.split(".csv")[0]


def clear_gpu_cache(force=False):
    if DEVICE is not None and DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        if force:
            torch.cuda.ipc_collect()
        # 少打一点 log，避免刷屏
        # print("🔧 已清理GPU显存缓存")


def read_expression_matrix(path: str) -> pd.DataFrame:
    """
    统一读取：返回 df [cells x genes]
    期望文件形态：行=gene, 列=cell（index_col=0） -> 读入后转置
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

    # to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty expression matrix after parsing: {df.shape}")

    # 转置：cells x genes
    df = df.T
    print(f"📊 Expression loaded: {df.shape[0]} cells x {df.shape[1]} genes")
    return df


# ==================== 3. 序列构建 ====================
def build_gene_sequences(expr_df: pd.DataFrame, vocab: GeneVocab, config: dict):
    """
    expr_df: [cells x genes]
    返回:
      sequences: List[List[int]]
      valid_gene_symbols: List[str]
      token_to_gene: Dict[int, str]
      used_cells: int
    """
    genes = expr_df.columns.tolist()
    expr_matrix = expr_df.values.astype(np.float32)
    n_cells, n_genes = expr_matrix.shape
    print(f"📊 Expression matrix: {n_cells} cells x {n_genes} genes")

    # gene -> token / token -> gene
    gene_to_token = {}
    token_to_gene = {}
    valid_gene_idx = []

    for i, gene in enumerate(genes):
        if gene in vocab:
            tid = int(vocab[gene])
            gene_to_token[gene] = tid
            token_to_gene[tid] = gene
            valid_gene_idx.append(i)

    print(f"✅ 匹配到的基因: {len(valid_gene_idx)}/{n_genes}")
    if len(valid_gene_idx) == 0:
        raise ValueError("⚠️ 无匹配的基因，无法继续处理！")

    expr_matrix = expr_matrix[:, valid_gene_idx]
    valid_gene_symbols = [genes[i] for i in valid_gene_idx]

    if config["USE_LOG1P"]:
        expr_matrix = np.log1p(np.maximum(expr_matrix, 0.0))

    # limit cells
    if config["N_CELLS"] is not None and int(config["N_CELLS"]) < n_cells:
        idx = np.random.choice(n_cells, int(config["N_CELLS"]), replace=False)
        expr_matrix = expr_matrix[idx, :]
        n_cells_use = int(config["N_CELLS"])
    else:
        n_cells_use = n_cells

    print(f"🔬 使用细胞数: {n_cells_use}")

    topk = min(int(config["SEQ_TOPK_GENES"]), len(valid_gene_symbols))
    max_len = int(config["MAX_SEQ_LEN"])

    # 每个细胞 topk indices (desc)
    topk_indices = np.argsort(expr_matrix, axis=1)[:, -topk:][:, ::-1]

    sequences = []
    for cell_idx in range(n_cells_use):
        seq = [gene_to_token[valid_gene_symbols[i]] for i in topk_indices[cell_idx]]
        sequences.append(seq[:max_len])

    return sequences, valid_gene_symbols, token_to_gene, n_cells_use


# ==================== 4. Hidden 提取（内存聚合） ====================
def extract_hidden_embeddings(model, sequences, gene_symbols, token_to_gene, vocab, config):
    """
    返回：
      gene_embeddings: [n_final_genes, hidden_dim]
      final_gene_list
    """
    hidden_dim = int(MODEL_CONFIGS["embsize"])
    batch_size = int(config["BATCH_SIZE"])
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    # 累加：token_id -> sum/cnt
    sum_vec = {}
    cnt = {}

    pad_id = int(vocab[PAD_TOKEN])

    print(f"\n🚀 Forward scGPT (batch_size={batch_size}, total_batches={n_batches})")

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Forward scGPT"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_seqs = sequences[start_idx:end_idx]
            if not batch_seqs:
                continue

            max_len = max(len(seq) for seq in batch_seqs)
            input_ids = []
            values = []

            for seq in batch_seqs:
                pad_len = max_len - len(seq)
                input_ids.append(seq + [pad_id] * pad_len)
                values.append([1.0] * len(seq) + [PAD_VALUE] * pad_len)

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
            values = torch.tensor(values, dtype=torch.float, device=DEVICE)

            # encoder + value encoder
            src = model.encoder(input_ids)
            val_embs = model.value_encoder(values)
            total_embs = src + val_embs

            # bn if exists
            if hasattr(model, "bn") and model.bn is not None:
                total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

            src_key_padding_mask = input_ids.eq(pad_id)

            # float16 transformer (GPU)
            total_embs = total_embs.half()
            transformer_output = model.transformer_encoder(
                total_embs,
                src_key_padding_mask=src_key_padding_mask
            )
            hidden_states = transformer_output.float().detach().cpu().numpy()  # [B, L, H]
            ids_np = input_ids.detach().cpu().numpy()
            mask_np = (~src_key_padding_mask).detach().cpu().numpy()  # True for valid

            B, L, H = hidden_states.shape
            for bi in range(B):
                for li in range(L):
                    if not mask_np[bi, li]:
                        break
                    tid = int(ids_np[bi, li])
                    if tid == pad_id:
                        continue
                    vec = hidden_states[bi, li].astype(np.float64)
                    if tid not in sum_vec:
                        sum_vec[tid] = vec
                        cnt[tid] = 1
                    else:
                        sum_vec[tid] += vec
                        cnt[tid] += 1

            if (batch_idx + 1) % int(config["CLEAR_CACHE_EVERY"]) == 0:
                clear_gpu_cache()

    # token -> gene -> embedding
    gene_emb = {}
    for tid, vec in sum_vec.items():
        gene = token_to_gene.get(int(tid), None)
        if gene is None:
            continue
        gene_emb[gene] = (vec / max(cnt.get(tid, 1), 1)).astype(np.float32)

    final_gene_list = sorted(gene_emb.keys())
    gene_embeddings = np.stack([gene_emb[g] for g in final_gene_list], axis=0)

    print(f"✅ 提取完成 | {len(final_gene_list)} genes x {hidden_dim} dim")

    clear_gpu_cache(force=True)
    return gene_embeddings, final_gene_list


# ==================== 5. 余弦边：流式写出（统一格式） ====================
def compute_and_save_all_cosine_edges_stream(gene_list, embeddings, output_tsv):
    """
    输出所有 i!=j 的有向边（流式写），列：Gene1 Gene2 EdgeWeight
    """
    gene_list = list(gene_list)
    emb = np.asarray(embeddings, dtype=np.float32)
    n_genes = len(gene_list)
    if n_genes < 2:
        raise ValueError("Need at least 2 genes to compute edges.")

    denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    E = emb / denom

    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    total_edges = 0
    with open(output_tsv, "w") as f:
        f.write("Gene1\tGene2\tEdgeWeight\n")
        for i in tqdm(range(n_genes), desc="All cosine edges (stream)"):
            sims = E[i] @ E.T
            sims[i] = -np.inf
            g1 = gene_list[i]
            for j in range(n_genes):
                if j == i:
                    continue
                f.write(f"{g1}\t{gene_list[j]}\t{float(sims[j]):.8f}\n")
            total_edges += (n_genes - 1)

    return {"mode": "all_stream", "n_genes": int(n_genes), "n_edges": int(total_edges)}


# ==================== 6. 单数据集处理（统一 IO 规则） ====================
def process_single_dataset(expr_path: str, folder_name: str, config: dict):
    start_time = datetime.datetime.now()
    dataset_name = extract_dataset_name(expr_path)

    # 输出目录：OUTPUT_ROOT/Folder/Dataset
    out_dir = OUTPUT_ROOT / folder_name / dataset_name
    out_dir.mkdir(exist_ok=True, parents=True)

    record = {
        "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model_Name": MODEL_NAME,
        "Folder_Name": folder_name,
        "Dataset_Name": dataset_name,
        "Input_File": str(expr_path),
        "Input_Genes_Count": 0,
        "Matched_Genes_Count": 0,
        "Final_Genes_Count": 0,
        "Embedding_Dim": 0,
        "Total_Edges_Generated": 0,
        "Output_TSV_Path": "",
        "Process_Status": "Success",
        "Process_Time_Seconds": 0.0,
        "Error_Message": "",
    }

    try:
        print(f"\n{'='*60}")
        print(f"🔍 开始处理数据集: {dataset_name} | Folder: {folder_name}")
        print(f"📂 输入文件: {expr_path}")

        # 1) read expression -> cells x genes
        df = read_expression_matrix(expr_path)
        record["Input_Genes_Count"] = int(df.shape[1])

        # 2) build sequences
        sequences, gene_symbols, token_to_gene, n_cells_used = build_gene_sequences(df, VOCAB, config)
        record["Matched_Genes_Count"] = int(len(gene_symbols))

        # 3) hidden embeddings (in memory)
        gene_embeddings, final_gene_list = extract_hidden_embeddings(
            MODEL, sequences, gene_symbols, token_to_gene, VOCAB, config
        )
        record["Final_Genes_Count"] = int(len(final_gene_list))
        record["Embedding_Dim"] = int(gene_embeddings.shape[1])

        # 4) only edges TSV: {MODEL_NAME}_{DATASET}.tsv
        edge_tsv = out_dir / f"{MODEL_NAME}_{dataset_name}.tsv"
        cosine_info = compute_and_save_all_cosine_edges_stream(final_gene_list, gene_embeddings, edge_tsv)
        record["Total_Edges_Generated"] = int(cosine_info["n_edges"])
        record["Output_TSV_Path"] = str(edge_tsv)
        print(f"✅ 边文件保存: {edge_tsv}")

        # 5) save run_params.json
        run_params = {
            "dataset": dataset_name,
            "folder": folder_name,
            "input_file": str(expr_path),
            "output_dir": str(out_dir),
            "outputs": {"edges_tsv": str(edge_tsv)},
            "model": {
                "model_name": MODEL_NAME,
                "model_dir": str(config["MODEL_DIR"]),
                "embsize": int(MODEL_CONFIGS.get("embsize", gene_embeddings.shape[1])),
                "nheads": int(MODEL_CONFIGS.get("nheads", -1)),
                "nlayers": int(MODEL_CONFIGS.get("nlayers", -1)),
                "d_hid": int(MODEL_CONFIGS.get("d_hid", -1)),
                "pad_token": PAD_TOKEN,
                "pad_value": PAD_VALUE,
                "n_bins": int(N_BINS),
            },
            "sequence": {
                "seq_topk_genes": int(config["SEQ_TOPK_GENES"]),
                "max_seq_len": int(config["MAX_SEQ_LEN"]),
                "use_log1p": bool(config["USE_LOG1P"]),
                "n_cells": config["N_CELLS"],
                "n_cells_used": int(n_cells_used),
                "batch_size": int(config["BATCH_SIZE"]),
            },
            "stats": {
                "n_input_genes": int(record["Input_Genes_Count"]),
                "n_matched_genes": int(record["Matched_Genes_Count"]),
                "n_final_genes_with_embedding": int(record["Final_Genes_Count"]),
                "embedding_dim": int(record["Embedding_Dim"]),
            },
            "cosine_export": cosine_info,
            "processing_time_seconds": round((datetime.datetime.now() - start_time).total_seconds(), 2),
        }

        with open(out_dir / "run_params.json", "w") as f:
            json.dump(run_params, f, indent=2, ensure_ascii=False)

        record["Process_Time_Seconds"] = run_params["processing_time_seconds"]
        print(f"✅ 数据集处理完成 | 总耗时: {record['Process_Time_Seconds']}秒")

    except Exception as e:
        traceback_msg = str(e)[:300]
        record["Process_Status"] = "Failed"
        record["Error_Message"] = traceback_msg
        print(f"❌ 处理失败: {dataset_name} | {traceback_msg}")
        import traceback
        traceback.print_exc()

    return record


# ==================== 7. 批量处理主函数（统一规则） ====================
def main():
    global MODEL, VOCAB, DEVICE, MODEL_CONFIGS, CONFIG
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    model_dir = Path(args.model_dir)

    CONFIG = {
        "SEQ_TOPK_GENES": int(args.seq_topk_genes),
        "MAX_SEQ_LEN": int(args.max_seq_len),
        "N_CELLS": args.n_cells,
        "USE_LOG1P": bool(args.use_log1p),
        "BATCH_SIZE": int(args.batch_size),
        "SAVE_ALL_EDGES": (not bool(args.no_save_all_edges)),
        "TOPK_PER_GENE": int(args.topk_per_gene),
        "CLEAR_CACHE_EVERY": int(args.clear_cache_every),
        "MODEL_DIR": str(model_dir),
    }

    output_root.mkdir(exist_ok=True, parents=True)
    print(f"\n🚀 启动 {MODEL_NAME} Hidden States 批量处理流程")
    print(f"📥 输入根目录: {input_root}")
    print(f"📤 输出根目录: {output_root}")
    print(f"📦 模型目录: {model_dir}")
    print(f"📂 Folders: {args.folders}")
    print(f"⚙️  当前配置: {json.dumps({k:v for k,v in CONFIG.items() if k!='MODEL_DIR'}, indent=2, ensure_ascii=False)}")

    # init model once
    MODEL, VOCAB, DEVICE, MODEL_CONFIGS = init_scgpt_model(model_dir=model_dir, seed=int(args.seed))

    # override module-level OUTPUT_ROOT/INPUT_ROOT behavior with local paths
    # (keep downstream functions unchanged by binding to globals here)
    globals()["INPUT_ROOT"] = input_root
    globals()["OUTPUT_ROOT"] = output_root

    all_records = []
    for folder in args.folders:
        folder_path = input_root / folder
        if not folder_path.exists():
            print(f"\n⚠️ 文件夹不存在，跳过: {folder_path}")
            continue

        print(f"\n{'='*60}")
        print(f"📂 处理文件夹: {folder}")

        if folder == "CHIP":
            expr_files = list(folder_path.glob("*_chip_matched-ExpressionData.csv"))
        else:
            expr_files = list(folder_path.glob("*_processed-ExpressionData.csv"))

        print(f"🔍 找到 {len(expr_files)} 个表达矩阵文件")

        for expr_file in expr_files:
            rec = process_single_dataset(str(expr_file), folder, CONFIG)
            all_records.append(rec)
            clear_gpu_cache(force=True)

    if all_records:
        summary_df = pd.DataFrame(all_records)
        summary_path = output_root / f"{MODEL_NAME}_processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n📊 汇总记录已保存: {summary_path}")

        success_count = sum(1 for r in all_records if r["Process_Status"] == "Success")
        fail_count = len(all_records) - success_count
        print(f"\n📈 处理统计: 成功 {success_count} / 失败 {fail_count}")

    print(f"\n{'='*60}")
    print(f"🎉 批量处理流程结束！结果保存在: {output_root}")


if __name__ == "__main__":
    main()
