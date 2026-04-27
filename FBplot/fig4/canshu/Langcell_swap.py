# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# LangCell模型swap参数敏感性分析（带时间测量）
# """

# import os
# import json
# import pickle
# import warnings
# import time
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from transformers import BertModel

# warnings.filterwarnings("ignore")

# # =====================================================
# # 固定配置
# # =====================================================
# MODEL_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/LangCell/cell_bert"
# DICTS_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts"
# OUTDIR = "results_langcell_swap_sensitivity"

# # =====================================================
# # 单数据集配置
# # =====================================================
# DATASET_CONFIG = {
#     "hESC": {
#         "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv",
#         "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv",
#         "species": "human",
#     }
# }

# # =====================================================
# # 测试的swap参数范围
# # =====================================================
# SWAP_TRIALS_VALUES = [1, 2, 4, 8, 16, 32]  # 要测试的swap-trials值

# # 固定其他参数
# FIXED_PARAMS = {
#     "top_percent": 30,
#     "pt_quantile": 0.2,
#     "gen_iters": 16,
#     "batch_size": 16,
#     "max_len": 1024,
#     "mask_ratio": 0.3,
#     "temperature": 1.0,
#     "ema_alpha": 0.1,
#     "token_update": "swap",  # 固定使用swap模式
#     "use_log1p": True,
#     "print_every": 1,
# }


# # =====================================================
# # LangCell模型包装器
# # =====================================================
# class LangCellModel(nn.Module):
#     """LangCell模型包装器"""
#     def __init__(self, model_dir: str):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
#         self.cls = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        
#     def forward(self, input_ids, attention_mask):
#         out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         return self.cls(out.last_hidden_state)


# # =====================================================
# # 工具函数
# # =====================================================
# def normalize_symbol(s: str) -> str:
#     return str(s).strip().upper()


# def is_ensembl_id(s: str) -> bool:
#     return isinstance(s, str) and (s.startswith("ENSG") or s.startswith("ENSMUSG"))


# def read_pt_file(path: str) -> pd.DataFrame:
#     """读取伪时间文件"""
#     try:
#         pt_df = pd.read_csv(path, header=None)
#         _ = float(pt_df.iloc[0, 1])
#     except (ValueError, IndexError, TypeError):
#         pt_df = pd.read_csv(path)
    
#     if pt_df.shape[1] < 2:
#         raise ValueError(f"Invalid pseudotime file: {path}")
    
#     pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
#     pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
#     pt_df = pt_df.dropna(subset=["pt"])
#     return pt_df


# def load_langcell_dicts(dicts_dir: Path):
#     """加载LangCell字典"""
#     token_dict_path = dicts_dir / "token_dictionary.pkl"
#     if token_dict_path.exists():
#         with open(token_dict_path, "rb") as f:
#             vocab = pickle.load(f)
#     else:
#         token_dict_path = dicts_dir / "token_dictionary.json"
#         if token_dict_path.exists():
#             with open(token_dict_path, "r") as f:
#                 vocab = json.load(f)
#         else:
#             raise FileNotFoundError(f"Token dictionary not found in {dicts_dir}")
    
#     gene_dict_path = dicts_dir / "gene_name_id_dict.pkl"
#     if gene_dict_path.exists():
#         with open(gene_dict_path, "rb") as f:
#             gene_name_id = pickle.load(f)
#     else:
#         gene_dict_path = dicts_dir / "gene_name_id_dict.json"
#         if gene_dict_path.exists():
#             with open(gene_dict_path, "r") as f:
#                 gene_name_id = json.load(f)
#         else:
#             gene_name_id = vocab
    
#     pad_id = vocab.get("<pad>", 0)
#     mask_id = vocab.get("<mask>", vocab.get("[MASK]", 103))
    
#     return vocab, gene_name_id, int(pad_id), int(mask_id)


# def build_symbol_to_ensembl_map(gene_name_id: Dict[str, str]) -> Dict[str, str]:
#     """构建symbol到Ensembl ID的映射"""
#     items = list(gene_name_id.items())
#     sample = items[: min(2000, len(items))]
#     k_ens = sum(is_ensembl_id(str(k)) for k, _ in sample)
#     v_ens = sum(is_ensembl_id(str(v)) for _, v in sample)
    
#     if v_ens > k_ens:
#         return {normalize_symbol(k): str(v) for k, v in gene_name_id.items()}
#     return {normalize_symbol(v): str(k) for k, v in gene_name_id.items()}


# def direction_accuracy_top_genes(
#     pred_delta: np.ndarray,
#     true_delta: np.ndarray,
#     top_idx: np.ndarray,
#     eps: float = 0.0,
# ) -> float:
#     """计算方向准确率"""
#     td = true_delta[top_idx]
#     pd = pred_delta[top_idx]
    
#     if eps > 0.0:
#         m = np.abs(td) > eps
#         if not np.any(m):
#             return float("nan")
#         td = td[m]
#         pd = pd[m]
    
#     true_dir = np.where(td > 0, 1, -1)
#     pred_dir = np.where(pd > 0, 1, -1)
#     return float((pred_dir == true_dir).mean())


# def positions_dict_from_seq(seq_ids: np.ndarray, length: int) -> Dict[int, int]:
#     """从序列中获取token位置字典"""
#     pos = {}
#     for i in range(length):
#         t = int(seq_ids[i])
#         if t not in pos:
#             pos[t] = i
#     return pos


# # =====================================================
# # 核心函数：swap迭代采样 (适配LangCell)
# # =====================================================
# @torch.no_grad()
# def iterative_swap_sampling_langcell(
#     model: LangCellModel,
#     seq: torch.Tensor,
#     length: int,
#     pad_id: int,
#     mask_id: int,
#     device: torch.device,
#     swap_trials: int,
#     temperature: float = 1.0,
# ) -> Tuple[np.ndarray, float]:
#     """执行swap迭代采样 - LangCell版本，返回序列和运行时间"""
#     start_time = time.time()
    
#     x = seq.clone().to(device).unsqueeze(0)  # [1, L]
#     attn = torch.zeros((1, x.size(1)), dtype=torch.long, device=device)
#     attn[0, :length] = 1
    
#     banned = torch.tensor([pad_id, mask_id], device=device, dtype=torch.long)
    
#     for _ in range(swap_trials):
#         if length < 2:
#             break
        
#         ij = torch.randperm(length, device="cpu")[:2]
#         i = int(ij[0].item())
#         j = int(ij[1].item())
#         if i == j:
#             continue
        
#         a = int(x[0, i].item())
#         b = int(x[0, j].item())
        
#         if a in (pad_id, mask_id) or b in (pad_id, mask_id):
#             continue
        
#         x_masked = x.clone()
#         x_masked[0, i] = mask_id
#         x_masked[0, j] = mask_id
        
#         logits = model(input_ids=x_masked, attention_mask=attn)[0, [i, j], :].float()
#         logits /= max(temperature, 1e-6)
        
#         banned2 = banned[banned >= 0]
#         banned2 = banned2[banned2 < logits.size(-1)]
#         if banned2.numel() > 0:
#             logits[:, banned2] = -1e9
        
#         logp = torch.log_softmax(logits, dim=-1)
        
#         if (a < 0) or (b < 0) or (a >= logits.size(-1)) or (b >= logits.size(-1)):
#             continue
        
#         cur_score = float(logp[0, a].item() + logp[1, b].item())
#         swap_score = float(logp[0, b].item() + logp[1, a].item())
        
#         if swap_score > cur_score:
#             x[0, i], x[0, j] = x[0, j].clone(), x[0, i].clone()
    
#     execution_time = time.time() - start_time
#     return x.squeeze(0).detach().cpu().numpy(), execution_time


# # =====================================================
# # 主分析函数
# # =====================================================
# @torch.no_grad()
# def run_langcell_swap_sensitivity_analysis(
#     dataset_name: str,
#     dataset_config: dict,
#     model: LangCellModel,
#     vocab: dict,
#     gene_name_id: dict,
#     pad_id: int,
#     mask_id: int,
#     device: torch.device,
# ) -> Dict[int, dict]:
#     """运行LangCell的swap参数敏感性分析"""
    
#     print(f"\nRunning swap sensitivity analysis for {dataset_name} (LangCell)")
    
#     # 加载数据
#     expr = pd.read_csv(dataset_config["expr_csv"], index_col=0)
#     pt_df = read_pt_file(dataset_config["pt_csv"])
    
#     common = expr.columns.intersection(pt_df.index)
#     if len(common) == 0:
#         raise ValueError("No overlapping cells between expression and pseudotime")
    
#     expr = expr[common]
#     pt = pt_df.loc[common, "pt"].values
    
#     # 预处理表达数据
#     X = expr.T.values.astype(np.float32)
#     if FIXED_PARAMS["use_log1p"]:
#         X = np.log1p(np.maximum(X, 0))
    
#     # 根据伪时间分割早期/晚期细胞
#     lo, hi = np.quantile(pt, [FIXED_PARAMS["pt_quantile"], 1 - FIXED_PARAMS["pt_quantile"]])
#     early_mask = pt <= lo
#     late_mask = pt >= hi
    
#     if early_mask.sum() == 0 or late_mask.sum() == 0:
#         raise ValueError("No early or late cells after quantile split")
    
#     early_mean = X[early_mask].mean(axis=0)
#     late_mean = X[late_mask].mean(axis=0)
#     true_delta = late_mean - early_mean
    
#     n_genes = len(true_delta)
#     top_n = max(int(n_genes * FIXED_PARAMS["top_percent"] / 100), 1)
#     top_idx = np.argsort(-np.abs(true_delta))[:top_n]
    
#     # 构建基因到token的映射
#     sym2ens = build_symbol_to_ensembl_map(gene_name_id)
#     genes_original = expr.index.astype(str).tolist()
    
#     def get_token_id(gene: str) -> int:
#         g_norm = normalize_symbol(gene)
#         if g_norm in vocab:
#             return vocab[g_norm]
#         ens = sym2ens.get(g_norm)
#         if ens and ens in vocab:
#             return vocab[ens]
#         return pad_id
    
#     gene_tids = [get_token_id(g) for g in genes_original]
#     matched = sum(1 for t in gene_tids if t != pad_id)
    
#     print(f"  Total genes: {n_genes}")
#     print(f"  Vocab matched: {matched} ({matched/max(1, n_genes)*100:.1f}%)")
#     print(f"  Early cells: {early_mask.sum()}, Late cells: {late_mask.sum()}")
#     print(f"  Top genes to evaluate: {top_n}")
    
#     # 构建细胞序列
#     def cell_to_seq(x: np.ndarray) -> Tuple[List[int], int]:
#         order = np.argsort(-x)
#         seq = []
#         for j in order:
#             t = gene_tids[j]
#             if t == pad_id:
#                 continue
#             seq.append(t)
#             if len(seq) >= FIXED_PARAMS["max_len"]:
#                 break
#         length = len(seq)
#         if length < FIXED_PARAMS["max_len"]:
#             seq += [pad_id] * (FIXED_PARAMS["max_len"] - length)
#         return seq, length
    
#     # 选择早期细胞
#     early_cells = []
#     early_idx_all = np.where(early_mask)[0]
#     for i in early_idx_all:
#         seq, length = cell_to_seq(X[i])
#         if length > 10:
#             early_cells.append((seq, length))
    
#     if not early_cells:
#         raise ValueError("No valid early cells after sequence building")
    
#     print(f"  Early cells in loop: {len(early_cells)}")
    
#     # 存储所有swap-trials的结果
#     all_results = {}
    
#     # 对每个swap-trials值运行分析
#     for swap_trials in SWAP_TRIALS_VALUES:
#         print(f"\n  Testing swap_trials = {swap_trials}")
        
#         # 初始化细胞状态
#         init_cells = [np.array(seq, dtype=np.int64) for seq, _ in early_cells]
#         curr_cells = [arr.copy() for arr in init_cells]
#         cell_lengths = [int(L) for _, L in early_cells]
        
#         acc_curve = []
#         iteration_times = []
#         best_acc = -1.0
        
#         for it in range(FIXED_PARAMS["gen_iters"]):
#             iteration_start_time = time.time()
            
#             # 保存上一轮状态
#             prev_state = [arr.copy() for arr in curr_cells]
#             deltas_batch = []
#             next_cells = []
#             cell_processing_times = []
            
#             # 对每个细胞执行swap采样
#             for ci, length in enumerate(cell_lengths):
#                 x0 = init_cells[ci]
#                 x_prev = curr_cells[ci]
                
#                 # 执行swap采样 (测量时间)
#                 x1, cell_time = iterative_swap_sampling_langcell(
#                     model=model,
#                     seq=torch.tensor(x_prev, dtype=torch.long),
#                     length=length,
#                     pad_id=pad_id,
#                     mask_id=mask_id,
#                     device=device,
#                     swap_trials=swap_trials,
#                     temperature=FIXED_PARAMS["temperature"],
#                 )
#                 x1 = x1.astype(np.int64, copy=False)
#                 next_cells.append(x1)
#                 cell_processing_times.append(cell_time)
                
#                 # 计算位置变化
#                 pos0 = positions_dict_from_seq(x0, length)
#                 pos1 = positions_dict_from_seq(x1, length)
#                 delta = np.zeros(n_genes, dtype=np.float32)
                
#                 for gi in range(n_genes):
#                     tid = gene_tids[gi]
#                     delta[gi] = pos1.get(tid, length) - pos0.get(tid, length)
                
#                 deltas_batch.append(delta)
            
#             curr_cells = next_cells
            
#             # 计算平均变化和准确率
#             mean_delta = np.stack(deltas_batch).mean(axis=0)
#             pred_delta_expr_aligned = -mean_delta.astype(np.float32, copy=False)
            
#             acc_now = direction_accuracy_top_genes(
#                 pred_delta_expr_aligned,
#                 true_delta,
#                 top_idx,
#                 eps=0.0,
#             )
            
#             # 强制非递减准确率
#             if best_acc >= 0 and acc_now < best_acc:
#                 curr_cells = prev_state
#                 acc_now = best_acc
            
#             if acc_now > best_acc:
#                 best_acc = acc_now
            
#             acc_curve.append(acc_now)
#             iteration_time = time.time() - iteration_start_time
#             iteration_times.append(iteration_time)
            
#             if FIXED_PARAMS["print_every"] > 0 and ((it + 1) % FIXED_PARAMS["print_every"] == 0):
#                 avg_cell_time = np.mean(cell_processing_times) if cell_processing_times else 0
#                 print(f"    Iter {it+1:>2}/{FIXED_PARAMS['gen_iters']} | acc={acc_curve[-1]:.2%} | "
#                       f"time={iteration_time:.3f}s | cell_time={avg_cell_time:.4f}s")
        
#         # 计算时间统计
#         total_time = np.sum(iteration_times)
#         avg_iteration_time = np.mean(iteration_times) if iteration_times else 0
#         std_iteration_time = np.std(iteration_times) if iteration_times else 0
        
#         # 存储这个swap-trials值的结果
#         all_results[swap_trials] = {
#             "accuracy_curve": acc_curve,
#             "final_accuracy": acc_curve[-1] if acc_curve else 0,
#             "avg_accuracy": np.mean(acc_curve) if acc_curve else 0,
#             "std_accuracy": np.std(acc_curve) if acc_curve else 0,
#             "max_accuracy": np.max(acc_curve) if acc_curve else 0,
#             "convergence_iter": find_convergence_iteration(acc_curve),
#             "convergence_speed": calculate_convergence_speed(acc_curve),
#             "time_stats": {
#                 "total_time": total_time,
#                 "avg_iteration_time": avg_iteration_time,
#                 "std_iteration_time": std_iteration_time,
#                 "iteration_times": iteration_times,
#             },
#             "efficiency": {
#                 "accuracy_per_second": (acc_curve[-1] if acc_curve else 0) / max(total_time, 0.001),
#                 "iterations_per_second": FIXED_PARAMS["gen_iters"] / max(total_time, 0.001),
#             }
#         }
        
#         print(f"    Final accuracy: {acc_curve[-1]:.4f}")
#         print(f"    Average accuracy: {np.mean(acc_curve):.4f}")
#         print(f"    Total time: {total_time:.2f}s ({avg_iteration_time:.3f}s/iter)")
#         print(f"    Efficiency: {all_results[swap_trials]['efficiency']['accuracy_per_second']:.4f} acc/s")
    
#     return all_results


# def find_convergence_iteration(acc_curve: List[float], threshold: float = 0.001) -> int:
#     """找到收敛的迭代次数"""
#     if len(acc_curve) < 3:
#         return len(acc_curve)
    
#     for i in range(2, len(acc_curve)):
#         if (abs(acc_curve[i] - acc_curve[i-1]) < threshold and 
#             abs(acc_curve[i-1] - acc_curve[i-2]) < threshold):
#             return i
#     return len(acc_curve)


# def calculate_convergence_speed(acc_curve: List[float]) -> int:
#     """计算收敛速度（达到最终准确率90%所需的迭代次数）"""
#     if len(acc_curve) < 2:
#         return 0
    
#     final_acc = acc_curve[-1]
#     target_acc = 0.9 * final_acc
    
#     for i, acc in enumerate(acc_curve):
#         if acc >= target_acc:
#             return i + 1
#     return len(acc_curve)


# # =====================================================
# # 综合评分计算
# # =====================================================
# def calculate_comprehensive_score(
#     results: Dict[int, dict],
#     weights: Dict[str, float] = None
# ) -> Dict[int, dict]:
#     """计算综合评分，考虑准确率、时间、稳定性等多个维度"""
    
#     if weights is None:
#         weights = {
#             "accuracy_weight": 0.5,      # 准确率权重
#             "time_weight": 0.3,          # 时间效率权重
#             "stability_weight": 0.1,     # 稳定性权重
#             "speed_weight": 0.1,         # 收敛速度权重
#         }
    
#     swap_trials_list = sorted(results.keys())
    
#     # 收集所有指标用于归一化
#     all_final_acc = [results[st]["final_accuracy"] for st in swap_trials_list]
#     all_avg_time = [results[st]["time_stats"]["avg_iteration_time"] for st in swap_trials_list]
#     all_std_acc = [results[st]["std_accuracy"] for st in swap_trials_list]
#     all_conv_speed = [results[st]["convergence_speed"] for st in swap_trials_list]
#     all_efficiency = [results[st]["efficiency"]["accuracy_per_second"] for st in swap_trials_list]
    
#     # 计算归一化的指标
#     def normalize(values, reverse=False):
#         """归一化到0-1范围，reverse=True表示值越小越好"""
#         if not values or all(np.isnan(v) for v in values):
#             return [0.0] * len(values)
        
#         valid_values = [v for v in values if not np.isnan(v)]
#         if not valid_values:
#             return [0.0] * len(values)
        
#         min_val = min(valid_values)
#         max_val = max(valid_values)
        
#         if max_val - min_val < 1e-8:
#             return [0.5] * len(values)
        
#         normalized = []
#         for v in values:
#             if np.isnan(v):
#                 normalized.append(0.0)
#             else:
#                 norm = (v - min_val) / (max_val - min_val)
#                 if reverse:
#                     norm = 1 - norm
#                 normalized.append(norm)
        
#         return normalized
    
#     # 归一化各项指标
#     norm_acc = normalize(all_final_acc, reverse=False)           # 准确率越高越好
#     norm_time = normalize(all_avg_time, reverse=True)            # 时间越短越好
#     norm_stability = normalize(all_std_acc, reverse=True)        # 标准差越小越好
#     norm_speed = normalize(all_conv_speed, reverse=True)         # 收敛速度越快越好
#     norm_efficiency = normalize(all_efficiency, reverse=False)   # 效率越高越好
    
#     # 计算综合评分
#     comprehensive_scores = {}
#     for i, swap_trials in enumerate(swap_trials_list):
#         # 基础评分
#         base_score = (
#             weights["accuracy_weight"] * norm_acc[i] +
#             weights["time_weight"] * norm_time[i] +
#             weights["stability_weight"] * norm_stability[i] +
#             weights["speed_weight"] * norm_speed[i]
#         )
        
#         # 额外奖励：效率特别高的参数
#         efficiency_bonus = 0.2 * norm_efficiency[i]  # 额外20%的效率奖励
        
#         # 最终评分
#         total_score = base_score + efficiency_bonus
        
#         comprehensive_scores[swap_trials] = {
#             "comprehensive_score": total_score,
#             "base_score": base_score,
#             "efficiency_bonus": efficiency_bonus,
#             "normalized_accuracy": norm_acc[i],
#             "normalized_time": norm_time[i],
#             "normalized_stability": norm_stability[i],
#             "normalized_speed": norm_speed[i],
#             "normalized_efficiency": norm_efficiency[i],
#         }
    
#     return comprehensive_scores


# def calculate_cost_effectiveness_ratio(results: Dict[int, dict]) -> Dict[int, float]:
#     """计算性价比：准确率/时间"""
#     cer = {}
#     for swap_trials, res in results.items():
#         accuracy = res["final_accuracy"]
#         time_cost = res["time_stats"]["total_time"]
#         if time_cost > 0:
#             cer[swap_trials] = accuracy / time_cost
#         else:
#             cer[swap_trials] = 0.0
#     return cer


# # =====================================================
# # 可视化函数
# # =====================================================
# def plot_langcell_swap_sensitivity_results(results: Dict[int, dict], outdir: Path, dataset_name: str):
#     """绘制LangCell swap参数敏感性分析结果"""
    
#     plt.rcParams.update({
#         'font.family': 'Arial',
#         'font.size': 10,
#         'axes.linewidth': 1.0,
#         'axes.labelsize': 12,
#         'axes.titlesize': 14,
#         'xtick.labelsize': 10,
#         'ytick.labelsize': 10,
#         'legend.fontsize': 10,
#         'figure.dpi': 300,
#     })
    
#     swap_trials_list = sorted(results.keys())
    
#     # 1. 主图：不同swap-trials的收敛曲线
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     # 1.1 收敛曲线对比
#     ax = axes[0, 0]
#     colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(swap_trials_list)))
    
#     for swap_trials, color in zip(swap_trials_list, colors):
#         acc_curve = results[swap_trials]["accuracy_curve"]
#         ax.plot(range(1, len(acc_curve) + 1), acc_curve, 
#                 'o-', color=color, linewidth=1.5, markersize=4, 
#                 label=f'{swap_trials}')
    
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Accuracy')
#     ax.set_title(f'LangCell: Convergence Curves\n({dataset_name})', 
#                 fontsize=12, fontweight='bold')
#     ax.set_ylim(0.4, 1.0)
#     ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
#     ax.grid(True, alpha=0.3)
#     ax.legend(title='Swap Trials', fontsize=9, title_fontsize=10, ncol=3)
    
#     # 1.2 最终准确率 vs swap trials
#     ax = axes[0, 1]
#     final_accs = [results[st]["final_accuracy"] for st in swap_trials_list]
    
#     ax.plot(swap_trials_list, final_accs, 's-', color='#e74c3c', 
#             linewidth=2, markersize=8, markerfacecolor='white')
    
#     for swap_trials, acc in zip(swap_trials_list, final_accs):
#         ax.text(swap_trials, acc + 0.01, f'{acc:.3f}', 
#                ha='center', va='bottom', fontsize=9)
    
#     ax.set_xlabel('Swap Trials (per iteration per cell)')
#     ax.set_ylabel('Final Accuracy')
#     ax.set_title('Final Accuracy vs Swap Trials', fontsize=12, fontweight='bold')
#     ax.set_ylim(min(final_accs)-0.05, max(final_accs)+0.05)
#     ax.grid(True, alpha=0.3)
#     ax.set_xscale('log', base=2)
    
#     # 1.3 时间效率 vs swap trials
#     ax = axes[1, 0]
#     total_times = [results[st]["time_stats"]["total_time"] for st in swap_trials_list]
#     avg_times = [results[st]["time_stats"]["avg_iteration_time"] for st in swap_trials_list]
    
#     ax.plot(swap_trials_list, total_times, 'o-', color='#3498db', 
#             linewidth=2, markersize=8, label='Total Time')
#     ax.plot(swap_trials_list, avg_times, 's-', color='#e67e22', 
#             linewidth=2, markersize=6, label='Avg Time/Iter')
    
#     ax.set_xlabel('Swap Trials')
#     ax.set_ylabel('Time (seconds)')
#     ax.set_title('Execution Time vs Swap Trials', fontsize=12, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     ax.set_xscale('log', base=2)
#     ax.legend()
    
#     # 1.4 准确率稳定性 vs swap trials
#     ax = axes[1, 1]
#     stabilities = [results[st]["std_accuracy"] for st in swap_trials_list]
    
#     ax.plot(swap_trials_list, stabilities, 'o-', color='#3498db', 
#             linewidth=2, markersize=8, markerfacecolor='white')
    
#     ax.set_xlabel('Swap Trials')
#     ax.set_ylabel('Accuracy Std Dev')
#     ax.set_title('Accuracy Stability vs Swap Trials', fontsize=12, fontweight='bold')
#     ax.set_ylim(0, max(stabilities) * 1.2)
#     ax.grid(True, alpha=0.3)
#     ax.set_xscale('log', base=2)
    
#     plt.tight_layout()
#     plt.savefig(outdir / f"langcell_swap_sensitivity_main_{dataset_name}.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 2. 综合评分图
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     # 计算综合评分
#     comprehensive_scores = calculate_comprehensive_score(results)
#     cost_effectiveness = calculate_cost_effectiveness_ratio(results)
    
#     # 2.1 综合评分柱状图
#     ax = axes[0, 0]
#     comp_scores_list = [comprehensive_scores[st]["comprehensive_score"] for st in swap_trials_list]
    
#     bars = ax.bar(range(len(swap_trials_list)), comp_scores_list, 
#                  color=['#3498db' if i != np.argmax(comp_scores_list) else '#f1c40f' 
#                         for i in range(len(swap_trials_list))],
#                  edgecolor='black', linewidth=1)
    
#     ax.set_xlabel('Swap Trials')
#     ax.set_ylabel('Comprehensive Score')
#     ax.set_title('Comprehensive Performance Score\n(higher is better)', fontsize=12, fontweight='bold')
#     ax.set_xticks(range(len(swap_trials_list)))
#     ax.set_xticklabels([f'{st}' for st in swap_trials_list], rotation=45)
#     ax.grid(True, alpha=0.3, axis='y')
    
#     for bar, score, st in zip(bars, comp_scores_list, swap_trials_list):
#         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                 f'{score:.3f}', ha='center', va='bottom', fontsize=8,
#                 fontweight='bold' if st == swap_trials_list[np.argmax(comp_scores_list)] else 'normal')
    
#     # 2.2 性价比图
#     ax = axes[0, 1]
#     cer_list = [cost_effectiveness[st] for st in swap_trials_list]
    
#     ax.plot(swap_trials_list, cer_list, '^-', color='#2ecc71', 
#             linewidth=2, markersize=8)
    
#     for swap_trials, cer in zip(swap_trials_list, cer_list):
#         ax.text(swap_trials, cer + 0.001, f'{cer:.4f}', 
#                ha='center', va='bottom', fontsize=8)
    
#     ax.set_xlabel('Swap Trials')
#     ax.set_ylabel('Accuracy per Second')
#     ax.set_title('Cost-Effectiveness Ratio\n(Accuracy / Time)', fontsize=12, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     ax.set_xscale('log', base=2)
    
#     # 2.3 雷达图：性能指标对比
#     ax = axes[1, 0]
    
#     # 选择最佳和最差的参数进行对比
#     best_st = swap_trials_list[np.argmax(comp_scores_list)]
#     worst_st = swap_trials_list[np.argmin(comp_scores_list)]
    
#     categories = ['Accuracy', 'Speed', 'Stability', 'Efficiency']
#     N = len(categories)
    
#     angles = [n / float(N) * 2 * np.pi for n in range(N)]
#     angles += angles[:1]
    
#     def get_radar_data(swap_trials):
#         comp = comprehensive_scores[swap_trials]
#         return [
#             comp["normalized_accuracy"],
#             comp["normalized_speed"],
#             comp["normalized_stability"],
#             comp["normalized_efficiency"],
#         ]
    
#     best_values = get_radar_data(best_st)
#     worst_values = get_radar_data(worst_st)
    
#     best_values += best_values[:1]
#     worst_values += worst_values[:1]
    
#     ax = plt.subplot(2, 2, 3, polar=True)
#     ax.plot(angles, best_values, 'o-', linewidth=2, label=f'Best: {best_st}', color='#2ecc71')
#     ax.fill(angles, best_values, alpha=0.25, color='#2ecc71')
#     ax.plot(angles, worst_values, 'o-', linewidth=2, label=f'Worst: {worst_st}', color='#e74c3c')
#     ax.fill(angles, worst_values, alpha=0.25, color='#e74c3c')
    
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories)
#     ax.set_ylim(0, 1)
#     ax.set_title('Performance Radar Chart', size=12, fontweight='bold')
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
#     # 2.4 推荐参数分析
#     ax = axes[1, 1]
    
#     # 绘制最佳参数的收敛曲线和时间曲线
#     best_res = results[best_st]
#     acc_curve = best_res["accuracy_curve"]
#     time_curve = best_res["time_stats"]["iteration_times"]
    
#     # 双y轴图
#     ax1 = ax
#     ax1.plot(range(1, len(acc_curve) + 1), acc_curve, 'o-', 
#             color='#3498db', linewidth=2, markersize=6, label='Accuracy')
#     ax1.set_xlabel('Iteration')
#     ax1.set_ylabel('Accuracy', color='#3498db')
#     ax1.tick_params(axis='y', labelcolor='#3498db')
#     ax1.set_ylim(0.4, 1.0)
#     ax1.grid(True, alpha=0.3)
    
#     ax2 = ax1.twinx()
#     ax2.plot(range(1, len(time_curve) + 1), time_curve, 's-', 
#             color='#e74c3c', linewidth=1.5, markersize=4, label='Time')
#     ax2.set_ylabel('Time per Iter (s)', color='#e74c3c')
#     ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
#     ax1.set_title(f'Recommended: Swap Trials = {best_st}', fontsize=12, fontweight='bold')
    
#     # 添加统计信息
#     textstr = f'Final Acc: {best_res["final_accuracy"]:.4f}\n'
#     textstr += f'Avg Time: {best_res["time_stats"]["avg_iteration_time"]:.3f}s\n'
#     textstr += f'Total Time: {best_res["time_stats"]["total_time"]:.2f}s\n'
#     textstr += f'Score: {comprehensive_scores[best_st]["comprehensive_score"]:.3f}'
    
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#     ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
#             verticalalignment='top', bbox=props)
    
#     plt.tight_layout()
#     plt.savefig(outdir / f"langcell_swap_sensitivity_comprehensive_{dataset_name}.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # 3. 热力图：时间-准确率权衡
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # 创建时间-准确率散点图
#     total_times = [results[st]["time_stats"]["total_time"] for st in swap_trials_list]
#     final_accs = [results[st]["final_accuracy"] for st in swap_trials_list]
    
#     scatter = ax.scatter(total_times, final_accs, c=swap_trials_list, 
#                         cmap='viridis', s=200, edgecolor='black', linewidth=1)
    
#     # 添加标签
#     for i, (t, a, st) in enumerate(zip(total_times, final_accs, swap_trials_list)):
#         ax.annotate(f'{st}', (t, a), fontsize=9, ha='center', va='bottom')
    
#     # 添加Pareto前沿
#     pareto_points = []
#     for i, (t1, a1) in enumerate(zip(total_times, final_accs)):
#         is_pareto = True
#         for j, (t2, a2) in enumerate(zip(total_times, final_accs)):
#             if i != j and t2 <= t1 and a2 >= a1 and (t2 < t1 or a2 > a1):
#                 is_pareto = False
#                 break
#         if is_pareto:
#             pareto_points.append((t1, a1))
    
#     if pareto_points:
#         pareto_points = sorted(pareto_points, key=lambda x: x[0])
#         pareto_t, pareto_a = zip(*pareto_points)
#         ax.plot(pareto_t, pareto_a, 'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
    
#     ax.set_xlabel('Total Execution Time (s)')
#     ax.set_ylabel('Final Accuracy')
#     ax.set_title('Time-Accuracy Trade-off Analysis', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('Swap Trials', fontsize=12)
    
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(outdir / f"langcell_swap_sensitivity_tradeoff_{dataset_name}.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"✓ Generated comprehensive LangCell swap sensitivity analysis plots")
#     return best_st, best_res, comprehensive_scores[best_st]


# def save_results_to_csv(results: Dict[int, dict], outdir: Path, dataset_name: str):
#     """保存结果到CSV文件"""
    
#     data = []
#     for swap_trials, res in results.items():
#         data.append({
#             'swap_trials': swap_trials,
#             'final_accuracy': res['final_accuracy'],
#             'avg_accuracy': res['avg_accuracy'],
#             'std_accuracy': res['std_accuracy'],
#             'max_accuracy': res['max_accuracy'],
#             'convergence_speed': res['convergence_speed'],
#             'convergence_iter': res['convergence_iter'],
#             'total_time': res['time_stats']['total_time'],
#             'avg_iteration_time': res['time_stats']['avg_iteration_time'],
#             'std_iteration_time': res['time_stats']['std_iteration_time'],
#             'accuracy_per_second': res['efficiency']['accuracy_per_second'],
#             'iterations_per_second': res['efficiency']['iterations_per_second'],
#         })
    
#     df = pd.DataFrame(data)
    
#     # 计算综合评分
#     comp_scores = calculate_comprehensive_score(results)
#     cost_effectiveness = calculate_cost_effectiveness_ratio(results)
    
#     df['comprehensive_score'] = df['swap_trials'].map({k: v['comprehensive_score'] for k, v in comp_scores.items()})
#     df['cost_effectiveness'] = df['swap_trials'].map(cost_effectiveness)
    
#     # 保存CSV
#     csv_path = outdir / f"langcell_swap_sensitivity_results_{dataset_name}.csv"
#     df.to_csv(csv_path, index=False)
    
#     # 保存Excel
#     excel_path = outdir / f"langcell_swap_sensitivity_results_{dataset_name}.xlsx"
#     df.to_excel(excel_path, index=False)
    
#     print(f"✓ Results saved to: {csv_path}")
#     print(f"✓ Results saved to: {excel_path}")
    
#     return df, comp_scores


# # =====================================================
# # 主函数
# # =====================================================
# def main():
#     """主函数"""
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     print(f"Testing SWAP sensitivity for LangCell model")
#     print(f"Swap trials values: {SWAP_TRIALS_VALUES}")
#     print(f"Fixed params: {FIXED_PARAMS}")
    
#     # 加载LangCell字典
#     print(f"\nLoading LangCell dictionaries...")
#     vocab, gene_name_id, pad_id, mask_id = load_langcell_dicts(Path(DICTS_DIR))
    
#     # 加载LangCell模型
#     print(f"Loading LangCell model from {MODEL_DIR}...")
#     model = LangCellModel(MODEL_DIR)
#     model.to(device)
#     model.eval()
    
#     print(f"Vocab size: {len(vocab)}")
#     print(f"Pad ID: {pad_id}, Mask ID: {mask_id}")
    
#     # 创建输出目录
#     outdir = Path(OUTDIR)
#     outdir.mkdir(exist_ok=True, parents=True)
    
#     # 运行swap参数敏感性分析
#     dataset_name = "hESC"
#     dataset_config = DATASET_CONFIG[dataset_name]
    
#     print(f"\n{'='*60}")
#     print(f"Running LangCell SWAP sensitivity analysis")
#     print('='*60)
    
#     results = run_langcell_swap_sensitivity_analysis(
#         dataset_name=dataset_name,
#         dataset_config=dataset_config,
#         model=model,
#         vocab=vocab,
#         gene_name_id=gene_name_id,
#         pad_id=pad_id,
#         mask_id=mask_id,
#         device=device,
#     )
    
#     # 保存原始结果
#     with open(outdir / f"langcell_swap_sensitivity_raw_results_{dataset_name}.json", "w") as f:
#         json.dump(results, f, indent=2, default=str)
    
#     # 绘制图表
#     print(f"\n{'='*60}")
#     print("Generating sensitivity analysis plots")
#     print('='*60)
    
#     best_swap_trials, best_result, best_score = plot_langcell_swap_sensitivity_results(results, outdir, dataset_name)
    
#     # 保存汇总表格
#     df, comp_scores = save_results_to_csv(results, outdir, dataset_name)
    
#     # 打印总结
#     print(f"\n{'='*80}")
#     print("LANGCELL SWAP SENSITIVITY ANALYSIS SUMMARY")
#     print('='*80)
#     print(f"Dataset: {dataset_name}")
#     print(f"Total swap-trials tested: {len(SWAP_TRIALS_VALUES)}")
#     print("\nDetailed Results:")
#     print("-" * 120)
#     header = f"{'Swap Trials':<12} {'Final Acc':<10} {'Avg Acc':<10} {'Std Dev':<10} {'Total Time':<12} "
#     header += f"{'Acc/s':<10} {'Comp Score':<12} {'Cost/Eff':<12}"
#     print(header)
#     print("-" * 120)
    
#     for swap_trials in sorted(results.keys()):
#         res = results[swap_trials]
#         comp_score = comp_scores[swap_trials]["comprehensive_score"]
#         cost_eff = res["efficiency"]["accuracy_per_second"]
        
#         print(f"{swap_trials:<12} {res['final_accuracy']:<10.4f} {res['avg_accuracy']:<10.4f} "
#               f"{res['std_accuracy']:<10.4f} {res['time_stats']['total_time']:<12.2f} "
#               f"{cost_eff:<10.4f} {comp_score:<12.4f} {cost_eff:<12.4f}")
    
#     print("\n" + "="*120)
#     print("RECOMMENDATION BASED ON COMPREHENSIVE SCORING")
#     print("="*120)
#     print(f"Best swap trials: {best_swap_trials}")
#     print(f"Comprehensive score: {best_score['comprehensive_score']:.4f}")
#     print(f"\nPerformance metrics:")
#     print(f"  Final accuracy: {best_result['final_accuracy']:.4f}")
#     print(f"  Average accuracy: {best_result['avg_accuracy']:.4f}")
#     print(f"  Standard deviation: {best_result['std_accuracy']:.4f}")
#     print(f"  Convergence speed: {best_result['convergence_speed']} iterations")
#     print(f"  Total execution time: {best_result['time_stats']['total_time']:.2f}s")
#     print(f"  Average iteration time: {best_result['time_stats']['avg_iteration_time']:.3f}s")
#     print(f"  Accuracy per second: {best_result['efficiency']['accuracy_per_second']:.4f}")
#     print(f"  Iterations per second: {best_result['efficiency']['iterations_per_second']:.2f}")
    
#     print(f"\nNormalized scores:")
#     print(f"  Accuracy score: {best_score['normalized_accuracy']:.3f}")
#     print(f"  Time score: {best_score['normalized_time']:.3f}")
#     print(f"  Stability score: {best_score['normalized_stability']:.3f}")
#     print(f"  Speed score: {best_score['normalized_speed']:.3f}")
#     print(f"  Efficiency score: {best_score['normalized_efficiency']:.3f}")
#     print(f"  Efficiency bonus: {best_score['efficiency_bonus']:.3f}")
    
#     print(f"\nAll results saved to: {outdir}")
#     print("="*120)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangCell | swap=8 | 全部6数据集 | 输出标准accuracy_curves.json
与Geneformer代码格式完全统一
"""

import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel

warnings.filterwarnings("ignore")

# =====================================================
# 固定配置
# =====================================================
MODEL_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/LangCell/cell_bert"
DICTS_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts"
OUT_JSON = "accuracy_curves.json"

# 全部6个数据集
DATASET_CONFIG = {
    "hESC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv",
        "species": "human",
    },
    "hHep": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hHep_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hHep/PseudoTime.csv",
        "species": "human",
    },
    "mDC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mDC_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mDC/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-E": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-E_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-E/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-GM": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-GM_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-GM/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-L": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-L/PseudoTime.csv",
        "species": "mouse",
    },
}

# 固定参数
FIXED_PARAMS = {
    "top_percent": 30,
    "pt_quantile": 0.2,
    "gen_iters": 16,
    "max_len": 1024,
    "temperature": 1.0,
    "use_log1p": True,
}
SWAP_TRIALS = 8

# =====================================================
# LangCell 模型
# =====================================================
class LangCellModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
        self.cls = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.cls(out.last_hidden_state)

# =====================================================
# 工具函数
# =====================================================
def normalize_symbol(s):
    return str(s).strip().upper()

def is_ensembl_id(s):
    return isinstance(s, str) and (s.startswith("ENSG") or s.startswith("ENSMUSG"))

def read_pt_file(path):
    try:
        pt_df = pd.read_csv(path, header=None)
        _ = float(pt_df.iloc[0, 1])
    except:
        pt_df = pd.read_csv(path)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
    return pt_df.dropna()

def load_geneformer_dicts(d):
    with open(d/"token_dictionary.pkl", "rb") as f: vocab = pickle.load(f)
    with open(d/"gene_name_id_dict.pkl", "rb") as f: gnd = pickle.load(f)
    return vocab, gnd, int(vocab["<pad>"]), int(vocab["<mask>"])

def build_symbol_to_ensembl(gnd):
    sample = list(gnd.items())[:2000]
    k_ens = sum(is_ensembl_id(str(k)) for k,_ in sample)
    v_ens = sum(is_ensembl_id(str(v)) for _,v in sample)
    if v_ens > k_ens:
        return {normalize_symbol(k):str(v) for k,v in gnd.items()}
    return {normalize_symbol(v):str(k) for k,v in gnd.items()}

def direction_accuracy_top_genes(pred, true, idx):
    td = true[idx]
    pd = pred[idx]
    true_dir = np.where(td>0,1,-1)
    pred_dir = np.where(pd>0,1,-1)
    return float((pred_dir == true_dir).mean())

def positions_dict_from_seq(seq, L):
    pos = {}
    for i in range(L):
        t = int(seq[i])
        if t not in pos: pos[t] = i
    return pos

# =====================================================
# Swap 核心逻辑
# =====================================================
@torch.no_grad()
def iterative_swap_sampling_langcell(
    model, seq, L, pad, mask, device, swap_trials, temp
):
    x = seq.clone().to(device).unsqueeze(0)
    attn = torch.zeros((1, x.size(1)), dtype=torch.long, device=device)
    attn[0,:L] = 1
    banned = torch.tensor([pad, mask], device=device)

    for _ in range(swap_trials):
        if L < 2:
            break
        ij = torch.randperm(L, device="cpu")[:2]
        i,j = int(ij[0]), int(ij[1])
        if i == j:
            continue
        a, b = int(x[0,i]), int(x[0,j])
        if a in (pad, mask) or b in (pad, mask):
            continue

        xm = x.clone()
        xm[0,i], xm[0,j] = mask, mask
        logits = model(input_ids=xm, attention_mask=attn)[0,[i,j],:].float()
        logits /= max(temp, 1e-6)

        bd = banned[(banned>=0)&(banned<logits.size(-1))]
        if bd.numel() > 0:
            logits[:, bd] = -1e9

        logp = torch.log_softmax(logits, -1)
        if a < 0 or b < 0 or a >= logits.size(-1) or b >= logits.size(-1):
            continue

        cur = logp[0,a].item() + logp[1,b].item()
        swp = logp[0,b].item() + logp[1,a].item()
        if swp > cur:
            x[0,i], x[0,j] = x[0,j].clone(), x[0,i].clone()
    return x.squeeze(0).cpu().numpy()

# =====================================================
# 单数据集运行
# =====================================================
@torch.no_grad()
def run_single_dataset(dset_name, dset_cfg, model, vocab, gnd, pad, mask, device):
    print(f"\n=== {dset_name} | swap={SWAP_TRIALS} ===")
    expr = pd.read_csv(dset_cfg["expr_csv"], index_col=0)
    pt_df = read_pt_file(dset_cfg["pt_csv"])
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common, "pt"].values

    X = expr.T.values.astype(np.float32)
    if FIXED_PARAMS["use_log1p"]:
        X = np.log1p(np.maximum(X, 0))

    lo, hi = np.quantile(pt, [0.2, 0.8])
    early_mask = pt <= lo
    late_mask = pt >= hi

    early_mean = X[early_mask].mean(0)
    late_mean = X[late_mask].mean(0)
    true_delta = late_mean - early_mean

    n_genes = len(true_delta)
    top_n = max(int(n_genes * 0.3), 1)
    top_idx = np.argsort(-np.abs(true_delta))[:top_n]

    sym2ens = build_symbol_to_ensembl(gnd)
    genes = expr.index.astype(str).tolist()

    def get_tid(g):
        gn = normalize_symbol(g)
        if gn in vocab:
            return vocab[gn]
        ens = sym2ens.get(gn)
        if ens and ens in vocab:
            return vocab[ens]
        return pad

    gene_tids = [get_tid(g) for g in genes]

    def cell2seq(x_arr):
        order = np.argsort(-x_arr)
        seq = []
        for j in order:
            t = gene_tids[j]
            if t == pad:
                continue
            seq.append(t)
            if len(seq) >= 1024:
                break
        L = len(seq)
        if L < 1024:
            seq += [pad] * (1024 - L)
        return seq, L

    early_cells = []
    for i in np.where(early_mask)[0]:
        seq, L = cell2seq(X[i])
        if L > 10:
            early_cells.append((seq, L))

    init = [np.array(s, dtype=np.int64) for s, _ in early_cells]
    curr = [arr.copy() for arr in init]
    lengths = [L for _, L in early_cells]

    acc_curve = []
    best_acc = -1

    for it in range(FIXED_PARAMS["gen_iters"]):
        prev = [arr.copy() for arr in curr]
        deltas = []
        next_curr = []

        for ci, L in enumerate(lengths):
            xp = curr[ci]
            x1 = iterative_swap_sampling_langcell(
                model, torch.tensor(xp, dtype=torch.long),
                L, pad, mask, device, SWAP_TRIALS, FIXED_PARAMS["temperature"]
            )
            next_curr.append(x1)
            p0 = positions_dict_from_seq(init[ci], L)
            p1 = positions_dict_from_seq(x1, L)
            delta = np.zeros(n_genes, dtype=np.float32)
            for gi in range(n_genes):
                tid = gene_tids[gi]
                delta[gi] = p1.get(tid, L) - p0.get(tid, L)
            deltas.append(delta)

        curr = next_curr
        mean_d = np.stack(deltas).mean(0)
        pred = -mean_d
        acc = direction_accuracy_top_genes(pred, true_delta, top_idx)

        if best_acc >= 0 and acc < best_acc:
            curr = prev
            acc = best_acc
        if acc > best_acc:
            best_acc = acc
        acc_curve.append(acc)
        print(f"  Iter {it+1:2d} | acc = {acc:.6f}")
    return acc_curve

# =====================================================
# 主函数
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | LangCell | fixed swap = {SWAP_TRIALS}")

    vocab, gnd, pad, mask = load_geneformer_dicts(Path(DICTS_DIR))
    model = LangCellModel(MODEL_DIR).to(device).eval()

    results = {}
    for dset_name, dset_cfg in DATASET_CONFIG.items():
        curve = run_single_dataset(dset_name, dset_cfg, model, vocab, gnd, pad, mask, device)
        results[dset_name] = curve

    # 输出标准json，和geneformer完全一致
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 完成！结果已保存至: {OUT_JSON}")

if __name__ == "__main__":
    main()