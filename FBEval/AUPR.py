"""
GRN AUPR评估脚本 (改进版)
功能：批量计算多个模型、数据集、基准类型的AUPR和AUPR Ratio
版本：1.0
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from itertools import permutations
from datetime import datetime
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')


def calculate_aupr_multi_gt(pred_file, true_file, algorithmName="model", TFEdges=True,
                            dataset_name="", groundtruth_type=""):
    """
    计算单个数据集的AUPR和AUPR Ratio
    
    参数:
    ----------
    pred_file : str
        预测网络文件路径（TSV格式，需包含Gene1, Gene2, EdgeWeight列）
    true_file : str
        真实网络文件路径（CSV格式，需包含Gene1, Gene2列）
    algorithmName : str, 可选
        算法名称，用于结果标识
    TFEdges : bool, 可选
        是否进行TF过滤（True: 仅保留真实网络中的TF作为Gene1）
    dataset_name : str, 可选
        数据集名称，用于结果标识
    groundtruth_type : str, 可选
        基准类型（如CHIP, Non_CHIP, STRING），用于结果标识
    
    返回:
    ----------
    dict : 包含所有评估指标的字典
    """
    result_dict = {
        'Run_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Algorithm': algorithmName,
        'Dataset': dataset_name,
        'GroundTruth_Type': groundtruth_type,
        'Prediction_File': pred_file,
        'True_Edges_File': true_file,
        'AUPR': 0.0,
        'AUPR_Ratio': 0.0,
        'Random_Baseline': 0.0,
        'TopK_Count': 0,
        'True_Edges_Count': 0,
        'Possible_Edges_Count': 0,
        'Correct_Predictions': 0,
        'Status': 'Success',
        'Pred_Edges_Original': 0,
        'Pred_Edges_TF_Filtered': 0,
        'TF_Filter_Rate': 0.0
    }

    try:
        # -------------------------- 读取真实边文件 --------------------------
        if not os.path.exists(true_file):
            result_dict['Status'] = f'Error: True edges file not found: {true_file}'
            return result_dict

        # 支持CSV和TSV格式
        sep = ',' if true_file.endswith('.csv') else '\t'
        try:
            trueEdgesDF = pd.read_csv(true_file, sep=sep, header=0)
        except:
            # 尝试自动检测分隔符
            trueEdgesDF = pd.read_csv(true_file, sep=None, engine='python', header=0)

        # 校验列名
        required_cols = {"Gene1", "Gene2"}
        available_cols = set(trueEdgesDF.columns)
        
        if not required_cols.issubset(available_cols):
            # 尝试大小写不敏感匹配
            col_map = {col.lower(): col for col in trueEdgesDF.columns}
            for req_col in required_cols:
                if req_col.lower() in col_map:
                    trueEdgesDF = trueEdgesDF.rename(columns={col_map[req_col.lower()]: req_col})
            
            if not required_cols.issubset(set(trueEdgesDF.columns)):
                result_dict['Status'] = (
                    f"Error: True file must contain columns {required_cols}. "
                    f"Found columns: {list(trueEdgesDF.columns)}"
                )
                return result_dict

        trueEdgesDF = trueEdgesDF[["Gene1", "Gene2"]].copy()
        
        # 转换为字符串类型
        trueEdgesDF['Gene1'] = trueEdgesDF['Gene1'].astype(str).str.strip()
        trueEdgesDF['Gene2'] = trueEdgesDF['Gene2'].astype(str).str.strip()
        
        # 剔除自环边
        trueEdgesDF = trueEdgesDF.loc[trueEdgesDF['Gene1'] != trueEdgesDF['Gene2']]
        trueEdgesDF.drop_duplicates(keep='first', inplace=True)
        trueEdgesDF.reset_index(drop=True, inplace=True)

        if trueEdgesDF.empty:
            result_dict['Status'] = 'Error: Empty true edges file'
            return result_dict

        result_dict['True_Edges_Count'] = len(trueEdgesDF)
        
        # 提取真实网络的节点信息
        TFs = set(trueEdgesDF['Gene1'])  # 真实文件的Gene1集合（合法调控源）
        Genes = set(trueEdgesDF['Gene1']) | set(trueEdgesDF['Gene2'])  # 真实文件所有节点并集
        true_edges_set = set(trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2'])  # 真实边的唯一集合

        # -------------------------- 读取预测边文件 --------------------------
        if not os.path.exists(pred_file):
            result_dict['Status'] = f'Error: Prediction file not found: {pred_file}'
            return result_dict

        # 支持多种分隔符
        try:
            predDF = pd.read_csv(pred_file, sep='\t', header=0)
        except:
            try:
                predDF = pd.read_csv(pred_file, sep=',', header=0)
            except:
                predDF = pd.read_csv(pred_file, sep=None, engine='python', header=0)

        # 校验预测文件列名
        pred_required = {"Gene1", "Gene2", "EdgeWeight"}
        available_pred_cols = set(predDF.columns)
        
        if not pred_required.issubset(available_pred_cols):
            # 尝试大小写不敏感匹配
            col_map = {col.lower(): col for col in predDF.columns}
            for req_col in pred_required:
                if req_col.lower() in col_map:
                    predDF = predDF.rename(columns={col_map[req_col.lower()]: req_col})
            
            if not pred_required.issubset(set(predDF.columns)):
                result_dict['Status'] = (
                    f"Error: Pred file must contain columns {pred_required}. "
                    f"Found columns: {list(predDF.columns)}"
                )
                return result_dict

        # 数据清洗
        predDF = predDF[["Gene1", "Gene2", "EdgeWeight"]].copy()
        predDF['Gene1'] = predDF['Gene1'].astype(str).str.strip()
        predDF['Gene2'] = predDF['Gene2'].astype(str).str.strip()
        
        # 剔除自环边
        predDF = predDF.loc[predDF['Gene1'] != predDF['Gene2']]
        predDF.drop_duplicates(subset=["Gene1", "Gene2"], keep='first', inplace=True)
        predDF.reset_index(drop=True, inplace=True)
        result_dict['Pred_Edges_Original'] = len(predDF)

        if predDF.empty:
            result_dict['Status'] = 'Error: Empty prediction file after filtering'
            return result_dict

        # -------------------------- TF过滤逻辑 --------------------------
        if TFEdges:
            # TF过滤：Gene1必须在真实网络的TF集合中，Gene2必须在真实网络的节点集合中
            pred_before_filter = len(predDF)
            predDF = predDF[predDF['Gene1'].isin(TFs)]
            predDF = predDF[predDF['Gene2'].isin(Genes)]
            
            result_dict['Pred_Edges_TF_Filtered'] = len(predDF)
            filter_rate = len(predDF) / pred_before_filter if pred_before_filter > 0 else 0.0
            result_dict['TF_Filter_Rate'] = round(filter_rate, 6)
            
            # 计算可能边数
            possible_edges_count = len(TFs) * len(Genes) - len(TFs)  # 减去自环
            result_dict['Possible_Edges_Count'] = possible_edges_count
        else:
            # 不进行TF过滤
            uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
            possible_edges_count = len(set(permutations(uniqueNodes, r=2)))
            result_dict['Possible_Edges_Count'] = possible_edges_count
            result_dict['Pred_Edges_TF_Filtered'] = len(predDF)
            result_dict['TF_Filter_Rate'] = 1.0

        if len(predDF) == 0:
            result_dict['Status'] = 'Warning: No valid edges after TF filter'
            return result_dict

        # -------------------------- AUPR核心计算逻辑 --------------------------
        # 确保EdgeWeight是数值类型
        try:
            predDF['EdgeWeight'] = pd.to_numeric(predDF['EdgeWeight'], errors='coerce')
        except:
            result_dict['Status'] = 'Error: EdgeWeight column must be numeric'
            return result_dict
        
        # 移除缺失值
        predDF = predDF.dropna(subset=['EdgeWeight'])
        predDF['EdgeWeight'] = predDF['EdgeWeight'].abs()  # 权重取绝对值
        
        # 1. 构建预测权重字典（Gene1|Gene2 作为键）
        pred_weight_dict = {}
        for _, row in predDF.iterrows():
            edge_key = f"{row['Gene1']}|{row['Gene2']}"
            pred_weight_dict[edge_key] = row['EdgeWeight']

        # 2. 构建标签向量l和预测得分向量p（遍历所有合法TF-Gene对）
        l = []  # 标签：1=真实边，0=非真实边
        p = []  # 预测得分：有预测则为权重，无预测则为-1
        
        for tf in TFs:
            for gene in Genes:
                if tf == gene:
                    continue  # 跳过自环边
                edge_key = f"{tf}|{gene}"
                
                # 构建标签向量
                if edge_key in true_edges_set:
                    l.append(1)
                else:
                    l.append(0)
                
                # 构建预测得分向量
                if edge_key in pred_weight_dict:
                    p.append(pred_weight_dict[edge_key])
                else:
                    p.append(-1)

        # 3. 转换为numpy数组（避免sklearn警告）
        l = np.array(l)
        p = np.array(p)

        # 4. 计算AUPR和AUPR Ratio
        random_baseline = len(true_edges_set) / result_dict['Possible_Edges_Count'] if result_dict['Possible_Edges_Count'] > 0 else 0.0
        result_dict['Random_Baseline'] = round(random_baseline, 8)
        
        if len(l) == 0:
            result_dict['AUPR'] = 0.0
            result_dict['AUPR_Ratio'] = 0.0
        else:
            # 计算AUPR（Average Precision）
            aupr = average_precision_score(l, p)
            result_dict['AUPR'] = round(aupr, 6)
            
            # 计算AUPR Ratio（AUPR / 随机基线）
            result_dict['AUPR_Ratio'] = round(aupr / random_baseline if random_baseline > 0 else 0.0, 6)

        # -------------------------- Top-K统计 --------------------------
        # Top-K的K值 = 真实边数
        maxk = min(predDF.shape[0], len(true_edges_set))
        result_dict['TopK_Count'] = maxk
        
        if maxk == 0:
            result_dict['Status'] = 'Warning: No valid edges for top-k selection'
            return result_dict

        # 按权重降序排序
        predDF_sorted = predDF.sort_values('EdgeWeight', ascending=False).reset_index(drop=True)
        topk_pred = predDF_sorted.iloc[:maxk]
        pred_edges = set(topk_pred['Gene1'] + "|" + topk_pred['Gene2'])

        # 计算正确预测数
        intersection = pred_edges.intersection(true_edges_set)
        result_dict['Correct_Predictions'] = len(intersection)
        result_dict['Status'] = 'Success'

    except Exception as e:
        result_dict['Status'] = f'Error: {str(e)}'
        import traceback
        print(f"处理文件时出错: {pred_file}")
        print(traceback.format_exc())

    return result_dict


def evaluate_all_combinations(config):
    """批量评估所有模型、数据集、基准类型的组合"""
    
    pred_root_dir = config['pred_root_dir']
    true_root_dir = config['true_root_dir']
    output_csv = config['output_csv']
    TFEdges = config['TFEdges']
    models = config['models']
    datasets = config['datasets']
    groundtruth_types = config['groundtruth_types']
    model_name_mapping = config.get('model_name_mapping', {})
    
    all_results = []
    total_combinations = len(models) * len(datasets) * len(groundtruth_types)
    processed = 0
    
    print(f"\n🚀 开始批量评估：{len(models)} 模型 × {len(datasets)} 数据集 × {len(groundtruth_types)} GT类型")
    print(f"预测文件根目录: {pred_root_dir}")
    print(f"真实文件根目录: {true_root_dir}")
    print(f"结果文件: {output_csv}")
    print(f"TF过滤: {'启用' if TFEdges else '禁用'}")
    print("=" * 100)
    
    for model in models:
        print(f"\n📌 模型: {model}")
        print("-" * 80)
        
        for dataset_name in datasets:
            # 构建预测文件路径
            model_short = model_name_mapping.get(model, model.split('_')[0])
            pred_filename = f"{model_short}_{dataset_name}.tsv"
            pred_file = os.path.join(pred_root_dir, model, pred_filename)
            
            if not os.path.exists(pred_file):
                # 尝试其他可能的扩展名
                for ext in ['.tsv', '.csv', '.txt']:
                    alt_file = pred_file.replace('.tsv', ext)
                    if os.path.exists(alt_file):
                        pred_file = alt_file
                        break
                
            if not os.path.exists(pred_file):
                print(f"  ⚠️  预测文件不存在，跳过: {pred_file}")
                continue
            
            for gt_type in groundtruth_types:
                processed += 1
                print(f"\r  📊 进度: {processed}/{total_combinations}", end="")
                
                # 构建真实文件路径
                if gt_type == "CHIP":
                    true_file = os.path.join(true_root_dir, "CHIP", f"{dataset_name}_chip_matched-network.csv")
                elif gt_type in ["Non_CHIP", "STRING"]:
                    true_file = os.path.join(true_root_dir, gt_type, f"{dataset_name}_processed-network.csv")
                else:
                    print(f"\n  ❌ 未知的基准类型: {gt_type}")
                    continue
                
                if not os.path.exists(true_file):
                    # 尝试其他可能的扩展名
                    for ext in ['.csv', '.tsv', '.txt']:
                        alt_file = true_file.replace('.csv', ext)
                        if os.path.exists(alt_file):
                            true_file = alt_file
                            break
                
                if not os.path.exists(true_file):
                    print(f"\n  ⚠️  真实文件不存在，跳过: {true_file}")
                    continue
                
                # 执行评估
                result = calculate_aupr_multi_gt(
                    pred_file=pred_file,
                    true_file=true_file,
                    algorithmName=model_short,
                    TFEdges=TFEdges,
                    dataset_name=dataset_name,
                    groundtruth_type=gt_type
                )
                
                # 保存结果
                save_results_to_csv(result, csv_path=output_csv)
                all_results.append(result)
                
                if result['Status'] == 'Success':
                    print(f"\n  ✅ {dataset_name}-{gt_type}: AUPR={result['AUPR']:.6f}, "
                          f"Ratio={result['AUPR_Ratio']:.6f}, Baseline={result['Random_Baseline']:.6f}")
                else:
                    print(f"\n  ❌ {dataset_name}-{gt_type}: {result['Status']}")
    
    print(f"\n\n🎉 批量评估完成!")
    
    # 生成汇总报告
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = output_csv.replace('.csv', '_summary.csv')
        
        # 按AUPR Ratio排序
        summary_df = summary_df.sort_values(
            ["Algorithm", "Dataset", "GroundTruth_Type", "AUPR_Ratio"],
            ascending=[True, True, True, False]
        ).reset_index(drop=True)
        
        summary_df.to_csv(summary_csv, index=False)
        print(f"📊 详细汇总已保存: {summary_csv}")
        
        # 生成简要统计
        print("\n📈 简要统计:")
        stats = summary_df.groupby(['Algorithm', 'GroundTruth_Type'])['AUPR_Ratio'].agg(['mean', 'std', 'count']).round(4)
        print(stats.to_string())
        
        # 保存统计结果
        stats.to_csv(output_csv.replace('.csv', '_stats.csv'))
        
    return all_results


def save_results_to_csv(result_dict, csv_path):
    """保存AUPR结果到CSV文件"""
    result_df = pd.DataFrame([result_dict])
    
    if not os.path.exists(csv_path):
        result_df.to_csv(csv_path, index=False, mode='w', encoding='utf-8')
    else:
        result_df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8')


def main():
    """主函数：解析参数并运行评估"""
    parser = argparse.ArgumentParser(description='批量评估GRN预测结果的AUPR和AUPR Ratio')
    
    # 必选参数
    parser.add_argument('--pred_root', type=str, required=True,
                       help='预测文件根目录路径')
    parser.add_argument('--true_root', type=str, required=True,
                       help='真实网络文件根目录路径')
    parser.add_argument('--output', type=str, default='aupr_results.csv',
                       help='输出结果CSV文件路径')
    
    # 可选参数
    parser.add_argument('--tf_edges', type=bool, default=True,
                       help='是否启用TF过滤 (默认: True)')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON配置文件路径 (可覆盖命令行参数)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=["scgpt_hidden", "Geneformer_hidden", "sccello_hidden", 
                               "scFoundation_hidden", "Langcell_hidden"],
                       help='模型名称列表')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"],
                       help='数据集名称列表')
    parser.add_argument('--gt_types', type=str, nargs='+',
                       default=["CHIP", "Non_CHIP", "STRING"],
                       help='基准类型列表')
    
    args = parser.parse_args()
    
    # 从配置文件加载配置（如果提供）
    config = {
        'pred_root_dir': args.pred_root,
        'true_root_dir': args.true_root,
        'output_csv': args.output,
        'TFEdges': args.tf_edges,
        'models': args.models,
        'datasets': args.datasets,
        'groundtruth_types': args.gt_types,
        'model_name_mapping': {
            "scgpt_hidden": "scGPT",
            "Geneformer_hidden": "Geneformer",
            "sccello_hidden": "scCello",
            "scFoundation_hidden": "scFoundation",
            "Langcell_hidden": "Langcell"
        }
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
        print(f"✅ 从配置文件加载: {args.config}")
    
    # 验证路径
    if not os.path.exists(config['pred_root_dir']):
        print(f"❌ 错误: 预测文件根目录不存在: {config['pred_root_dir']}")
        sys.exit(1)
    
    if not os.path.exists(config['true_root_dir']):
        print(f"❌ 错误: 真实文件根目录不存在: {config['true_root_dir']}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(config['output_csv'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 显示配置
    print("\n" + "="*80)
    print("GRN AUPR评估工具")
    print("="*80)
    print(f"📁 预测文件目录: {config['pred_root_dir']}")
    print(f"📁 真实文件目录: {config['true_root_dir']}")
    print(f"💾 输出文件: {config['output_csv']}")
    print(f"🔄 TF过滤: {'启用' if config['TFEdges'] else '禁用'}")
    print(f"🤖 模型: {', '.join(config['models'])}")
    print(f"📊 数据集: {', '.join(config['datasets'])}")
    print(f"🎯 基准类型: {', '.join(config['groundtruth_types'])}")
    print("="*80)
    
    # 运行评估
    evaluate_all_combinations(config)


if __name__ == "__main__":
    main()