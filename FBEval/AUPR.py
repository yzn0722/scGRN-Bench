"""
GRN AUPR evaluation script
Batch AUPR and AUPR-ratio evaluation across models, datasets, and ground-truth types.
Version: 1.0
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
    Compute AUPR and AUPR ratio for one dataset.
    
    Args:
    ----------
    pred_file : str
        Prediction file path (TSV with Gene1, Gene2, EdgeWeight).
    true_file : str
        Ground-truth file path (CSV with Gene1 and Gene2).
    algorithmName : str, 
        Algorithm name used in the result table.
    TFEdges : bool, 
        Whether to apply TF filtering (Gene1 must be a TF from the ground truth).
    dataset_name : str, 
        Dataset name used in the result table.
    groundtruth_type : str, 
        Ground-truth type used in the result table, such as CHIP or STRING.
    
    Returns:
    ----------
    dict : A dictionary with all evaluation metrics.
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
        # -------------------------- Read the ground-truth edge file --------------------------
        if not os.path.exists(true_file):
            result_dict['Status'] = f'Error: True edges file not found: {true_file}'
            return result_dict

        # Support both CSV and TSV formats
        sep = ',' if true_file.endswith('.csv') else '\t'
        try:
            trueEdgesDF = pd.read_csv(true_file, sep=sep, header=0)
        except:
            # Try to auto-detect the delimiter
            trueEdgesDF = pd.read_csv(true_file, sep=None, engine='python', header=0)

        # Validate column names
        required_cols = {"Gene1", "Gene2"}
        available_cols = set(trueEdgesDF.columns)
        
        if not required_cols.issubset(available_cols):
            # Try case-insensitive matching
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
        
        # Convert to string type
        trueEdgesDF['Gene1'] = trueEdgesDF['Gene1'].astype(str).str.strip()
        trueEdgesDF['Gene2'] = trueEdgesDF['Gene2'].astype(str).str.strip()
        
        # Remove self-loop edges
        trueEdgesDF = trueEdgesDF.loc[trueEdgesDF['Gene1'] != trueEdgesDF['Gene2']]
        trueEdgesDF.drop_duplicates(keep='first', inplace=True)
        trueEdgesDF.reset_index(drop=True, inplace=True)

        if trueEdgesDF.empty:
            result_dict['Status'] = 'Error: Empty true edges file'
            return result_dict

        result_dict['True_Edges_Count'] = len(trueEdgesDF)
        
        # Extract node information from the ground-truth network
        TFs = set(trueEdgesDF['Gene1'])  # Gene1 set from the ground-truth file (valid regulators)
        Genes = set(trueEdgesDF['Gene1']) | set(trueEdgesDF['Gene2'])  # Union of all nodes in the ground-truth file
        true_edges_set = set(trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2'])  # Unique set of ground-truth edges

        # -------------------------- Read the prediction edge file --------------------------
        if not os.path.exists(pred_file):
            result_dict['Status'] = f'Error: Prediction file not found: {pred_file}'
            return result_dict

        # Support multiple delimiters
        try:
            predDF = pd.read_csv(pred_file, sep='\t', header=0)
        except:
            try:
                predDF = pd.read_csv(pred_file, sep=',', header=0)
            except:
                predDF = pd.read_csv(pred_file, sep=None, engine='python', header=0)

        # Validate prediction file columns
        pred_required = {"Gene1", "Gene2", "EdgeWeight"}
        available_pred_cols = set(predDF.columns)
        
        if not pred_required.issubset(available_pred_cols):
            # Try case-insensitive matching
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

        # Clean data
        predDF = predDF[["Gene1", "Gene2", "EdgeWeight"]].copy()
        predDF['Gene1'] = predDF['Gene1'].astype(str).str.strip()
        predDF['Gene2'] = predDF['Gene2'].astype(str).str.strip()
        
        # Remove self-loop edges
        predDF = predDF.loc[predDF['Gene1'] != predDF['Gene2']]
        predDF.drop_duplicates(subset=["Gene1", "Gene2"], keep='first', inplace=True)
        predDF.reset_index(drop=True, inplace=True)
        result_dict['Pred_Edges_Original'] = len(predDF)

        if predDF.empty:
            result_dict['Status'] = 'Error: Empty prediction file after filtering'
            return result_dict

        # -------------------------- TF filtering logic --------------------------
        if TFEdges:
            # Under TF filtering, Gene1 must be a TF in the ground truth and Gene2 must be in the ground-truth node set.
            pred_before_filter = len(predDF)
            predDF = predDF[predDF['Gene1'].isin(TFs)]
            predDF = predDF[predDF['Gene2'].isin(Genes)]
            
            result_dict['Pred_Edges_TF_Filtered'] = len(predDF)
            filter_rate = len(predDF) / pred_before_filter if pred_before_filter > 0 else 0.0
            result_dict['TF_Filter_Rate'] = round(filter_rate, 6)
            
            # Compute the number of possible edges
            possible_edges_count = len(TFs) * len(Genes) - len(TFs)  # subtract self-loops
            result_dict['Possible_Edges_Count'] = possible_edges_count
        else:
            # Skip TF filtering
            uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
            possible_edges_count = len(set(permutations(uniqueNodes, r=2)))
            result_dict['Possible_Edges_Count'] = possible_edges_count
            result_dict['Pred_Edges_TF_Filtered'] = len(predDF)
            result_dict['TF_Filter_Rate'] = 1.0

        if len(predDF) == 0:
            result_dict['Status'] = 'Warning: No valid edges after TF filter'
            return result_dict

        # -------------------------- Core AUPR calculation logic --------------------------
        # Ensure EdgeWeight is numeric
        try:
            predDF['EdgeWeight'] = pd.to_numeric(predDF['EdgeWeight'], errors='coerce')
        except:
            result_dict['Status'] = 'Error: EdgeWeight column must be numeric'
            return result_dict
        
        # Remove missing values
        predDF = predDF.dropna(subset=['EdgeWeight'])
        predDF['EdgeWeight'] = predDF['EdgeWeight'].abs()  # take the absolute value of the weights
        
        # 1. Build a prediction weight dictionary keyed by Gene1|Gene2
        pred_weight_dict = {}
        for _, row in predDF.iterrows():
            edge_key = f"{row['Gene1']}|{row['Gene2']}"
            pred_weight_dict[edge_key] = row['EdgeWeight']

        # 2. Build the label vector l and score vector p by iterating over all valid TF-gene pairs
        l = []  # Labels: 1 for true edges and 0 for non-edges
        p = []  # Prediction scores: use the weight if present, otherwise -1
        
        for tf in TFs:
            for gene in Genes:
                if tf == gene:
                    continue  # Skip self-loop edges
                edge_key = f"{tf}|{gene}"
                
                # 
                if edge_key in true_edges_set:
                    l.append(1)
                else:
                    l.append(0)
                
                # 
                if edge_key in pred_weight_dict:
                    p.append(pred_weight_dict[edge_key])
                else:
                    p.append(-1)

        # 3. Convert to numpy arrays to avoid sklearn warnings
        l = np.array(l)
        p = np.array(p)

        # 4. Compute AUPR and AUPR ratio
        random_baseline = len(true_edges_set) / result_dict['Possible_Edges_Count'] if result_dict['Possible_Edges_Count'] > 0 else 0.0
        result_dict['Random_Baseline'] = round(random_baseline, 8)
        
        if len(l) == 0:
            result_dict['AUPR'] = 0.0
            result_dict['AUPR_Ratio'] = 0.0
        else:
            # Compute AUPR (Average Precision)
            aupr = average_precision_score(l, p)
            result_dict['AUPR'] = round(aupr, 6)
            
            # Compute AUPR ratio (AUPR / random baseline)
            result_dict['AUPR_Ratio'] = round(aupr / random_baseline if random_baseline > 0 else 0.0, 6)

        # -------------------------- Top-K statistics --------------------------
        # Set K for Top-K to the number of ground-truth edges
        maxk = min(predDF.shape[0], len(true_edges_set))
        result_dict['TopK_Count'] = maxk
        
        if maxk == 0:
            result_dict['Status'] = 'Warning: No valid edges for top-k selection'
            return result_dict

        # Sort by weight in descending order
        predDF_sorted = predDF.sort_values('EdgeWeight', ascending=False).reset_index(drop=True)
        topk_pred = predDF_sorted.iloc[:maxk]
        pred_edges = set(topk_pred['Gene1'] + "|" + topk_pred['Gene2'])

        # Count correct predictions
        intersection = pred_edges.intersection(true_edges_set)
        result_dict['Correct_Predictions'] = len(intersection)
        result_dict['Status'] = 'Success'

    except Exception as e:
        result_dict['Status'] = f'Error: {str(e)}'
        import traceback
        print(f"Error while processing file: {pred_file}")
        print(traceback.format_exc())

    return result_dict


def evaluate_all_combinations(config):
    """Batch-evaluate all combinations of models, datasets, and ground-truth types."""
    
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
    
    print(f"\n🚀 Start batch evaluation:{len(models)} Model × {len(datasets)} Datasets × {len(groundtruth_types)} GT")
    print(f"Prediction root: {pred_root_dir}")
    print(f"Ground-truth root: {true_root_dir}")
    print(f"Output file: {output_csv}")
    print(f"TF filtering: {'enabled' if TFEdges else 'disabled'}")
    print("=" * 100)
    
    for model in models:
        print(f"\n📌 Model: {model}")
        print("-" * 80)
        
        for dataset_name in datasets:
            # Build the prediction file path
            model_short = model_name_mapping.get(model, model.split('_')[0])
            pred_filename = f"{model_short}_{dataset_name}.tsv"
            pred_file = os.path.join(pred_root_dir, model, pred_filename)
            
            if not os.path.exists(pred_file):
                # Try other possible extensions
                for ext in ['.tsv', '.csv', '.txt']:
                    alt_file = pred_file.replace('.tsv', ext)
                    if os.path.exists(alt_file):
                        pred_file = alt_file
                        break
                
            if not os.path.exists(pred_file):
                print(f"  ⚠️  Prediction file not found, skip: {pred_file}")
                continue
            
            for gt_type in groundtruth_types:
                processed += 1
                print(f"\r  📊 Progress: {processed}/{total_combinations}", end="")
                
                # Build the ground-truth file path
                if gt_type == "CHIP":
                    true_file = os.path.join(true_root_dir, "CHIP", f"{dataset_name}_chip_matched-network.csv")
                elif gt_type in ["Non_CHIP", "STRING"]:
                    true_file = os.path.join(true_root_dir, gt_type, f"{dataset_name}_processed-network.csv")
                else:
                    print(f"\n  ❌ Unknown ground-truth type: {gt_type}")
                    continue
                
                if not os.path.exists(true_file):
                    # Try other possible extensions
                    for ext in ['.csv', '.tsv', '.txt']:
                        alt_file = true_file.replace('.csv', ext)
                        if os.path.exists(alt_file):
                            true_file = alt_file
                            break
                
                if not os.path.exists(true_file):
                    print(f"\n  ⚠️  Ground-truth file not found, skip: {true_file}")
                    continue
                
                # Run evaluation
                result = calculate_aupr_multi_gt(
                    pred_file=pred_file,
                    true_file=true_file,
                    algorithmName=model_short,
                    TFEdges=TFEdges,
                    dataset_name=dataset_name,
                    groundtruth_type=gt_type
                )
                
                # Save results
                save_results_to_csv(result, csv_path=output_csv)
                all_results.append(result)
                
                if result['Status'] == 'Success':
                    print(f"\n  ✅ {dataset_name}-{gt_type}: AUPR={result['AUPR']:.6f}, "
                          f"Ratio={result['AUPR_Ratio']:.6f}, Baseline={result['Random_Baseline']:.6f}")
                else:
                    print(f"\n  ❌ {dataset_name}-{gt_type}: {result['Status']}")
    
    print(f"\n\n🎉 Batch evaluation completed!")
    
    # Generated
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = output_csv.replace('.csv', '_summary.csv')
        
        # Sort by AUPR ratio
        summary_df = summary_df.sort_values(
            ["Algorithm", "Dataset", "GroundTruth_Type", "AUPR_Ratio"],
            ascending=[True, True, True, False]
        ).reset_index(drop=True)
        
        summary_df.to_csv(summary_csv, index=False)
        print(f"📊 Detailed summary saved: {summary_csv}")
        
        # GeneratedSummary statistics
        print("\n📈 Summary statistics:")
        stats = summary_df.groupby(['Algorithm', 'GroundTruth_Type'])['AUPR_Ratio'].agg(['mean', 'std', 'count']).round(4)
        print(stats.to_string())
        
        # Save summary statistics
        stats.to_csv(output_csv.replace('.csv', '_stats.csv'))
        
    return all_results


def save_results_to_csv(result_dict, csv_path):
    """Save AUPR results to a CSV file."""
    result_df = pd.DataFrame([result_dict])
    
    if not os.path.exists(csv_path):
        result_df.to_csv(csv_path, index=False, mode='w', encoding='utf-8')
    else:
        result_df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8')


def main():
    """Main function: parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Batch evaluation of GRN prediction AUPR and AUPR ratio')
    
    # Required arguments
    parser.add_argument('--pred_root', type=str, required=True,
                       help='Prediction root path')
    parser.add_argument('--true_root', type=str, required=True,
                       help='Ground-truth root directory')
    parser.add_argument('--output', type=str, default='aupr_results.csv',
                       help='Output CSV path')
    
    # Optional arguments
    parser.add_argument('--tf_edges', type=bool, default=True,
                       help='Enable TF filtering (default: True)')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON config path (overrides CLI arguments)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=["scgpt_hidden", "Geneformer_hidden", "sccello_hidden", 
                               "scFoundation_hidden", "Langcell_hidden"],
                       help='Model name list')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"],
                       help='Dataset name list')
    parser.add_argument('--gt_types', type=str, nargs='+',
                       default=["CHIP", "Non_CHIP", "STRING"],
                       help='Ground-truth type list')
    
    args = parser.parse_args()
    
    # Load configuration from the config file if provided
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
        print(f"✅ Loaded from config: {args.config}")
    
    # Validate paths
    if not os.path.exists(config['pred_root_dir']):
        print(f"❌ Error: Prediction rootdoes not exist: {config['pred_root_dir']}")
        sys.exit(1)
    
    if not os.path.exists(config['true_root_dir']):
        print(f"❌ Error: Ground-truth rootdoes not exist: {config['true_root_dir']}")
        sys.exit(1)
    
    # Create the output directory
    output_dir = os.path.dirname(config['output_csv'])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Show configuration
    print("\n" + "="*80)
    print("GRN AUPR evaluation tool")
    print("="*80)
    print(f"📁 Prediction directory: {config['pred_root_dir']}")
    print(f"📁 Ground-truth directory: {config['true_root_dir']}")
    print(f"💾 Output file: {config['output_csv']}")
    print(f"🔄 TF filtering: {'enabled' if config['TFEdges'] else 'disabled'}")
    print(f"🤖 Model: {', '.join(config['models'])}")
    print(f"📊 Datasets: {', '.join(config['datasets'])}")
    print(f"🎯 Ground-truth types: {', '.join(config['groundtruth_types'])}")
    print("="*80)
    
    # Run evaluation
    evaluate_all_combinations(config)


if __name__ == "__main__":
    main()