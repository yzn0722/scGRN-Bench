import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

# 设置路径（请根据实际修改，或使用命令行参数）
data_dir = "/mnt/10T/yzn/FoundBench/FBplot/fig4/results_multidataset_pseudotime_227_random"
file_pattern = os.path.join(data_dir, "hESC_gene_result_run*.csv")

# 获取所有匹配的文件，并按 run 编号排序
file_list = sorted(glob.glob(file_pattern))

if not file_list:
    print(f"未找到匹配的文件: {file_pattern}")
    exit()

bal_acc_list = []
mcc_list = []

for file_path in file_list:
    df = pd.read_csv(file_path)
    
    # 确保存在 dir_true 和 dir_pred 列，且值为 'Up'/'Down'
    if 'dir_true' not in df.columns or 'dir_pred' not in df.columns:
        print(f"文件 {file_path} 缺少 dir_true 或 dir_pred 列，跳过")
        continue
    
    # 转换为二值标签：Up -> 1, Down -> 0（注意：MCC 和平衡准确率对标签顺序不敏感，但需一致）
    y_true = (df['dir_true'] == 'Up').astype(int)
    y_pred = (df['dir_pred'] == 'Up').astype(int)
    
    # 计算平衡准确率
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    # 计算 MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    
    bal_acc_list.append(bal_acc)
    mcc_list.append(mcc)
    
    print(f"{os.path.basename(file_path)}: Balanced Accuracy = {bal_acc:.4f}, MCC = {mcc:.4f}")

# 汇总统计
bal_acc_mean = np.mean(bal_acc_list)
bal_acc_std = np.std(bal_acc_list, ddof=1)  # 样本标准差
mcc_mean = np.mean(mcc_list)
mcc_std = np.std(mcc_list, ddof=1)

print("\n========== Summary across 10 runs ==========")
print(f"Balanced Accuracy: {bal_acc_mean:.4f} ± {bal_acc_std:.4f}")
print(f"MCC: {mcc_mean:.4f} ± {mcc_std:.4f}")