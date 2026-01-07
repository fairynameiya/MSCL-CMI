import pandas as pd
import numpy as np

# 读取CSV文件
assoc_matrix = pd.read_csv(r'D:\YD\MFERL-main\datasets\CircBANK\matrix_821-2115.csv', header=None)
pred_matrix = pd.read_csv('casestudy_matrix.csv', header=None)
m_names = pd.read_csv(r'D:\YD\MFERL-main\datasets\SHUJU\CMI9859\miRNA.csv', usecols=[0], names=['M'], header=None)
c_names = pd.read_csv(r'D:\YD\MFERL-main\datasets\SHUJU\CMI9859\circRNA.csv', usecols=[0], names=['C'], header=None)

# 将名称列表转换为Python列表
M_names = m_names['M'].tolist()
C_names = c_names['C'].tolist()

# 将DataFrame转换为NumPy数组
assoc_matrix_np = assoc_matrix.to_numpy()
pred_matrix_np = pred_matrix.to_numpy()

# 提取负样本的预测值及其对应的名称对
negative_samples = []
for i in range(pred_matrix_np.shape[0]):
    for j in range(pred_matrix_np.shape[1]):
        if assoc_matrix_np[i, j] == 0:  # 0表示负样本
            negative_samples.append((M_names[i], C_names[j], pred_matrix_np[i, j]))

# 按预测值从高到低排序
negative_samples_sorted = sorted(negative_samples, key=lambda x: x[2], reverse=True)

# 转换为DataFrame保存
df_negative_samples = pd.DataFrame(negative_samples_sorted, columns=["M_name", "C_name", "Prediction"])

# 保存为CSV文件
df_negative_samples.to_csv("sorted_negative_samples.csv", index=False)

# 显示排序后的结果
print(df_negative_samples)