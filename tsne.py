import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# 加载 .npy 文件
array1 = np.load('./prediction.npy')
array2 = np.load('./240820.npy')

# 合并两个数组，假设沿行合并（即纵向）
combined_array = np.concatenate((array1, array2), axis=0)

# 检查合并后数组的形状
print(f"Combined array shape: {combined_array.shape}")

# 使用 t-SNE 将 4096 维降到 2 维
tsne = TSNE(n_components=3, random_state=42)
reduced_array = tsne.fit_transform(combined_array)

# 检查降维后数组的形状
print(f"Reduced array shape: {reduced_array.shape}")

# 将降维后的数据保存为 Excel 文件
df = pd.DataFrame(reduced_array, columns=['Dim1', 'Dim2','Dim3'])
df.to_excel('reduced_3d_array_tsne.xlsx', index=False)

# 打印一些数据以确认
print(df.head())
