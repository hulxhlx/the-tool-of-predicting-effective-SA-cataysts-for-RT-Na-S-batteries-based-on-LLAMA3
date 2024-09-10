# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:38:26 2024

@author: Lingxiang
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the Excel files
file_path_1 = "./pred_emb.xlsx"
file_path_2 = "./SA_new_all_emb.xlsx"


df_A = pd.read_excel(file_path_1)
df_B = pd.read_excel(file_path_2)

# Convert the 'Embedding' column from string to numpy array
df_A['Embedding'] = df_A['Embedding'].apply(lambda x: np.array(list(map(float, x.split()))))
embedding_lengths = df_A['Embedding'].apply(len)
print("Unique lengths of embeddings:", embedding_lengths.unique())
length_counts = embedding_lengths.value_counts().sort_index()

# 输出结果
print(length_counts)

#df_B['Embedding'] = df_B['Embedding'].apply(lambda x: np.array(list(map(float, x.split()))))

converted_embeddings = []

# 遍历 df_B 中的每一行
for i in range(len(df_B)):
    # 获取当前行的 embedding 字符串
    embedding_str = df_B['Embedding'].iloc[i]
    
    # 将字符串分割并转换为浮点数
    embedding_list = embedding_str.split()  # 将字符串按空格分割成列表
    embedding_floats = []
    
    # 遍历每个元素并尝试转换为浮点数
    for value in embedding_list:
        try:
            embedding_floats.append(float(value))
        except ValueError:
            # 如果转换失败（例如遇到非数字字符），可以选择忽略或处理
            pass
    
    # 将浮点数列表转换为 numpy 数组
    embedding_array = np.array(embedding_floats)
    # 将转换后的 numpy 数组添加到列表中
    converted_embeddings.append(embedding_array)

print(converted_embeddings[0])
print(converted_embeddings[0].shape)

# 将新列表中的 numpy 数组赋值回 df_B 的 'Embedding' 列
df_B['Embedding'] = converted_embeddings

# 输出结果检查
print(df_B['Embedding'].head())
# Function to get top 50 similar embeddings
def get_top50_similar_indices(embedding, embeddings_B):
    similarities = cosine_similarity([embedding], embeddings_B)[0]
    top50_indices = np.argsort(similarities)[-50:][::-1]
    return top50_indices



embedding_lengths = df_B['Embedding'].apply(len)
print("Unique lengths of embeddings:", embedding_lengths.unique())
# 使用 value_counts() 来统计每个长度出现的次数
length_counts = embedding_lengths.value_counts().sort_index()

# 输出结果
print(length_counts)
# Calculate the top 50 indices for each embedding in df_A
embeddings_B = np.array(df_B['Embedding'].tolist())
df_A['Top50_Indices'] = df_A['Embedding'].apply(lambda emb: get_top50_similar_indices(emb, embeddings_B))


# Save the result to a new Excel file
output_path = "./pred_emb_top50.xlsx"
df_A.to_excel(output_path, index=False)
