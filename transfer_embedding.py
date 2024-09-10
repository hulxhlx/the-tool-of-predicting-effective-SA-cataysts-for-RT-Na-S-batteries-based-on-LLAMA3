# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:51:02 2024

@author: Lingxiang
"""

import pandas as pd
import ollama
import numpy as np

# 读取.xlsx文件
df = pd.read_excel('ceshiji3.xlsx')
print(df.columns)


# 假设B列包含需要转换的文本
texts = df['Abstract'].tolist()

# 初始化一个空列表，用于存储嵌入向量
embeddings = []

for i,text in enumerate(texts):
    response = ollama.embeddings(model="llama3", prompt=text)
    embedding = response["embedding"]
    embeddings.append(embedding)
    print('transfer'+str(i)+'/'+str(len(texts)))

embeddings_array = np.array(embeddings)
np.save('ceshiji3.npy', embeddings_array)


print(embeddings_array.shape)


# 将嵌入向量转换为numpy数组，并添加到数据框中
embeddings_str = [' '.join(map(str, embedding)) for embedding in embeddings]


# 添加一个新列用于存储嵌入向量
df['Embedding'] = embeddings_str

# 保存修改后的数据框到新的.xlsx文件
df.to_excel('dataset_240820_ceshiji3.xlsx', index=False)

print("嵌入向量已成功添加到数据框并保存到新的.xlsx文件中。")
