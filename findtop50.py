# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:51:02 2024

@author: Lingxiang
"""
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import ollama
import numpy as np


# =============================================================================
# 
# # 读取.xlsx文件
# df = pd.read_excel('SA_new_all.xlsx')
# print(df.columns)
# 
# # 假设B列包含需要转换的文本
# texts = df['Column2'].tolist()
# 
# # 初始化一个空列表，用于存储嵌入向量
# embeddings = []
# 
# for text in texts:
#     response = ollama.embeddings(model="llama3", prompt=text)
#     embedding = response["embedding"]  # 提取嵌入向量
#     embeddings.append(embedding)
#     print(len(embedding))
# 
# embeddings_array = np.array(embeddings)
# np.save('SA.npy', embeddings_array)
# 
# 
# print(embeddings_array.shape)
# 
# 
# # 将嵌入向量转换为numpy数组，并添加到数据框中
# embeddings_str = [' '.join(map(str, embedding)) for embedding in embeddings]
# 
# # 添加一个新列用于存储嵌入向量
# df['Embedding'] = embeddings_str
# =============================================================================


# =============================================================================
# 
# # 读取.xlsx文件
# df = pd.read_excel('./prediction dataset.xlsx')
# 
# # 假设B列包含需要转换的文本
# texts = df['Column2'].tolist()
# 
# # 初始化一个空列表，用于存储嵌入向量
# embeddings = []
# 
# for text in texts:
#     response = ollama.embeddings(model="llama3", prompt=text)
#     embedding = response["embedding"]  # 提取嵌入向量
#     embeddings.append(embedding)
#     print(len(embedding))
# 
# embeddings_array = np.array(embeddings)
# np.save('prediction.npy', embeddings_array)
# 
# embeddings_array = np.load('./SA.npy')
# 
# =============================================================================


#df = pd.read_excel('./prediction dataset.xlsx')

prediction = np.load('./prediction.npy')
embeddings_array = np.load('./SA.npy')
top50indices = [] 


for predict in prediction:


    
    # 计算查询嵌入向量与所有嵌入向量的余弦相似度
    similarities = cosine_similarity([predict], embeddings_array)[0]
    # 找到最相似的文本
    most_similar_indices = np.argsort(similarities)[-50:][::-1]
    top50indices.append(most_similar_indices)
    print(most_similar_indices)
# =============================================================================
# top50indices_str = [' '.join(map(str, indice)) for indice in top50indices]
# df['top50indices'] = top50indices_str
# df.to_excel('pred_top50indices.xlsx', index=False)
# =============================================================================
# =============================================================================
#     most_similar_text = texts[most_similar_index]
#     most_similar_score = similarities[0][most_similar_index]
#     
#     print(f"最相似的文本是: {most_similar_text}")
#     print(f"相似度得分: {most_similar_score:.4f}")
# =============================================================================



# =============================================================================
# for text in texts:
#     response = ollama.embeddings(model="llama3", prompt=text)
#     query_embedding = response["embedding"]  # 提取嵌入向量
# 
#     
#     # 计算查询嵌入向量与所有嵌入向量的余弦相似度
#     similarities = cosine_similarity([query_embedding], embeddings_array)[0]
#     # 找到最相似的文本
#     most_similar_indices = np.argsort(similarities)[-50:][::-1]
#     top50indices.append(most_similar_indices)
# top50indices_str = [' '.join(map(str, indice)) for indice in top50indices]
# df['top50indices'] = top50indices_str
# df.to_excel('pred_top50indices.xlsx', index=False)
# # =============================================================================
# #     most_similar_text = texts[most_similar_index]
# #     most_similar_score = similarities[0][most_similar_index]
# #     
# #     print(f"最相似的文本是: {most_similar_text}")
# #     print(f"相似度得分: {most_similar_score:.4f}")
# # =============================================================================
# 
# =============================================================================
