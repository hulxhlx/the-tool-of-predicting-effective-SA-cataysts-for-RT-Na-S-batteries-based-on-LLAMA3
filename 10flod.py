# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:23:13 2024

@author: Lingxiang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_validate

# # 假设数据已经读取到一个 DataFrame 中
# # 例如，使用 pd.read_excel() 方法从文件中读取数据
file_path = 'ceshiji3_emb.xlsx'  # 替换为你的Excel文件路径
data = pd.read_excel(file_path)


embeddings = np.load('./ceshiji3.npy')

file_path = 'ceshiji3.xlsx'  # 替换为你的Excel文件路径
data = pd.read_excel(file_path)

labels = data['Label'].values

# 确保嵌入向量与标签数量匹配
if len(embeddings) != len(labels):
    print(f"Mismatch: {len(embeddings)} embeddings and {len(labels)} labels")
else:
    print("Embeddings and labels match in size. Proceeding with model training...")
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 定义MLP模型
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, learning_rate_init=0.01,random_state=42)
    
    # 进行10折交叉验证
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc_auc': make_scorer(roc_auc_score, multi_class='ovo', needs_proba=True),
        'average_precision': make_scorer(average_precision_score, average='weighted', needs_proba=True)
    }

    # 使用cross_validate进行10折交叉验证，并获取这些指标
    results = cross_validate(mlp, embeddings_scaled, labels, cv=3, scoring=scoring)

    # 显示每个指标的平均分数
    print("Accuracy (平均):", np.mean(results['test_accuracy']))
    print("Precision (平均):", np.mean(results['test_precision']))
    print("F1-Score (平均):", np.mean(results['test_f1']))
    print("AUROC (平均):", np.mean(results['test_roc_auc']))
    print("AUPR (平均):", np.mean(results['test_average_precision']))