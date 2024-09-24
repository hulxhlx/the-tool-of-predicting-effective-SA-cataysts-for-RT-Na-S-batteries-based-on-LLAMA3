# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:38:26 2024

@author: Lingxiang
"""

import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings

df_all = pd.read_excel ("./dataset_240820.xlsx", header = None).drop_duplicates()
df_pre = pd.read_excel ("prediction dataset.xlsx").drop_duplicates()

embedding_model = "text-embedding-3-small"
openai_api_key = "sk-proj-fuzKGig2v3GR1nOS8WrJT3BlbkFJlehB6npoULxZkAc9hCW2"
embeddings = OpenAIEmbeddings(model = embedding_model, openai_api_key = openai_api_key)

texts = df_all.iloc[:, 1].tolist()  # Abstract -> Embedding (vectorize)
# metadata for output: title, abstract, journal name
metadata = df_all.iloc[:, [0, 1, 2]].rename(columns={df_all.columns[0]: 'TITLE', df_all.columns[1]: 'ABSTRACT', df_all.columns[2]: 'JOURNAL_NAME'}).to_dict(orient='records')
#%%
from opensearchpy import OpenSearch
opensearch_url = "http://localhost:9200"
client = OpenSearch(opensearch_url)

# Check if the index already exists
if client.indices.exists(index="sa_new_all"):
    print("Index already exists. Skipping ingestion.")
else:
    vector_store = OpenSearchVectorSearch.from_texts(
        texts,
        embeddings,
        opensearch_url=opensearch_url,
        metadatas=metadata,
        index_name="sa_new_all",
        bulk_size=12000,
        timeout=60
    )
#%%    
from langchain.vectorstores import OpenSearchVectorSearch
opensearch_url = "http://localhost:9200"
vector_store = OpenSearchVectorSearch.from_texts(
    texts,
    embeddings,
    opensearch_url = opensearch_url,
    metadatas = metadata,
    index_name = "sa_new_all",
    bulk_size = 12000,
    timeout = 60
)

if vector_store.index_exists(index_name="sa_new_all"):
    vector_store.delete_index(index_name="sa_new_all")
query = df_pre.iloc[0,0] +" "+ df_pre.iloc[0,1]

results = vector_store.similarity_search_with_score(query, k=50, space_type = 'cosinesimil')

data = []
for result, score in results:
    data.append({
        'Title': result.metadata.get('TITLE', ''),
        'Abstract': result.page_content,
        'Journal_name': result.metadata.get('JOURNAL_NAME', ''),
        'Cosinesimil_Score': score
    })
    
df_results = pd.DataFrame(data)

df_results = pd.concat([pd.DataFrame([{'Title': df_pre.iloc[0,0], 'Abstract': df_pre.iloc[0,1]}]), df_results], ignore_index=True)
01
dfs_results = []

# Loop 
for i in range(df_pre.shape[0]):
    # query = df_pre.iloc[i, 0] + " " + df_pre.iloc[i, 1]
    query = df_pre.iloc[i, 1]
    results = vector_store.similarity_search_with_score(query, k=100, space_type='cosinesimil')

    data = []
    for result, score in results:
        data.append({
            'Title': result.metadata.get('TITLE', ''),
            'Abstract': result.page_content,
            'Journal_name': result.metadata.get('JOURNAL_NAME', ''),
            'Cosinesimil_Score': score
        })
    
    df_results = pd.DataFrame(data)
    df_results = pd.concat([pd.DataFrame([{'Title': df_pre.iloc[i, 0], 'Abstract': df_pre.iloc[i, 1]}]), df_results], ignore_index=True)
    
    dfs_results.append(df_results)
  #%% 
from openai import OpenAI
client = OpenAI(api_key = "sk-proj-fuzKGig2v3GR1nOS8WrJT3BlbkFJlehB6npoULxZkAc9hCW2")
completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
      {"role": "user", "content": f"Based on the abstract, summarize what chemical reaction the article is about and what catalyst is used. The abstrac: {dfs_results[0].iloc[0,1]}"}
  ]
)
