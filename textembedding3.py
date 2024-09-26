# -*- coding: utf-8 -*-


import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings

df_all = pd.read_excel ("./dataset_240820.xlsx", header = None).drop_duplicates()
df_pre = pd.read_excel ("prediction dataset.xlsx").drop_duplicates()

embedding_model = "text-embedding-3-small"
openai_api_key = "your API key"
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
    
output_name = 'output_multiple_sheets_top100_abs.xlsx'

with pd.ExcelWriter(output_name, engine='xlsxwriter') as writer:
    for i, df in enumerate(dfs_results):
        sheet_name = f'Sheet{i+1}'
        
        df.to_excel(writer, index=False, sheet_name=sheet_name)
#%%

from sklearn.feature_extraction.text import TfidfVectorizer

# Combine titles and abstracts for vectorization
documents = [row['Title'] + " " + row['Abstract'] for df in dfs_results for _, row in df.iterrows()]

# Vectorize the combined text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
#%%
from sklearn.cluster import KMeans

# Perform K-Means clustering with an arbitrary number of clusters (e.g., 5)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster labels for each document
cluster_labels = kmeans.labels_
#%%
from sklearn.manifold import TSNE

# Apply t-SNE to reduce the dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())  # Convert sparse matrix to dense if needed
df_tsne = pd.DataFrame(X_tsne, columns=['Dimension 1', 'Dimension 2'])
df_tsne.to_csv('tsne_results.csv', index=False)
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame for the t-SNE results with cluster labels
df_tsne = pd.DataFrame({'x': X_tsne[:, 0], 'y': X_tsne[:, 1], 'Cluster': cluster_labels})

# Plot the t-SNE results with clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_tsne['x'], df_tsne['y'], c=df_tsne['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.title('t-SNE Visualization of Document Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Cluster')
plt.show()

