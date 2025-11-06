
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import umap
import matplotlib.pyplot as plt

# Prepare Sample Data
data = [
    {"id": 1, "text": "How to fix SSL certificate error in Python requests?"},
    {"id": 2, "text": "Best way to implement retries with exponential backoff"},
    {"id": 3, "text": "Design review: event-driven architecture for telemetry pipeline"},
    {"id": 4, "text": "How to optimize SQL queries for large datasets"},
    {"id": 5, "text": "Difference between TCP and UDP protocols"}
]
df = pd.DataFrame(data)

# Compute Embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), normalize_embeddings=True)

# Build FAISS Index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

def search(query, k=3):
    qv = model.encode([query], normalize_embeddings=True)
    scores, idx = index.search(np.array(qv), k)
    return df.iloc[idx[0]].assign(score=scores[0])

# Test Search
print(search("How to handle SSL errors?", k=3))

# Visualize Embedding Space
reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, metric="cosine")
xy = reducer.fit_transform(embeddings)

plt.scatter(xy[:,0], xy[:,1], s=30)
for i, row in df.iterrows():
    plt.text(xy[i,0], xy[i,1], str(row["id"]), fontsize=9)
plt.title("UMAP projection of embeddings")
plt.savefig("embedding_visualization.png")
print("Embedding visualization saved to embedding_visualization.png")
