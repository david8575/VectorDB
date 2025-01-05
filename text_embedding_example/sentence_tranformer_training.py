from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing the violin."
]

embeddings = model.encode(sentences)

similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print("[문장 유사도]: ", similarity.item())

similarity = util.pytorch_cos_sim(embeddings[0], embeddings[3])
print("[문장 유사도]: ", similarity.item())

similarity = util.pytorch_cos_sim(embeddings[0], embeddings[4])
print("[문장 유사도]: ", similarity.item())

num_clusters = 3

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

for i in range(num_clusters):
    print(f"\nCluster {i+1}")
    cluster_sentences = [sentences[j] for j in range(len(sentences)) if cluster_assignment[j] == i]
    for sentence in cluster_sentences:
        print(sentence)