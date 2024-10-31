import numpy as np
from numpy.linalg import norm

# For similarity search: comparing an embedding with multiple embeddings
def perform_semantic_search(query_embedding, embeddings):
    similarities = []
    for idx, embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(query_embedding, embedding)
        similarities.append((idx, similarity_score))
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Function to calculate Cosine Similarity
def cosine_similarity(gt_embedding, generated_embedding):
    return np.dot(gt_embedding, generated_embedding) / (norm(gt_embedding) * norm(generated_embedding))

