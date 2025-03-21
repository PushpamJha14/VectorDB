import numpy as np

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2 + 1e-9)  # Avoid division by zero

def euclidean_distance(vec1, vec2):
    """Computes Euclidean distance between two vectors."""
    return np.linalg.norm(vec1 - vec2)
