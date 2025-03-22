import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock, minkowski, jaccard

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)

def euclidean_distance(vec1, vec2):
    """Computes Euclidean distance between two vectors."""
    return euclidean(vec1, vec2)

def manhattan_distance(vec1, vec2):
    """Computes Manhattan distance (L1 norm) between two vectors."""
    return cityblock(vec1, vec2)

def minkowski_distance(vec1, vec2, p=3):
    """Computes Minkowski distance between two vectors."""
    return minkowski(vec1, vec2, p)

def jaccard_similarity(vec1, vec2):
    """Computes Jaccard similarity (for binary or set-like data)."""
    return 1 - jaccard(vec1, vec2)

def cosine_index(vec1, vec2):
    """Computes Cosine Index, an alternative to Cosine Similarity."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / (norm_product + 1e-9)  # Avoid division by zero

def get_distance_metric(metric):
    """Returns the correct function for the given metric name."""
    metric_functions = {
        "cosine": cosine_similarity,
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "minkowski": minkowski_distance,
        "jaccard": jaccard_similarity,
        "cosine_index": cosine_index,
    }
    return metric_functions.get(metric, None)
