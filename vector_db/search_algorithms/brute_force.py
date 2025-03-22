import numpy as np
from ..search import get_distance_metric

class BruteForceSearch:
    def search(self, vectors, query_vector, k, metric):
        """Brute-force search over all stored vectors."""
        metric_fn = get_distance_metric(metric)
        if not metric_fn:
            raise ValueError(f"Unsupported metric: {metric}")

        scores = []
        query_vector = np.array(query_vector)

        for key, vector in vectors.items():
            score = metric_fn(query_vector, vector)
            scores.append((key, score))

        # Sort ascending for distance-based metrics; descending for similarity
        reverse = metric in ["cosine", "jaccard", "cosine_index"]
        scores.sort(key=lambda x: x[1], reverse=reverse)
        
        return scores[:k]
