import numpy as np
from .storage import save_vectors, load_vectors
from .search import cosine_similarity, euclidean_distance

class VectorDatabase:
    def __init__(self, persist_path=None):
        self.vectors = {}
        self.persist_path = persist_path
        if persist_path:
            self.vectors = load_vectors(persist_path)

    def add_vector(self, key, vector):
        """Stores a vector with an associated key."""
        self.vectors[key] = np.array(vector)

    def search(self, query_vector, k=1, metric="cosine"):
        """Finds the top-k nearest vectors based on a similarity metric."""
        scores = []
        query_vector = np.array(query_vector)

        for key, vector in self.vectors.items():
            if metric == "cosine":
                score = cosine_similarity(query_vector, vector)
            elif metric == "euclidean":
                score = -euclidean_distance(query_vector, vector)  # Negative for sorting
            scores.append((key, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def save(self):
        """Saves vectors to disk."""
        if self.persist_path:
            save_vectors(self.persist_path, self.vectors)
