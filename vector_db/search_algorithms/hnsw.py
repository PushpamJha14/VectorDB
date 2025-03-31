import hnswlib
import numpy as np

class HNSWSearch:
    def __init__(self, dim=128, max_elements=10000, ef=200, M=16):
        """
        Initializes an HNSW index.
        - dim: Dimensionality of the vectors
        - max_elements: Maximum number of vectors
        - ef: Controls search accuracy (higher = better but slower)
        - M: Controls connectivity in the graph (higher = better recall)
        """
        self.dim = dim
        self.index = hnswlib.Index(space='l2', dim=dim)  # Using L2 (Euclidean distance)
        self.index.init_index(max_elements=max_elements, ef_construction=ef, M=M)
        self.index.set_ef(ef)
        self.keys = []

    def add_vectors(self, vectors):
        """Adds multiple vectors to the HNSW index."""
        vector_array = np.array(list(vectors.values()))
        keys = list(vectors.keys())
        self.keys.extend(keys)
        self.index.add_items(vector_array)

    def search(self, vectors, query_vector, k=1, metric="euclidean"):
        """Searches for the top-k nearest neighbors using HNSW."""
        if metric not in ["euclidean", "cosine"]:
            raise ValueError("HNSW only supports 'euclidean' and 'cosine'.")

        query_vector = np.array([query_vector])
        labels, distances = self.index.knn_query(query_vector, k=k)

        return [(self.keys[idx], -distances[0][i]) for i, idx in enumerate(labels[0])]
