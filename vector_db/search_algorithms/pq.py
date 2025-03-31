import faiss
import numpy as np

class PQSearch:
    def __init__(self, dim=128, num_clusters=64):
        """
        Initializes a FAISS Product Quantization index.
        - dim: Dimensionality of vectors
        - num_clusters: Number of clusters for quantization
        """
        self.dim = dim
        self.index = faiss.IndexPQ(dim, num_clusters, faiss.METRIC_L2)
        self.keys = []
        self.vectors = {}

    def add_vectors(self, vectors):
        """Adds vectors to the PQ index."""
        vector_array = np.array(list(vectors.values()), dtype=np.float32)
        self.keys.extend(vectors.keys())
        self.vectors.update(vectors)
        self.index.train(vector_array)  # Train PQ
        self.index.add(vector_array)  # Add vectors

    def search(self, vectors, query_vector, k=1, metric="euclidean"):
        """Searches for the top-k nearest neighbors using PQ."""
        if metric != "euclidean":
            raise ValueError("PQ only supports 'euclidean'.")

        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector, k)

        return [(self.keys[i], -distances[0][idx]) for idx, i in enumerate(indices[0])]
