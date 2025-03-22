from scipy.spatial import KDTree
import numpy as np

class KDTreeSearch:
    def search(self, vectors, query_vector, k, metric):
        """Uses k-d tree for nearest neighbor search."""
        if metric not in ["euclidean", "manhattan"]:
            raise ValueError("KDTree only supports 'euclidean' or 'manhattan'.")

        keys = list(vectors.keys())
        data = np.array(list(vectors.values()))
        tree = KDTree(data)

        distances, indices = tree.query(query_vector, k)
        return [(keys[i], distances[i]) for i in range(len(indices))]
