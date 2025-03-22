from sklearn.neighbors import BallTree
import numpy as np

class BallTreeSearch:
    def search(self, vectors, query_vector, k, metric):
        """Uses BallTree for fast nearest neighbor search."""
        if metric not in ["euclidean", "manhattan", "minkowski", "cosine"]:
            raise ValueError("BallTree supports 'euclidean', 'manhattan', 'minkowski', and 'cosine'.")

        keys = list(vectors.keys())
        data = np.array(list(vectors.values()))
        tree = BallTree(data, metric=metric)

        distances, indices = tree.query([query_vector], k)
        return [(keys[i], distances[0][i]) for i in range(k)]
