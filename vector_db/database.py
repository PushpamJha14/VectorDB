import numpy as np
from .storage import save_vectors, load_vectors
from .search_algorithms.brute_force import BruteForceSearch
from .search_algorithms.kd_tree import KDTreeSearch
from .search_algorithms.ball_tree import BallTreeSearch
from .search_algorithms.hnsw import HNSWSearch
from .search_algorithms.pq import PQSearch

class VectorDatabase:
    def __init__(self, persist_path=None, search_method="brute_force", dim=128):
        self.vectors = {}
        self.persist_path = persist_path
        self.dim = dim

        if persist_path:
            self.vectors = load_vectors(persist_path)

        # Choose search mechanism
        search_methods = {
            "brute_force": BruteForceSearch(),
            "kd_tree": KDTreeSearch(),
            "ball_tree": BallTreeSearch(),
            "hnsw": HNSWSearch(dim=dim),
            "pq": PQSearch(dim=dim)
        }
        self.search_engine = search_methods.get(search_method, None)

        if not self.search_engine:
            raise ValueError(f"Invalid search method: {search_method}")

        if search_method in ["hnsw", "pq"]:
            self.search_engine.add_vectors(self.vectors)

    def add_vector(self, key, vector):
        """Stores a vector with an associated key."""
        self.vectors[key] = np.array(vector)
        if isinstance(self.search_engine, (HNSWSearch, PQSearch)):
            self.search_engine.add_vectors({key: vector})

    def search(self, query_vector, k=1, metric="cosine"):
        """Finds the top-k nearest vectors using the chosen search algorithm."""
        return self.search_engine.search(self.vectors, query_vector, k, metric)

    def save(self):
        """Saves vectors to disk."""
        if self.persist_path:
            save_vectors(self.persist_path, self.vectors)
