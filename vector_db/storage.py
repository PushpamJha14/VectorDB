import pickle

def save_vectors(filepath, vectors):
    """Saves vectors to a file using pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(vectors, f)

def load_vectors(filepath):
    """Loads vectors from a file."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}
