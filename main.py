from vector_db.database import VectorDatabase

# Initialize with brute-force search
db = VectorDatabase(persist_path="vectors.pkl", search_method="brute_force")

# Add vectors
db.add_vector("vector_1", [1, 2, 3])
db.add_vector("vector_2", [4, 5, 6])
db.add_vector("vector_3", [7, 8, 9])

# Search with different metrics
query = [2, 3, 4]
print("Cosine similarity:", db.search(query, k=2, metric="cosine"))
print("Euclidean distance:", db.search(query, k=2, metric="euclidean"))
print("Manhattan distance:", db.search(query, k=2, metric="manhattan"))
print("Jaccard similarity:", db.search(query, k=2, metric="jaccard"))

db.save()
