from vector_db.database import VectorDatabase

# Initialize the vector database
db = VectorDatabase(persist_path="vectors.pkl")

# Add vectors
db.add_vector("vector_1", [1, 2, 3])
db.add_vector("vector_2", [4, 5, 6])
db.add_vector("vector_3", [7, 8, 9])

# Search for similar vectors
query = [2, 3, 4]
result = db.search(query, k=2, metric="cosine")

print("Top matches:", result)

# Save the database
db.save()
