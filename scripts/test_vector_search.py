from time import perf_counter

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.vector_store import VectorStore

documents = [
    "Employees can work remotely three days per week.",
    "Remote work is permitted with manager approval.",
    "Passwords must contain at least 12 characters.",
    "Expense claims must be submitted within 30 days.",
    "Multi-factor authentication is mandatory.",
]

# Initialize the embedding model
embedding_service = EmbeddingService()

# Generate document embeddings
start = perf_counter()

embeddings = embedding_service.embed_documents(documents)

embedding_time = perf_counter() - start

# Create and populate the FAISS index
store = VectorStore(dimension=embedding_service.dimension)

store.add(embeddings)

print("Embedding shape:", embeddings.shape)
print("Indexed vectors:", store.size)
print(f"Embedding time: {embedding_time:.4f} seconds")

# Generate a query embedding
query = "How often can employees work from home?"

start = perf_counter()

query_embedding = embedding_service.embed_query(query)

query_time = perf_counter() - start

# Search the index
start = perf_counter()

scores, indices = store.search(
    query_embedding,
    top_k=3,
)

search_time = perf_counter() - start

print(f"\nQuery: {query}")
print("\nTop 3 results:")

for rank, (score, index) in enumerate(
    zip(scores, indices),
    start=1,
):
    print(f"{rank}. Score: {score:.4f} | {documents[index]}")

print(f"\nQuery embedding time: {query_time:.4f}s")
print(f"FAISS search time: {search_time:.6f}s")

# Test index persistence
store.save("artifacts/demo_vector_index.faiss")

loaded_store = VectorStore.load("artifacts/demo_vector_index.faiss")

print("\nReloaded index size:", loaded_store.size)

loaded_scores, loaded_indices = loaded_store.search(
    query_embedding,
    top_k=3,
)

print("Reloaded search indices:", loaded_indices)
