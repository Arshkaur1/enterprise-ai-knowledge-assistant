from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

sentences = [
    "Employees can work remotely three days per week.",
    "Remote work is permitted with manager approval.",
    "Passwords must contain at least 12 characters.",
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
)

print("Model loaded successfully")
print("Embedding shape:", embeddings.shape)
print("Embedding dimensions:", embeddings.shape[1])

similarities = embeddings @ embeddings.T

print("\nCosine similarity matrix:")
print(similarities.round(4))