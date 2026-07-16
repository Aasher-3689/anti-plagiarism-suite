# Imports
from sentence_transformers import SentenceTransformer

# Model Loadings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper Functions
def generate_embeddings(answers: list[str]):
    embeddings = model.encode(answers, batch_size=32)
    return embeddings