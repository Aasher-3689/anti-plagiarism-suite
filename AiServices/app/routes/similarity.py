# Imports
import numpy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Model Loadings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper Functions
def generate_embeddings(answers: list[str]):
    """Convert a list of answers into list of vectors in batch
       Return Type: numpy.ndarray"""
    embeddings = model.encode(answers, batch_size=32)
    return embeddings

def generate_cosine_similarity(embeddings: numpy.ndarray):
    """Compute similarity of one embedding with every other embedding
       Return Type: numpy.ndarray"""
    similarities = cosine_similarity(embeddings)
    return similarities