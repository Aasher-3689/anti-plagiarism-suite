# Imports
import numpy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.schemas.similarity import FlaggedPair


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


def get_flagged_pairs(similarities: numpy.ndarray,
                     submissions: list,
                     threshold: float):
    """Iterate over similarities matrix, and find flagged pair
       based on the threshold. Avoiding duplicate comparisons
       and self comparisons.
       Return Type: list[FlaggedPair]"""
    flagged: list[FlaggedPair] = []
    length = len(submissions)
    for i in range(length):
        for j in range(i+1, length):
            score = similarities[i,j]
            if score >= threshold:
                flagged.append(FlaggedPair(
                    student1_id = submissions[i].student_id,
                    student2_id = submissions[j].student_id,
                    similarity_score = float(score)
                ))
    return flagged