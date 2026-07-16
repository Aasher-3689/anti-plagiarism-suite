# Imports
from pydantic import BaseModel


# Schemas
class FlaggedPair(BaseModel):
    student1_id: str
    student2_id: str
    similarity_score: float
