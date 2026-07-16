# Imports
from pydantic import BaseModel, Field


# Schemas
class FlaggedPair(BaseModel):
    student1_id: str = Field(min_length=1)
    student2_id: str = Field(min_length=1)
    similarity_score: float = Field(ge=0, le=1)

class Submission(BaseModel):
    """A single student submission"""
    student_id: str = Field(min_length=1)
    answer: str


class SimilarityResponse(BaseModel):
    exam_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    total_submissions: int
    total_comparisons: int
    total_flagged_pairs: int
    threshold: float = Field(default=0.85, ge=0, le=1)
    results: list[FlaggedPair]


class SimilarityRequest(BaseModel):
    exam_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    threshold: float = Field(default=0.85, ge=0, le=1)
    submissions: list[Submission]