# Imports
from pydantic import BaseModel, Field


# Schemas
class CodeSubmission(BaseModel):
    student_id: str = Field(min_length=1)
    code: str


class ASTFlaggedPair(BaseModel):
    student1_id: str = Field(min_length=1)
    student2_id: str = Field(min_length=1)
    similarity_score: float = Field(ge=0, le=1)


class ASTRequest(BaseModel):
    exam_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    threshold: float = Field(default=0.85, ge=0, le=1)
    submissions: list[CodeSubmission]


class ASTResponse(BaseModel):
    exam_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    total_submissions: int
    total_comparisons: int
    total_flagged_pairs: int
    threshold: float = Field(ge=0, le=1)
    results: list[ASTFlaggedPair]