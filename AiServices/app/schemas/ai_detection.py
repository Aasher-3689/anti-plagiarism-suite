# Imports
from pydantic import BaseModel, Field
from enum import Enum


# Enums
class SignalEnum(Enum):
    ai = "Possible AI Content"
    human = "Likely Human"


# Schemas
class Submission(BaseModel):
    """A single student submission"""
    student_id: str = Field(min_length=1)
    answer: str


class Result(BaseModel):
    student_id: str = Field(min_length=1)
    ai_score: float = Field(ge=0, le=100)
    signal: SignalEnum = Field(default=SignalEnum.human)


class AIDetectionRequest(BaseModel):
    exam_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    threshold: float = Field(default=70, ge=0, le=100)
    submissions: list[Submission]


class AIDetectionResponse(BaseModel):
    exam_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    total_submissions: int
    total_flagged: int
    threshold: float = Field(ge=0, le=100)
    results: list[Result]