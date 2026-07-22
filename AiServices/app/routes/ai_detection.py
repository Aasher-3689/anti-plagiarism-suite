# Imports
from fastapi import APIRouter, HTTPException
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from app.schemas.ai_detection import Submission, Result, SignalEnum, AIDetectionRequest, AIDetectionResponse


# Router
router = APIRouter(prefix="/api")

@router.post("/ai-detection", response_model=AIDetectionResponse)
async def check_ai_detection(request: AIDetectionRequest):
    submissions = len(request.submissions)
    if submissions < 1:
        raise HTTPException(
            status_code=400,
            detail="At least 1 submission is required for AI Detection."
        )
    results, flagged = get_result(request.submissions, request.threshold)
    return AIDetectionResponse(
        exam_id = request.exam_id,
        question_id = request.question_id,
        total_submissions = submissions,
        total_flagged = flagged,
        threshold = request.threshold,
        results = results
    )


# GPT2 Load
tokenizer = GPT2TokenizerFast.from_pretrained("GPT2")
model = GPT2LMHeadModel.from_pretrained("GPT2")
model.eval()

# Helper Functions
def calculate_perplexity(answer: str):
    """Calculate GPT2 perplexity score
       Low perplexity = AI likely
       High perplexity = human likely
       Return: float """
    inputs = tokenizer(answer, return_tensors="pt",
                       max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity


def perplexity_to_score(perplexity: float):
    """Convert perplexity to score between 0 and 100
       Lower perplexity = AI likely (high score)
       Return: float """
    # /3 is scaling factor, which spread the perplexity between 0 and 100
    # This is approximate, not a scientific formula
    score = max(0, min(100, 100 - perplexity / 3))
    return round(score, 4)


def get_result(submissions: list[Submission],
               threshold: float):
    """Signal flagged submissions based on the threshold
       Return: (list[Result], int) """
    results: list[Result] = []
    flagged = 0
    for submission in submissions:
        perplexity = calculate_perplexity(submission.answer)
        ai_score = perplexity_to_score(perplexity)
        signal = SignalEnum.human
        if ai_score >= threshold:
            flagged += 1
            signal = SignalEnum.ai
        results.append(Result(
            student_id = submission.student_id,
            ai_score = ai_score,
            signal = signal
        ))
    return results, flagged