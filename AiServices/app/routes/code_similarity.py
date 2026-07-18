# Imports
from fastapi import APIRouter, HTTPException
from app.schemas.code_similarity import  ASTFlaggedPair, ASTRequest, ASTResponse, CodeSubmission
import ast
import zss


# Router
router = APIRouter(prefix="/api")

@router.post("/code-similarity", response_model=ASTResponse)
async def check_ast(request: ASTRequest):
    """Compare code submission using AST
       Current Languages: Python"""
    total_submissions = len(request.submissions)
    if total_submissions < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 submissions are required for code similarity check."
        )
    total_comparisons = total_submissions * (total_submissions - 1) // 2
    flagged = get_flagged_pairs(request.submissions,
                                request.threshold)
    return ASTResponse(
        exam_id = request.exam_id,
        question_id = request.question_id,
        total_submissions = total_submissions,
        total_comparisons = total_comparisons,
        total_flagged_pairs = len(flagged),
        threshold = request.threshold,
        results = flagged
    )


# Helper Functions
def get_flagged_pairs(submissions: list[CodeSubmission],
                      threshold: float):
    flagged: list[ASTFlaggedPair] = []
    length = len(submissions)
    for i in range(length):
        for j in range(i+1, length):
            score = compute_tree_edit_dist(submissions[i].code,
                                           submissions[j].code)
            if score >= threshold:
                flagged.append(ASTFlaggedPair(
                    student1_id = submissions[i].student_id,
                    student2_id = submissions[j].student_id,
                    similarity_score = round(score, 4)
                ))
    return flagged


def parse_ast(code: str):
    """Parse Python code into AST( abstract syntax tree)"""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def convert_ast_to_zss(node: ast.AST):
    """Variable & arguments names are ignored
       Convert AST to ZSS recursively """
    if isinstance(node, ast.Name):
        label = "VARIABLE"
    elif isinstance(node, ast.arg):
        label = "ARGUMENT"
    else:
        label = type(node).__name__
    zss_node = zss.Node(label)

    for child in ast.iter_child_nodes(node):
        zss_node.addkid(convert_ast_to_zss(child))

    return zss_node


def compute_tree_edit_dist(code1: str, code2: str):
    """Compare two codes by tree-edit-distannce
       Return similarity score between 0 & 1 """
    tree1: ast.AST | None = parse_ast(code1)
    tree2: ast.AST | None = parse_ast(code2)
    if not tree1 or not tree2:
        return 0.0
    zss_tree1 = convert_ast_to_zss(tree1)
    zss_tree2 = convert_ast_to_zss(tree2)

    distance = zss.simple_distance(zss_tree1, zss_tree2)
    # Converting distance to score between 0 & 1
    size1 = sum(1 for _ in ast.walk(tree1))
    size2 = sum(1 for _ in ast.walk(tree2))
    max_size = max(size1, size2)

    if max_size == 0:
        return 0.0

    distance = distance / max_size
    similarity_score = 1 - distance
    similarity_score = round(similarity_score, 4)
    return max(0.0, float(similarity_score))