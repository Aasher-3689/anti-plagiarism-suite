# Imports
from fastapi import FastAPI
from app.routes.similarity import router as similarity_router
from app.routes.code_similarity import router as code_similarity_router


# Creating app
app = FastAPI(title="Anti Plagiarism Suite & Academic Integrity Suite")


# Adding routes
app.include_router(similarity_router)
app.include_router(code_similarity_router)