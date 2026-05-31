"""
Main FastAPI application entry point.

Run with:
  uvicorn main:app --reload --port 8000

Interactive API docs at: http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router
from config import API_VERSION, ENVIRONMENT

load_dotenv()

app = FastAPI(
    title="Financial Document Q&A API",
    description=(
        "RAG-powered question answering over financial documents. "
        "Upload 10-K reports, earnings transcripts, or SEC filings and ask "
        "natural language questions — get grounded answers with page citations."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow browser frontends (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENVIRONMENT == "development" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Financial Document Q&A API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
