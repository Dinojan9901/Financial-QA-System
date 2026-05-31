"""
API Schemas — Pydantic request/response models for the FastAPI endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Ingest endpoint ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    source: str
    pages: int
    chunks: int
    avg_tokens_per_chunk: int
    message: str


# ── Ask endpoint ─────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        description="Natural language question about the financial document",
        examples=["What was the total revenue in fiscal 2023?"],
    )
    document_name: Optional[str] = Field(
        None,
        description="Filter to a specific document (file name). If None, searches all documents.",
    )
    max_results: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve (1–10)",
    )


class Source(BaseModel):
    page: Optional[int]
    source: Optional[str]
    relevance: float


class QuestionResponse(BaseModel):
    question: str
    answer: str
    model: str
    sources: List[Source]
    tokens_used: int
    chunks_retrieved: int


# ── Documents list endpoint ───────────────────────────────────────────────────

class DocumentListResponse(BaseModel):
    documents: List[str]
    total_count: int


# ── Delete endpoint ───────────────────────────────────────────────────────────

class DeleteResponse(BaseModel):
    status: str
    source: str


# ── Health endpoint ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    total_chunks_indexed: int
