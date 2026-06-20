"""
API Routes — FastAPI endpoint definitions.

Endpoints:
  POST /api/v1/ingest    — upload + index a financial PDF
  POST /api/v1/ask       — ask a question about indexed documents
  GET  /api/v1/documents — list all indexed documents
  DELETE /api/v1/documents/{name} — remove a document
  GET  /api/v1/health    — health check
"""

import os
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas import (
    IngestResponse,
    QuestionRequest,
    QuestionResponse,
    DocumentListResponse,
    DeleteResponse,
    HealthResponse,
)
from pipeline import ingest_document, answer_question, list_indexed_documents, delete_document
from retrieval.vector_store import VectorStore
from config import API_VERSION

router = APIRouter(prefix="/api/v1", tags=["Financial Q&A"])


# ── POST /ingest ──────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="Upload and index a financial PDF")
async def ingest_pdf(file: UploadFile = File(..., description="Financial PDF document")):
    """
    Upload a financial PDF (10-K, earnings call, SEC filing).
    The document is extracted, chunked, embedded, and stored for Q&A.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to a temp file so pdfplumber can open it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_document(tmp_path, source_name=file.filename)
        return IngestResponse(
            status=result["status"],
            source=file.filename,
            pages=result["pages"],
            chunks=result["chunks"],
            avg_tokens_per_chunk=result["avg_tokens_per_chunk"],
            message=f"'{file.filename}' is indexed and ready for questions!",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ── POST /ask ─────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=QuestionResponse, summary="Ask a question about documents")
async def ask_question(request: QuestionRequest):
    """
    Ask a natural language question. The system retrieves relevant document
    sections and returns a grounded answer with page citations.
    """
    try:
        result = answer_question(
            question=request.question,
            source_filter=request.document_name,
            n_results=request.max_results,
        )
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents():
    """Return a list of all financial documents currently indexed."""
    docs = list_indexed_documents()
    return DocumentListResponse(documents=docs, total_count=len(docs))


# ── DELETE /documents/{name} ──────────────────────────────────────────────────

@router.delete(
    "/documents/{document_name}",
    response_model=DeleteResponse,
    summary="Remove a document from the index",
)
async def remove_document(document_name: str):
    """Delete all chunks for the specified document from the vector store."""
    try:
        result = delete_document(document_name)
        return DeleteResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Returns system status and the number of indexed chunks."""
    store = VectorStore()
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        total_chunks_indexed=store.get_document_count(),
    )
