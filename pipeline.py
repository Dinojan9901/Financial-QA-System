"""
Pipeline — the full RAG (Retrieval-Augmented Generation) system.

Two entry points:
  • ingest_document(pdf_path)  — run ONCE per document (indexes it)
  • answer_question(question)  — run on EVERY user query (retrieves + generates)
"""

from pathlib import Path
from typing import Dict, Optional

from ingestion.pdf_loader import FinancialPDFLoader
from ingestion.text_chunker import FinancialTextChunker
from ingestion.embedder import EmbeddingGenerator
from retrieval.vector_store import VectorStore
from retrieval.retriever import FinancialRetriever
from generation.qa_chain import get_qa_chain
from config import TOP_K_RESULTS


# ── Phase 1: Indexing ─────────────────────────────────────────────────────────

def ingest_document(pdf_path: str, source_name: Optional[str] = None) -> Dict:
    """
    Full ingestion pipeline for one financial PDF.

    Steps:
      1. Extract text + tables from PDF pages
      2. Split into overlapping token-counted chunks
      3. Generate dense embeddings for each chunk
      4. Upsert into ChromaDB vector store

    Run this ONCE per document. Takes 30 sec – 5 min depending on size.
    """
    print(f"\n{'='*60}")
    print(f"INGESTING: {pdf_path}")
    print(f"{'='*60}")

    # Step 1 — Extract
    print("\nStep 1/4: Extracting text from PDF...")
    loader = FinancialPDFLoader(pdf_path)
    pages = loader.extract_text()
    doc_info = loader.get_document_info()
    source = Path(source_name).name if source_name else doc_info["file_name"]
    for page in pages:
        page["source"] = source
    print(f"  Extracted {len(pages)} non-empty pages from {doc_info['total_pages']} total")

    if not pages:
        raise ValueError(f"No text could be extracted from {pdf_path}")

    # Step 2 — Chunk
    print("\nStep 2/4: Chunking text...")
    chunker = FinancialTextChunker()
    chunks = chunker.chunk_pages(pages)
    stats = chunker.get_chunk_stats(chunks)
    print(f"  Stats: {stats}")

    # Step 3 — Embed
    print("\nStep 3/4: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.embed_chunks(chunks)

    # Step 4 — Store
    print("\nStep 4/4: Storing in vector database...")
    store = VectorStore()
    store.add_chunks(embedded_chunks)

    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE — '{pdf_path}' is ready for questions!")
    print(f"{'='*60}\n")

    return {
        "status": "success",
        "source": source,
        "pages": len(pages),
        "chunks": len(chunks),
        "avg_tokens_per_chunk": stats.get("avg_tokens", 0),
    }


# ── Phase 2: Querying ─────────────────────────────────────────────────────────

def answer_question(
    question: str,
    source_filter: Optional[str] = None,
    n_results: int = TOP_K_RESULTS,
) -> Dict:
    """
    Full querying pipeline for a natural language question.

    Steps:
      1. Embed the question (same model as documents)
      2. Search vector DB for top-k relevant chunks
      3. Feed chunks + question to LLM → grounded answer

    Runs on every user query (fast: ~1–3 seconds).
    """
    # Step 1+2 — Retrieve relevant context
    retriever = FinancialRetriever()
    context_chunks = retriever.retrieve(
        question=question,
        n_results=n_results,
        source_filter=source_filter,
    )

    if not context_chunks:
        return {
            "answer": "No documents have been ingested yet, or no relevant content found.",
            "question": question,
            "sources": [],
            "chunks_retrieved": 0,
        }

    # Step 3 — Generate grounded answer
    qa_chain = get_qa_chain()
    result = qa_chain.answer(question=question, context_chunks=context_chunks)
    return result


# ── Utility helpers ───────────────────────────────────────────────────────────

def list_indexed_documents() -> list:
    """Return names of all documents currently in the vector store."""
    store = VectorStore()
    return store.list_documents()


def delete_document(source_name: str) -> Dict:
    """Remove a document and all its chunks from the vector store."""
    store = VectorStore()
    store.delete_document(source_name)
    return {"status": "deleted", "source": source_name}


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        result = ingest_document(pdf)
        print(result)

        question = sys.argv[2] if len(sys.argv) > 2 else "What is this document about?"
        answer = answer_question(question)
        print("\nANSWER:", answer["answer"])
        print("\nSOURCES:")
        for s in answer.get("sources", []):
            print(f"  Page {s['page']} | Relevance: {s['relevance']}")
    else:
        print("Usage: python pipeline.py <path_to_pdf> [question]")
        print("\nCurrently indexed documents:")
        docs = list_indexed_documents()
        print(docs if docs else "  (none)")
