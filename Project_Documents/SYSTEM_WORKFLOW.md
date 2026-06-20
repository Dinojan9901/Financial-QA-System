# System Workflow

## Phase 1: Index a PDF

```text
PDF upload
  -> extract text and tables per page
  -> create overlapping chunks
  -> convert chunks into 384-dimensional MiniLM vectors
  -> store vectors, text, and metadata in ChromaDB
```

1. Streamlit or FastAPI saves an upload to a temporary path.
2. The original filename is passed separately to `ingest_document()`.
3. `FinancialPDFLoader` uses pdfplumber, with PyPDF2 as fallback.
4. Each non-empty page retains its page number and original filename.
5. `FinancialTextChunker` uses a 200-token size and 40-token overlap.
6. Each chunk receives an ID, page number, token count, and source.
7. `all-MiniLM-L6-v2` creates the embedding for every chunk.
8. `VectorStore` upserts the chunks into the local ChromaDB collection.

Indexing normally happens once for each document.

## Phase 2: Ask a Question

```text
Question
  -> embed with the same MiniLM model
  -> cosine-similarity search
  -> retrieve the best chunks
  -> return an excerpt or generate a grounded answer
```

1. `FinancialRetriever` embeds the question.
2. Search can optionally filter by the original document filename.
3. ChromaDB ranks chunks using cosine similarity.
4. The configured top results become the answer context.
5. `LocalQAChain` returns the strongest excerpt in free local mode.
6. `FinancialQAChain` can instead generate a constrained answer.
7. Responses include source names, page numbers, and relevance scores.

## Entry Points

- `streamlit run app.py`: visual upload and Q&A interface.
- `uvicorn main:app --reload`: REST API and Swagger documentation.
- `python pipeline.py <pdf> [question]`: direct pipeline smoke test.
- `python run_evaluation.py`: evaluation experiments.

## Important Invariant

Documents and questions must use the same embedding model. After changing the
model or dimension, remove `data/chroma_db/` and index every PDF again. Vectors
from different models must not share a collection.
