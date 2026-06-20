"""
inspect_pipeline.py  --  Manual step-by-step pipeline inspector

Runs your real PDF through every pipeline stage and saves a human-readable
output file for each stage so you can open them and see exactly what happened.

Usage:
    python inspect_pipeline.py <path_to_pdf> "<your question>"

Example:
    python inspect_pipeline.py data/uploads/acme_report.pdf "What were the operating expenses?"

Output files saved to:  process/<pdf_name>/
    1_extracted.txt    raw text pulled from the PDF
    2_chunks.txt       text split into numbered chunks
    3_embeddings.txt   each chunk + first 8 embedding values
    4_stored.txt       confirmation of what was saved to ChromaDB
    5_retrieved.txt    top chunks returned for your question
    6_answer.txt       the final extracted sentence answer
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows so special characters don't crash the terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Argument check ────────────────────────────────────────────────────────────

if len(sys.argv) < 3:
    print("Usage: python inspect_pipeline.py <path_to_pdf> \"<your question>\"")
    print()
    print("Example:")
    print("  python inspect_pipeline.py data/uploads/report.pdf \"What were the operating expenses?\"")
    sys.exit(1)

PDF_PATH = Path(sys.argv[1])
QUESTION = sys.argv[2]

if not PDF_PATH.exists():
    print(f"ERROR: PDF not found at: {PDF_PATH}")
    sys.exit(1)

PDF_NAME   = PDF_PATH.stem                       # filename without .pdf
OUTPUT_DIR = Path("process") / PDF_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIVIDER = "─" * 56
HEADER  = "=" * 60

print(f"\n{HEADER}")
print("  PIPELINE INSPECTOR")
print(HEADER)
print(f"  PDF      : {PDF_PATH}")
print(f"  Question : {QUESTION}")
print(f"  Output   : {OUTPUT_DIR}/")
print(f"{HEADER}\n")


def save_file(filename: str, lines: list) -> None:
    path = OUTPUT_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved -> {path}")


# ── STEP 1: PDF Extraction ────────────────────────────────────────────────────

print("STEP 1/6: Extracting text from PDF ...")

from ingestion.pdf_loader import FinancialPDFLoader

loader   = FinancialPDFLoader(str(PDF_PATH))
pages    = loader.extract_text()
doc_info = loader.get_document_info()
source   = doc_info["file_name"]

for page in pages:
    page["source"] = source

lines = [
    "=== STEP 1: PDF EXTRACTION ===",
    f"Source         : {source}",
    f"Total pages    : {doc_info.get('total_pages', '?')}",
    f"Non-empty pages: {len(pages)}",
    "",
    "Each section below is one page extracted from the PDF.",
    "This is the raw text the rest of the pipeline will work with.",
    "",
]

for page in pages:
    lines += [
        DIVIDER,
        f"PAGE {page['page_number']}",
        DIVIDER,
        page["text"],
        "",
    ]

save_file("1_extracted.txt", lines)
print(f"  Extracted {len(pages)} page(s) from {doc_info.get('total_pages', '?')} total\n")


# ── STEP 2: Chunking ──────────────────────────────────────────────────────────

print("STEP 2/6: Splitting text into chunks ...")

from ingestion.text_chunker import FinancialTextChunker

chunker = FinancialTextChunker()
chunks  = chunker.chunk_pages(pages)
stats   = chunker.get_chunk_stats(chunks)

lines = [
    "=== STEP 2: CHUNKING ===",
    f"Total chunks  : {stats.get('total_chunks', len(chunks))}",
    f"Avg tokens    : {stats.get('avg_tokens', '?')}",
    f"Min tokens    : {stats.get('min_tokens', '?')}",
    f"Max tokens    : {stats.get('max_tokens', '?')}",
    "",
    "Each chunk is a piece of text that will be embedded and stored.",
    "Look here if answers feel incomplete — small or broken chunks",
    "mean the chunker couldn't find good split points (e.g. no blank lines).",
    "",
]

for i, chunk in enumerate(chunks, 1):
    meta = chunk["metadata"]
    lines += [
        DIVIDER,
        f"CHUNK {i}  |  Page {meta.get('page_number')}  "
        f"|  Tokens: {meta.get('token_count')}  "
        f"|  ID: {meta.get('chunk_id')}",
        DIVIDER,
        chunk["text"],
        "",
    ]

save_file("2_chunks.txt", lines)
print(f"  Created {len(chunks)} chunk(s)  |  avg {stats.get('avg_tokens', '?')} tokens\n")


# ── STEP 3: Embedding ─────────────────────────────────────────────────────────

print("STEP 3/6: Generating embeddings ...")

from ingestion.embedder import EmbeddingGenerator
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

embedder        = EmbeddingGenerator()
embedded_chunks = embedder.embed_chunks(chunks)

lines = [
    "=== STEP 3: EMBEDDINGS ===",
    f"Model          : {EMBEDDING_MODEL}",
    f"Dimensions     : {EMBEDDING_DIMENSIONS}",
    f"Chunks embedded: {len(embedded_chunks)}",
    "",
    "Each chunk now has a 384-number vector that represents its meaning.",
    "Only the first 8 values are shown here — the full vector lives in ChromaDB.",
    "If the first 8 values look like small decimals near zero, the model is working.",
    "",
]

for i, chunk in enumerate(embedded_chunks, 1):
    emb     = chunk.get("embedding", [])
    preview = [round(v, 5) for v in emb[:8]]
    meta    = chunk["metadata"]
    lines += [
        DIVIDER,
        f"CHUNK {i}  |  Page {meta.get('page_number')}  "
        f"|  Tokens: {meta.get('token_count')}",
        DIVIDER,
        f"Text preview : {chunk['text'][:120].strip()} ...",
        f"Emb [0..7]   : {preview}",
        f"Vector norm  : {round(sum(v**2 for v in emb)**0.5, 5)}",
        "",
    ]

save_file("3_embeddings.txt", lines)
print(f"  Embedded {len(embedded_chunks)} chunk(s) at {EMBEDDING_DIMENSIONS} dimensions\n")


# ── STEP 4: Vector Store ──────────────────────────────────────────────────────

print("STEP 4/6: Storing chunks in ChromaDB ...")

from retrieval.vector_store import VectorStore
from config import COLLECTION_NAME, CHROMA_DB_PATH

store  = VectorStore()
before = store.get_document_count()
store.add_chunks(embedded_chunks)
after  = store.get_document_count()
docs   = store.list_documents()

lines = [
    "=== STEP 4: VECTOR STORE ===",
    f"Collection    : {COLLECTION_NAME}",
    f"ChromaDB path : {CHROMA_DB_PATH}",
    f"Chunks before : {before}",
    f"Chunks after  : {after}",
    f"Added now     : {after - before}",
    "",
    "All documents currently indexed in the database:",
]
for doc in docs:
    lines.append(f"  - {doc}")

lines += [
    "",
    "If 'Added now' is 0, this PDF was already in the database (that is fine).",
    "If ChromaDB path does not exist or collection is empty, re-run ingestion.",
]

save_file("4_stored.txt", lines)
print(f"  Stored {after - before} new chunk(s)  (total in DB: {after})\n")


# ── STEP 5: Retrieval ─────────────────────────────────────────────────────────

print(f"STEP 5/6: Retrieving top chunks for your question ...")
print(f"  \"{QUESTION}\"\n")

from retrieval.retriever import FinancialRetriever
from config import TOP_K_RESULTS

retriever      = FinancialRetriever()
context_chunks = retriever.retrieve(question=QUESTION, n_results=TOP_K_RESULTS)

lines = [
    "=== STEP 5: RETRIEVAL ===",
    f"Question      : {QUESTION}",
    f"Top-K setting : {TOP_K_RESULTS}",
    f"Results found : {len(context_chunks)}",
    "",
    "These are the chunks the system thinks best answer your question.",
    "Score is cosine similarity (0 = no match, 1 = identical meaning).",
    "Above 0.5 is a strong match. Below 0.3 means the answer may not be in the document.",
    "",
]

for i, chunk in enumerate(context_chunks, 1):
    score = chunk.get("similarity_score", 0)
    meta  = chunk.get("metadata", {})
    lines += [
        DIVIDER,
        f"RESULT {i}  |  Score: {score:.4f}  "
        f"|  Page {meta.get('page_number')}  "
        f"|  Source: {meta.get('source')}",
        DIVIDER,
        chunk.get("text", ""),
        "",
    ]

save_file("5_retrieved.txt", lines)
print(f"  Retrieved {len(context_chunks)} chunk(s)\n")


# ── STEP 6: Extractive Answer ─────────────────────────────────────────────────

print("STEP 6/6: Extracting best answer sentence ...")

from generation.qa_chain import get_qa_chain

qa     = get_qa_chain()
result = qa.answer(question=QUESTION, context_chunks=context_chunks)
answer = result.get("answer", "No answer generated.")

lines = [
    "=== STEP 6: EXTRACTIVE ANSWER ===",
    f"Question      : {QUESTION}",
    f"Model         : {result.get('model', '?')}",
    f"Chunks used   : {result.get('chunks_retrieved', '?')}",
    "",
    "The system split all retrieved chunks into individual sentences,",
    "embedded each sentence, and returned the one most similar to your question.",
    "",
    DIVIDER,
    "ANSWER",
    DIVIDER,
    answer,
    "",
    DIVIDER,
    "ALL SOURCES (ranked by chunk relevance)",
    DIVIDER,
]

for s in result.get("sources", []):
    lines.append(
        f"  Page {s.get('page')}  |  {s.get('source')}  "
        f"|  chunk relevance={s.get('relevance')}"
    )

save_file("6_answer.txt", lines)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{HEADER}")
print("  ALL STEPS COMPLETE")
print(HEADER)
print(f"\n  Output folder: {OUTPUT_DIR}/")
print()
print("    1_extracted.txt    raw text pulled from the PDF")
print("    2_chunks.txt       text split into numbered chunks")
print("    3_embeddings.txt   chunks with embedding preview")
print("    4_stored.txt       what was saved to ChromaDB")
print("    5_retrieved.txt    top matches for your question")
print("    6_answer.txt       final extracted sentence answer")
print()
print(f"  ANSWER:\n")
print(f"  {answer}")
print()
