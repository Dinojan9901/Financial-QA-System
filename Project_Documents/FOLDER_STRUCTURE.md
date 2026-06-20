# Folder Structure

```text
Financial-QA-System/
|-- Project_Documents/  Explanations, report, presentation, and checklist
|-- api/                FastAPI routes and schemas
|-- ingestion/          PDF loading, chunking, and embeddings
|-- retrieval/          ChromaDB storage and semantic retrieval
|-- generation/         Prompt construction and answer generation
|-- evaluation/         Benchmarks and saved results
|-- tests/              Automated tests
|-- notebooks/          Interactive walkthrough
|-- data/               Runtime uploads and vector data
|-- app.py              Streamlit entry point
|-- main.py             FastAPI entry point
|-- pipeline.py         Ingestion and Q&A orchestration
|-- config.py           Environment and model configuration
|-- run_evaluation.py   Evaluation runner
|-- demo.py             Terminal demonstration
|-- requirements.txt    Python dependencies
|-- Dockerfile          Container definition
|-- .env.example        Environment template
`-- README.md           Setup and usage landing page
```

## Source Modules

### `ingestion/`

- `pdf_loader.py`: extracts page text and tables and cleans PDF artifacts.
- `text_chunker.py`: creates chunks with source and page metadata.
- `embedder.py`: creates local all-MiniLM-L6-v2 embeddings.

### `retrieval/`

- `vector_store.py`: persists vectors, text, and metadata in ChromaDB.
- `retriever.py`: embeds questions and returns similar chunks.

### `generation/`

- `prompt_builder.py`: builds grounded prompts with retrieved passages.
- `qa_chain.py`: selects retrieval-only or generated-answer behavior.

### `api/`

- `schemas.py`: validates API inputs and outputs.
- `routes.py`: provides upload, ask, list, delete, and health endpoints.

### `evaluation/`

- `baseline_comparison.py`: compares retrieval approaches.
- `embedding_benchmark.py`: measures embedding ranking quality.
- `rag_vs_norag.py`: compares RAG, no-RAG, and random context.
- `results/`: contains timestamped JSON output.

## Documentation and Runtime Data

- `Project_Documents/report/`: technical report files.
- `Project_Documents/presentation/`: slides, script, and export tools.
- `data/uploads/`: uploaded PDFs.
- `data/chroma_db/`: local index; rebuild after changing embeddings.
- `__init__.py`: marks a directory as a Python package.
