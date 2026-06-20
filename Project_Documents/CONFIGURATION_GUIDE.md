# Configuration Guide

## Initial Setup

```powershell
Copy-Item .env.example .env
```

The `.env` file is optional unless custom paths or ports are needed.

## Environment Variables

| Variable | Example | Purpose |
|---|---|---|
| `CHROMA_DB_PATH` | `./data/chroma_db` | Vector index location |
| `UPLOAD_DIR` | `./data/uploads` | PDF upload location |
| `ENVIRONMENT` | `development` | API environment behavior |
| `API_PORT` | `8000` | FastAPI port |
| `STREAMLIT_PORT` | `8501` | Streamlit port |

## Embeddings

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
```

Sentence Transformers downloads MiniLM on first use and then loads it from the
local cache. The conservative chunk size avoids exceeding MiniLM's input limit.

## Changing Embedding Models

1. Stop Streamlit and FastAPI.
2. Change the model and dimension in `config.py`.
3. Remove `data/chroma_db/`.
4. Restart the application and re-index every PDF.

Vectors produced by different models must not share a collection.
