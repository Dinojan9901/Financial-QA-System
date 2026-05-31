import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "chroma_db"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

# ── Model selection ──────────────────────────────────────────────────────────
# Set USE_LOCAL_MODELS=true in .env to run fully free with sentence-transformers
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Embedding settings ───────────────────────────────────────────────────────
if USE_LOCAL_MODELS:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, 384-dim
    EMBEDDING_DIMENSIONS = 384
else:
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI, 1536-dim
    EMBEDDING_DIMENSIONS = 1536

# ── LLM settings ─────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"          # cheap + good for Q&A
LLM_TEMPERATURE = 0.1              # low = factual, not creative
LLM_MAX_TOKENS = 1000

# ── Chunking settings ────────────────────────────────────────────────────────
CHUNK_SIZE = 800        # tokens per chunk (sweet spot for financial docs)
CHUNK_OVERLAP = 150     # ~18% overlap to preserve context at boundaries

# ── Retrieval settings ───────────────────────────────────────────────────────
TOP_K_RESULTS = 5       # number of chunks to retrieve per query
COLLECTION_NAME = "financial_documents"

# ── API settings ─────────────────────────────────────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = "1.0.0"
