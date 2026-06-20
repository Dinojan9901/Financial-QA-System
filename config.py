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
# The project uses one free local embedding model.

# ── Embedding settings ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
TOKENIZER_ENCODING = "cl100k_base"

# ── LLM settings ─────────────────────────────────────────────────────────────

# ── Chunking settings ────────────────────────────────────────────────────────
CHUNK_SIZE = 200        # avoids truncation by MiniLM's 256-token input limit
CHUNK_OVERLAP = 40      # 20% overlap preserves context at boundaries

# ── Retrieval settings ───────────────────────────────────────────────────────
TOP_K_RESULTS = 5       # number of chunks to retrieve per query
COLLECTION_NAME = "financial_documents"

# ── HuggingFace Inference API ─────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL  = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

# ── API settings ─────────────────────────────────────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = "1.0.0"
