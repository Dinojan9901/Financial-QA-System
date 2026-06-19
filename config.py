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
# Set USE_LOCAL_MODELS=true in .env to run embeddings fully free (MiniLM).
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Embedding settings (retrieval) ───────────────────────────────────────────
# Embeddings default to the free local MiniLM model; OpenAI embeddings are only
# used when USE_LOCAL_MODELS=false *and* an OpenAI key is present. (Groq does not
# serve embeddings, so Groq users keep MiniLM embeddings -- still fully free.)
if USE_LOCAL_MODELS or not OPENAI_API_KEY.strip():
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"        # free, local, 384-dim
    EMBEDDING_DIMENSIONS = 384
else:
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI, 1536-dim
    EMBEDDING_DIMENSIONS = 1536


# ── Chat-LLM provider (answer generation) ────────────────────────────────────
def resolve_llm_provider():
    """Resolve the chat-LLM provider from the CURRENT environment (evaluated
    live so a key entered in the web UI at runtime takes effect immediately).

    Preference when LLM_PROVIDER=auto: Groq (free) -> OpenAI -> local.
    Returns a 4-tuple: (provider, api_key, base_url, model).
    Both Groq and OpenAI speak the OpenAI Chat Completions API, so the same
    client code drives either one -- only the base URL and model differ.
    """
    def clean(value: str) -> str:
        value = (value or "").strip()
        # treat .env.example placeholders (e.g. "sk-your-key-here") as unset
        if not value or "your-" in value or "-here" in value:
            return ""
        return value

    forced = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    groq_key = clean(os.getenv("GROQ_API_KEY"))
    openai_key = clean(os.getenv("OPENAI_API_KEY"))

    if forced == "groq" or (forced == "auto" and groq_key):
        return ("groq", groq_key, "https://api.groq.com/openai/v1",
                os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile")
    if forced == "openai" or (forced == "auto" and openai_key):
        return ("openai", openai_key, None,           # None = default OpenAI URL
                os.getenv("LLM_MODEL") or "gpt-4o-mini")
    return ("local", "", None, "local-retrieval-only")


# Snapshot at import (used by modules that read these as constants).
LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = resolve_llm_provider()
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
