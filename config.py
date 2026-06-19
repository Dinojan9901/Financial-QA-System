import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "chroma_db"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if USE_LOCAL_MODELS or not OPENAI_API_KEY.strip():
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS = 384
else:
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536

def resolve_llm_provider():
    """Resolve the chat-LLM provider from the current environment.

    Preference when LLM_PROVIDER=auto: Groq (free) -> OpenAI -> local.
    Returns a 4-tuple: (provider, api_key, base_url, model).
    """
    def clean(value: str) -> str:
        value = (value or "").strip()
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
        return ("openai", openai_key, None,
                os.getenv("LLM_MODEL") or "gpt-4o-mini")
    return ("local", "", None, "local-retrieval-only")


LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = resolve_llm_provider()
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1000

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K_RESULTS = 5
COLLECTION_NAME = "financial_documents"

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = "1.0.0"
