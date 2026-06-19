# Dockerfile for Financial Document Q&A System (Streamlit web UI)
# Railway-ready: binds to the platform-provided $PORT at runtime.

FROM python:3.13-slim

WORKDIR /app

# System build deps (needed by some scientific wheels) + curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Runtime data dirs (ephemeral on Railway — fine for demo/session use)
RUN mkdir -p data/uploads data/chroma_db

# Hugging Face cache dir (model weights download here on first query)
ENV HF_HOME=/app/.cache/huggingface
ENV USE_LOCAL_MODELS=true

# Streamlit listens on the platform port. ${PORT:-8501} = Railway's $PORT in
# production, or 8501 for a local `docker run`. Shell form lets $PORT expand.
EXPOSE 8501
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
