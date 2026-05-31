# Multi-stage Dockerfile for Financial Document Q&A System
# Supports both Streamlit web UI and FastAPI REST API

FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/chroma_db .streamlit

# Streamlit configuration
RUN echo '\
[client]\n\
showErrorDetails = true\n\
[server]\n\
headless = true\n\
port = 8501\n\
enableXsrfProtection = false\n\
' > .streamlit/config.toml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports: 8501 for Streamlit, 8000 for FastAPI
EXPOSE 8501 8000

# Default: Run Streamlit web UI
# To run FastAPI instead: docker run -p 8000:8000 financial-qa uvicorn main:app --host 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
