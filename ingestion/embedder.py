"""
Embedder — converts text chunks into dense vector embeddings.

Supports two modes:
  • OpenAI  (text-embedding-3-small, 1536-dim) — best quality, requires API key
  • Local   (all-MiniLM-L6-v2, 384-dim)        — free, runs on CPU, no API key

Set USE_LOCAL_MODELS=true in .env to use the free local mode.
"""

import os
from typing import List, Dict

import numpy as np

from config import USE_LOCAL_MODELS, EMBEDDING_MODEL, OPENAI_API_KEY


class EmbeddingGenerator:
    """Generates and attaches embeddings to chunk dicts."""

    def __init__(self):
        self.use_local = USE_LOCAL_MODELS
        if self.use_local:
            self._load_local_model()
        else:
            self._load_openai_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """Generate a single embedding vector for a text string."""
        if self.use_local:
            return self.model.encode(text).tolist()
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Attach an 'embedding' key to each chunk dict.
        Uses batch processing for efficiency.
        """
        print(f"[embedder] Generating embeddings for {len(chunks)} chunks...")
        if self.use_local:
            texts = [c["text"] for c in chunks]
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True).tolist()
        else:
            embeddings = self._batch_openai(chunks)

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        print(f"[embedder] Done — {len(embeddings)} embeddings generated.")
        return chunks

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_local_model(self):
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading local model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def _load_openai_client(self):
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Either add it to .env or set USE_LOCAL_MODELS=true"
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _batch_openai(self, chunks: List[Dict]) -> List[List[float]]:
        """OpenAI supports up to 2048 inputs per call — batch accordingly."""
        batch_size = 100
        texts = [c["text"] for c in chunks]
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)
            print(f"  [{min(i + batch_size, len(texts))}/{len(texts)}] embedded")
        return all_embeddings
