"""Generate free, local MiniLM embeddings for document chunks and queries."""

from typing import Dict, List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingGenerator:
    """Generate embeddings with the project's proposed MiniLM model."""

    def __init__(self):
        print(f"[embedder] Loading local model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_text(self, text: str) -> List[float]:
        """Return one 384-dimensional embedding vector."""
        return self.model.encode(text).tolist()

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings in batches and attach them to chunks."""
        print(f"[embedder] Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
        ).tolist()

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        print(f"[embedder] Done - {len(embeddings)} embeddings generated.")
        return chunks
