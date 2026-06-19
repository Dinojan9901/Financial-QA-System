"""
Vector Store — ChromaDB interface for storing and searching embeddings.

ChromaDB runs locally with zero config. Each financial document's chunks
are stored with their embeddings and metadata, enabling cosine-similarity
search at query time.
"""

from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_PATH, COLLECTION_NAME


class VectorStore:
    """Wraps ChromaDB: add chunks, search by embedding, delete by source."""

    def __init__(self, persist_directory: str = CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._get_or_create_collection()



    def add_chunks(self, chunks: List[Dict]) -> None:
        """Upsert embedded chunks into the collection."""
        if not chunks:
            return
        ids = [c["metadata"]["chunk_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[vector_store] Stored {len(chunks)} chunks in ChromaDB")

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return the top-k most similar chunks to the query embedding.
        Each result: {text, metadata, similarity_score}
        """
        where = {"source": source_filter} if source_filter else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": round(1 - results["distances"][0][i], 4),
            })
        return chunks

    def delete_document(self, source_name: str) -> None:
        """Remove all chunks belonging to a specific document."""
        results = self.collection.get(where={"source": source_name})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"[vector_store] Deleted {len(results['ids'])} chunks for '{source_name}'")

    def list_documents(self) -> List[str]:
        """Return unique document names stored in the collection."""
        results = self.collection.get(include=["metadatas"])
        sources = {m["source"] for m in results["metadatas"] if m}
        return sorted(sources)

    def get_document_count(self) -> int:
        return self.collection.count()



    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
