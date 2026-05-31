"""
Retriever — combines the embedder and vector store to answer
"what chunks are relevant to this question?"
"""

from typing import List, Dict, Optional

from ingestion.embedder import EmbeddingGenerator
from retrieval.vector_store import VectorStore
from config import TOP_K_RESULTS


class FinancialRetriever:
    """
    High-level retrieval interface:
      1. Embed the user question
      2. Similarity-search the vector store
      3. Return ranked relevant chunks
    """

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.store = VectorStore()

    def retrieve(
        self,
        question: str,
        n_results: int = TOP_K_RESULTS,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Return the top-k relevant chunks for a natural language question."""
        question_embedding = self.embedder.embed_text(question)
        chunks = self.store.search(
            query_embedding=question_embedding,
            n_results=n_results,
            source_filter=source_filter,
        )
        return chunks

    def retrieve_with_scores(self, question: str, n_results: int = TOP_K_RESULTS) -> List[Dict]:
        """Same as retrieve() but prints similarity scores for debugging."""
        chunks = self.retrieve(question, n_results)
        print(f"\n[retriever] Top {len(chunks)} chunks for: '{question}'")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. Score={c['similarity_score']:.3f} | "
                  f"Page={c['metadata'].get('page_number')} | "
                  f"{c['text'][:80]}...")
        return chunks
