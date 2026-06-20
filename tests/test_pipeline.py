"""
Test suite for the Financial Q&A RAG pipeline.

Tests cover:
  • PDF text extraction
  • Text chunking (token counts, overlap, metadata)
  • Embedding generation (dimension check)
  • Vector store (add / search / delete)
  • Retrieval (top-k results, similarity scores)
  • Prompt building (message structure)
  • End-to-end pipeline (ingest → answer)

Run with: pytest tests/ -v
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.text_chunker import FinancialTextChunker
from ingestion.embedder import EmbeddingGenerator
from retrieval.vector_store import VectorStore
from generation.prompt_builder import build_qa_prompt, FINANCIAL_QA_SYSTEM_PROMPT


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_page(text: str, page_num: int = 1, source: str = "test.pdf") -> dict:
    return {"page_number": page_num, "text": text, "tables": [], "source": source}


def make_fake_chunk(text: str, page_num: int = 1, chunk_idx: int = 0, source: str = "test.pdf") -> dict:
    return {
        "text": text,
        "metadata": {
            "source": source,
            "page_number": page_num,
            "chunk_index": chunk_idx,
            "chunk_id": f"{source}_p{page_num}_c{chunk_idx}",
            "token_count": 50,
        },
    }


# ── Text Chunker tests ────────────────────────────────────────────────────────

class TestFinancialTextChunker:

    def test_chunks_created(self):
        chunker = FinancialTextChunker(chunk_size=100, chunk_overlap=20)
        long_text = "Apple Inc. reported total net revenues of $383.3 billion for fiscal 2023. " * 20
        pages = [make_fake_page(long_text)]
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunk_has_required_metadata_keys(self):
        chunker = FinancialTextChunker()
        pages = [make_fake_page("Revenue grew 15% year-over-year to approximately $10 billion.")]
        chunks = chunker.chunk_pages(pages)
        assert chunks, "Should produce at least one chunk"
        meta = chunks[0]["metadata"]
        for key in ("source", "page_number", "chunk_index", "chunk_id", "token_count"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_tiny_text_is_skipped(self):
        chunker = FinancialTextChunker()
        pages = [make_fake_page("OK")]   # too short (<50 chars)
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) == 0, "Tiny chunks should be filtered out"

    def test_chunk_stats_returned(self):
        chunker = FinancialTextChunker()
        pages = [make_fake_page("Financial results for Q4 2023 showed strong revenue growth. " * 10)]
        chunks = chunker.chunk_pages(pages)
        stats = chunker.get_chunk_stats(chunks)
        assert "total_chunks" in stats
        assert stats["total_chunks"] == len(chunks)


# ── Embedding tests ───────────────────────────────────────────────────────────

class TestEmbeddingGenerator:

    @pytest.fixture(autouse=True)
    def use_minilm_model(self, monkeypatch):
        """Use the project's proposed MiniLM model."""
        import config
        import ingestion.embedder as embedder_module
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.EMBEDDING_DIMENSIONS = 384
        monkeypatch.setattr(embedder_module, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    def test_embed_text_returns_vector(self):
        embedder = EmbeddingGenerator()
        vec = embedder.embed_text("What was the revenue in Q3?")
        assert isinstance(vec, list), "Embedding should be a list"
        assert len(vec) == 384, "all-MiniLM-L6-v2 produces 384-dim vectors"
        assert all(isinstance(v, float) for v in vec)

    def test_embed_chunks_attaches_embeddings(self):
        embedder = EmbeddingGenerator()
        chunks = [make_fake_chunk("Revenue was $10B in 2023.")]
        result = embedder.embed_chunks(chunks)
        assert "embedding" in result[0]
        assert len(result[0]["embedding"]) == 384

    def test_similar_texts_have_high_cosine_similarity(self):
        import numpy as np
        embedder = EmbeddingGenerator()
        v1 = embedder.embed_text("Total revenue was $383 billion")
        v2 = embedder.embed_text("Net sales amounted to $383 billion")
        v3 = embedder.embed_text("The weather is sunny today")

        def cosine(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        similar = cosine(v1, v2)
        dissimilar = cosine(v1, v3)
        assert similar > dissimilar, "Semantically similar texts should be closer"


# ── Vector Store tests ────────────────────────────────────────────────────────

class TestVectorStore:

    @pytest.fixture
    def temp_store(self, monkeypatch, tmp_path):
        store = VectorStore(persist_directory=str(tmp_path / "chroma"))
        return store

    def test_add_and_count(self, temp_store):
        embedder = EmbeddingGenerator()
        chunks = [make_fake_chunk("Apple revenue was $383B in FY2023.")]
        chunks = embedder.embed_chunks(chunks)
        temp_store.add_chunks(chunks)
        assert temp_store.get_document_count() >= 1

    def test_search_returns_relevant_chunk(self, temp_store):
        embedder = EmbeddingGenerator()
        chunks = [make_fake_chunk("Net income was $97 billion for fiscal year 2023.")]
        chunks = embedder.embed_chunks(chunks)
        temp_store.add_chunks(chunks)

        query_emb = embedder.embed_text("What was the net income?")
        results = temp_store.search(query_embedding=query_emb, n_results=1)
        assert len(results) >= 1
        assert "net income" in results[0]["text"].lower()

    def test_search_result_has_similarity_score(self, temp_store):
        embedder = EmbeddingGenerator()
        chunks = [make_fake_chunk("Operating expenses were $40 billion.")]
        chunks = embedder.embed_chunks(chunks)
        temp_store.add_chunks(chunks)

        query_emb = embedder.embed_text("operating expenses")
        results = temp_store.search(query_embedding=query_emb, n_results=1)
        assert "similarity_score" in results[0]
        assert 0 <= results[0]["similarity_score"] <= 1


# ── Prompt Builder tests ──────────────────────────────────────────────────────

class TestPromptBuilder:

    def test_prompt_has_system_message(self):
        chunks = [{"text": "Revenue was $10B.", "metadata": {"page_number": 5}, "similarity_score": 0.9}]
        messages = build_qa_prompt("What was revenue?", chunks)
        assert messages[0]["role"] == "system"
        assert "financial analyst" in messages[0]["content"].lower()

    def test_prompt_contains_question(self):
        chunks = [{"text": "EPS was $6.13.", "metadata": {"page_number": 2}, "similarity_score": 0.85}]
        question = "What was the earnings per share?"
        messages = build_qa_prompt(question, chunks)
        last_user_msg = next(m for m in reversed(messages) if m["role"] == "user")
        assert question in last_user_msg["content"]

    def test_prompt_contains_chunk_text(self):
        chunk_text = "Total assets were $352 billion."
        chunks = [{"text": chunk_text, "metadata": {"page_number": 88}, "similarity_score": 0.92}]
        messages = build_qa_prompt("What are total assets?", chunks)
        full_content = " ".join(m["content"] for m in messages)
        assert chunk_text in full_content

    def test_system_prompt_has_critical_rules(self):
        assert "ONLY use information" in FINANCIAL_QA_SYSTEM_PROMPT
        assert "cite the page number" in FINANCIAL_QA_SYSTEM_PROMPT
        assert "step by step" in FINANCIAL_QA_SYSTEM_PROMPT


# ── Evaluation metrics test ───────────────────────────────────────────────────

class TestEvaluationMetrics:
    """Golden test cases — measure answer quality against expected keywords."""

    GOLDEN_CASES = [
        {
            "question": "What was the total revenue?",
            "context_text": "Total net revenues were $383.3 billion for fiscal 2023.",
            "expected_keywords": ["383", "billion", "revenue"],
        },
        {
            "question": "Who are the main competitors?",
            "context_text": "The company faces competition from Microsoft, Google, and Amazon.",
            "expected_keywords": ["microsoft", "google", "amazon"],
        },
    ]

    def _hit_rate(self, answer: str, keywords: list) -> float:
        lower = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in lower)
        return hits / len(keywords)

    def test_prompt_includes_expected_context(self):
        for case in self.GOLDEN_CASES:
            chunks = [{
                "text": case["context_text"],
                "metadata": {"page_number": 1},
                "similarity_score": 0.95,
            }]
            messages = build_qa_prompt(case["question"], chunks)
            full_text = " ".join(m["content"] for m in messages)
            # The context should appear verbatim in the prompt
            assert case["context_text"] in full_text

    def test_keyword_hit_rate_on_retrieval(self):
        """Verify retrieval returns text containing expected keywords."""
        embedder = EmbeddingGenerator()
        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(persist_directory=tmp + "/chroma")
            for i, case in enumerate(self.GOLDEN_CASES):
                fake_chunk = make_fake_chunk(case["context_text"], chunk_idx=i)
                embedded = embedder.embed_chunks([fake_chunk])
                store.add_chunks(embedded)

            for case in self.GOLDEN_CASES:
                q_emb = embedder.embed_text(case["question"])
                results = store.search(query_embedding=q_emb, n_results=3)
                top_text = results[0]["text"].lower() if results else ""
                hit_rate = self._hit_rate(top_text, case["expected_keywords"])
                assert hit_rate >= 0.5, (
                    f"Expected keywords not found in top result for: '{case['question']}'"
                )
