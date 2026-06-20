"""Format retrieved document passages as grounded answers.

Uses HuggingFace Inference API when HF_TOKEN is set in .env.
Falls back to extractive sentence matching when it is not.
"""

import re
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, HF_TOKEN, HF_MODEL

# Module-level singleton — loaded once per process, reused across calls.
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _extractive_answer(question: str, context_chunks: List[Dict]):
    """Return (best_sentence, score, metadata) using sentence-level similarity."""
    model = _get_model()
    q_vec = model.encode(question).tolist()

    best_sentence = ""
    best_score = -1.0
    best_meta = context_chunks[0]["metadata"]

    for chunk in context_chunks:
        sentences = _split_sentences(chunk["text"])
        if not sentences:
            continue
        s_vecs = model.encode(sentences).tolist()
        for sentence, s_vec in zip(sentences, s_vecs):
            score = _cosine_similarity(q_vec, s_vec)
            if score > best_score:
                best_score = score
                best_sentence = sentence
                best_meta = chunk["metadata"]

    return best_sentence, best_score, best_meta


def _hf_llm_answer(question: str, context_chunks: List[Dict]) -> str:
    """Call HuggingFace Inference API and return the generated answer."""
    from huggingface_hub import InferenceClient
    from generation.prompt_builder import build_qa_prompt

    client   = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    messages = build_qa_prompt(question=question, context_chunks=context_chunks)

    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


class LocalQAChain:
    """
    Answer questions from retrieved chunks.
    - If HF_TOKEN is set in .env  → calls HuggingFace LLM for a synthesised answer.
    - If HF_TOKEN is missing       → falls back to extractive sentence matching.
    """

    def answer(self, question: str, context_chunks: List[Dict], **kwargs) -> Dict:
        if not context_chunks:
            return {
                "answer": "No relevant context found in the document.",
                "question": question,
                "model": "none",
                "sources": [],
                "tokens_used": 0,
                "chunks_retrieved": 0,
            }

        sources = [
            {
                "page": chunk["metadata"].get("page_number"),
                "source": chunk["metadata"].get("source"),
                "relevance": round(chunk.get("similarity_score", 0), 3),
            }
            for chunk in context_chunks
        ]

        # ── Path 1: HuggingFace LLM ───────────────────────────────────────────
        if HF_TOKEN:
            try:
                answer_text = _hf_llm_answer(question, context_chunks)
                return {
                    "answer": answer_text,
                    "question": question,
                    "model": HF_MODEL,
                    "sources": sources,
                    "tokens_used": 0,
                    "chunks_retrieved": len(context_chunks),
                }
            except Exception as e:
                print(f"[qa_chain] HF API failed ({e}) — falling back to extractive.")

        # ── Path 2: Extractive fallback ───────────────────────────────────────
        best_sentence, best_score, best_meta = _extractive_answer(question, context_chunks)
        page = best_meta.get("page_number", "?")
        answer_text = (
            f"Answer (Page {page}, relevance={best_score:.2f}):\n\n"
            f"{best_sentence}"
        )
        return {
            "answer": answer_text,
            "question": question,
            "model": "all-MiniLM-L6-v2-extractive",
            "sources": sources,
            "tokens_used": 0,
            "chunks_retrieved": len(context_chunks),
        }


def get_qa_chain() -> LocalQAChain:
    return LocalQAChain()
