"""
QA Chain — end-to-end: retrieved chunks → grounded LLM answer.

Generation runs through any OpenAI-compatible provider:
  • Groq   (free, default when GROQ_API_KEY is set) — e.g. Llama 3.3 70B
  • OpenAI (GPT-4o-mini) when OPENAI_API_KEY is set
  • Local  retrieval-only fallback when no key is configured
Temperature=0.1 keeps answers factual, not creative.
"""

import re
from typing import List, Dict

from generation.prompt_builder import build_qa_prompt
from config import resolve_llm_provider, LLM_TEMPERATURE, LLM_MAX_TOKENS, EMBEDDING_MODEL


class FinancialQAChain:
    """
    Takes a question + retrieved context chunks and returns a grounded answer
    from the configured LLM (Groq or OpenAI), plus citations and token usage.
    """

    def __init__(self, model: str = None):
        from openai import OpenAI
        provider, api_key, base_url, default_model = resolve_llm_provider()
        if not api_key:
            raise EnvironmentError(
                "No LLM API key configured. Set GROQ_API_KEY (free) or "
                "OPENAI_API_KEY in .env, or use local retrieval-only mode."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider = provider
        self.model = model or default_model

    def answer(
        self,
        question: str,
        context_chunks: List[Dict],
        temperature: float = LLM_TEMPERATURE,
    ) -> Dict:
        """
        Generate an answer grounded in context_chunks.

        Returns:
            {answer, question, model, sources, tokens_used}
        """
        messages = build_qa_prompt(question, context_chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=LLM_MAX_TOKENS,
        )
        answer_text = response.choices[0].message.content
        sources = [
            {
                "page": c["metadata"].get("page_number"),
                "source": c["metadata"].get("source"),
                "relevance": round(c.get("similarity_score", 0), 3),
            }
            for c in context_chunks
        ]
        return {
            "answer": answer_text,
            "question": question,
            "model": self.model,
            "sources": sources,
            "tokens_used": response.usage.total_tokens,
            "chunks_retrieved": len(context_chunks),
        }


# ── Extractive fallback helpers (used when no LLM key is configured) ──────────

_extractive_model = None  # SentenceTransformer singleton, loaded once per process


def _get_extractive_model():
    global _extractive_model
    if _extractive_model is None:
        from sentence_transformers import SentenceTransformer
        _extractive_model = SentenceTransformer(EMBEDDING_MODEL)
    return _extractive_model


def _split_sentences(text: str) -> List[str]:
    """Split a chunk into sentences, keeping only substantial ones."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _best_sentence(question: str, context_chunks: List[Dict]):
    """Return (sentence, score, metadata) for the chunk sentence most similar
    to the question, using cosine similarity on MiniLM embeddings."""
    import numpy as np

    model = _get_extractive_model()
    q_vec = np.asarray(model.encode(question))
    best = ("", -1.0, context_chunks[0]["metadata"])

    for chunk in context_chunks:
        sentences = _split_sentences(chunk["text"])
        if not sentences:
            continue
        s_vecs = model.encode(sentences)
        for sentence, s_vec in zip(sentences, s_vecs):
            s_vec = np.asarray(s_vec)
            denom = np.linalg.norm(q_vec) * np.linalg.norm(s_vec)
            score = float(np.dot(q_vec, s_vec) / denom) if denom else 0.0
            if score > best[1]:
                best = (sentence, score, chunk["metadata"])
    return best


class LocalQAChain:
    """
    Free fallback used when no LLM key is configured.

    Instead of dumping a whole chunk, it returns the single most relevant
    *sentence* from the retrieved context (extractive answering via MiniLM
    sentence similarity) — concise and still fully grounded with a citation.
    """

    def answer(self, question: str, context_chunks: List[Dict], **kwargs) -> Dict:
        if not context_chunks:
            return {
                "answer": "No relevant context found in the document.",
                "question": question,
                "model": "local-extractive",
                "sources": [],
                "tokens_used": 0,
                "chunks_retrieved": 0,
            }

        sentence, score, meta = _best_sentence(question, context_chunks)
        page = meta.get("page_number", "?")
        answer = (
            f"{sentence}\n\n"
            f"_(Extractive answer — Page {page}, relevance {score:.2f}. "
            f"Add a free GROQ_API_KEY for a full generated answer.)_"
        )
        return {
            "answer": answer,
            "question": question,
            "model": "local-extractive (MiniLM)",
            "sources": [
                {
                    "page": c["metadata"].get("page_number"),
                    "source": c["metadata"].get("source"),
                    "relevance": round(c.get("similarity_score", 0), 3),
                }
                for c in context_chunks
            ],
            "tokens_used": 0,
            "chunks_retrieved": len(context_chunks),
        }


def get_qa_chain():
    """Factory: pick the real LLM chain if a provider key is configured
    (evaluated live), else the free local retrieval-only fallback."""
    provider, api_key, _, model = resolve_llm_provider()
    if provider == "local" or not api_key:
        print("[qa_chain] Using LocalQAChain (no LLM key — retrieval-only mode)")
        return LocalQAChain()
    print(f"[qa_chain] Using {provider} LLM for generation: {model}")
    return FinancialQAChain()
