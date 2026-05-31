"""
QA Chain — end-to-end: retrieved chunks → grounded LLM answer.

Uses GPT-4o-mini (cheap, fast, great for extraction tasks).
Temperature=0.1 keeps answers factual, not creative.
"""

import os
from typing import List, Dict

from generation.prompt_builder import build_qa_prompt
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, OPENAI_API_KEY


class FinancialQAChain:
    """
    Takes a question + retrieved context chunks and returns a grounded answer
    from the LLM, plus source citations and token usage stats.
    """

    def __init__(self, model: str = LLM_MODEL):
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is required for answer generation. "
                "Set it in .env or use the local-model demo mode."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

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


class LocalQAChain:
    """
    Fallback QA chain that works without an OpenAI key.
    Returns the most relevant chunk text as the "answer" — useful for testing
    the retrieval pipeline without incurring API costs.
    """

    def answer(self, question: str, context_chunks: List[Dict], **kwargs) -> Dict:
        if not context_chunks:
            return {
                "answer": "No relevant context found in the document.",
                "question": question,
                "model": "local-retrieval-only",
                "sources": [],
                "tokens_used": 0,
                "chunks_retrieved": 0,
            }
        top_chunk = context_chunks[0]
        answer = (
            f"[Local mode — no LLM generation]\n\n"
            f"Most relevant excerpt (Page {top_chunk['metadata'].get('page_number')}, "
            f"relevance={top_chunk.get('similarity_score', 0):.2f}):\n\n"
            f"{top_chunk['text']}"
        )
        return {
            "answer": answer,
            "question": question,
            "model": "local-retrieval-only",
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
    """Factory: returns the right chain based on config."""
    from config import USE_LOCAL_MODELS
    if USE_LOCAL_MODELS or not OPENAI_API_KEY:
        print("[qa_chain] Using LocalQAChain (no OpenAI key / local mode)")
        return LocalQAChain()
    return FinancialQAChain()
