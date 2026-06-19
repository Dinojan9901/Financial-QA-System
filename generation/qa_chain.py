"""
QA Chain — end-to-end: retrieved chunks → grounded LLM answer.

Generation runs through any OpenAI-compatible provider:
  • Groq   (free, default when GROQ_API_KEY is set) — e.g. Llama 3.3 70B
  • OpenAI (GPT-4o-mini) when OPENAI_API_KEY is set
  • Local  retrieval-only fallback when no key is configured
Temperature=0.1 keeps answers factual, not creative.
"""

import os
from typing import List, Dict

from generation.prompt_builder import build_qa_prompt
from config import resolve_llm_provider, LLM_TEMPERATURE, LLM_MAX_TOKENS


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
        # Both Groq and OpenAI use the OpenAI SDK; only base_url/model differ.
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
    """Factory: pick the real LLM chain if a provider key is configured
    (evaluated live), else the free local retrieval-only fallback."""
    provider, api_key, _, model = resolve_llm_provider()
    if provider == "local" or not api_key:
        print("[qa_chain] Using LocalQAChain (no LLM key — retrieval-only mode)")
        return LocalQAChain()
    print(f"[qa_chain] Using {provider} LLM for generation: {model}")
    return FinancialQAChain()
