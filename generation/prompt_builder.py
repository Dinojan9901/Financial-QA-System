"""
Prompt Builder — constructs the LLM messages for financial Q&A.

Key design choices:
  • System prompt forces the model to stay grounded in retrieved excerpts
  • Chain-of-thought instruction ("think step by step") improves accuracy
  • Page citations are required so users can verify answers
  • "I don't know" is preferred over hallucination
"""

from typing import List, Dict

# ── System prompt (Prompt Engineering technique) ──────────────────────────────
FINANCIAL_QA_SYSTEM_PROMPT = """You are an expert financial analyst assistant.
You answer questions about financial documents (10-K reports, earnings calls, SEC filings)
accurately, concisely, and with clear source citations.

CRITICAL RULES:
1. ONLY use information from the provided document excerpts below.
2. If the answer is not found in the excerpts, respond exactly:
   "This information is not found in the provided document sections."
3. Always cite the page number when referencing specific data (e.g., "[Page 12]").
4. For numerical data (revenue, EPS, margins), quote the EXACT figures from the document.
5. Do NOT speculate or add information from your general knowledge.
6. Use bullet points for multi-part answers to improve clarity.
7. Flag any uncertainty explicitly with "Note: ...".
8. Think step by step before answering to ensure accuracy."""

# ── Few-shot examples (in-context learning) ───────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "DOCUMENT EXCERPTS:\n"
            "[Excerpt 1 | Page 23 | Relevance: 0.94]\n"
            "Total net revenues for fiscal 2023 were $383.3 billion, "
            "compared to $394.3 billion in fiscal 2022.\n\n"
            "QUESTION: What was the total revenue in fiscal 2023?\n\nANSWER:"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Total net revenues for fiscal 2023 were **$383.3 billion** [Page 23], "
            "representing a decrease from $394.3 billion in fiscal 2022."
        ),
    },
]


def build_qa_prompt(question: str, context_chunks: List[Dict]) -> List[Dict]:
    """
    Build the full message list for the LLM.

    Args:
        question:       The user's natural language question.
        context_chunks: Retrieved chunks from the vector store.

    Returns:
        List of chat-style message dictionaries (system / few-shot / user).
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        page = chunk["metadata"].get("page_number", "Unknown")
        score = chunk.get("similarity_score", 0)
        context_parts.append(
            f"[Excerpt {i} | Page {page} | Relevance: {score:.2f}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Please answer the following question using ONLY the document excerpts below.\n\n"
        f"DOCUMENT EXCERPTS:\n{context_text}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    messages = [
        {"role": "system", "content": FINANCIAL_QA_SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_message},
    ]
    return messages
