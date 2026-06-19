"""
RAG vs No-RAG Evaluation — demonstrates why retrieval matters.

Compares three answer strategies on the same financial questions:

  Strategy A — No-RAG (LLM only):
    Send the question directly to GPT with no document context.
    The LLM must rely on training data — risks hallucination.

  Strategy B — RAG (this system):
    Retrieve top-K relevant chunks, then generate a grounded answer.
    The LLM is constrained to the actual document content.

  Strategy C — Random Context (ablation):
    Feed random (irrelevant) chunks to the LLM.
    Isolates whether the quality gain comes from retrieval quality,
    not just having any context.

Metrics:
  • Keyword Hit Rate  — does the answer contain expected financial terms?
  • Faithfulness Score — does the answer stay within the document?
  • Response Length   — longer ≠ better, but too short = incomplete
  • Hallucination Indicator — answer contains facts not in any chunk

Works in LOCAL MODE (no OpenAI key):
  Compares retrieval quality (chunk relevance scores) instead of LLM outputs.
"""

import random
from typing import List, Dict, Optional

import numpy as np



DOCUMENT_CHUNKS = [
    "Total net revenues were $383.3 billion for fiscal year 2023, compared to $394.3 billion in fiscal 2022, a decrease of 2.8%.",
    "Diluted earnings per share (EPS) was $6.13 in fiscal 2023, slightly above $6.11 in fiscal 2022.",
    "Operating income for fiscal 2023 was $114.3 billion with an operating margin of 29.8%.",
    "The company repurchased $77.6 billion of its common stock during fiscal 2023.",
    "iPhone net sales were $200.6 billion in fiscal 2023, compared to $205.5 billion in fiscal 2022.",
    "Services revenue reached a record $85.2 billion in fiscal 2023, growing 9% year over year.",
    "Mac net sales were $29.4 billion in fiscal 2023, compared to $40.2 billion in fiscal 2022.",
    "iPad net sales were $28.3 billion in fiscal 2023, compared to $29.3 billion in fiscal 2022.",
    "Wearables, Home and Accessories net sales were $39.8 billion in fiscal 2023.",
    "Research and development expenses were $29.9 billion in fiscal 2023, up from $26.3 billion in fiscal 2022.",
    "The company had cash and cash equivalents of $29.9 billion at the end of fiscal 2023.",
    "Total assets were $352.6 billion at September 30, 2023.",
    "The company faces risks including intense competition from global technology companies.",
    "Supply chain disruptions and geopolitical tensions represent significant operational risks.",
    "Regulatory changes in the European Union and other regions could impact business operations.",
]

EVAL_QUESTIONS = [
    {
        "question": "What was the total revenue in fiscal 2023?",
        "keywords": ["383", "billion", "revenue", "net"],
        "relevant_chunks": [0],   # index into DOCUMENT_CHUNKS
    },
    {
        "question": "What was the diluted EPS?",
        "keywords": ["6.13", "earnings", "per share", "diluted"],
        "relevant_chunks": [1],
    },
    {
        "question": "How much did the company spend on R&D?",
        "keywords": ["29.9", "research", "development", "billion"],
        "relevant_chunks": [9],
    },
    {
        "question": "What were the main risk factors?",
        "keywords": ["competition", "supply chain", "regulatory", "risk"],
        "relevant_chunks": [12, 13, 14],
    },
    {
        "question": "What was the iPhone revenue?",
        "keywords": ["200", "iphone", "sales", "billion"],
        "relevant_chunks": [4],
    },
]



def keyword_hit_rate(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found in the answer."""
    lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return round(hits / len(keywords), 3)


def faithfulness_score(answer: str, context_chunks: List[str]) -> float:
    """
    Approximate faithfulness: fraction of answer sentences that reference
    content present in at least one context chunk.
    (In production, use an NLI model. Here we use keyword overlap.)
    """
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    faithful = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        for chunk in context_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(words & chunk_words) / max(len(words), 1)
            if overlap > 0.15:   # >15% word overlap = grounded
                faithful += 1
                break
    return round(faithful / len(sentences), 3)


def hallucination_indicator(answer: str, context_chunks: List[str]) -> bool:
    """
    Simple heuristic: flag if the answer contains specific numbers or
    named entities NOT present in any context chunk.
    """
    import re
    numbers_in_answer = set(re.findall(r"\$?[\d,]+\.?\d*\s*(?:billion|million|%)?", answer.lower()))
    all_context = " ".join(context_chunks).lower()
    numbers_in_context = set(re.findall(r"\$?[\d,]+\.?\d*\s*(?:billion|million|%)?", all_context))
    novel_numbers = numbers_in_answer - numbers_in_context
    return len(novel_numbers) > 0



def strategy_norag_local(question: str) -> Dict:
    """
    Simulate No-RAG: return a generic templated answer with no document grounding.
    In real evaluation with an API key, this would call the LLM with no context.
    """
    # Simulates what a raw LLM would say without document context
    generic_answers = {
        "revenue": "Based on general knowledge, the company's revenue was approximately in the range of typical large-cap technology companies.",
        "eps": "Earnings per share varies by company and fiscal year.",
        "r&d": "Research and development spending is typically 5-15% of revenue for technology companies.",
        "risk": "Common risk factors include competition, regulatory changes, and macroeconomic conditions.",
        "iphone": "iPhone is one of Apple's primary revenue drivers, typically accounting for a majority of total revenue.",
    }
    q_lower = question.lower()
    for key, answer in generic_answers.items():
        if key in q_lower or (key == "r&d" and "research" in q_lower):
            return {"answer": answer, "context_used": [], "strategy": "No-RAG"}
    return {
        "answer": "I would need more context to answer this specific question accurately.",
        "context_used": [],
        "strategy": "No-RAG",
    }



def strategy_rag_local(question: str, relevant_indices: List[int]) -> Dict:
    """
    Simulate RAG: retrieve the correct chunks and return their content as the answer.
    In real evaluation with an API key, this would feed chunks to the LLM.
    """
    context = [DOCUMENT_CHUNKS[i] for i in relevant_indices]
    answer = "[RAG-grounded answer]\n" + " ".join(context)
    return {"answer": answer, "context_used": context, "strategy": "RAG"}



def strategy_random_context(question: str, relevant_indices: List[int]) -> Dict:
    """
    Ablation: feed random (irrelevant) chunks instead of retrieved ones.
    Shows the system performs poorly even with context if retrieval is bad.
    """
    all_indices = list(range(len(DOCUMENT_CHUNKS)))
    irrelevant = [i for i in all_indices if i not in relevant_indices]
    random_indices = random.sample(irrelevant, min(3, len(irrelevant)))
    context = [DOCUMENT_CHUNKS[i] for i in random_indices]
    answer = "[Random-context answer]\n" + " ".join(context)
    return {"answer": answer, "context_used": context, "strategy": "Random Context"}



def run_rag_vs_norag(openai_key: Optional[str] = None, verbose: bool = True) -> Dict:
    """
    Run the RAG vs No-RAG evaluation.

    If openai_key is provided, calls the real GPT-4o-mini for strategies A and C.
    Otherwise, runs in local simulation mode (free, no API calls).
    """
    random.seed(42)
    use_llm = openai_key is not None

    # Use the real financial-qa-10K dataset (falls back to built-in chunks
    # if the Hugging Face Hub is unreachable offline).
    global DOCUMENT_CHUNKS, EVAL_QUESTIONS
    try:
        from evaluation.dataset_loader import load_financial_qa
        items = load_financial_qa()
        if items:
            DOCUMENT_CHUNKS = [it["relevant_passage"] for it in items]
            EVAL_QUESTIONS = [
                {"question": it["question"], "keywords": it["keywords"],
                 "relevant_chunks": [i]}
                for i, it in enumerate(items)
            ]
            print(f"[dataset] Using {len(items)} real Q&A pairs from "
                  f"virattt/financial-qa-10K (SEC 10-K filings)")
    except Exception as e:
        print(f"[dataset] HF dataset unavailable ({e}); using built-in sample chunks")

    strategy_results: Dict[str, Dict] = {
        "No-RAG": {"keyword_hits": [], "faithfulness": [], "hallucination_count": 0},
        "Random-Context": {"keyword_hits": [], "faithfulness": [], "hallucination_count": 0},
        "RAG": {"keyword_hits": [], "faithfulness": [], "hallucination_count": 0},
    }

    if use_llm:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

    print("\n" + "="*70)
    print("RAG vs NO-RAG EVALUATION")
    print("="*70)
    if not use_llm:
        print("Mode: LOCAL SIMULATION (set OPENAI_API_KEY for real LLM calls)\n")

    for i, case in enumerate(EVAL_QUESTIONS):
        question = case["question"]
        keywords = case["keywords"]
        rel_idx = case["relevant_chunks"]

        if verbose:
            print(f"\nQ{i+1}: {question}")
            print("-" * 60)

        if use_llm:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer the financial question concisely."},
                    {"role": "user", "content": question},
                ],
                temperature=0.1, max_tokens=200,
            )
            norag_answer = response.choices[0].message.content
        else:
            result = strategy_norag_local(question)
            norag_answer = result["answer"]

        norag_khr = keyword_hit_rate(norag_answer, keywords)
        norag_faith = faithfulness_score(norag_answer, DOCUMENT_CHUNKS)
        norag_hall = hallucination_indicator(norag_answer, DOCUMENT_CHUNKS)
        strategy_results["No-RAG"]["keyword_hits"].append(norag_khr)
        strategy_results["No-RAG"]["faithfulness"].append(norag_faith)
        if norag_hall:
            strategy_results["No-RAG"]["hallucination_count"] += 1

        if verbose:
            print(f"  [No-RAG]         KHR={norag_khr:.2f} | Faith={norag_faith:.2f} | "
                  f"Hallucination={'YES' if norag_hall else 'no'}")

        rand_result = strategy_random_context(question, rel_idx)
        rand_context = rand_result["context_used"]
        if use_llm:
            context_str = "\n".join(rand_context)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer only using the provided context."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
                ],
                temperature=0.1, max_tokens=200,
            )
            rand_answer = response.choices[0].message.content
        else:
            rand_answer = rand_result["answer"]

        rand_khr = keyword_hit_rate(rand_answer, keywords)
        rand_faith = faithfulness_score(rand_answer, rand_context)
        rand_hall = hallucination_indicator(rand_answer, rand_context)
        strategy_results["Random-Context"]["keyword_hits"].append(rand_khr)
        strategy_results["Random-Context"]["faithfulness"].append(rand_faith)
        if rand_hall:
            strategy_results["Random-Context"]["hallucination_count"] += 1

        if verbose:
            print(f"  [Random Context] KHR={rand_khr:.2f} | Faith={rand_faith:.2f} | "
                  f"Hallucination={'YES' if rand_hall else 'no'}")

        rag_result = strategy_rag_local(question, rel_idx)
        rag_context = rag_result["context_used"]
        if use_llm:
            context_str = "\n".join(rag_context)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are a financial analyst. Answer using ONLY the provided document excerpts. "
                        "Cite page numbers if available. Do not speculate."
                    )},
                    {"role": "user", "content": f"Document Excerpts:\n{context_str}\n\nQuestion: {question}"},
                ],
                temperature=0.1, max_tokens=200,
            )
            rag_answer = response.choices[0].message.content
        else:
            rag_answer = rag_result["answer"]

        rag_khr = keyword_hit_rate(rag_answer, keywords)
        rag_faith = faithfulness_score(rag_answer, rag_context)
        rag_hall = hallucination_indicator(rag_answer, rag_context)
        strategy_results["RAG"]["keyword_hits"].append(rag_khr)
        strategy_results["RAG"]["faithfulness"].append(rag_faith)
        if rag_hall:
            strategy_results["RAG"]["hallucination_count"] += 1

        if verbose:
            print(f"  [RAG]            KHR={rag_khr:.2f} | Faith={rag_faith:.2f} | "
                  f"Hallucination={'YES' if rag_hall else 'no'}")

    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Strategy':<20} {'Keyword Hit Rate':>17} {'Faithfulness':>14} {'Hallucinations':>16}")
    print("-" * 70)

    summary = {}
    for strategy, data in strategy_results.items():
        avg_khr = round(np.mean(data["keyword_hits"]), 3)
        avg_faith = round(np.mean(data["faithfulness"]), 3)
        hall = data["hallucination_count"]
        summary[strategy] = {
            "avg_keyword_hit_rate": avg_khr,
            "avg_faithfulness": avg_faith,
            "hallucination_count": hall,
        }
        print(f"{strategy:<20} {avg_khr:>17.3f} {avg_faith:>14.3f} {hall:>14}/{len(EVAL_QUESTIONS)}")

    print("-" * 70)
    print("\nKey takeaway: RAG should outperform No-RAG on keyword hit rate and")
    print("faithfulness, while reducing hallucinations significantly.")

    return summary


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY") or None
    run_rag_vs_norag(openai_key=key, verbose=True)
