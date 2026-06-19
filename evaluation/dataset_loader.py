"""
dataset_loader.py — Real financial Q&A dataset for evaluation.
================================================================
Loads the **virattt/financial-qa-10K** dataset from the Hugging Face Hub
(https://huggingface.co/datasets/virattt/financial-qa-10K).

The dataset contains 7,000 question/answer pairs derived from real SEC 10-K
annual-report filings (NVIDIA, etc.). Each record has:
    question : a natural-language financial question
    answer   : the ground-truth answer
    context  : the exact passage from the 10-K supporting the answer
    ticker   : company stock ticker (e.g. NVDA)
    filing   : source filing (e.g. 2023_10K)

We sample a fixed, reproducible subset and cache it to a local JSON so the
evaluation is deterministic and runs offline after the first download.

This replaces the previous hand-written sample passages with a genuine,
publicly available dataset (satisfies the project's dataset requirement).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

HF_DATASET_NAME = "virattt/financial-qa-10K"
DEFAULT_SAMPLE_SIZE = 50
RANDOM_SEED = 42

_CACHE_DIR = Path(__file__).parent / "results"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Common English + finance stop words to strip when deriving answer keywords.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "by", "with", "as", "at", "from", "that",
    "this", "these", "those", "it", "its", "their", "his", "her", "they",
    "which", "what", "who", "whom", "how", "much", "many", "did", "does",
    "do", "has", "have", "had", "will", "would", "can", "could", "company",
    "companys", "year", "fiscal", "during", "about", "into", "than", "such",
    "some", "other", "also", "including", "include", "includes", "mentioned",
}


def _cache_path(sample_size: int) -> Path:
    return _CACHE_DIR / f"financial_qa_10k_sample_{sample_size}.json"


def _derive_keywords(answer: str, max_keywords: int = 5) -> List[str]:
    """Extract the most informative tokens from a gold answer.

    Keeps numbers (e.g. '383.3', '1999') and content words >2 chars, drops
    stop words. Used to score Keyword Hit Rate against generated answers.
    """
    tokens = re.findall(r"[A-Za-z0-9.%$]+", answer.lower())
    keywords: List[str] = []
    for tok in tokens:
        clean = tok.strip(".%$")
        has_digit = any(c.isdigit() for c in clean)
        if not clean or clean in _STOPWORDS:
            continue
        if len(clean) <= 2 and not has_digit:
            continue
        if clean not in keywords:
            keywords.append(clean)
    return keywords[:max_keywords]


def load_financial_qa(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    use_cache: bool = True,
) -> List[Dict]:
    """Return a reproducible sample of the financial-qa-10K dataset.

    Each item: {question, keywords, relevant_passage, answer, ticker, filing}.
    Falls back to the cached JSON if the Hugging Face Hub is unreachable.

    The list is structurally compatible with the evaluation harness, where
    each item's `relevant_passage` is the one correct context to retrieve and
    every other item's passage acts as a distractor in the shared corpus.
    """
    cache_file = _cache_path(sample_size)

    # 1. Use cached sample if present (fast, offline, deterministic).
    if use_cache and cache_file.exists():
        with open(cache_file, encoding="utf-8") as fh:
            return json.load(fh)

    # 2. Otherwise download from the Hugging Face Hub.
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_NAME, split="train")
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(sample_size))

    items: List[Dict] = []
    for row in ds:
        passage = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not (passage and question and answer):
            continue
        items.append({
            "question": question,
            "answer": answer,
            "keywords": _derive_keywords(answer),
            "relevant_passage": passage,
            "ticker": row.get("ticker", ""),
            "filing": row.get("filing", ""),
        })

    # 3. Cache for reproducible, offline re-runs.
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)

    return items


def dataset_info() -> Dict:
    """Lightweight provenance block for reports / logging."""
    return {
        "name": HF_DATASET_NAME,
        "source": "Hugging Face Hub",
        "url": f"https://huggingface.co/datasets/{HF_DATASET_NAME}",
        "description": (
            "7,000 question-answer pairs derived from real SEC 10-K annual "
            "report filings, each with the supporting source passage."
        ),
        "license": "Publicly available (Hugging Face Datasets)",
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "seed": RANDOM_SEED,
    }


if __name__ == "__main__":
    data = load_financial_qa()
    print(f"Loaded {len(data)} real financial Q&A items "
          f"from {HF_DATASET_NAME}\n")
    for item in data[:3]:
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        print(f"Keywords: {item['keywords']}")
        print(f"Source: {item['ticker']} {item['filing']}")
        print("-" * 60)
