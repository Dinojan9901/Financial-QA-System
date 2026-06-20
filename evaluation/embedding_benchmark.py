"""
Embedding Model Benchmark — compares all embedding strategies side-by-side.

Models benchmarked:
  1. Word2Vec (skip-gram, trained on corpus)      — classic NLP baseline
  2. TF-IDF (sparse bag-of-words)                 — traditional IR baseline
  3. all-MiniLM-L6-v2 (sentence-transformers)     — our free local model

This module provides:
  • Semantic similarity tests (do similar sentences get close vectors?)
  • Retrieval quality benchmarks (Precision@K, MRR, NDCG)
  • Speed benchmarks (embedding latency)
  • Visualization-ready data (cosine similarity matrices)

The results table is designed to be directly included in the project report
under "Experimental Setup & Results" and "Evaluation & Performance Metrics".
"""

import time
import warnings
from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# ── Test data ─────────────────────────────────────────────────────────────────

# Sentence pairs that SHOULD be semantically similar (expected high cosine sim)
SIMILAR_PAIRS = [
    ("Revenue increased by 15% to $383 billion.", "Net sales grew 15 percent to 383 billion dollars."),
    ("Earnings per share was $6.13.", "EPS for the year came in at six dollars and thirteen cents."),
    ("The company faces regulatory risk.", "Regulatory changes pose a challenge to the business."),
    ("Operating income rose to $114 billion.", "Operating profit increased to one hundred fourteen billion."),
    ("iPhone sales declined year over year.", "iPhone revenue fell compared to the previous fiscal year."),
]

# Sentence pairs that are DISSIMILAR (expected low cosine sim)
DISSIMILAR_PAIRS = [
    ("Revenue increased by 15% to $383 billion.", "The board approved the meeting date for shareholders."),
    ("Earnings per share was $6.13.", "The company was founded in California in 1976."),
    ("Operating income rose to $114 billion.", "Winter storms disrupted logistics in the midwest."),
]

# Retrieval corpus: one correct passage per query + distractors
RETRIEVAL_CORPUS = [
    # Index 0-4: relevant passages
    "Total net revenues were $383.3 billion for fiscal year 2023, a 2.8% decrease.",
    "Diluted EPS for fiscal 2023 was $6.13 compared to $6.11 in fiscal 2022.",
    "R&D expenses increased to $29.9 billion in fiscal 2023 from $26.3 billion.",
    "iPhone net sales were $200.6 billion, Services revenue reached $85.2 billion.",
    "Key risks include competition, supply chain disruption, and regulatory exposure.",
    # Index 5-9: distractors
    "The annual shareholders meeting was scheduled for February 2024.",
    "Board members received compensation packages in line with market standards.",
    "The company was incorporated under the laws of California.",
    "Stock options were granted under the 2022 Employee Stock Plan.",
    "Certain leases are classified as operating leases under ASC 842.",
]

RETRIEVAL_QUERIES = [
    {"question": "What was the total net revenue?",       "correct_idx": 0, "keywords": ["383", "revenue"]},
    {"question": "What was the diluted earnings per share?", "correct_idx": 1, "keywords": ["6.13", "eps"]},
    {"question": "How much was spent on research and development?", "correct_idx": 2, "keywords": ["29.9", "r&d"]},
    {"question": "What are iPhone and Services revenues?", "correct_idx": 3, "keywords": ["iphone", "services"]},
    {"question": "What are the main risk factors?",        "correct_idx": 4, "keywords": ["risk", "competition"]},
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def precision_at_k(ranked_indices: List[int], correct_idx: int, k: int) -> float:
    return 1.0 if correct_idx in ranked_indices[:k] else 0.0


def reciprocal_rank(ranked_indices: List[int], correct_idx: int) -> float:
    try:
        rank = ranked_indices.index(correct_idx) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(ranked_indices: List[int], correct_idx: int, k: int) -> float:
    """NDCG@K — discounted cumulative gain for binary relevance."""
    import math
    dcg = 0.0
    for i, idx in enumerate(ranked_indices[:k]):
        if idx == correct_idx:
            dcg = 1.0 / math.log2(i + 2)
            break
    ideal_dcg = 1.0  # best possible: correct doc at rank 1
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def rank_documents(
    query_vec: np.ndarray, doc_vecs: np.ndarray
) -> List[int]:
    sims = cosine_similarity([query_vec], doc_vecs)[0]
    return list(np.argsort(sims)[::-1])


# ── Model wrappers ────────────────────────────────────────────────────────────

def _load_word2vec(corpus: List[str]):
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess

    tokenized = [simple_preprocess(d) for d in corpus]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5,
                     min_count=1, workers=4, epochs=15, sg=1)

    def embed(text: str) -> np.ndarray:
        tokens = simple_preprocess(text)
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(100)

    return embed, "Word2Vec (sg, d=100)"


def _load_tfidf(corpus: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    vectorizer.fit(corpus)

    def embed(text: str) -> np.ndarray:
        return vectorizer.transform([text]).toarray()[0]

    return embed, "TF-IDF (n-gram 1-2)"


def _load_minilm():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(text: str) -> np.ndarray:
        return model.encode(text)

    return embed, "MiniLM-L6-v2 (384-dim)"


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_embedding_benchmark(verbose: bool = True) -> Dict:
    """
    Benchmark all embedding models on semantic similarity + retrieval tasks.
    Returns a results dict for report inclusion.
    """
    # Build training corpus for Word2Vec + TF-IDF
    all_text = RETRIEVAL_CORPUS + [p for pair in SIMILAR_PAIRS + DISSIMILAR_PAIRS for p in pair]

    # Load models
    models = []
    print("\n[embedding_benchmark] Loading models...")
    w2v_embed, w2v_name = _load_word2vec(all_text)
    tfidf_embed, tfidf_name = _load_tfidf(all_text)
    minilm_embed, minilm_name = _load_minilm()
    models = [(w2v_embed, w2v_name), (tfidf_embed, tfidf_name), (minilm_embed, minilm_name)]

    results = {}

    for embed_fn, model_name in models:
        print(f"\n{'='*65}")
        print(f"MODEL: {model_name}")
        print("="*65)

        # ── Semantic similarity test ─────────────────────────────────────────
        sim_scores, dissim_scores = [], []
        for a, b in SIMILAR_PAIRS:
            va, vb = embed_fn(a), embed_fn(b)
            s = float(cosine_similarity([va], [vb])[0][0])
            sim_scores.append(s)
        for a, b in DISSIMILAR_PAIRS:
            va, vb = embed_fn(a), embed_fn(b)
            s = float(cosine_similarity([va], [vb])[0][0])
            dissim_scores.append(s)

        avg_sim = round(np.mean(sim_scores), 3)
        avg_dissim = round(np.mean(dissim_scores), 3)
        separability = round(avg_sim - avg_dissim, 3)  # higher = better discrimination

        if verbose:
            print(f"  Semantic similarity  (similar pairs):    {avg_sim:.3f}")
            print(f"  Semantic similarity  (dissimilar pairs): {avg_dissim:.3f}")
            print(f"  Separability gap:                        {separability:.3f} (higher=better)")

        # ── Retrieval benchmark ──────────────────────────────────────────────
        t0 = time.perf_counter()
        doc_vecs = np.array([embed_fn(d) for d in RETRIEVAL_CORPUS])
        index_time = (time.perf_counter() - t0) * 1000

        p1_scores, p3_scores, mrr_scores, ndcg_scores, query_times = [], [], [], [], []
        for case in RETRIEVAL_QUERIES:
            t0 = time.perf_counter()
            q_vec = embed_fn(case["question"])
            elapsed = (time.perf_counter() - t0) * 1000
            query_times.append(elapsed)

            ranked = rank_documents(q_vec, doc_vecs)
            p1_scores.append(precision_at_k(ranked, case["correct_idx"], k=1))
            p3_scores.append(precision_at_k(ranked, case["correct_idx"], k=3))
            mrr_scores.append(reciprocal_rank(ranked, case["correct_idx"]))
            ndcg_scores.append(ndcg_at_k(ranked, case["correct_idx"], k=3))

            if verbose:
                top3 = [RETRIEVAL_CORPUS[i][:50] for i in ranked[:3]]
                correct = ranked.index(case["correct_idx"]) + 1
                print(f"  Q: {case['question'][:50]}")
                print(f"     Rank of correct doc: #{correct} | "
                      f"P@1={p1_scores[-1]:.0f} | MRR={mrr_scores[-1]:.2f}")

        model_results = {
            "avg_semantic_sim_similar": avg_sim,
            "avg_semantic_sim_dissimilar": avg_dissim,
            "separability": separability,
            "precision_at_1": round(np.mean(p1_scores), 3),
            "precision_at_3": round(np.mean(p3_scores), 3),
            "mrr": round(np.mean(mrr_scores), 3),
            "ndcg_at_3": round(np.mean(ndcg_scores), 3),
            "avg_query_time_ms": round(np.mean(query_times), 2),
            "index_time_ms": round(index_time, 2),
        }
        results[model_name] = model_results

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY — Embedding Model Comparison")
    print("="*90)
    header = f"{'Model':<28} {'Sep.Gap':>8} {'P@1':>6} {'P@3':>6} {'MRR':>6} {'NDCG@3':>8} {'QTime(ms)':>11}"
    print(header)
    print("-" * 90)
    for model_name, m in results.items():
        print(
            f"{model_name:<28} "
            f"{m['separability']:>8.3f} "
            f"{m['precision_at_1']:>6.3f} "
            f"{m['precision_at_3']:>6.3f} "
            f"{m['mrr']:>6.3f} "
            f"{m['ndcg_at_3']:>8.3f} "
            f"{m['avg_query_time_ms']:>9.1f}ms"
        )
    print("-" * 90)
    print("\nMetric Guide:")
    print("  Separability Gap : avg_sim(similar) - avg_sim(dissimilar). Larger = better discrimination.")
    print("  Precision@1      : correct doc is the very first result (1.0 = perfect).")
    print("  Precision@3      : correct doc is in top-3 results.")
    print("  MRR              : mean reciprocal rank. 1.0 = always first.")
    print("  NDCG@3           : normalized discounted cumulative gain at rank 3.")
    print("  Query Time       : average embedding + search latency per question.")

    return results


if __name__ == "__main__":
    run_embedding_benchmark(verbose=True)
