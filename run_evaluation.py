"""
run_evaluation.py — Single entry point for all evaluation experiments.

Usage:
  python run_evaluation.py                    # Run all three evaluations
  python run_evaluation.py --baseline         # Only embedding baseline comparison
  python run_evaluation.py --ragvsnot         # Only RAG vs No-RAG
  python run_evaluation.py --benchmark        # Only embedding benchmark
  python run_evaluation.py --save             # Save results to evaluation/results/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

def run_all(save: bool = False):
    results = {}

    print("\n" + "#"*70)
    print("# EVALUATION 1: EMBEDDING BASELINE COMPARISON")
    print("#   Word2Vec vs TF-IDF vs Sentence-Transformers")
    print("#"*70)
    from evaluation.baseline_comparison import run_baseline_comparison
    results["embedding_baseline"] = run_baseline_comparison(verbose=True)

    print("\n" + "#"*70)
    print("# EVALUATION 2: RAG vs NO-RAG")
    print("#   Keyword Hit Rate · Faithfulness · Hallucination Rate")
    print("#"*70)
    from evaluation.rag_vs_norag import run_rag_vs_norag
    results["rag_vs_norag"] = run_rag_vs_norag(verbose=True)

    print("\n" + "#"*70)
    print("# EVALUATION 3: EMBEDDING MODEL BENCHMARK")
    print("#   Precision@K · MRR · NDCG · Query Latency")
    print("#"*70)
    from evaluation.embedding_benchmark import run_embedding_benchmark
    results["embedding_benchmark"] = run_embedding_benchmark(verbose=True)

    if save:
        out_dir = Path("evaluation/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"evaluation_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] Results written to {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Financial Q&A evaluations")
    parser.add_argument("--baseline",   action="store_true", help="Run embedding baseline comparison")
    parser.add_argument("--ragvsnot",   action="store_true", help="Run RAG vs No-RAG evaluation")
    parser.add_argument("--benchmark",  action="store_true", help="Run embedding benchmark")
    parser.add_argument("--save",       action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    run_specific = args.baseline or args.ragvsnot or args.benchmark

    if not run_specific:
        run_all(save=args.save)
        return

    if args.baseline:
        from evaluation.baseline_comparison import run_baseline_comparison
        run_baseline_comparison(verbose=True)

    if args.ragvsnot:
        from evaluation.rag_vs_norag import run_rag_vs_norag
        run_rag_vs_norag(verbose=True)

    if args.benchmark:
        from evaluation.embedding_benchmark import run_embedding_benchmark
        run_embedding_benchmark(verbose=True)


if __name__ == "__main__":
    main()
