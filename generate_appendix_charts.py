"""
generate_appendix_charts.py
----------------------------
Generates the three appendix figures for the Financial Document Q&A report.
All data is taken directly from Tables 3, 4, and 5 of the report.

Requirements:
    pip install matplotlib

Output files (saved in the same folder as this script):
    figure_B1_retrieval_comparison.png
    figure_B2_rag_vs_norag.png
    figure_B3_embedding_benchmark.png
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})

BLUE   = '#378ADD'
TEAL   = '#1D9E75'
PURPLE = '#7F77DD'
GRAY   = '#888780'

# ── Figure B1: Retrieval quality comparison (Table 3) ─────────────────────────

methods = ['TF-IDF', 'Word2Vec', 'SentenceTransformer\n(MiniLM-L6-v2)']
hit1  = [0.940, 0.640, 1.000]
hit3  = [1.000, 0.760, 1.000]
mrr   = [0.963, 0.717, 1.000]

x = np.arange(len(methods))
w = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w,   hit1, w, label='Hit@1',  color=BLUE,   zorder=3)
b2 = ax.bar(x,       hit3, w, label='Hit@3',  color=TEAL,   zorder=3)
b3 = ax.bar(x + w,   mrr,  w, label='MRR',    color=PURPLE, zorder=3)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Figure B1 — Retrieval quality: Hit@1, Hit@3, MRR\n(50 real SEC 10-K Q&A pairs)', fontsize=11)
ax.legend(loc='lower right', framealpha=0.7)
plt.tight_layout()
plt.savefig('figure_B1_retrieval_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: figure_B1_retrieval_comparison.png")

# ── Figure B2: RAG vs No-RAG keyword hit rate (Table 4) ───────────────────────

strategies = ['No-RAG\n(LLM only)', 'Random context\n(irrelevant chunks)', 'RAG\n(proposed system)']
khr = [0.008, 0.133, 0.840]
colors = [GRAY, GRAY, BLUE]

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.barh(strategies, khr, color=colors, zorder=3, height=0.5)

for bar, val in zip(bars, khr):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10)

ax.set_xlim(0, 1.0)
ax.set_xlabel('Keyword hit rate (KHR)')
ax.set_title('Figure B2 — RAG vs No-RAG: keyword hit rate\n(50 real SEC 10-K Q&A pairs)', fontsize=11)
highlight = mpatches.Patch(color=BLUE, label='Proposed RAG system')
ax.legend(handles=[highlight], loc='lower right', framealpha=0.7)
plt.tight_layout()
plt.savefig('figure_B2_rag_vs_norag.png', bbox_inches='tight')
plt.close()
print("Saved: figure_B2_rag_vs_norag.png")

# ── Figure B3: Intrinsic embedding benchmark (Table 5) ────────────────────────

models = ['Word2Vec\n(sg, d=100)', 'TF-IDF\n(n-gram 1-2)', 'MiniLM-L6-v2\n(384-dim)']
p1  = [0.200, 0.600, 1.000]
mrr2= [0.370, 0.658, 1.000]
sep = [0.163, 0.130, 0.548]

x = np.arange(len(models))
w = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w,   p1,   w, label='Precision@1',      color=BLUE,   zorder=3)
b2 = ax.bar(x,       mrr2, w, label='MRR',               color=TEAL,   zorder=3)
b3 = ax.bar(x + w,   sep,  w, label='Separability gap',  color=PURPLE, zorder=3)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Figure B3 — Intrinsic embedding benchmark\n(curated similar/dissimilar financial sentence pairs)', fontsize=11)
ax.legend(loc='upper left', framealpha=0.7)
plt.tight_layout()
plt.savefig('figure_B3_embedding_benchmark.png', bbox_inches='tight')
plt.close()
print("Saved: figure_B3_embedding_benchmark.png")

print("\nAll three figures generated successfully.")
print("Insert them into Appendix B of your report with captions as labelled.")