# AI-Powered Financial Document Q&A System

**EC7203 Advanced Artificial Intelligence — Final Group Project**

A **RAG (Retrieval-Augmented Generation)** system that lets users upload financial PDFs (10-K reports, earnings calls, SEC filings) and ask natural-language questions — returning **grounded answers with page citations**.

🔗 **Live demo:** https://financial-app-system-bcffpkdt88dxp8krfjg7pd.streamlit.app/

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the interactive web app
streamlit run app.py
```

Opens at **http://localhost:8501** — upload a PDF, ask questions, view evaluation results, explore the system internals.

> Runs **100% free** by default (local MiniLM embeddings + retrieval-only answers). To get real LLM-written answers for free, add a Groq key (see *Configuration* below).

---

## Key Features

✅ **Three AI Techniques** (Course Requirement)
- **NLP:** Text preprocessing, Word2Vec / TF-IDF baselines, Sentence-Transformer embeddings
- **LLM:** Groq (Llama 3.3 70B, free) or OpenAI GPT-4o-mini for grounded generation — with a free local fallback
- **Prompt Engineering:** Systematic 7-rule system prompt, chain-of-thought, few-shot in-context learning

✅ **Evaluated on a real public dataset** — [`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K) (Q&A pairs from genuine SEC 10-K filings)

✅ **Four Deliverables**
1. **Final Report** — `report/main.pdf` (body within the 20-page limit, excluding references & appendices)
2. **Demonstrable Output** — `app.py` (Streamlit web app, deployed live) + `demo.py` (terminal demo)
3. **Code & Notebooks** — modular source + `notebooks/financial_qa_walkthrough.ipynb`
4. **Presentation** — `presentation/` (slides + 5-minute speaker script)

---

## Architecture — Two-Phase RAG Pipeline

**Phase 1: Indexing** (once per document)
```
PDF → Extract text/tables (pdfplumber) → Chunk (800 tok, 150 overlap)
    → Embed (all-MiniLM-L6-v2) → Store (ChromaDB)
```

**Phase 2: Querying** (every question)
```
Question → Embed → Cosine search (top-K) → LLM + prompt → Grounded answer + citations
```

---

## Configuration

Copy the template and (optionally) add a key:

```bash
cp .env.example .env
```

```ini
# .env — all keys optional
GROQ_API_KEY=          # free at console.groq.com/keys → real LLM answers (Llama 3.3 70B)
OPENAI_API_KEY=        # alternative (GPT-4o-mini)
LLM_PROVIDER=auto      # auto = Groq if set, else OpenAI, else local
USE_LOCAL_MODELS=true  # free local MiniLM embeddings (recommended)
```

| Mode | Setup | Answers |
|---|---|---|
| **Free local** (default) | no key | Returns the most relevant retrieved passage |
| **Free + Groq** | `GROQ_API_KEY` | Llama 3.3 70B writes a grounded, cited answer |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4o-mini writes the answer |

Embeddings always run on the free local MiniLM model, so retrieval costs nothing in every mode.

---

## Usage

### Web app (recommended)
```bash
streamlit run app.py
```
Upload a PDF → **Index** → ask questions → grounded answers with page citations. Tabs also show live **Evaluation Results** and **System Internals** (prompt, architecture).

### Terminal demo
```bash
python demo.py          # NLP + RAG + evaluation experiments with formatted tables
```

### REST API
```bash
python main.py          # FastAPI docs at http://localhost:8000/docs
```
Endpoints: `POST /api/v1/ingest`, `POST /api/v1/ask`, `GET /api/v1/documents`, `GET /api/v1/health`

### Run the evaluations
```bash
python run_evaluation.py --save   # saves JSON to evaluation/results/
```

---

## Evaluation & Results

Evaluated on **50 real Q&A pairs** sampled (seed 42) from the `financial-qa-10K` dataset.

### Experiment 1 — Embedding retrieval (gold source-passage)

| Method | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF | 0.940 | 1.000 | 0.963 |
| Word2Vec | 0.640 | 0.760 | 0.717 |
| **SentenceTransformer (MiniLM-L6-v2)** | **1.000** | **1.000** | **1.000** |

The semantic model retrieves the exact source passage **first for every query**; TF-IDF is a strong lexical baseline; Word2Vec is weakest.

### Experiment 2 — RAG vs No-RAG (Keyword Hit Rate)

| Strategy | Keyword Hit Rate |
|---|---|
| No-RAG (LLM only) | 0.008 |
| Random Context | 0.133 |
| **RAG (proposed)** | **0.840** |

Relevant retrieved context is decisive — RAG recovers 84% of gold-answer keywords vs under 1% with no grounding.

### Experiment 3 — Intrinsic embedding benchmark
Separability gap, P₁, MRR, NDCG₃ on curated probe pairs (MiniLM leads on all).

---

## Project Structure

```
financial-qa-system/
├── README.md            # This file
├── DEPLOY.md            # Free deployment guide (Streamlit Cloud / HF Spaces)
├── .env.example         # Environment template
├── requirements.txt     # Python 3.13 dependencies
│
├── app.py               # Streamlit web app (main entry point, deployed live)
├── demo.py              # Terminal demonstration
├── main.py              # FastAPI REST API server
├── pipeline.py          # High-level RAG pipeline (ingest + answer)
├── config.py            # Configuration + LLM provider resolution
│
├── ingestion/           # Phase 1: PDF → chunks → embeddings
│   ├── pdf_loader.py    # pdfplumber text & table extraction
│   ├── text_chunker.py  # 800-token chunking, 150-token overlap
│   └── embedder.py      # MiniLM-L6-v2 (local) or OpenAI embeddings
├── retrieval/
│   ├── vector_store.py  # ChromaDB interface (add, cosine search, delete)
│   └── retriever.py     # High-level retrieval API
├── generation/
│   ├── prompt_builder.py # 7-rule system prompt + few-shot examples
│   └── qa_chain.py       # Groq/OpenAI LLM + local fallback
├── api/
│   ├── routes.py        # FastAPI endpoints
│   └── schemas.py       # Pydantic request/response models
├── evaluation/          # Three experiments on the real dataset
│   ├── dataset_loader.py        # Loads/caches financial-qa-10K
│   ├── baseline_comparison.py   # TF-IDF vs Word2Vec vs MiniLM
│   ├── rag_vs_norag.py          # RAG vs No-RAG vs Random Context
│   ├── embedding_benchmark.py   # P@K, MRR, NDCG, separability
│   └── results/                 # Saved JSON results (read by the app)
│
├── notebooks/financial_qa_walkthrough.ipynb   # Interactive walkthrough
├── report/              # Final report (main.tex → main.pdf)
├── presentation/        # Slides (.pptx/.pdf) + speaker script
├── tests/test_pipeline.py
├── Dockerfile           # Container build (for HF Spaces / any Docker host)
└── data/uploads/        # Temporary PDF storage
```

---

## Deployment

The app is deployed on **Streamlit Community Cloud** (free). To deploy your own copy, see **`DEPLOY.md`** — in short:

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io): New app → pick the repo → main file `app.py`.
3. In **Secrets**, add `GROQ_API_KEY`, `USE_LOCAL_MODELS = "true"`, `LLM_PROVIDER = "auto"`.

`DEPLOY.md` also covers Hugging Face Spaces (more RAM) and the included `Dockerfile`.

> Pushing to the connected GitHub branch **auto-redeploys** the live Streamlit app.

---

## AI Techniques Implemented

| Technique | File | Implementation |
|---|---|---|
| NLP — preprocessing | `ingestion/pdf_loader.py` | Ligature fixing, page-number removal, table extraction |
| NLP — word embeddings | `evaluation/baseline_comparison.py` | Word2Vec (skip-gram) + TF-IDF baselines vs SentenceTransformer |
| NLP — chunking | `ingestion/text_chunker.py` | RecursiveCharacterTextSplitter, 800 tok / 150 overlap |
| LLM generation | `generation/qa_chain.py` | Groq (Llama 3.3 70B) / OpenAI GPT-4o-mini + local fallback |
| Prompt engineering | `generation/prompt_builder.py` | 7-rule system prompt, chain-of-thought, 2-shot in-context |
| Vector search | `retrieval/vector_store.py` | ChromaDB cosine ANN, top-K retrieval |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| App falls back to local mode | Add `GROQ_API_KEY` to `.env` (or the app sidebar) |
| First answer is slow (~20–30 s) | MiniLM model downloads once, then it's fast |
| Switched embedding mode | Delete `data/chroma_db/` and re-index |

---

## References

- **RAG:** Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- **Sentence-Transformers:** Reimers & Gurevych (2019) — *Sentence-BERT*
- **Chain-of-Thought:** Wei et al. (2022)
- **Word2Vec:** Mikolov et al. (2013)
- **Dataset:** `virattt/financial-qa-10K` (Hugging Face)

Full references in `report/main.pdf`.

---

*Academic project for EC7203 Advanced Artificial Intelligence — educational use.*
