# AI-Powered Financial Document Q&A System

> New to the repository? Start with the
> [project documentation index](Project_Documents/README.md) and
> [folder structure guide](Project_Documents/FOLDER_STRUCTURE.md).

**EC7203 Advanced Artificial Intelligence — Final Group Project**

A production-grade **RAG (Retrieval-Augmented Generation)** system that lets users upload financial PDFs (10-K reports, earnings calls, SEC filings) and ask natural language questions in plain English — returning **grounded answers with page citations** and **zero hallucinations**.

## Quick Start (Recommended)

### Run the Streamlit Web App
```bash
# Install dependencies
pip install -r requirements.txt

# Start the interactive web UI
py -3.13 -m streamlit run app.py
```

Opens at **http://localhost:8501** — upload a PDF, ask questions, view evaluation results, explore the system internals.

---

## Key Features

✅ **Three AI Techniques** (Course Requirement)
- **NLP:** Text preprocessing, Word2Vec/TF-IDF baselines, Sentence-Transformers embeddings
- **Transformer embeddings:** all-MiniLM-L6-v2 for semantic retrieval
- **Prompt Engineering:** Systematic 7-rule system prompt, chain-of-thought, few-shot learning

✅ **Demonstrated Results**
- MRR = 1.000 (perfect retrieval on 5 test cases)
- 0 hallucinations (vs 1/5 for No-RAG baseline)
- Keyword Hit Rate = 1.000 (100% financial metric extraction)

✅ **Three Deliverables** (All Included)
1. **Final Report** — `Project_Documents/report/main.pdf`
2. **Demonstrable Output** — `app.py` (Streamlit web UI) + `demo.py` (terminal demo)
3. **Python Code & Notebooks** — 28 source files + Jupyter notebook + evaluation modules

---

## Architecture

### Two-Phase RAG Pipeline

**Phase 1: Indexing** (runs once per document)
```
PDF Upload
  ↓ [pdfplumber]
Extract Text & Tables
  ↓ [RecursiveCharacterTextSplitter]
Chunk (200 tokens, 40-token overlap)
  ↓ [all-MiniLM-L6-v2]
Generate Embeddings
  ↓ [ChromaDB]
Store in Vector Database
```

**Phase 2: Querying** (runs on every user question)
```
User Question
  ↓ [all-MiniLM-L6-v2]
Embed Question
  ↓ [ChromaDB]
Cosine Similarity Search
  ↓
Retrieve Top-K Chunks
  ↓ [LocalQAChain]
Return Grounded Excerpt with Citations
```

---

## Installation & Setup

### 1. Clone and Create Environment
```bash
cd financial-qa-system
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Local Paths (Optional)
```bash
cp .env.example .env
# Edit .env only when custom storage paths or ports are needed
```

---

## Usage

### Web Interface (Recommended for Demo)
```bash
py -3.13 -m streamlit run app.py
```

**Features:**
- Upload any financial PDF
- Watch real-time indexing progress
- Ask questions in plain English
- View grounded answers with page citations and relevance scores
- Switch to Evaluation tab to see experiment results
- Switch to System Internals tab to view prompt engineering examples

### Terminal Demo
```bash
py -3.13 -X utf8 demo.py
# Shows NLP pipeline, RAG pipeline, evaluation experiments with formatted tables
```

### REST API Server
```bash
python main.py
# API docs at http://localhost:8000/docs
```

**Endpoints:**
- `POST /api/v1/ingest` — Upload and index a PDF
- `POST /api/v1/ask` — Ask a question
- `GET /api/v1/documents` — List indexed documents
- `GET /api/v1/health` — System status

### Command-Line (Direct Python)
```bash
python pipeline.py apple_10k_2023.pdf "What was the total revenue in fiscal 2023?"
```

---

## Evaluation & Results

Three evaluation experiments with **live results** in the Streamlit app:

### Experiment 1: Embedding Baseline Comparison
Compares TF-IDF, Word2Vec, and SentenceTransformer on 5 financial queries:

| Method | Precision@3 | MRR | Query Time |
|---|---|---|---|
| TF-IDF | 0.333 | 0.800 | 4.9 ms |
| Word2Vec | 0.200 | 0.517 | 2.0 ms |
| **MiniLM-L6-v2** | **0.400** | **1.000** | 89.0 ms |

**Key Finding:** Sentence-transformers achieve perfect (1.000) MRR — correct document always ranked first.

### Experiment 2: RAG vs No-RAG
Compares three answer strategies:

| Strategy | Keyword Hit Rate | Faithfulness | Hallucinations |
|---|---|---|---|
| No-RAG (LLM only) | 0.400 | 0.600 | 1/5 ❌ |
| Random Context | 0.200 | 1.000 | 0/5 |
| **RAG (Proposed)** | **1.000** | **1.000** | **0/5** ✅ |

**Key Finding:** RAG achieves 100% keyword hit rate and zero hallucinations.

### Experiment 3: Full Embedding Benchmark
Comprehensive metrics for all models (P₁, MRR, NDCG₃, separability gap).

**Run evaluations:**
```bash
python run_evaluation.py --save
# Results saved to evaluation/results/evaluation_TIMESTAMP.json
```

---

## Project Structure

```
financial-qa-system/
├── README.md              # This file
├── .env.example          # Environment template
├── requirements.txt      # Python 3.13+ dependencies
│
├── app.py                # Streamlit web UI (main entry point)
├── demo.py               # Terminal demonstration with formatted tables
├── main.py               # FastAPI REST API server
├── pipeline.py           # High-level RAG pipeline functions
├── config.py             # Centralized configuration
│
├── ingestion/            # Phase 1: PDF → Chunks → Embeddings
│   ├── pdf_loader.py     # pdfplumber text & table extraction
│   ├── text_chunker.py   # 200-token chunking with 40-token overlap
│   └── embedder.py       # all-MiniLM-L6-v2 local embeddings
│
├── retrieval/            # Similarity Search
│   ├── vector_store.py   # ChromaDB interface (CRUD + cosine search)
│   └── retriever.py      # High-level retrieval API
│
├── generation/           # Phase 2: Question → Answer
│   ├── prompt_builder.py # System prompt (7 critical rules), few-shot examples
│   └── qa_chain.py       # Retrieval-only grounded response formatter
│
├── api/                  # REST API (FastAPI)
│   ├── routes.py         # Endpoints: /ingest, /ask, /documents, /health
│   └── schemas.py        # Pydantic models for request/response validation
│
├── evaluation/           # Three evaluation experiments
│   ├── baseline_comparison.py    # TF-IDF vs Word2Vec vs MiniLM
│   ├── rag_vs_norag.py           # RAG vs No-RAG vs Random Context
│   ├── embedding_benchmark.py    # P@K, MRR, NDCG@3, separability, latency
│   ├── run_evaluation.py         # Entry point for all experiments
│   └── results/                  # Timestamped JSON results
│
├── notebooks/            # Jupyter notebooks (documentation + walkthrough)
│   └── financial_qa_walkthrough.ipynb  # 10-section interactive demo
│
├── report/               # Final Report (LaTeX → PDF)
│   ├── main.tex          # 19-page LaTeX source
│   └── main.pdf          # Compiled PDF report
│
├── presentation/         # Video presentation materials
│   ├── Financial_QA_Presentation.pptx  # 11 slides, 16:9 format
│   ├── Financial_QA_Presentation.pdf   # PDF backup of slides
│   ├── SPEAKER_SCRIPT.md               # 5-minute narration script
│   └── build_presentation.py           # Slide generator (python-pptx)
│
├── tests/                # Unit & integration tests
│   └── test_pipeline.py  # pytest-based tests
│
└── data/
    └── uploads/          # Temporary PDF storage during indexing
```

---

## Video Presentation

**Files:** `presentation/Financial_QA_Presentation.pptx` + `SPEAKER_SCRIPT.md`

**11-slide deck (5:00 total):**
1. Title & Course Info (0:15)
2. Problem & Motivation (0:30)
3. Proposed Solution — RAG (0:25)
4. Three AI Techniques (0:35)
5. System Architecture (0:30)
6. Implementation Stack (0:25)
7. **LIVE DEMO** — Streamlit app (1:00)
8. Experiment 1 Results (0:25)
9. Experiment 2 Results (0:25)
10. Conclusions & Future Work (0:20)
11. Thank You (0:10)

**Recording checklist:**
- [ ] Pre-index a financial PDF (saves time during demo)
- [ ] Test Streamlit app: `py -3.13 -m streamlit run app.py`
- [ ] Record at 1080p using OBS Studio or Zoom
- [ ] Each team member narrates 2–3 slides
- [ ] Export as MP4, upload to YouTube/Google Drive
- [ ] Share video link in the final report submission

---

## AI Techniques Implemented

| Technique | Files | Key Implementation |
|---|---|---|
| **NLP: Text Preprocessing** | `ingestion/pdf_loader.py` | Ligature fixing, page number removal, table extraction via pdfplumber |
| **NLP: Word Embeddings** | `ingestion/embedder.py` | Word2Vec (skip-gram, d=100) baseline + TF-IDF (sparse) baseline + SentenceTransformer (proposed) |
| **NLP: Text Chunking** | `ingestion/text_chunker.py` | RecursiveCharacterTextSplitter, 200 tokens, 40-token overlap |
| **Transformer Encoder** | `ingestion/embedder.py` | all-MiniLM-L6-v2 semantic embeddings |
| **Prompt Engineering** | `generation/prompt_builder.py` | 7-rule system prompt, chain-of-thought, 2-shot in-context examples |
| **Vector Similarity Search** | `retrieval/vector_store.py` | ChromaDB with cosine ANN, top-K retrieval |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Only evaluation tests
pytest tests/test_pipeline.py::TestEvaluation -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Deployment

### Docker
```bash
docker build -t financial-qa .
docker run -p 8501:8501 --env-file .env financial-qa streamlit run app.py
```

### Railway (Cloud)
1. Push repo to GitHub
2. Connect on railway.app
3. Railway auto-detects Python and deploys

---

## Free Local Mode

The system always uses free local `all-MiniLM-L6-v2` embeddings. No API key is required.

**Behavior:** Retrieves relevant chunks and returns the strongest grounded excerpt with page citations.

---

## Troubleshooting

**Q: "ModuleNotFoundError: No module named X"**
```bash
pip install -r requirements.txt
```

**Q: Streamlit app won't start**
```bash
# Use explicit Python 3.13 with UTF-8
py -3.13 -X utf8 -m streamlit run app.py
```

**Q: PDF indexing is slow**
- First run generates embeddings (~1–2 min for 100 pages)
- Subsequent queries are instant (cached in ChromaDB)

---

## Citation & References

This project builds on:
- **RAG:** Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **Sentence-Transformers:** Reimers & Gurevych (2019) — "Sentence-BERT"
- **Prompt Engineering:** Wei et al. (2022) — "Chain-of-Thought Prompting"
- **Word Embeddings:** Mikolov et al. (2013) — "Efficient Estimation of Word Representations"

Full references in `report/main.pdf`.

---

## License

Academic project for EC7203 Advanced Artificial Intelligence. All code is provided for educational purposes.

---

## Contact & Support

- **Final Report:** `report/main.pdf`
- **Evaluation Results:** `evaluation/results/evaluation_*.json`
- **Slides:** `presentation/Financial_QA_Presentation.pptx`
- **Speaker Notes:** `presentation/SPEAKER_SCRIPT.md`
