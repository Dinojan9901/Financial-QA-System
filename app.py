"""
app.py — Streamlit Web UI for the Financial Document Q&A System
EC7203 Advanced Artificial Intelligence — Demonstrable Output

Run:  streamlit run app.py
"""

import os
import sys
import tempfile
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("USE_LOCAL_MODELS", "true")

from dotenv import load_dotenv
load_dotenv()

# On Streamlit Community Cloud / HF Spaces, secrets are provided via st.secrets.
# Bridge them into os.environ so the env-based config (config.py) picks them up.
try:
    for _k in ("GROQ_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "USE_LOCAL_MODELS"):
        if _k in st.secrets and str(st.secrets[_k]).strip():
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally — that's fine

st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.0rem; font-weight: 700; color: #1a3c6e;
    border-bottom: 3px solid #1a3c6e; padding-bottom: 0.5rem; margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.1rem; color: #444; margin-bottom: 1.5rem;
}
.metric-card {
    background: #f0f4f8; border-radius: 8px; padding: 1rem;
    border-left: 4px solid #1a3c6e; margin: 0.5rem 0; color: #1a1a1a;
}
.answer-box {
    background: #ffffff; border-radius: 8px; padding: 1.2rem;
    border-left: 4px solid #2e7d32; font-size: 1.05rem; color: #1a1a1a;
}
.source-box {
    background: #fff8e1; border-radius: 6px; padding: 0.8rem;
    border-left: 3px solid #f57c00; margin: 0.3rem 0; font-size: 0.9rem; color: #1a1a1a;
}
.step-badge {
    background: #1a3c6e; color: white; border-radius: 50%;
    width: 28px; height: 28px; display: inline-flex;
    align-items: center; justify-content: center; font-weight: bold;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner=False)
def load_embedder():
    from ingestion.embedder import EmbeddingGenerator
    return EmbeddingGenerator()

@st.cache_resource(show_spinner=False)
def load_store():
    from retrieval.vector_store import VectorStore
    return VectorStore()



with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/1px-placeholder.png",
             width=80)

    st.markdown("## ⚙️ Settings")

    st.markdown("**LLM for answer generation** (optional)")
    groq_key = st.text_input(
        "Groq API Key — free",
        type="password",
        placeholder="gsk_... (free at console.groq.com/keys)",
        help="Free, fast LLM generation (Llama 3.3 70B). With a key the system "
             "writes a synthesised grounded answer; without one it returns the "
             "most relevant retrieved passage.",
    )
    openai_key = st.text_input(
        "OpenAI API Key — alternative",
        type="password",
        placeholder="sk-... (optional; GPT-4o-mini)",
        help="Alternative to Groq. Leave blank if using Groq or local mode.",
    )

    os.environ["USE_LOCAL_MODELS"] = "true"
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["LLM_PROVIDER"] = "groq"
    elif openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["LLM_PROVIDER"] = "openai"
    elif os.getenv("LLM_PROVIDER") not in ("groq", "openai"):
        os.environ["LLM_PROVIDER"] = "auto"  # let config resolve from secrets/.env

    from config import resolve_llm_provider
    _active, _, _, _model = resolve_llm_provider()
    if _active == "groq":
        st.success("🟢 Generation: Groq (Llama 3.3 70B)")
    elif _active == "openai":
        st.success("🟢 Generation: OpenAI (GPT-4o-mini)")
    else:
        st.info("🔵 Generation: local retrieval-only (no key)")

    n_chunks = st.slider("Chunks retrieved per question", min_value=1, max_value=8, value=3)

    st.markdown("---")
    st.markdown("### 🏛️ About")
    st.markdown(
        "**EC7203 Advanced AI**  \n"
        "AI-Powered Financial Document Q&A  \n"
        "Using RAG · NLP · LLM · Prompt Engineering"
    )
    st.markdown("---")
    st.markdown("### 🔧 Architecture")
    st.markdown(
        "**Phase 1 — Indexing**  \n"
        "PDF → Extract → Chunk → Embed → Store  \n\n"
        "**Phase 2 — Querying**  \n"
        "Question → Embed → Search → Generate"
    )



st.markdown('<div class="main-header">📊 Financial Document Q&A System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a financial PDF (10-K, earnings call, SEC filing) '
    'and ask questions in plain English — get grounded answers with page citations.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📄 Upload & Ask", "📈 Evaluation Results", "🔬 System Internals"])



with tab1:
    col_upload, col_qa = st.columns([1, 1.6], gap="large")

    with col_upload:
        st.subheader("Step 1 — Upload Financial PDF")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Accepts 10-K annual reports, earnings transcripts, SEC filings, "
                 "or any financial PDF document.",
        )

        if uploaded_file:
            st.success(f"✅ File received: **{uploaded_file.name}** "
                       f"({uploaded_file.size / 1024:.1f} KB)")

            if st.button("🚀 Index This Document", type="primary", width="stretch"):
                with st.spinner("Indexing — extracting text, chunking, generating embeddings..."):
                    try:
                        from pipeline import ingest_document

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf",
                            prefix=uploaded_file.name.replace(".pdf", "_"),
                        ) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name

                        t0 = time.perf_counter()
                        result = ingest_document(tmp_path)
                        elapsed = time.perf_counter() - t0
                        os.unlink(tmp_path)

                        st.session_state["indexed_doc"] = uploaded_file.name
                        st.session_state["index_result"] = result

                        st.success(f"✅ Document indexed in {elapsed:.1f}s")
                        st.json({
                            "Pages extracted": result["pages"],
                            "Chunks created": result["chunks"],
                            "Avg tokens/chunk": result.get("avg_tokens_per_chunk", "—"),
                        })

                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

        st.markdown("---")
        st.subheader("📚 Indexed Documents")

        if "confirm_clear_all" not in st.session_state:
            st.session_state["confirm_clear_all"] = False

        try:
            from pipeline import list_indexed_documents, delete_document
            docs = list_indexed_documents()
            if docs:
                for d in docs:
                    col_doc, col_del = st.columns([3, 1])
                    col_doc.markdown(f"📄 `{d}`")
                    if col_del.button("🗑️", key=f"del_{d}", help=f"Remove {d}"):
                        delete_document(d)
                        st.success(f"Removed **{d}** from the index.")
                        st.rerun()

                st.markdown("")
                if not st.session_state["confirm_clear_all"]:
                    if st.button("🗑️ Clear All Documents", type="secondary", width="stretch"):
                        st.session_state["confirm_clear_all"] = True
                        st.rerun()
                else:
                    st.warning("⚠️ This will permanently remove all indexed documents. Are you sure?")
                    col_yes, col_no = st.columns(2)
                    if col_yes.button("✅ Yes, Clear All", type="primary"):
                        for d in docs:
                            delete_document(d)
                        st.session_state["confirm_clear_all"] = False
                        st.success("All documents cleared from the index.")
                        st.rerun()
                    if col_no.button("❌ Cancel"):
                        st.session_state["confirm_clear_all"] = False
                        st.rerun()
            else:
                st.session_state["confirm_clear_all"] = False
                st.info("No documents indexed yet. Upload and index a PDF above.")
        except Exception:
            st.info("Index not yet initialised.")

    with col_qa:
        st.subheader("Step 2 — Ask a Question")

        example_qs = [
            "Custom question...",
            "What was the total net revenue or sales?",
            "What was the net income or profit?",
            "What were the main risk factors?",
            "What was the earnings per share (EPS)?",
            "How much was spent on research and development?",
            "What are the main business segments?",
        ]
        selected_example = st.selectbox("Or choose an example question:", example_qs)

        if selected_example == "Custom question...":
            question = st.text_area(
                "Your question:",
                placeholder="e.g. What was the total revenue in fiscal 2023?",
                height=100,
            )
        else:
            question = st.text_area("Your question:", value=selected_example, height=100)

        try:
            from pipeline import list_indexed_documents
            docs = list_indexed_documents()
            doc_filter = st.selectbox(
                "Filter to document (optional):",
                ["Search all documents"] + docs,
            )
            source_filter = None if doc_filter == "Search all documents" else doc_filter
        except Exception:
            source_filter = None

        ask_btn = st.button("🔍 Ask", type="primary", width="stretch",
                            disabled=not question.strip())

        if ask_btn and question.strip():
            with st.spinner("Retrieving relevant passages and generating answer..."):
                try:
                    from ingestion.embedder import EmbeddingGenerator
                    from retrieval.vector_store import VectorStore
                    from generation.qa_chain import get_qa_chain

                    embedder = load_embedder()
                    store = load_store()
                    qa = get_qa_chain()

                    t0 = time.perf_counter()
                    q_emb = embedder.embed_text(question)
                    chunks = store.search(
                        query_embedding=q_emb,
                        n_results=n_chunks,
                        source_filter=source_filter,
                    )
                    result = qa.answer(question=question, context_chunks=chunks)
                    elapsed = time.perf_counter() - t0

                    st.markdown("#### 💬 Answer")
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    if result.get("sources"):
                        st.markdown(f"#### 📎 Sources ({len(result['sources'])} retrieved)")
                        for s in result["sources"]:
                            relevance_pct = int(s.get("relevance", 0) * 100)
                            bar = "█" * (relevance_pct // 10) + "░" * (10 - relevance_pct // 10)
                            st.markdown(
                                f'<div class="source-box">'
                                f'📄 <b>{s.get("source", "unknown")}</b> &nbsp;|&nbsp; '
                                f'Page <b>{s.get("page", "?")}</b> &nbsp;|&nbsp; '
                                f'Relevance: <b>{relevance_pct}%</b> {bar}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    with st.expander("⚡ Performance stats"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total latency", f"{elapsed:.2f}s")
                        col_b.metric("Chunks retrieved", result.get("chunks_retrieved", n_chunks))
                        col_c.metric("Tokens used", result.get("tokens_used", 0))
                        st.caption(f"Model: {result.get('model', 'local')}")

                except Exception as e:
                    st.error(f"Q&A failed: {e}")
                    st.info("Make sure you have indexed at least one document first.")



with tab2:
    st.subheader("Evaluation Results — Live Experimental Data")
    st.markdown(
        "Produced by `python run_evaluation.py --save`. Experiments 1–2 run on "
        "**50 real Q&A pairs** sampled from the public "
        "[`virattt/financial-qa-10K`](https://huggingface.co/datasets/virattt/financial-qa-10K) "
        "dataset (derived from genuine **SEC 10-K filings**); the free local model "
        "is used, so no API key is required."
    )

    import json, glob as _glob
    result_files = sorted(_glob.glob(
        "evaluation/results/evaluation_*.json"
    ), reverse=True)

    if result_files:
        with open(result_files[0]) as f:
            eval_data = json.load(f)

        st.markdown("---")
        st.markdown("### Experiment 1 — Embedding Baseline Comparison")
        st.markdown(
            "Gold source-passage retrieval over 50 real SEC 10-K questions: does each "
            "method rank the question's exact source passage first (**Hit@1**) or "
            "within the top 3 (**Hit@3**)? **MRR** is the mean reciprocal rank."
        )

        baseline = eval_data.get("embedding_baseline", {})
        if baseline:
            col1, col2, col3 = st.columns(3)
            for col, (method, metrics) in zip([col1, col2, col3], baseline.items()):
                with col:
                    hit1 = metrics.get("hit_at_1", metrics.get("avg_precision_at_3", 0))
                    hit3 = metrics.get("hit_at_3", 0)
                    mrr = metrics.get("avg_mrr", 0)
                    star = " 🏆" if method == "SentenceTransformer" else ""
                    col.markdown(f"**{method}{star}**")
                    col.metric("Hit@1", f"{hit1:.3f}")
                    col.metric("Hit@3", f"{hit3:.3f}")
                    col.metric("MRR", f"{mrr:.3f}")

        st.markdown("---")
        st.markdown("### Experiment 2 — RAG vs No-RAG")
        st.markdown(
            "Three answer strategies on the same 50 questions, measured by **Keyword "
            "Hit Rate** (fraction of gold-answer keywords recovered). RAG should "
            "dominate — relevant context is what makes grounded answers possible."
        )

        rag_data = eval_data.get("rag_vs_norag", {})
        if rag_data:
            col1, col2, col3 = st.columns(3)
            colours = {"No-RAG": "🔴", "Random-Context": "🟡", "RAG": "🟢"}
            labels = {
                "No-RAG": "LLM only, no context",
                "Random-Context": "irrelevant chunks",
                "RAG": "retrieved relevant chunks",
            }
            for col, (strategy, metrics) in zip([col1, col2, col3], rag_data.items()):
                with col:
                    khr = metrics.get("avg_keyword_hit_rate", 0)
                    emoji = colours.get(strategy, "⚪")
                    col.markdown(f"**{emoji} {strategy}**")
                    col.metric("Keyword Hit Rate", f"{khr:.3f}")
                    col.caption(labels.get(strategy, ""))
            st.caption(
                "Faithfulness / hallucination heuristics are degenerate on this "
                "larger set in offline simulation mode, so Keyword Hit Rate is the "
                "reported signal (see report §6.3.2)."
            )

        st.markdown("---")
        st.markdown("### Experiment 3 — Intrinsic Embedding Benchmark")
        st.markdown(
            "Intrinsic embedding quality on curated similar/dissimilar probe pairs: "
            "P₁, MRR, NDCG₃, **Separability Gap**, and query latency."
        )

        bench = eval_data.get("embedding_benchmark", {})
        if bench:
            import pandas as pd
            rows = []
            for model, m in bench.items():
                rows.append({
                    "Model": model,
                    "Sep. Gap": f"{m.get('separability', 0):.3f}",
                    "P1": f"{m.get('precision_at_1', 0):.3f}",
                    "P3": f"{m.get('precision_at_3', 0):.3f}",
                    "MRR": f"{m.get('mrr', 0):.3f}",
                    "NDCG3": f"{m.get('ndcg_at_3', 0):.3f}",
                    "Query Time": f"{m.get('avg_query_time_ms', 0):.1f} ms",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)

        st.caption(f"Results from: `{result_files[0]}`")

    else:
        st.info(
            "No saved results found. Run `python run_evaluation.py --save` first "
            "to generate evaluation data."
        )
        if st.button("Run Evaluation Now"):
            with st.spinner("Running all 3 evaluation experiments (may take 3-5 min)..."):
                from evaluation.baseline_comparison import run_baseline_comparison
                from evaluation.rag_vs_norag import run_rag_vs_norag
                st.write("Experiment 1: Baseline Comparison")
                r1 = run_baseline_comparison(verbose=False)
                st.success("Experiment 1 done")
                st.write("Experiment 2: RAG vs No-RAG")
                r2 = run_rag_vs_norag(verbose=False)
                st.success("Experiment 2 done")
                st.json({"embedding_baseline": r1, "rag_vs_norag": r2})



with tab3:
    st.subheader("System Internals — How the Pipeline Works")

    st.markdown("### RAG Pipeline Architecture")
    st.markdown("""
```
PHASE 1 - INDEXING (run once per document)
----------------------------------------------------------
  PDF Upload
    --> Text Extraction  (pdfplumber + table support)
         --> Text Chunking  (800 tokens, 150-token overlap)
              --> Embedding  (MiniLM-L6-v2 or OpenAI)
                   --> ChromaDB Vector Store (cosine ANN)

PHASE 2 - QUERYING (run on every user question)
----------------------------------------------------------
  User Question
    --> Query Embedding  (same model as documents)
         --> Cosine Search  (top-K relevant chunks)
              --> LLM + Prompt Engineering
                   --> Grounded Answer + Page Citations
```
""")

    st.markdown("### AI Techniques Applied")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. NLP")
        st.markdown("""
- Text preprocessing & cleaning
- Ligature fixing, noise removal
- Token-counted chunking
- **Word2Vec** (skip-gram baseline)
- **TF-IDF** (sparse IR baseline)
- **Sentence-Transformers** (proposed)
        """)

    with col2:
        st.markdown("#### 2. LLM")
        st.markdown("""
- GPT-4o-mini (OpenAI API)
- Transformer decoder architecture
- Temperature = 0.1 (factual)
- Grounded generation from context
- Source citation enforcement
- Free fallback: retrieval-only mode
        """)

    with col3:
        st.markdown("#### 3. Prompt Engineering")
        st.markdown("""
- **Systematic design**: 7 critical rules
- **Chain-of-thought**: "think step by step"
- **Few-shot learning**: 2 demo examples
- Hallucination prevention constraints
- Page citation requirements
- "I don't know" > hallucination
        """)

    st.markdown("---")
    st.markdown("### Current System Status")
    try:
        from retrieval.vector_store import VectorStore
        store = VectorStore()
        count = store.get_document_count()
        docs = store.list_documents()
        col_a, col_b = st.columns(2)
        col_a.metric("Total chunks indexed", count)
        col_b.metric("Documents in store", len(docs))
        if docs:
            st.markdown("**Indexed documents:**")
            for d in docs:
                st.markdown(f"- `{d}`")
    except Exception as e:
        st.warning(f"Could not read vector store: {e}")

    st.markdown("---")
    st.markdown("### Prompt Engineering Example")
    with st.expander("View the system prompt (7 critical rules)"):
        from generation.prompt_builder import FINANCIAL_QA_SYSTEM_PROMPT
        st.code(FINANCIAL_QA_SYSTEM_PROMPT, language="text")

    with st.expander("View a few-shot in-context learning example"):
        from generation.prompt_builder import FEW_SHOT_EXAMPLES
        for msg in FEW_SHOT_EXAMPLES:
            st.markdown(f"**[{msg['role'].upper()}]**")
            st.code(msg["content"], language="text")
