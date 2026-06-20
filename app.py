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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
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


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embedder():
    from ingestion.embedder import EmbeddingGenerator
    return EmbeddingGenerator()

@st.cache_resource(show_spinner=False)
def load_store():
    from retrieval.vector_store import VectorStore
    return VectorStore()

@st.cache_resource(show_spinner=False)
def load_qa_chain():
    from generation.qa_chain import get_qa_chain
    return get_qa_chain()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/1px-placeholder.png",
             width=80)

    st.markdown("## ⚙️ Settings")

    n_chunks = st.slider("Chunks retrieved per question", min_value=1, max_value=8, value=3)
    st.caption("Embeddings: all-MiniLM-L6-v2 (free and local)")

    st.markdown("---")
    st.markdown("### 🏛️ About")
    st.markdown(
        "**EC7203 Advanced AI**  \n"
        "AI-Powered Financial Document Q&A  \n"
        "Using semantic retrieval · NLP · sentence transformers"
    )
    st.markdown("---")
    st.markdown("### 🔧 Architecture")
    st.markdown(
        "**Phase 1 — Indexing**  \n"
        "PDF → Extract → Chunk → Embed → Store  \n\n"
        "**Phase 2 — Querying**  \n"
        "Question → Embed → Search → Return grounded excerpts"
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📊 Financial Document Q&A System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a financial PDF (10-K, earnings call, SEC filing) '
    'and ask questions in plain English — get grounded answers with page citations.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📄 Upload & Ask", "📈 Evaluation Results", "🔬 System Internals"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD & ASK
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    col_upload, col_qa = st.columns([1, 1.6], gap="large")

    # ── Left column: upload ────────────────────────────────────────────────────
    with col_upload:
        st.subheader("Step 1 — Upload Financial PDF")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Accepts 10-K annual reports, earnings transcripts, SEC filings, "
                 "or any financial PDF document.",
        )

        if uploaded_file is not None:
            st.success(f"✅ File received: **{uploaded_file.name}** "
                       f"({uploaded_file.size / 1024:.1f} KB)")
        else:
            st.caption("Select a PDF above to enable document indexing.")

        index_clicked = st.button(
            "🚀 Index This Document",
            type="primary",
            width="stretch",
            disabled=uploaded_file is None,
            help="Upload a PDF first, then click here to add it to the vector index.",
        )

        if index_clicked and uploaded_file is not None:
            with st.spinner("Indexing — extracting text, chunking, generating embeddings..."):
                tmp_path = None
                try:
                    from pipeline import ingest_document

                    # pdfplumber needs a filesystem path for the uploaded file.
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    t0 = time.perf_counter()
                    result = ingest_document(tmp_path, source_name=uploaded_file.name)
                    elapsed = time.perf_counter() - t0

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
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # Show currently indexed documents
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

    # ── Right column: Q&A ─────────────────────────────────────────────────────
    with col_qa:
        st.subheader("Step 2 — Ask a Question")

        # Preset example questions
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

        # Document filter
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
                    qa = load_qa_chain()

                    t0 = time.perf_counter()
                    q_emb = embedder.embed_text(question)
                    chunks = store.search(
                        query_embedding=q_emb,
                        n_results=n_chunks,
                        source_filter=source_filter,
                    )
                    result = qa.answer(question=question, context_chunks=chunks)
                    elapsed = time.perf_counter() - t0

                    # ── Answer ─────────────────────────────────────────────────
                    st.markdown("#### 💬 Answer")
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Sources ────────────────────────────────────────────────
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

                    # ── Stats ──────────────────────────────────────────────────
                    with st.expander("⚡ Performance stats"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total latency", f"{elapsed:.2f}s")
                        col_b.metric("Chunks retrieved", result.get("chunks_retrieved", n_chunks))
                        col_c.metric("Tokens used", result.get("tokens_used", 0))
                        st.caption(f"Model: {result.get('model', 'local')}")

                except Exception as e:
                    st.error(f"Q&A failed: {e}")
                    st.info("Make sure you have indexed at least one document first.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — EVALUATION RESULTS
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Evaluation Results — Live Experimental Data")
    st.markdown(
        "These results were produced by running `python run_evaluation.py --save` "
        "on the local system. All three experiments use the **free local model** "
        "(no API key required)."
    )

    # Load saved results
    import json, glob as _glob
    result_files = sorted(_glob.glob(
        "evaluation/results/evaluation_*.json"
    ), reverse=True)

    if result_files:
        with open(result_files[0]) as f:
            eval_data = json.load(f)

        # ── Experiment 1: Baseline Comparison ─────────────────────────────────
        st.markdown("---")
        st.markdown("### Experiment 1 — Embedding Baseline Comparison")
        st.markdown(
            "Compares TF-IDF, Word2Vec, and SentenceTransformer on retrieval quality "
            "for 5 financial queries."
        )

        baseline = eval_data.get("embedding_baseline", {})
        if baseline:
            col1, col2, col3 = st.columns(3)
            for col, (method, metrics) in zip([col1, col2, col3], baseline.items()):
                with col:
                    p3 = metrics.get("avg_precision_at_3", 0)
                    mrr = metrics.get("avg_mrr", 0)
                    col.metric(f"Precision (top-3)", f"{p3:.3f}",
                               delta=f"{(p3 - 0.333):+.3f} vs TF-IDF" if method != "TF-IDF" else None)
                    col.metric("MRR", f"{mrr:.3f}",
                               delta=f"{(mrr - 0.800):+.3f} vs TF-IDF" if method != "TF-IDF" else None)
                    col.caption(f"**{method}**")

        # ── Experiment 2: RAG vs No-RAG ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### Experiment 2 — RAG vs No-RAG")
        st.markdown(
            "Compares three answer strategies: No-RAG (LLM only), Random Context "
            "(irrelevant chunks), and RAG (correctly retrieved chunks)."
        )

        rag_data = eval_data.get("rag_vs_norag", {})
        if rag_data:
            col1, col2, col3 = st.columns(3)
            colours = {"No-RAG": "🔴", "Random-Context": "🟡", "RAG": "🟢"}
            for col, (strategy, metrics) in zip([col1, col2, col3], rag_data.items()):
                with col:
                    khr = metrics.get("avg_keyword_hit_rate", 0)
                    faith = metrics.get("avg_faithfulness", 0)
                    hall = metrics.get("hallucination_count", 0)
                    emoji = colours.get(strategy, "⚪")
                    col.markdown(f"**{emoji} {strategy}**")
                    col.metric("Keyword Hit Rate", f"{khr:.3f}")
                    col.metric("Faithfulness", f"{faith:.3f}")
                    col.metric("Hallucinations", f"{hall} / 5")

        # ── Experiment 3: Full Benchmark ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### Experiment 3 — Full Embedding Benchmark")
        st.markdown(
            "P₁ (Precision at rank 1), MRR, NDCG₃, Separability Gap, and Query Latency "
            "for Word2Vec, TF-IDF, and MiniLM-L6-v2."
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


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — SYSTEM INTERNALS
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("System Internals — How the Pipeline Works")

    st.markdown("### RAG Pipeline Architecture")
    st.markdown("""
```
PHASE 1 - INDEXING (run once per document)
----------------------------------------------------------
  PDF Upload
    --> Text Extraction  (pdfplumber + table support)
         --> Text Chunking  (200 tokens, 40-token overlap)
              --> Embedding  (all-MiniLM-L6-v2)
                   --> ChromaDB Vector Store (cosine ANN)

PHASE 2 - QUERYING (run on every user question)
----------------------------------------------------------
  User Question
    --> Query Embedding  (same model as documents)
         --> Cosine Search  (top-K relevant chunks)
              --> Ranked Document Excerpts
                   --> Grounded Response + Page Citations
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
        st.markdown("#### 2. Semantic Retrieval")
        st.markdown("""
- all-MiniLM-L6-v2 embeddings
- Transformer encoder architecture
- 384-dimensional dense vectors
- Cosine similarity ranking
- Grounded document excerpts
- Fully local operation
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
