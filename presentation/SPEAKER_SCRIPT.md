# Video Presentation Speaker Script (5 Minutes)

**Project:** AI-Powered Financial Document Q&A System
**Course:** EC7203 Advanced Artificial Intelligence
**Format:** 11 slides + live demo, 5:00 total

**Meets the requirement:** ≤ 5 minutes · covers **problem statement, approach, key
results, and demonstration** · **all 4 team members speak** with equal time · upload
the recording to YouTube/Drive and paste the shareable link in the report and below.

🔗 **Video link:** _<paste shareable link here after upload>_

---

## Speaking Roles (all 4 members — equal contribution)

Each speaker presents the part that matches the work they actually did, so the
narration is authentic.

| # | Speaker | Slides | Covers (requirement) | Built / responsible for | Time |
|---|---------|--------|----------------------|--------------------------|------|
| 1 | **Senevirathna P.U.S.** (EG/2021/4805) | 1–3 | **Problem statement** + solution | NLP preprocessing, chunking, prompt engineering | ~1:10 |
| 2 | **Dinojan V.** (EG/2021/4487) | 4–5 | **Approach** (AI techniques + architecture) | Core RAG pipeline, embeddings, ChromaDB, retrieval, deployment | ~1:05 |
| 3 | **Jackshan Venujan G.S.** (EG/2021/4566) | 6–7 | **Demonstration** (live app) | LLM integration, answer generation, web app UI | ~1:25 |
| 4 | **Tharshihan R.** (EG/2021/4825) | 8–11 | **Key results** + conclusion | Evaluation framework, baseline/benchmark experiments, charts | ~1:20 |

> Hand-off line between speakers: *"…and now <Name> will take you through <next part>."*

---

## Time Budget

| Slide | Content | Speaker | Duration | Cumulative |
|---|---|---|---|---|
| 1 | Title | Senevirathna | 0:15 | 0:15 |
| 2 | Problem & Motivation | Senevirathna | 0:30 | 0:45 |
| 3 | Proposed Solution (RAG) | Senevirathna | 0:25 | 1:10 |
| 4 | Three AI Techniques | Dinojan | 0:35 | 1:45 |
| 5 | System Architecture | Dinojan | 0:30 | 2:15 |
| 6 | Implementation Stack | Jackshan | 0:25 | 2:40 |
| 7 | **LIVE DEMO** | Jackshan | 1:00 | 3:40 |
| 8 | Experiment 1: Baselines | Tharshihan | 0:25 | 4:05 |
| 9 | Experiment 2: RAG vs No-RAG | Tharshihan | 0:25 | 4:30 |
| 10 | Conclusions & Future Work | Tharshihan | 0:20 | 4:50 |
| 11 | Thank You | Tharshihan | 0:10 | 5:00 |

**Tip:** Use a slide timer overlay. Practice 2–3 times before recording.

---

## Slide-by-Slide Script

### Slide 1 — Title (0:15) — *Senevirathna*

> "Hello and welcome to our final project for EC7203 Advanced Artificial Intelligence.
> We are presenting an AI-Powered Financial Document Q&A System using Retrieval-Augmented
> Generation, or RAG. I'm Senevirathna, and I'll be joined by Dinojan, Jackshan, and Tharshihan."

### Slide 2 — Problem & Motivation (0:30) — *Senevirathna*

> "Financial analysts routinely deal with 300-page 10-K filings. Manual search is slow.
> Keyword tools miss paraphrased content. And sending the document to a raw LLM like GPT-4
> costs five to fifteen dollars per question — and the LLM often hallucinates financial
> figures that don't exist in the document.
> Our domain — Banking and Finance — is one of the suggested industries in the course guidelines."

### Slide 3 — Proposed Solution (0:25) — *Senevirathna*

> "Our solution combines retrieval with generation. The system has three key advantages:
> it's accurate because answers come only from the document; it's cited with page numbers
> for verification; and it's efficient because only the most relevant excerpts are sent to
> the LLM — about ten times cheaper than sending the full document.
> Now Dinojan will explain how we built it."

### Slide 4 — Three AI Techniques (0:35) — *Dinojan*

> "The course required a minimum of three AI techniques. We applied:
> One: NLP — including text preprocessing, Word2Vec, TF-IDF, and Sentence-Transformers.
> Two: Large Language Models — Groq's Llama 3.3 70B, free, or OpenAI GPT-4o-mini, through an OpenAI-compatible API.
> Three: Prompt Engineering — using systematic prompt design, chain-of-thought reasoning,
> and few-shot in-context learning."

### Slide 5 — System Architecture (0:30) — *Dinojan*

> "The system runs in two phases.
> Phase one — indexing — runs once per document: we extract text with pdfplumber, chunk
> it into 800-token blocks, embed each chunk with MiniLM-L6-v2, and store them in ChromaDB.
> Phase two — querying — runs on every user question: we embed the question, find the
> top-K similar chunks via cosine search, and pass them to the LLM with our engineered prompt.
> Jackshan will now show this working live."

### Slide 6 — Implementation Stack (0:25) — *Jackshan*

> "The system is built in Python across five layers — from PDF preprocessing at the bottom,
> through embedding and retrieval, up to the Streamlit web UI and FastAPI service at the top.
> The codebase is fully modular, with three evaluation experiments run on a real public
> dataset — financial-qa-10K, derived from genuine SEC 10-K filings."

### Slide 7 — LIVE DEMO (1:00) — *Jackshan*

> "Let me switch to the live demo." [SWITCH TO BROWSER — the deployed Streamlit app]
>
> [1] "I'll upload Apple's 2023 10-K filing... and click Index."
> [2] "Watch as the system extracts pages, chunks them, generates embeddings, and stores them."
> [3] "Now I'll ask: 'What was the total net revenue in fiscal 2023?'"
> [4] "Here's the answer — 383.3 billion dollars — with a citation to page 23 at 94% relevance."
> [5] "Let me ask about risk factors..." [show multi-part answer with multiple citations]
> "Notice every answer is grounded in the document, with page numbers we can verify.
> Tharshihan will now share how well it performs."

[SWITCH BACK TO SLIDES]

### Slide 8 — Experiment 1: Baselines (0:25) — *Tharshihan*

> "We benchmarked three embedding methods on 50 real SEC 10-K question-answer pairs,
> measuring whether each retrieves the exact source passage.
> TF-IDF is a strong lexical baseline at Hit@1 of 0.940.
> Word2Vec is weakest at 0.640 — limited by its small in-corpus vocabulary.
> Our proposed MiniLM-L6-v2 retrieves the correct passage first for every query —
> Hit@1 and MRR are both a perfect 1.000."

### Slide 9 — Experiment 2: RAG vs No-RAG (0:25) — *Tharshihan*

> "We then compared three answer strategies by Keyword Hit Rate.
> No-RAG — the LLM with no document — recovered under one percent of the expected keywords.
> Random Context — irrelevant chunks — managed only thirteen percent.
> RAG — our system — recovered eighty-four percent. This proves that relevant retrieved
> context, not just any context, is what drives answer quality."

### Slide 10 — Conclusions & Future Work (0:20) — *Tharshihan*

> "We delivered a complete RAG pipeline satisfying all three required AI techniques, with
> perfect retrieval — Hit@1 of 1.000 — and an 84 percent keyword hit rate on real filings,
> deployed live on Streamlit Cloud. Future work includes hybrid dense-plus-BM25 retrieval,
> FinBERT for financial domain embeddings, and cross-document queries comparing companies."

### Slide 11 — Thank You (0:10) — *Tharshihan*

> "Thank you for watching. We've submitted the final report, working web app, full source
> code, and this video. Happy to answer any questions."

---

## Recording Checklist

- [ ] Test the Streamlit app starts cleanly: `py -3.13 -m streamlit run app.py`
- [ ] Have a sample financial PDF ready (Apple 10-K from SEC EDGAR)
- [ ] Pre-index the PDF before recording (saves time during demo)
- [ ] Close all other browser tabs and notifications
- [ ] Use OBS Studio or Zoom recording at 1080p
- [ ] **All 4 members record their own segment** (above) — equal participation
- [ ] Aim for 4:50 total — leaves 10s buffer for transitions
- [ ] Re-record any section that runs over time
- [ ] Export as MP4, upload to YouTube/Drive, paste the **shareable link** at the top of this file and in the report
