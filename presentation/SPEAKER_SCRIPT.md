# Video Presentation Speaker Script (5 Minutes)

**Project:** AI-Powered Financial Document Q&A System
**Course:** EC7203 Advanced Artificial Intelligence
**Format:** 11 slides + live demo, 5:00 total

---

## Time Budget

| Slide | Content | Duration | Cumulative |
|---|---|---|---|
| 1 | Title | 0:15 | 0:15 |
| 2 | Problem & Motivation | 0:30 | 0:45 |
| 3 | Proposed Solution (RAG) | 0:25 | 1:10 |
| 4 | Three AI Techniques | 0:35 | 1:45 |
| 5 | System Architecture | 0:30 | 2:15 |
| 6 | Implementation Stack | 0:25 | 2:40 |
| 7 | **LIVE DEMO** | 1:00 | 3:40 |
| 8 | Experiment 1: Baselines | 0:25 | 4:05 |
| 9 | Experiment 2: RAG vs No-RAG | 0:25 | 4:30 |
| 10 | Conclusions & Future Work | 0:20 | 4:50 |
| 11 | Thank You | 0:10 | 5:00 |

**Tip:** Use a slide timer overlay. Practice 2–3 times before recording.

---

## Slide-by-Slide Script

### Slide 1 — Title (0:15)

> "Hello and welcome to our final project for EC7203 Advanced Artificial Intelligence.
> We are presenting an AI-Powered Financial Document Q&A System using Retrieval-Augmented
> Generation, or RAG."

### Slide 2 — Problem & Motivation (0:30)

> "Financial analysts routinely deal with 300-page 10-K filings. Manual search is slow.
> Keyword tools miss paraphrased content. And sending the document to a raw LLM like GPT-4
> costs five to fifteen dollars per question — and the LLM often hallucinates financial
> figures that don't exist in the document.
> Our domain — Banking and Finance — is one of the suggested industries in the course guidelines."

### Slide 3 — Proposed Solution (0:25)

> "Our solution combines retrieval with generation. The system has three key advantages:
> it's accurate because answers come only from the document; it's cited with page numbers
> for verification; and it's efficient because only the most relevant excerpts are sent to
> the LLM — about ten times cheaper than sending the full document."

### Slide 4 — Three AI Techniques (0:35)

> "The course required a minimum of three AI techniques. We applied:
> One: NLP — including text preprocessing, Word2Vec, TF-IDF, and Sentence-Transformers.
> Two: Large Language Models — Groq's Llama 3.3 70B, free, or OpenAI GPT-4o-mini, through an OpenAI-compatible API.
> Three: Prompt Engineering — using systematic prompt design, chain-of-thought reasoning,
> and few-shot in-context learning."

### Slide 5 — System Architecture (0:30)

> "The system runs in two phases.
> Phase one — indexing — runs once per document: we extract text with pdfplumber, chunk
> it into 800-token blocks, embed each chunk with MiniLM-L6-v2, and store them in ChromaDB.
> Phase two — querying — runs on every user question: we embed the question, find the
> top-K similar chunks via cosine search, and pass them to the LLM with our engineered prompt."

### Slide 6 — Implementation Stack (0:25)

> "The system is built in Python across five layers — from PDF preprocessing at the bottom,
> through embedding and retrieval, up to the Streamlit web UI and FastAPI service at the top.
> The codebase is fully modular, with three evaluation experiments run on a real public
> dataset — financial-qa-10K, derived from genuine SEC 10-K filings."

### Slide 7 — LIVE DEMO (1:00)

> "Let me switch to the live demo." [SWITCH TO BROWSER — http://localhost:8501]
>
> [1] "I'll upload Apple's 2023 10-K filing... and click Index."
> [2] "Watch as the system extracts pages, chunks them, generates embeddings, and stores them."
> [3] "Now I'll ask: 'What was the total net revenue in fiscal 2023?'"
> [4] "Here's the answer — 383.3 billion dollars — with a citation to page 23 at 94% relevance."
> [5] "Let me ask about risk factors..." [show multi-part answer with multiple citations]
> "Notice every answer is grounded in the document, with page numbers we can verify."

[SWITCH BACK TO SLIDES]

### Slide 8 — Experiment 1: Baselines (0:25)

> "We benchmarked three embedding methods on 50 real SEC 10-K question-answer pairs,
> measuring whether each retrieves the exact source passage.
> TF-IDF is a strong lexical baseline at Hit@1 of 0.940.
> Word2Vec is weakest at 0.640 — limited by its small in-corpus vocabulary.
> Our proposed MiniLM-L6-v2 retrieves the correct passage first for every query —
> Hit@1 and MRR are both a perfect 1.000."

### Slide 9 — Experiment 2: RAG vs No-RAG (0:25)

> "We then compared three answer strategies by Keyword Hit Rate.
> No-RAG — the LLM with no document — recovered under one percent of the expected keywords.
> Random Context — irrelevant chunks — managed only thirteen percent.
> RAG — our system — recovered eighty-four percent. This proves that relevant retrieved
> context, not just any context, is what drives answer quality."

### Slide 10 — Conclusions & Future Work (0:20)

> "We delivered a complete RAG pipeline satisfying all three required AI techniques, with
> perfect retrieval — Hit@1 of 1.000 — and an 84 percent keyword hit rate on real filings,
> deployed live on Streamlit Cloud. Future work includes hybrid dense-plus-BM25 retrieval,
> FinBERT for financial domain embeddings, and cross-document queries comparing companies."

### Slide 11 — Thank You (0:10)

> "Thank you for watching. We've submitted the final report, working web app, full source
> code, and this video. Happy to answer any questions."

---

## Recording Checklist

- [ ] Test the Streamlit app starts cleanly: `py -3.13 -m streamlit run app.py`
- [ ] Have a sample financial PDF ready (Apple 10-K from SEC EDGAR)
- [ ] Pre-index the PDF before recording (saves time during demo)
- [ ] Close all other browser tabs and notifications
- [ ] Use OBS Studio or Zoom recording at 1080p
- [ ] Each team member should narrate at least one slide
- [ ] Aim for 4:50 total — leaves 10s buffer for transitions
- [ ] Re-record any section that runs over time
- [ ] Export as MP4 and upload to YouTube/Drive — share link in report

---

## Team Speaking Roles (4 members)

| Member | Slides |
|---|---|
| Member 1 | Slides 1–3 (Intro + Solution) |
| Member 2 | Slides 4–5 (Techniques + Architecture) |
| Member 3 | Slides 6 + 7 (Stack + Live Demo) |
| Member 4 | Slides 8–11 (Results + Conclusion) |
