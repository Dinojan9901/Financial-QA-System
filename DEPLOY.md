# Deploying to Railway (Free, ~10 minutes)

This guide deploys the **Streamlit web app** to a public URL on
[Railway](https://railway.app). Railway auto-detects the `Dockerfile` and runs
the app on its platform-provided `$PORT`.

> The app runs in **free local mode** by default (MiniLM embeddings, retrieval
> only). Add a free `GROQ_API_KEY` to enable real LLM-generated answers.

---

## What's already configured for you

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the app; binds Streamlit to Railway's `$PORT` |
| `railway.json` | Start command, health check (`/_stcore/health`), restart policy |
| `.dockerignore` | Keeps the image lean (excludes venv, PDFs, local DB, caches) |

You don't need to edit any of these.

---

## Step 1 — Push the project to GitHub

If the repo isn't on GitHub yet:

```bash
cd financial-qa-system
git init
git add .
git commit -m "Financial Document Q&A System — Railway ready"
# create an empty repo on github.com first, then:
git remote add origin https://github.com/<your-username>/financial-qa-system.git
git branch -M main
git push -u origin main
```

> ✅ `.env` is gitignored, so your `GROQ_API_KEY` is **not** pushed. You'll add
> it as a Railway environment variable instead (Step 3).

---

## Step 2 — Create the Railway project

1. Go to **https://railway.app** and sign in with GitHub (free).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your `financial-qa-system` repository.
4. Railway detects the `Dockerfile` and starts the first build automatically.
   (The first build takes ~3–6 min — it installs PyTorch and friends.)

---

## Step 3 — Add your environment variables

In the Railway project → **Variables** tab, add:

| Variable | Value | Notes |
|----------|-------|-------|
| `GROQ_API_KEY` | `gsk_...` | Free key from console.groq.com/keys — enables real LLM answers |
| `USE_LOCAL_MODELS` | `true` | Keep embeddings on free local MiniLM |
| `LLM_PROVIDER` | `auto` | Uses Groq when the key is present |

Click **Deploy** (or it redeploys automatically when variables change).

> Without `GROQ_API_KEY` the app still works — it returns the most relevant
> retrieved passage instead of an LLM-written answer.

---

## Step 4 — Generate a public URL

1. Go to **Settings** → **Networking** → **Generate Domain**.
2. Railway gives you a URL like `https://financial-qa-system-production.up.railway.app`.
3. Open it — the Streamlit app loads. 🎉

Put this URL in your report and video as the **live deployment**.

---

## Step 5 — Verify it works

1. Open the URL.
2. In the sidebar, confirm the generation badge (🟢 Groq if you added the key).
3. Upload a financial PDF → **Index** → ask a question.
4. Check the **Evaluation Results** tab shows the real-dataset metrics.

---

## Important notes

- **Ephemeral storage:** Railway's filesystem resets on each redeploy/restart,
  so uploaded PDFs and the ChromaDB index do **not** persist across deploys.
  This is fine for a live demo (upload + query in one session). For permanent
  storage, attach a Railway **Volume** mounted at `/app/data`, or swap ChromaDB
  for a hosted vector DB (e.g. Pinecone) as described in the project guide.
- **Cold start:** the MiniLM model (~90 MB) downloads on the first question of a
  fresh container, so the first answer may take ~20–30 s. Subsequent queries are
  fast.
- **Memory:** PyTorch + sentence-transformers need ~1 GB RAM. Railway's Hobby
  plan is comfortable; if a build/run is killed for memory, upgrade the plan or
  reduce concurrency.
- **Cost:** Railway's trial credit covers a demo; the Hobby plan is low-cost.
  Groq generation stays free.

---

## Optional — deploy the REST API instead of the web app

The same image can serve the FastAPI backend. Override the start command in
`railway.json` (or Railway → Settings → Deploy → Custom Start Command):

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Then the interactive API docs are live at `https://<your-domain>/docs`.

---

## Test the container locally first (optional)

If you have Docker installed:

```bash
cd financial-qa-system
docker build -t financial-qa .
docker run -p 8501:8501 -e GROQ_API_KEY=gsk_... financial-qa
# open http://localhost:8501
```
