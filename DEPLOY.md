# Deploying the Web App (Free Options)

The demo is a **Streamlit app**. The two genuinely free, no-credit-card ways to
put it online are **Streamlit Community Cloud** (simplest) and **Hugging Face
Spaces** (more RAM). Railway is also covered at the bottom, but note Railway is
now a **paid/trial** service (its "$5.00 left / upgrade" banner means the free
trial credit is limited) — prefer the free options below.

> The app runs in **free local mode** by default (MiniLM embeddings, retrieval
> only). Add a free `GROQ_API_KEY` to enable real LLM-generated answers.

---

## ⭐ Option 1 — Streamlit Community Cloud (recommended, 100% free)

Purpose-built for Streamlit apps. No credit card, no trial limit.

### Step 1 — Push the project to GitHub (public repo)

```bash
cd financial-qa-system
git init
git add .
git commit -m "Financial Document Q&A System"
# create an empty PUBLIC repo on github.com first, then:
git remote add origin https://github.com/<your-username>/financial-qa-system.git
git branch -M main
git push -u origin main
```

> ✅ `.env` is gitignored, so your `GROQ_API_KEY` is **not** pushed — you'll add
> it in the Streamlit Secrets UI instead.

### Step 2 — Create the app

1. Go to **https://share.streamlit.io** and sign in with GitHub (free).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/financial-qa-system`
   - **Branch:** `main`
   - **Main file path:** `app.py`  *(if your repo's root is the parent folder,
     use `financial-qa-system/app.py`)*

### Step 3 — Add your secret (enables free Groq answers)

Before clicking Deploy, open **Advanced settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
USE_LOCAL_MODELS = "true"
LLM_PROVIDER = "auto"
```

(Optional) set **Python version** to 3.12 or 3.13 in Advanced settings.

### Step 4 — Deploy

Click **Deploy**. First build takes ~3–7 min (installs PyTorch). You get a URL
like `https://<your-app>.streamlit.app`. Put this in your report and video.

> **Resource note:** the free tier gives ~1 GB RAM. The MiniLM model is small,
> so it normally fits, but if the app is killed for memory, use **Option 2
> (Hugging Face Spaces)** which offers far more RAM for free.

---

## Option 2 — Hugging Face Spaces (free, more RAM — best for ML apps)

Free "CPU basic" Spaces give **16 GB RAM** — very comfortable for PyTorch.

1. Go to **https://huggingface.co/new-space**, sign in (free).
2. **Space name:** `financial-qa-system` · **SDK:** select **Streamlit** ·
   **Hardware:** CPU basic (free) · Visibility: Public.
3. Push your code to the Space's git repo (it gives you the remote URL), or use
   **Files → Upload** to add the project files. Ensure `app.py` and
   `requirements.txt` are at the Space root.
4. In **Settings → Variables and secrets**, add a secret:
   `GROQ_API_KEY = gsk_...` (and optionally `LLM_PROVIDER=auto`,
   `USE_LOCAL_MODELS=true`).
5. The Space builds and serves the app at
   `https://huggingface.co/spaces/<you>/financial-qa-system`.

> Spaces secrets are exposed as environment variables, which the app reads
> automatically (no code change needed).

---

## Option 3 — Railway (Dockerfile-based; trial/paid)

Railway gives a one-time **$5 trial credit**, then requires a paid plan to keep
services online — so use it only if you specifically want a Docker deploy. The
repo already includes `Dockerfile`, `railway.json`, and `.dockerignore`.

1. Push to GitHub (as in Option 1, Step 1).
2. **railway.app** → New Project → Deploy from GitHub repo → pick your repo
   (it auto-detects the `Dockerfile`).
3. **Variables** tab → add `GROQ_API_KEY`, `USE_LOCAL_MODELS=true`,
   `LLM_PROVIDER=auto`.
4. **Settings → Networking → Generate Domain** → public URL.

---

## Verify any deployment

1. Open the URL.
2. Sidebar shows the generation badge (🟢 Groq when the key is set).
3. Upload a financial PDF → **Index** → ask a question.
4. The **Evaluation Results** tab shows the real-dataset metrics.

---

## Important notes

- **Ephemeral storage:** on all these platforms the filesystem resets on each
  redeploy/restart, so uploaded PDFs and the ChromaDB index do **not** persist
  across deploys. This is fine for a live demo (upload + query in one session).
  For permanent storage, attach a persistent volume or swap ChromaDB for a
  hosted vector DB (e.g. Pinecone) as described in the project guide.
- **Cold start:** the MiniLM model (~90 MB) downloads on the first question of a
  fresh container, so the first answer may take ~20–30 s. Subsequent queries are
  fast.
- **Memory:** PyTorch + sentence-transformers need ~1 GB RAM. Hugging Face Spaces
  (16 GB free) is the most comfortable; Streamlit Community Cloud (~1 GB) usually
  fits the small MiniLM model but is tighter.
- **Cost:** Streamlit Community Cloud and Hugging Face Spaces are free with no
  card. Groq generation stays free. Railway is trial/paid.

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
