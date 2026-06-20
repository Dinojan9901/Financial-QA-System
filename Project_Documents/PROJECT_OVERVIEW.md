# Project Overview

## Purpose

The Financial Document Q&A System lets users upload financial PDFs and ask
natural-language questions about them. It uses semantic retrieval to return
the most relevant grounded passages with document and page citations.

## Main Capabilities

- Extract page text and tables from financial PDFs.
- Split documents into overlapping, token-aware chunks.
- Create free local embeddings with `all-MiniLM-L6-v2`.
- Store and search vectors locally with ChromaDB.
- Filter retrieval to a selected document.
- Return grounded excerpts without an external API.
- Run through Streamlit, FastAPI, or direct Python calls.
- Compare retrieval approaches with repeatable evaluations.

## Technology Map

| Area | Technology |
|---|---|
| Language | Python |
| Web UI | Streamlit |
| REST API | FastAPI |
| PDF extraction | pdfplumber and PyPDF2 |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | all-MiniLM-L6-v2 through Sentence Transformers |
| Vector database | ChromaDB with cosine similarity |
| Testing | pytest |

## Operating Mode

The system is fully local. `all-MiniLM-L6-v2` produces 384-dimensional
embeddings for document chunks and questions. ChromaDB ranks the chunks, and
`LocalQAChain` returns the strongest excerpt with source information. No API
key is required.
