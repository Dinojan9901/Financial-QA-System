# Project Documentation

This folder is the starting point for understanding the Financial Q&A System.
The application source remains grouped by responsibility in the project root.

## Recommended Reading Order

1. [Project Overview](PROJECT_OVERVIEW.md) - purpose, features, and technologies.
2. [Folder Structure](FOLDER_STRUCTURE.md) - where each file belongs and why.
3. [System Workflow](SYSTEM_WORKFLOW.md) - PDF ingestion and question-answering flow.
4. [Configuration Guide](CONFIGURATION_GUIDE.md) - local embeddings, environment variables, and storage.
5. [Project Summary](PROJECT_SUMMARY.md) - deliverables, experiments, and results.
6. [Final Checklist](FINAL_CHECKLIST.txt) - submission and demonstration checks.

## Related Deliverables

- Main setup instructions: [`../README.md`](../README.md)
- Technical report: [`report/main.pdf`](report/main.pdf)
- Report source: [`report/main.tex`](report/main.tex)
- Presentation: [`presentation/Financial_QA_Presentation.pptx`](presentation/Financial_QA_Presentation.pptx)
- Speaker script: [`presentation/SPEAKER_SCRIPT.md`](presentation/SPEAKER_SCRIPT.md)
- Walkthrough notebook: [`../notebooks/financial_qa_walkthrough.ipynb`](../notebooks/financial_qa_walkthrough.ipynb)

## Quick Commands

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

For the REST API:

```powershell
uvicorn main:app --reload --port 8000
```
