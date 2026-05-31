"""
PDF Ingestion — extracts and cleans text from financial PDFs.

Uses pdfplumber (preferred for financial docs with tables) with a PyPDF2
fallback for simple layouts.
"""

import re
from pathlib import Path
from typing import List, Dict

import pdfplumber
import PyPDF2


class FinancialPDFLoader:
    """Loads financial PDFs and extracts clean text page-by-page."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_text(self) -> List[Dict]:
        """
        Returns a list of page dicts:
          {page_number, text, tables, source}
        """
        pages = self._extract_with_pdfplumber()
        if not pages:
            pages = self._extract_with_pypdf2()
        return pages

    def get_document_info(self) -> Dict:
        """Basic metadata about the PDF."""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                return {
                    "file_name": self.file_path.name,
                    "total_pages": len(pdf.pages),
                    "file_size_kb": round(self.file_path.stat().st_size / 1024, 2),
                }
        except Exception:
            return {"file_name": self.file_path.name, "total_pages": 0, "file_size_kb": 0}

    # ── Extraction strategies ─────────────────────────────────────────────────

    def _extract_with_pdfplumber(self) -> List[Dict]:
        """Primary: pdfplumber handles tables and complex layouts well."""
        pages = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    raw_text = page.extract_text() or ""
                    tables = page.extract_tables() or []
                    table_text = self._tables_to_text(tables)
                    full_text = raw_text + ("\n" + table_text if table_text else "")
                    clean = self._clean_text(full_text)
                    if clean.strip():
                        pages.append({
                            "page_number": i + 1,
                            "text": clean,
                            "tables": tables,
                            "source": self.file_path.name,
                        })
        except Exception as e:
            print(f"[pdfplumber] failed: {e}; trying PyPDF2 fallback")
        return pages

    def _extract_with_pypdf2(self) -> List[Dict]:
        """Fallback: PyPDF2 for simple, well-structured PDFs."""
        pages = []
        try:
            with open(self.file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    clean = self._clean_text(raw_text)
                    if clean.strip():
                        pages.append({
                            "page_number": i + 1,
                            "text": clean,
                            "tables": [],
                            "source": self.file_path.name,
                        })
        except Exception as e:
            print(f"[PyPDF2] failed: {e}")
        return pages

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tables_to_text(self, tables: list) -> str:
        """Convert extracted table rows to pipe-separated readable text."""
        lines = []
        for table in tables:
            for row in table:
                row_text = " | ".join(str(cell) for cell in row if cell is not None)
                if row_text.strip():
                    lines.append(row_text)
        return "\n".join(lines)

    def _clean_text(self, text: str) -> str:
        """Remove common PDF noise: excess whitespace, page markers, ligatures."""
        text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank lines
        text = re.sub(r" {3,}", " ", text)           # collapse spaces
        text = re.sub(r"Page \d+ of \d+", "", text)  # remove "Page X of Y"
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # lone page numbers
        # Fix common PDF ligature artifacts
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        return text.strip()
