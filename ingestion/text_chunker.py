"""
Text Chunking — splits financial document pages into overlapping chunks.

Uses LangChain's RecursiveCharacterTextSplitter which tries paragraph
breaks first, then sentences, then words — best all-around for financial text.
"""

from typing import List, Dict

import tiktoken
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, TOKENIZER_ENCODING


class FinancialTextChunker:
    """Splits page dicts into overlapping token-counted chunks with metadata."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        tokenizer_encoding: str = TOKENIZER_ENCODING,
    ):
        self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Takes list of page dicts from FinancialPDFLoader.
        Returns list of chunk dicts:
          {text, metadata: {source, page_number, chunk_index, chunk_id, token_count}}
        """
        all_chunks: List[Dict] = []
        for page in pages:
            raw_chunks = self.splitter.split_text(page["text"])
            for i, chunk_text in enumerate(raw_chunks):
                if len(chunk_text.strip()) < 50:
                    continue
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": page["source"],
                        "page_number": page["page_number"],
                        "chunk_index": i,
                        "chunk_id": f"{page['source']}_p{page['page_number']}_c{i}",
                        "token_count": self._count_tokens(chunk_text),
                    },
                })
        print(f"[chunker] Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """Summary statistics on the chunk set."""
        if not chunks:
            return {}
        token_counts = [c["metadata"]["token_count"] for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_tokens": round(sum(token_counts) / len(token_counts)),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
