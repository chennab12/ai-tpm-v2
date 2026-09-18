"""
TF-IDF vector index with cosine-similarity retrieval.

Pure numpy implementation so RAG works offline without extra heavy
dependencies. Chunks are sentences/paragraphs from the knowledge base.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import DATA_DIR


@dataclass
class Chunk:
    """A retrievable text chunk with source metadata."""

    text: str
    source: str = ""
    title: str = ""
    doc_id: int = 0


@dataclass
class RetrievalResult:
    """A ranked retrieval hit."""

    chunk: Chunk
    score: float
    rank: int


@dataclass
class CorpusIndex:
    """Stores chunks + TF-IDF vectors and supports cosine retrieval."""

    chunks: list[Chunk] = field(default_factory=list)
    _vocab: dict[str, int] = field(default_factory=dict)
    _vectors: np.ndarray | None = None
    _doc_norms: np.ndarray | None = None

    # --------------------------------------------------------
    # Build / index
    # --------------------------------------------------------

    def add_documents(self, documents: Iterable[tuple[str, str, str]]):
        """Add (title, source_path, text) documents, chunking each."""
        for title, source, text in documents:
            parsed = _chunk_markdown(text, title=title, source=source)
            start = len(self.chunks)
            for i, chunk_text in enumerate(parsed):
                self.chunks.append(
                    Chunk(text=chunk_text, source=source, title=title, doc_id=start + i)
                )
        self._build_index()
        return self

    def _build_index(self):
        """Compute TF-IDF vectors for all chunks."""
        tokenized = [_tokenize(c.text) for c in self.chunks]
        vocab: dict[str, int] = {}
        for toks in tokenized:
            for t in set(toks):
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        n_docs = len(self.chunks)
        df = np.zeros(len(vocab), dtype=float)
        for toks in tokenized:
            for t in set(toks):
                if t in vocab:
                    df[vocab[t]] += 1.0
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        vectors = np.zeros((n_docs, len(vocab)), dtype=np.float32)
        for d, toks in enumerate(tokenized):
            for t in toks:
                vectors[d, vocab[t]] += 1.0
            tf = vectors[d]
            vectors[d] = tf * idf
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0
        self._vectors = vectors
        self._doc_norms = norms

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def build_query_vector(self, query: str) -> np.ndarray:
        """Build a normalized TF-IDF query vector."""
        q_vec = np.zeros(len(self._vocab), dtype=np.float32)
        toks = _tokenize(query)
        for t in toks:
            if t in self._vocab:
                q_vec[self._vocab[t]] += 1.0
        n = np.linalg.norm(q_vec)
        return q_vec / n if n > 0 else q_vec

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Cosine-similarity search, returning ranked results."""
        if not self.chunks or self._vectors is None:
            return []
        q = self.build_query_vector(query)
        scores = self._vectors @ q / self._doc_norms
        order = np.argsort(-scores)
        results = []
        for rank, idx in enumerate(order[:top_k]):
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(RetrievalResult(chunk=self.chunks[idx], score=score, rank=rank))
        return results

    @property
    def size(self) -> int:
        return len(self.chunks)


# ============================================================
# Chunking + tokenizing helpers
# ============================================================


def _chunk_markdown(text: str, title: str = "", source: str = "") -> list[str]:
    """Split a markdown document into chunks by headings/paragraphs."""
    chunks: list[str] = []
    current_title = title
    buffer: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            if buffer:
                pieces = _split_sentences(" ".join(buffer).strip())
                chunks.extend(p for p in pieces if len(p) > 40)
            if s.startswith("##"):
                current_title = s.lstrip("#").strip()
            buffer = []
        elif s:
            buffer.append(s)
    if buffer:
        pieces = _split_sentences(" ".join(buffer).strip())
        chunks.extend(p for p in pieces if len(p) > 40)
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    merged: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) > 400:
            if buf:
                merged.append(buf)
            buf = p
        else:
            buf = (buf + " " + p).strip()
    if buf:
        merged.append(buf)
    return merged


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "up", "down", "so", "than", "can", "could", "will", "would", "should",
    "may", "might", "must", "not", "no", "yes", "do", "does", "did",
    "have", "has", "had", "how", "what", "when", "where", "which", "who",
    "why", "if", "then", "else", "also", "out", "over", "under", "about",
    "into", "per", "via", "when", "your", "you", "has", "more", "most",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


# ============================================================
# Default corpus builder
# ============================================================

DEFAULT_KB_DIR = DATA_DIR / "knowledge_base"


def build_default_corpus(kb_dir: Path | None = None) -> CorpusIndex:
    """Build the corpus from data/knowledge_base/*.md."""
    kb_dir = kb_dir or DEFAULT_KB_DIR
    index = CorpusIndex()
    docs: list[tuple[str, str, str]] = []
    for md in sorted(kb_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        title = md.stem.replace("_", " ").title()
        docs.append((title, str(md), text))
    index.add_documents(docs)
    return index


def retrieve(query: str, top_k: int = 3, index: CorpusIndex | None = None) -> list[RetrievalResult]:
    """Retrieve top-k chunks for a query (creates default index on first use)."""
    if index is None:
        index = _GLOBAL_INDEX
    return index.search(query, top_k=top_k)


_GLOBAL_INDEX: CorpusIndex | None = None


def get_global_index() -> CorpusIndex:
    """Lazily built, cached global index."""
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None or _GLOBAL_INDEX.size == 0:
        _GLOBAL_INDEX = build_default_corpus()
    return _GLOBAL_INDEX