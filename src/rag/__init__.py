"""V3 - Retrieval Augmented Generation (RAG) over the project knowledge base."""

from .index import CorpusIndex, RetrievalResult, build_default_corpus, retrieve
from .pipeline import rag_query

__all__ = [
    "CorpusIndex",
    "RetrievalResult",
    "build_default_corpus",
    "retrieve",
    "rag_query",
]