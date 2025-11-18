"""
RAG (Retrieval-Augmented Generation) Package

This package provides a complete RAG system for medical documents.

Main components:
- CorpusRetriever: Main class for document retrieval
- load_corpus: Load PDFs with YAML metadata
- get_gemini_embedder: Initialize Gemini embeddings
- format_docs_with_citations: Format results with citations
"""

from .retriever import CorpusRetriever
from .loader import load_corpus
from .embedder import get_gemini_embedder
from .postprocessor import (
    reorder_documents,
    format_docs_with_citations,
    extract_metadata_field
)

__all__ = [
    "CorpusRetriever",
    "load_corpus",
    "get_gemini_embedder",
    "reorder_documents",
    "format_docs_with_citations",
    "extract_metadata_field",
]