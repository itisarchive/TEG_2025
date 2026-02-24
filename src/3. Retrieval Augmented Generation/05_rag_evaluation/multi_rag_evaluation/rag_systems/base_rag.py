"""
🧱 Abstract Base Class for RAG Implementations
===============================================

Every RAG strategy in this comparative evaluation (naive, metadata filtering,
hybrid search, query expansion, reranking) inherits from BaseRAG.  The base
class enforces a consistent two-phase lifecycle:

1. **build()** — initialize embeddings, vector store, retriever and chain.
2. **query()** — run a question through the built chain and return both
   the generated answer and the raw retrieved context strings.
"""

from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore

from ..config.settings import PipelineSettings


class BaseRAG(ABC):
    """Abstract base class for all RAG implementations."""

    def __init__(self, document_chunks: list[Document], pipeline_settings: PipelineSettings) -> None:
        self.document_chunks = document_chunks
        self.pipeline_settings = pipeline_settings
        self.rag_chain: Runnable | None = None
        self.retriever = None
        self.vector_store: InMemoryVectorStore | None = None

    @abstractmethod
    def build(self) -> None:
        """Build the RAG system components (embeddings, retriever, chain)."""

    def query(self, question: str) -> tuple[str, list[str]]:
        """Query the RAG system and return (answer, list_of_context_strings)."""
        if not self.rag_chain or not self.retriever:
            raise RuntimeError(f"{self.name} not built. Call build() first.")

        answer = self.rag_chain.invoke(question)

        if callable(self.retriever):
            retrieved_documents = self.retriever(question)
        else:
            retrieved_documents = self.retriever.invoke(question)

        context_texts = [document.page_content for document in retrieved_documents]
        return answer, context_texts

    @property
    @abstractmethod
    def name(self) -> str:
        """Descriptive display name for this RAG strategy."""
