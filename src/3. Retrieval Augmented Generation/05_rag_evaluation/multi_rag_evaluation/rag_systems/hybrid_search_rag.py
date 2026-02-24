"""
🔀 Hybrid Search RAG — BM25 Keyword Search + Vector Similarity
===============================================================

Combines two complementary retrieval signals:

• **BM25 (keyword match)** — excels at finding exact terms and names.
• **Vector similarity (semantic match)** — excels at paraphrased queries
  and conceptual similarity.

Scores from both methods are normalized to [0, 1] and combined with a
configurable weight (default 0.3 for BM25, 0.7 for vector).  The fused
ranking often outperforms either method alone.
"""

import textwrap

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from rank_bm25 import BM25Okapi

from .base_rag import BaseRAG
from ..config.settings import PipelineSettings

RAG_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {question}

    Context: {context}

    Answer:""")


class HybridSearchRAG(BaseRAG):
    """RAG combining BM25 keyword search with vector similarity for fused ranking."""

    def __init__(self, document_chunks: list[Document], pipeline_settings: PipelineSettings) -> None:
        super().__init__(document_chunks, pipeline_settings)
        self.bm25_index: BM25Okapi | None = None

    @property
    def name(self) -> str:
        return "Hybrid Search RAG"

    def _build_bm25_index(self) -> BM25Okapi:
        """Tokenize all chunks and build a BM25Okapi index."""
        tokenized_chunks = [
            chunk.page_content.lower().split()
            for chunk in self.document_chunks
        ]
        return BM25Okapi(tokenized_chunks)

    def _bm25_search(self, user_query: str, *, result_count: int = 5) -> list[tuple[Document, float]]:
        """Return (document, bm25_score) pairs with score > 0."""
        query_tokens = user_query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        top_indices = np.argsort(bm25_scores)[::-1][:result_count]
        return [
            (self.document_chunks[idx], bm25_scores[idx])
            for idx in top_indices
            if bm25_scores[idx] > 0
        ]

    def _vector_search(self, user_query: str, *, result_count: int = 5) -> list[tuple[Document, float]]:
        """Return (document, similarity_score) pairs from the vector store."""
        return self.vector_store.similarity_search_with_score(user_query, k=result_count)

    @staticmethod
    def _normalize_scores(scored_documents: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        """Min-max normalize scores to [0, 1]."""
        if not scored_documents:
            return []
        scores = [score for _, score in scored_documents]
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [(document, 1.0) for document, _ in scored_documents]
        score_range = max_score - min_score
        return [
            (document, (score - min_score) / score_range)
            for document, score in scored_documents
        ]

    def _hybrid_search(self, user_query: str, *, result_count: int = 3) -> list[Document]:
        """Fuse BM25 and vector results using weighted normalized scores."""
        broad_count = result_count * 2

        normalised_bm25 = self._normalize_scores(
            self._bm25_search(user_query, result_count=broad_count)
        )
        normalised_vector = self._normalize_scores(
            self._vector_search(user_query, result_count=broad_count)
        )

        bm25_weight = self.pipeline_settings.retrieval.bm25_weight
        vector_weight = 1.0 - bm25_weight

        combined_scores: dict[int, float] = {}
        document_by_id: dict[int, Document] = {}

        for document, score in normalised_bm25:
            doc_id = id(document)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score * bm25_weight
            document_by_id[doc_id] = document

        for document, score in normalised_vector:
            doc_id = id(document)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score * vector_weight
            document_by_id[doc_id] = document

        sorted_doc_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)

        return [
            document_by_id[doc_id]
            for doc_id in sorted_doc_ids[:result_count]
            if doc_id in document_by_id
        ]

    def build(self) -> None:
        """Build BM25 index, vector store and the hybrid retrieval chain."""
        embeddings = AzureOpenAIEmbeddings(model=self.pipeline_settings.models.embedding_model)
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(documents=self.document_chunks)

        self.bm25_index = self._build_bm25_index()

        retrieval_top_k = self.pipeline_settings.retrieval.top_k

        def hybrid_retriever(query: str) -> list[Document]:
            return self._hybrid_search(query, result_count=retrieval_top_k)

        self.retriever = hybrid_retriever

        rag_llm = AzureChatOpenAI(model=self.pipeline_settings.models.rag_chat_model)
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | rag_llm
                | StrOutputParser()
        )
