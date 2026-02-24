"""
🏆 Reranking RAG — Post-Retrieval Relevance Re-scoring
=======================================================

Retrieves more documents than needed (rerank_initial_top_k) and then
re-scores each one against the query to keep only the most relevant
(rerank_final_top_k).  Two reranking backends are supported:

• **Cross-encoder** (preferred) — a lightweight `ms-marco-MiniLM-L-6-v2`
  model that jointly encodes the query-document pair for accurate relevance
  scoring.
• **LLM-based scoring** (fallback) — asks an LLM to rate relevance 1-10
  when the cross-encoder is not available.
"""

import re
import textwrap

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from sentence_transformers import CrossEncoder

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

RELEVANCE_SCORING_PROMPT_TEMPLATE = textwrap.dedent("""\
    Rate the relevance of the following document to the query on a scale of 1-10.
    Consider how well the document answers the question or provides relevant information.

    Query: {query}

    Document: {document}

    Provide only a numeric score (1-10):""")

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM_FALLBACK_SCORE = 5.0
MAX_DOCUMENT_CHARS_FOR_LLM_SCORING = 500


class RerankingRAG(BaseRAG):
    """RAG with post-retrieval reranking via cross-encoder or LLM fallback."""

    def __init__(self, document_chunks: list[Document], pipeline_settings: PipelineSettings) -> None:
        super().__init__(document_chunks, pipeline_settings)
        self.cross_encoder_model: CrossEncoder | None = None

    @property
    def name(self) -> str:
        return "Reranking RAG"

    @staticmethod
    def _load_cross_encoder() -> CrossEncoder | None:
        """Attempt to load the cross-encoder model; return None on failure."""
        try:
            return CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        except Exception as load_error:
            print(f"Warning: Failed to load cross-encoder: {load_error}")
            return None

    def _cross_encoder_rerank(
            self,
            user_query: str,
            candidate_documents: list[Document],
            *,
            final_count: int = 3,
    ) -> list[Document]:
        """Re-rank candidates using the cross-encoder relevance model."""
        if self.cross_encoder_model is None:
            return candidate_documents[:final_count]

        query_document_pairs = [
            (user_query, document.page_content)
            for document in candidate_documents
        ]
        relevance_scores = self.cross_encoder_model.predict(query_document_pairs)

        scored_pairs = sorted(
            zip(candidate_documents, relevance_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [document for document, _score in scored_pairs[:final_count]]

    def _llm_relevance_rerank(
            self,
            user_query: str,
            candidate_documents: list[Document],
            *,
            final_count: int = 3,
    ) -> list[Document]:
        """Fallback: use an LLM to score each document's relevance 1-10."""
        scoring_prompt = ChatPromptTemplate.from_template(RELEVANCE_SCORING_PROMPT_TEMPLATE)
        scoring_llm = AzureChatOpenAI(
            model=self.pipeline_settings.models.rag_chat_model,
            temperature=0,
        )
        scoring_chain = scoring_prompt | scoring_llm | StrOutputParser()

        scored_documents: list[tuple[Document, float]] = []

        for document in candidate_documents:
            try:
                raw_score_text = scoring_chain.invoke({
                    "query": user_query,
                    "document": document.page_content[:MAX_DOCUMENT_CHARS_FOR_LLM_SCORING],
                })
                score_match = re.search(r"\b(\d+(?:\.\d+)?)\b", raw_score_text)
                relevance_score = float(score_match.group(1)) if score_match else DEFAULT_LLM_FALLBACK_SCORE
            except (ValueError, RuntimeError, AttributeError):
                relevance_score = DEFAULT_LLM_FALLBACK_SCORE

            scored_documents.append((document, relevance_score))

        scored_documents.sort(key=lambda pair: pair[1], reverse=True)
        return [document for document, _score in scored_documents[:final_count]]

    def _rerank_documents(
            self,
            user_query: str,
            candidate_documents: list[Document],
            *,
            final_count: int = 3,
    ) -> list[Document]:
        """Dispatch to cross-encoder or LLM-based reranking."""
        if self.cross_encoder_model is not None:
            return self._cross_encoder_rerank(user_query, candidate_documents, final_count=final_count)
        return self._llm_relevance_rerank(user_query, candidate_documents, final_count=final_count)

    def build(self) -> None:
        """Load cross-encoder, build vector store, and wire the reranking chain."""
        self.cross_encoder_model = self._load_cross_encoder()

        embeddings = AzureOpenAIEmbeddings(model=self.pipeline_settings.models.embedding_model)
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(documents=self.document_chunks)

        initial_retrieval_count = self.pipeline_settings.retrieval.rerank_initial_top_k
        final_retrieval_count = self.pipeline_settings.retrieval.rerank_final_top_k

        def reranking_retriever(query: str) -> list[Document]:
            initial_documents = self.vector_store.similarity_search(query, k=initial_retrieval_count)
            return self._rerank_documents(query, initial_documents, final_count=final_retrieval_count)

        self.retriever = reranking_retriever

        rag_llm = AzureChatOpenAI(model=self.pipeline_settings.models.rag_chat_model)
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | rag_llm
                | StrOutputParser()
        )
