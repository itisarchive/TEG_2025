"""
🏷️ Metadata Filtering RAG — Precision Through Rich Document Metadata
=====================================================================

Enhances each chunk with structured metadata extracted from its content and
filename (scientist name, scientific field, time period).  At query time the
retriever detects the relevant scientific domain from the question and uses
it to filter out irrelevant chunks *before* returning results, improving
precision without sacrificing recall.
"""

import os
import re
import textwrap

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from .base_rag import BaseRAG

RAG_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {question}

    Context: {context}

    Answer:""")

FIELD_KEYWORDS: dict[str, list[str]] = {
    "mathematics": ["mathematician", "algorithm", "analytical", "computation"],
    "physics": ["physicist", "relativity", "Nobel Prize", "photoelectric", "radioactivity"],
    "chemistry": ["chemist", "chemical", "elements", "research"],
    "computer_science": ["computer", "programming", "algorithm", "machine"],
}

QUERY_FIELD_HINTS: dict[str, list[str]] = {
    "physics": ["physics", "relativity", "nobel"],
    "chemistry": ["chemistry", "elements", "radioactive"],
    "computer_science": ["computing", "programming", "algorithm"],
    "mathematics": ["mathematics", "mathematical"],
}


class MetadataFilteringRAG(BaseRAG):
    """RAG with rich metadata extraction and field-aware filtering at retrieval time."""

    @property
    def name(self) -> str:
        return "Metadata Filtering RAG"

    @staticmethod
    def _extract_enhanced_metadata(document: Document) -> Document:
        """Enrich a single document chunk with structured metadata."""
        source_file = document.metadata.get("source", "")
        scientist_name = os.path.basename(source_file).replace(".txt", "")
        content = document.page_content

        enriched_metadata: dict = {
            "scientist_name": scientist_name,
            "content_type": "biography",
            "source_type": "text_file",
            "language": "english",
        }

        year_matches = re.findall(r"\((\d{4})-(\d{4})\)", content)
        if year_matches:
            birth_year, death_year = year_matches[0]
            enriched_metadata.update({
                "birth_year": int(birth_year),
                "death_year": int(death_year),
                "century": f"{birth_year[:2]}th century",
                "time_period": "historical",
            })

        content_lower = content.lower()
        detected_fields = [
            field
            for field, keywords in FIELD_KEYWORDS.items()
            if any(keyword in content_lower for keyword in keywords)
        ]

        enriched_metadata["scientific_fields"] = detected_fields
        enriched_metadata["primary_field"] = detected_fields[0] if detected_fields else "unknown"

        document.metadata.update(enriched_metadata)
        return document

    @staticmethod
    def _detect_target_field_from_query(user_query: str) -> str | None:
        """Infer the scientific domain the user is asking about."""
        query_lower = user_query.lower()
        for field, hint_words in QUERY_FIELD_HINTS.items():
            if any(word in query_lower for word in hint_words):
                return field
        return None

    def _search_with_metadata_filter(self, user_query: str, *, result_count: int = 3) -> list[Document]:
        """Retrieve documents, optionally filtering by the detected scientific field."""
        broad_results = self.vector_store.similarity_search(user_query, k=result_count * 3)

        target_field = self._detect_target_field_from_query(user_query)
        if not target_field:
            return broad_results[:result_count]

        field_filtered_results = [
            document for document in broad_results
            if document.metadata.get("primary_field") == target_field
        ]
        return (field_filtered_results[:result_count] if field_filtered_results
                else broad_results[:result_count])

    def build(self) -> None:
        """Enrich chunks with metadata, build the vector store and chain."""
        enhanced_chunks = [
            self._extract_enhanced_metadata(chunk)
            for chunk in self.document_chunks
        ]

        embeddings = AzureOpenAIEmbeddings(model=self.pipeline_settings.models.embedding_model)
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(documents=enhanced_chunks)

        retrieval_top_k = self.pipeline_settings.retrieval.top_k

        def metadata_filter_retriever(query: str) -> list[Document]:
            return self._search_with_metadata_filter(query, result_count=retrieval_top_k)

        self.retriever = metadata_filter_retriever

        rag_llm = AzureChatOpenAI(model=self.pipeline_settings.models.rag_chat_model)
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | rag_llm
                | StrOutputParser()
        )
