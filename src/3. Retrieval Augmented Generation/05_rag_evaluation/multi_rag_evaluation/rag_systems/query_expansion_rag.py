"""
🔄 Query Expansion RAG — Broader Retrieval Through Multi-Query Search
=====================================================================

Improves recall by generating multiple reformulations of the user's question:

1. **LLM-based expansion** — an LLM produces synonym-rich and more specific
   alternative queries (e.g. "What did Curie discover?" → "Marie Curie
   radioactivity research contributions").
2. **Concept-based expansion** — domain-specific keywords for known scientists
   are appended to the original query.

All query variants are searched in parallel and deduplicated so that the
final top-k results cover a wider semantic space than a single query could.
"""

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

QUERY_EXPANSION_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are a search query expert. Given a user's search query about scientists \
    and their work, generate {max_variations} alternative versions that might \
    help find relevant information.

    Focus on:
    1. Synonyms and related terms
    2. More specific scientific terminology
    3. Alternative phrasings

    Original query: {query}

    Return only the alternative queries, one per line, without numbering or explanation:""")

SCIENTIST_CONCEPT_MAP: dict[str, list[str]] = {
    "einstein": ["relativity", "photon", "spacetime"],
    "newton": ["gravity", "motion", "calculus"],
    "curie": ["radioactivity", "radiation", "polonium"],
    "darwin": ["evolution", "selection", "species"],
    "lovelace": ["programming", "algorithm", "computer"],
}


class QueryExpansionRAG(BaseRAG):
    """RAG with LLM-driven and concept-driven query expansion for broader recall."""

    @property
    def name(self) -> str:
        return "Query Expansion RAG"

    def _expand_query_with_llm(self, original_query: str) -> list[str]:
        """Use an LLM to generate alternative phrasings of the original query."""
        expansion_prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT_TEMPLATE)
        expansion_llm = AzureChatOpenAI(
            model=self.pipeline_settings.models.rag_chat_model,
            temperature=0.3,
        )
        expansion_chain = expansion_prompt | expansion_llm | StrOutputParser()

        max_variations = self.pipeline_settings.retrieval.max_query_variations
        raw_output = expansion_chain.invoke({
            "query": original_query,
            "max_variations": max_variations,
        })

        generated_variations = [
            line.strip() for line in raw_output.split("\n") if line.strip()
        ]
        return [original_query] + generated_variations[:max_variations]

    @staticmethod
    def _expand_query_with_domain_concepts(original_query: str) -> str:
        """Append domain-specific keywords when a known scientist is mentioned."""
        query_lower = original_query.lower()
        for scientist, concepts in SCIENTIST_CONCEPT_MAP.items():
            if scientist in query_lower:
                return f"{original_query} {' '.join(concepts[:2])}"
        return original_query

    def _multi_query_search(self, user_query: str, *, result_count: int = 3) -> list[Document]:
        """Search with all query variations and return deduplicated top-k results."""
        query_variations = self._expand_query_with_llm(user_query)

        concept_expanded_query = self._expand_query_with_domain_concepts(user_query)
        if concept_expanded_query != user_query:
            query_variations.append(concept_expanded_query)

        unique_results: list[Document] = []
        seen_document_ids: set[int] = set()

        for variation in query_variations:
            variation_results = self.vector_store.similarity_search(variation, k=result_count * 2)
            for document in variation_results:
                doc_id = id(document)
                if doc_id not in seen_document_ids:
                    unique_results.append(document)
                    seen_document_ids.add(doc_id)

        return unique_results[:result_count]

    def build(self) -> None:
        """Build vector store and query-expansion retrieval chain."""
        embeddings = AzureOpenAIEmbeddings(model=self.pipeline_settings.models.embedding_model)
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(documents=self.document_chunks)

        retrieval_top_k = self.pipeline_settings.retrieval.top_k

        def query_expansion_retriever(query: str) -> list[Document]:
            return self._multi_query_search(query, result_count=retrieval_top_k)

        self.retriever = query_expansion_retriever

        rag_llm = AzureChatOpenAI(model=self.pipeline_settings.models.rag_chat_model)
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | rag_llm
                | StrOutputParser()
        )
