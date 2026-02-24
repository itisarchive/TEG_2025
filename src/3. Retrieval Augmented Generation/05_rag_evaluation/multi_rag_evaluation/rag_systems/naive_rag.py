"""
🔍 Naive RAG — Baseline Vector Similarity Search
=================================================

The simplest possible RAG strategy: embed chunks into a vector store and
retrieve the top-k most similar documents for each query.  This implementation
serves as the baseline against which all advanced strategies are compared.
"""

import textwrap

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


class NaiveRAG(BaseRAG):
    """Baseline RAG using plain vector similarity search — no reranking, no filtering."""

    @property
    def name(self) -> str:
        return "Naive RAG"

    def build(self) -> None:
        """Build the vector store, retriever and LLM chain."""
        embeddings = AzureOpenAIEmbeddings(model=self.pipeline_settings.models.embedding_model)
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=self.document_chunks)

        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.pipeline_settings.retrieval.top_k},
        )

        rag_llm = AzureChatOpenAI(model=self.pipeline_settings.models.rag_chat_model)
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | rag_prompt
                | rag_llm
                | StrOutputParser()
        )
