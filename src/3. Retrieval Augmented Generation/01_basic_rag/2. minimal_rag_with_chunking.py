#!/usr/bin/env python3
"""
🚀 RAG with Text Chunking - Interactive Educational Journey
============================================================

This script builds on the minimal RAG example by introducing *text chunking* —
the practice of splitting large documents into smaller, focused segments before
embedding them.  Smaller chunks mean the retriever can pinpoint exactly the
paragraph that answers a question, instead of returning an entire biography.

🎯 What You'll Learn:
- Why chunking improves retrieval accuracy
- How RecursiveCharacterTextSplitter breaks text at natural boundaries
- How chunk size and overlap affect the number of chunks and answer quality
- How to compare different chunking strategies on the same question

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, and python-dotenv packages
"""

import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/01_basic_rag/data/scientists_bios"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHAT_MODEL_NAME = "gpt-5-nano"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
RETRIEVER_TOP_K = 4
TEXT_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

CHUNKED_RAG_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
The context consists of multiple text chunks that may contain relevant information.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""

SAMPLE_QUESTIONS = [
    "What collaboration did Ada Lovelace have with Charles Babbage?",
    "How did Newton's work during the Great Plague contribute to his discoveries?",
    "What awards did Marie Curie receive during her lifetime?",
]


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int

    def __str__(self) -> str:
        return f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"


CHUNKING_CONFIGS_TO_COMPARE = [
    ChunkingConfig(chunk_size=500, chunk_overlap=50),
    ChunkingConfig(chunk_size=1000, chunk_overlap=200),
    ChunkingConfig(chunk_size=2000, chunk_overlap=400),
]


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def load_scientist_biographies(*, directory_path: str) -> list[Document]:
    """
    Document Loading — reads every file from the given directory
    and wraps each one in a LangChain Document object.
    """
    print_section_header("LOADING DOCUMENTS")
    print(textwrap.dedent(load_scientist_biographies.__doc__))

    directory_loader = DirectoryLoader(directory_path)
    loaded_documents = directory_loader.load()
    print(f"Loaded {len(loaded_documents)} documents from '{directory_path}'")
    return loaded_documents


def split_documents_into_chunks(
        *,
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int,
) -> list[Document]:
    """
    Text Chunking — splits each document into smaller segments.

    RecursiveCharacterTextSplitter tries the separators in order (paragraph break,
    line break, sentence end, word boundary, character) and picks the first one
    that produces chunks within the requested size.  The overlap ensures that
    context at chunk boundaries is not lost.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=TEXT_SPLIT_SEPARATORS,
    )
    document_chunks = recursive_splitter.split_documents(documents)
    return document_chunks


def build_chunked_vector_store(
        *,
        document_chunks: list[Document],
        embedding_model_name: str,
) -> InMemoryVectorStore:
    """
    Embedding & Storing — each chunk is embedded and placed in an
    InMemoryVectorStore for later similarity search.
    """
    azure_embeddings = AzureOpenAIEmbeddings(model=embedding_model_name)
    vector_store = InMemoryVectorStore(embedding=azure_embeddings)
    vector_store.add_documents(documents=document_chunks)
    return vector_store


def create_chunked_rag_chain(
        *,
        vector_store: InMemoryVectorStore,
        chat_model_name: str,
        retriever_top_k: int = RETRIEVER_TOP_K,
) -> RunnableSerializable:
    """Assembles a RAG chain backed by a chunked vector store."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retriever_top_k},
    )
    chat_llm = AzureChatOpenAI(model=chat_model_name)
    rag_prompt = ChatPromptTemplate.from_template(CHUNKED_RAG_PROMPT_TEMPLATE)

    return (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | chat_llm
            | StrOutputParser()
    )


def demonstrate_chunked_rag(
        *,
        documents: list[Document],
        questions: list[str],
) -> None:
    """
    Builds a complete chunked-RAG pipeline with default settings and answers
    a batch of sample questions.

    The default configuration uses chunk_size=1000 and chunk_overlap=200 —
    a balanced starting point that works well for mid-length prose like
    scientist biographies.
    """
    print_section_header("CHUNKED RAG — DEFAULT CONFIGURATION")
    print(textwrap.dedent(demonstrate_chunked_rag.__doc__))

    document_chunks = split_documents_into_chunks(
        documents=documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    print(f"Created {len(document_chunks)} text chunks "
          f"(chunk_size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP})")

    chunked_vector_store = build_chunked_vector_store(
        document_chunks=document_chunks,
        embedding_model_name=EMBEDDING_MODEL_NAME,
    )

    rag_chain = create_chunked_rag_chain(
        vector_store=chunked_vector_store,
        chat_model_name=CHAT_MODEL_NAME,
    )

    for question_number, current_question in enumerate(questions, start=1):
        print(f"\nQ{question_number}: {current_question}\n"
              f"{'-' * 50}")
        answer = rag_chain.invoke(current_question)
        print(f"A{question_number}: {answer}")


def compare_chunking_strategies(
        *,
        documents: list[Document],
        configs: list[ChunkingConfig],
        test_question: str,
) -> None:
    """
    Compares how different chunk-size / overlap combinations affect answers.

    For each configuration the pipeline is rebuilt from scratch so that the
    vector store contains chunks of the requested size.  This lets you see
    how granularity influences both the total number of chunks and the
    quality of the retrieved answer.
    """
    print_section_header("COMPARING CHUNKING STRATEGIES")
    print(textwrap.dedent(compare_chunking_strategies.__doc__))
    print(f"Question: {test_question}\n")

    for current_config in configs:
        print(f"\n📊 {current_config}\n"
              f"{'-' * 40}")

        document_chunks = split_documents_into_chunks(
            documents=documents,
            chunk_size=current_config.chunk_size,
            chunk_overlap=current_config.chunk_overlap,
        )

        chunked_vector_store = build_chunked_vector_store(
            document_chunks=document_chunks,
            embedding_model_name=EMBEDDING_MODEL_NAME,
        )

        rag_chain = create_chunked_rag_chain(
            vector_store=chunked_vector_store,
            chat_model_name=CHAT_MODEL_NAME,
        )

        answer = rag_chain.invoke(test_question)
        print(f"Total chunks: {len(document_chunks)}\n"
              f"Answer: {answer}")


if __name__ == "__main__":
    load_dotenv(override=True)

    scientist_documents = load_scientist_biographies(directory_path=SCIENTISTS_BIOS_DIR)

    demonstrate_chunked_rag(
        documents=scientist_documents,
        questions=SAMPLE_QUESTIONS,
    )

    compare_chunking_strategies(
        documents=scientist_documents,
        configs=CHUNKING_CONFIGS_TO_COMPARE,
        test_question="What was Ada Lovelace's contribution to computer programming?",
    )
