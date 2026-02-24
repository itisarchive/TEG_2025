"""
🧠 InMemoryVectorStore – Simplest RAG Pipeline
================================================

This script demonstrates the simplest vector store option available in LangChain:
storing document embeddings entirely in memory. It walks you through the complete
RAG (Retrieval Augmented Generation) pipeline — from loading documents, through
chunking and embedding, to answering questions grounded in retrieved context.

🎯 What You'll Learn:
- How to load documents from a directory and split them into chunks
- How to embed chunks and store them in an InMemoryVectorStore
- How to perform similarity search against the store
- How to wire a full RAG chain: retriever → prompt → LLM → answer

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, and python-dotenv packages

✅ When to use InMemoryVectorStore:
- Development and testing workflows
- Small datasets that fit in RAM
- Scenarios where persistence is not required

❌ Limitations:
- Data is lost when the process ends
- Limited by available RAM
- No sharing between processes
- Must rebuild on every restart
"""

import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/02_vector_stores/data/scientists_bios"

RAG_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""

DEMO_QUESTIONS = [
    "What programming concept did Ada Lovelace pioneer?",
    "How did Einstein's work change physics?",
    "What was Marie Curie's most important discovery?",
]


@dataclass(frozen=True)
class ChunkingConfig:
    """Controls how documents are split into smaller pieces for embedding."""

    chunk_size: int = 1000
    chunk_overlap: int = 200


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def load_and_chunk_documents(
        *,
        directory_path: str,
        chunking_config: ChunkingConfig,
) -> list[Document]:
    """Loads all documents from a directory and splits them into overlapping chunks."""
    raw_documents = DirectoryLoader(directory_path).load()
    print(f"Loaded {len(raw_documents)} documents from: {directory_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
    )
    document_chunks = text_splitter.split_documents(raw_documents)
    print(
        f"Created {len(document_chunks)} chunks (size={chunking_config.chunk_size}, overlap={chunking_config.chunk_overlap})")
    return document_chunks


def build_in_memory_vector_store(
        *,
        document_chunks: list[Document],
        embeddings: AzureOpenAIEmbeddings,
) -> InMemoryVectorStore:
    """Embeds document chunks and stores them in an InMemoryVectorStore."""
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=document_chunks)
    print(f"✅ Added {len(document_chunks)} chunks to in-memory vector store")
    return vector_store


def demonstrate_similarity_search(
        *,
        vector_store: InMemoryVectorStore,
        query: str,
        top_k: int = 3,
) -> None:
    """
    Similarity search finds chunks whose embeddings are closest to the query embedding.
    This is the core retrieval mechanism that powers RAG — the model answers questions
    using only the most relevant pieces of your documents.
    """
    print_section_header("SIMILARITY SEARCH DEMO")
    print(textwrap.dedent(demonstrate_similarity_search.__doc__))

    matched_documents = vector_store.similarity_search(query=query, k=top_k)

    print(f"Query: {query}\n"
          f"Found {len(matched_documents)} most similar chunks:\n")
    for chunk_index, matched_doc in enumerate(matched_documents, start=1):
        preview = matched_doc.page_content[:200]
        print(f"  Chunk {chunk_index}: {preview}...")


def build_rag_chain(
        *,
        vector_store: InMemoryVectorStore,
        retriever_top_k: int = 3,
):
    """
    A RAG chain wires together four stages:
    1. Retriever – fetches the most relevant document chunks for a question
    2. Prompt    – injects retrieved context alongside the user question
    3. LLM       – generates an answer grounded in the provided context
    4. Parser    – extracts the plain-text response
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retriever_top_k},
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = AzureChatOpenAI(model="gpt-4o-mini")

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


def demonstrate_rag_question_answering(
        *,
        rag_chain,
        questions: list[str],
) -> None:
    """
    End-to-end RAG demonstration: each question is answered using only the knowledge
    retrieved from the scientist biography documents stored in the vector store.
    """
    print_section_header("IN-MEMORY VECTOR STORE RAG DEMO")
    print(textwrap.dedent(demonstrate_rag_question_answering.__doc__))

    for question_number, question_text in enumerate(questions, start=1):
        print(f"\nQ{question_number}: {question_text}\n"
              f"{'-' * 40}")
        answer = rag_chain.invoke(question_text)
        print(f"A{question_number}: {answer}")


def print_in_memory_store_summary(*, total_chunks: int) -> None:
    print_section_header("IN-MEMORY STORE PROPERTIES")

    print(textwrap.dedent(f"""\
        ✅ Advantages:
          • Fast – no disk I/O overhead
          • Simple – no external dependencies
          • Perfect for development and testing
          • No setup required

        ❌ Limitations:
          • Data lost when process ends
          • Limited by available RAM
          • No data sharing between processes
          • Rebuild required on restart

        💡 Current store contains {total_chunks} chunks in memory
        💡 Try: rag_chain.invoke('Your question here')
        💡 Or:  vector_store.similarity_search('Your query', k=5)"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    chunking_configuration = ChunkingConfig()

    scientist_bio_chunks = load_and_chunk_documents(
        directory_path=SCIENTISTS_BIOS_DIR,
        chunking_config=chunking_configuration,
    )

    azure_embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

    in_memory_vector_store = build_in_memory_vector_store(
        document_chunks=scientist_bio_chunks,
        embeddings=azure_embeddings,
    )

    demonstrate_similarity_search(
        vector_store=in_memory_vector_store,
        query="What did Ada Lovelace contribute to computing?",
    )

    in_memory_rag_chain = build_rag_chain(vector_store=in_memory_vector_store)

    demonstrate_rag_question_answering(
        rag_chain=in_memory_rag_chain,
        questions=DEMO_QUESTIONS,
    )

    print_in_memory_store_summary(total_chunks=len(scientist_bio_chunks))
