"""
💾 ChromaDB – Persistent Vector Store for RAG
==============================================

This script demonstrates persistent vector storage using ChromaDB. Unlike the
InMemoryVectorStore from the previous lesson, ChromaDB saves embeddings to disk
so they survive process restarts — no need to re-embed documents each time.

🎯 What You'll Learn:
- How to create and persist a ChromaDB collection on disk
- How to detect and reuse an existing collection across sessions
- How to perform similarity search with relevance scores
- How to reload a persisted store and verify data integrity

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-chroma, and python-dotenv packages

✅ When to use ChromaDB:
- Data must survive restarts (persistent storage on disk)
- You need similarity scores alongside results
- Incremental updates to the collection are required
- Production-grade RAG pipelines

❌ Limitations compared to hosted solutions:
- Single-machine storage (no distributed cluster)
- No built-in access control or multi-tenancy
"""

import os
import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/02_vector_stores/data/scientists_bios"
CHROMA_PERSIST_DIR = "src/3. Retrieval Augmented Generation/02_vector_stores/chroma_db"
CHROMA_COLLECTION_NAME = "scientists_bios"

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
    "What awards did Marie Curie receive?",
    "How did Charles Darwin develop his theory of evolution?",
    "What was Newton's contribution to mathematics?",
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


def collection_already_persisted(*, persist_directory: str) -> bool:
    """Checks whether a ChromaDB directory already exists on disk."""
    already_exists = os.path.exists(persist_directory)
    if already_exists:
        print(f"📁 Found existing ChromaDB at: {persist_directory}")
    return already_exists


def build_chroma_vector_store(
        *,
        document_chunks: list[Document],
        embeddings: AzureOpenAIEmbeddings,
        persist_directory: str,
        collection_name: str,
) -> Chroma:
    """
    Creates (or loads) a ChromaDB collection backed by persistent disk storage.
    If the collection already exists and contains data, the existing embeddings are
    reused. Otherwise, documents are embedded and persisted for future sessions.
    """
    chroma_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    is_existing = collection_already_persisted(persist_directory=persist_directory)

    if not is_existing:
        print("➕ Adding documents to new ChromaDB collection...")
        chroma_store.add_documents(documents=document_chunks)
        print(f"✅ Added {len(document_chunks)} chunks to ChromaDB")
    else:
        existing_document_count = len(chroma_store.get()["ids"])
        print(f"📊 Existing collection has {existing_document_count} documents")

        if existing_document_count == 0:
            print("➕ Collection is empty, adding documents...")
            chroma_store.add_documents(documents=document_chunks)
            print(f"✅ Added {len(document_chunks)} chunks to ChromaDB")

    return chroma_store


def demonstrate_similarity_search_with_scores(
        *,
        chroma_store: Chroma,
        query: str,
        top_k: int = 3,
) -> None:
    """
    ChromaDB extends basic similarity search by returning a relevance score alongside
    each result. Lower scores indicate higher similarity (distance-based metric).
    This helps you understand how confident the retriever is about each match.
    """
    print_section_header("SIMILARITY SEARCH WITH SCORES")
    print(textwrap.dedent(demonstrate_similarity_search_with_scores.__doc__))

    matched_docs_with_scores = chroma_store.similarity_search_with_score(query=query, k=top_k)

    print(f"Query: {query}")
    print("Results with similarity scores:\n")
    for chunk_index, (matched_doc, similarity_score) in enumerate(matched_docs_with_scores, start=1):
        preview = matched_doc.page_content[:150]
        print(f"  Chunk {chunk_index} (score: {similarity_score:.3f}): {preview}...")


def build_chroma_rag_chain(
        *,
        chroma_store: Chroma,
        retriever_top_k: int = 4,
):
    """
    Builds a RAG chain backed by a persistent ChromaDB retriever.
    Identical in structure to the in-memory RAG chain, but the retriever
    fetches context from disk-persisted embeddings.
    """
    retriever = chroma_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retriever_top_k},
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = AzureChatOpenAI(model="gpt-5-nano")

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
    retrieved from the scientist biography documents stored in ChromaDB.
    """
    print_section_header("CHROMA VECTOR STORE RAG DEMO")
    print(textwrap.dedent(demonstrate_rag_question_answering.__doc__))

    for question_number, question_text in enumerate(questions, start=1):
        print(f"\nQ{question_number}: {question_text}\n"
              f"{'-' * 40}")
        answer = rag_chain.invoke(question_text)
        print(f"A{question_number}: {answer}")


def demonstrate_persistence_across_reloads(
        *,
        embeddings: AzureOpenAIEmbeddings,
        persist_directory: str,
        collection_name: str,
        verification_question: str,
) -> None:
    """
    The key advantage of ChromaDB over InMemoryVectorStore is persistence.
    Here we simulate a "restart" by creating a brand-new Chroma instance that
    points to the same on-disk directory — proving that data survives reloads.
    """
    print_section_header("PERSISTENCE VERIFICATION")
    print(textwrap.dedent(demonstrate_persistence_across_reloads.__doc__))

    print("🔄 Reloading ChromaDB from disk to verify persistence...")
    reloaded_chroma_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    reloaded_retriever = reloaded_chroma_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = AzureChatOpenAI(model="gpt-5-nano")

    reloaded_rag_chain = (
            {"context": reloaded_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    reloaded_answer = reloaded_rag_chain.invoke(verification_question)
    print(f"\nAfter reload — Q: {verification_question}\n"
          f"A: {reloaded_answer}")


def print_chroma_store_summary(
        *,
        chroma_store: Chroma,
        collection_name: str,
        persist_directory: str,
) -> None:
    print_section_header("CHROMADB STORE PROPERTIES")

    collection_info = chroma_store.get()
    total_documents = len(collection_info["ids"])

    print(textwrap.dedent(f"""\
        📊 Collection '{collection_name}' contains {total_documents} documents
        💾 Persisted to: {persist_directory}

        ✅ ChromaDB Advantages:
          • Persistent storage — survives restarts
          • Efficient similarity search
          • Built-in metadata support
          • Incremental updates possible
          • Great for production use

        🔄 Restart behavior:
          • Next run will load existing data
          • No need to re-embed documents
          • Fast startup for existing collections

        💡 Try: rag_chain.invoke('Your question here')
        💡 Or:  chroma_store.similarity_search('query', k=5)
        💡 With scores: chroma_store.similarity_search_with_score('query', k=3)"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    chunking_configuration = ChunkingConfig()

    scientist_bio_chunks = load_and_chunk_documents(
        directory_path=SCIENTISTS_BIOS_DIR,
        chunking_config=chunking_configuration,
    )

    azure_embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

    persistent_chroma_store = build_chroma_vector_store(
        document_chunks=scientist_bio_chunks,
        embeddings=azure_embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    demonstrate_similarity_search_with_scores(
        chroma_store=persistent_chroma_store,
        query="What did Marie Curie discover about radioactivity?",
    )

    persistent_rag_chain = build_chroma_rag_chain(chroma_store=persistent_chroma_store)

    demonstrate_rag_question_answering(
        rag_chain=persistent_rag_chain,
        questions=DEMO_QUESTIONS,
    )

    demonstrate_persistence_across_reloads(
        embeddings=azure_embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        verification_question=DEMO_QUESTIONS[0],
    )

    print_chroma_store_summary(
        chroma_store=persistent_chroma_store,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
