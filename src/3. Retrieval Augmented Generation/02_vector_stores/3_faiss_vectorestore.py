"""
⚡ FAISS – High-Performance Vector Store for RAG
=================================================

This script demonstrates FAISS (Facebook AI Similarity Search) for
high-performance vector storage. FAISS is optimized for raw speed and can
handle large-scale similarity search far more efficiently than general-purpose
stores like ChromaDB or InMemoryVectorStore.

🎯 What You'll Learn:
- How to create a FAISS index from document chunks
- How to save and reload an index to/from disk
- How to perform similarity search with distance scores
- How FAISS compares to ChromaDB and InMemoryVectorStore

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, faiss-cpu, and python-dotenv packages

✅ When to use FAISS:
- Speed-critical similarity search at scale
- Read-heavy workloads with infrequent updates
- GPU acceleration needed (faiss-gpu package)
- Memory-efficient indexing of large datasets

❌ Limitations:
- No built-in metadata filtering (unlike ChromaDB)
- Less feature-rich than full vector databases
- Updating an existing index is cumbersome
- Persistence is manual (.faiss + .pkl files)
"""

import os
import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/02_vector_stores/data/scientists_bios"
FAISS_INDEX_PATH = "src/3. Retrieval Augmented Generation/02_vector_stores/faiss_index"

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
    "What was Einstein's theory of relativity about?",
    "How did Darwin's voyage influence his thinking?",
    "What programming concepts did Ada Lovelace pioneer?",
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


def index_already_saved(*, index_path: str) -> bool:
    """Checks whether a FAISS index file already exists on disk."""
    faiss_file = f"{index_path}.faiss"
    already_exists = os.path.exists(faiss_file)
    if already_exists:
        print(f"📁 Found existing FAISS index at: {faiss_file}")
    return already_exists


def build_faiss_vector_store(
        *,
        document_chunks: list[Document],
        embeddings: AzureOpenAIEmbeddings,
        index_path: str,
) -> FAISS:
    """
    Creates (or loads) a FAISS index with persistent disk storage.
    If the index already exists, it is deserialized from disk — skipping the
    expensive embedding step. Otherwise, documents are embedded and the new
    index is saved for future sessions.
    """
    if index_already_saved(index_path=index_path):
        print("📖 Loading existing FAISS index...")
        faiss_store = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"✅ Loaded existing FAISS index with {faiss_store.index.ntotal} vectors")
    else:
        print("🔨 Creating new FAISS index from documents...")
        faiss_store = FAISS.from_documents(
            documents=document_chunks,
            embedding=embeddings,
        )
        print("💾 Saving FAISS index to disk...")
        faiss_store.save_local(folder_path=index_path)
        print(f"✅ Created and saved FAISS index with {faiss_store.index.ntotal} vectors")

    return faiss_store


def demonstrate_similarity_search(
        *,
        faiss_store: FAISS,
        query: str,
        top_k: int = 3,
) -> None:
    """
    Basic similarity search returns the chunks whose embeddings are nearest
    to the query vector. FAISS uses optimized index structures (Flat, IVF, HNSW)
    to make this lookup extremely fast — even over millions of vectors.
    """
    print_section_header("FAISS SIMILARITY SEARCH")
    print(textwrap.dedent(demonstrate_similarity_search.__doc__))

    matched_documents = faiss_store.similarity_search(query=query, k=top_k)

    print(f"Query: {query}")
    print(f"Found {len(matched_documents)} most similar chunks:\n")
    for chunk_index, matched_doc in enumerate(matched_documents, start=1):
        preview = matched_doc.page_content[:200]
        print(f"  Chunk {chunk_index}: {preview}...")


def demonstrate_similarity_search_with_scores(
        *,
        faiss_store: FAISS,
        query: str,
        top_k: int = 3,
) -> None:
    """
    FAISS returns L2 (Euclidean) distance scores alongside each result.
    Lower distance means higher similarity. This is useful for setting
    relevance thresholds — e.g. discarding chunks with distance > 1.0.
    """
    print_section_header("SIMILARITY SEARCH WITH DISTANCE SCORES")
    print(textwrap.dedent(demonstrate_similarity_search_with_scores.__doc__))

    matched_docs_with_distances = faiss_store.similarity_search_with_score(query=query, k=top_k)

    print(f"Query: {query}")
    print("Results with FAISS distance scores (lower = more similar):\n")
    for chunk_index, (matched_doc, distance) in enumerate(matched_docs_with_distances, start=1):
        preview = matched_doc.page_content[:150]
        print(f"  Chunk {chunk_index} (distance: {distance:.3f}): {preview}...")


def build_faiss_rag_chain(
        *,
        faiss_store: FAISS,
        retriever_top_k: int = 4,
):
    """
    Builds a RAG chain backed by FAISS retriever.
    Structurally identical to the ChromaDB and InMemory chains, but retrieval
    is backed by FAISS's optimized ANN (Approximate Nearest Neighbor) engine.
    """
    retriever = faiss_store.as_retriever(
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
    retrieved from the scientist biography documents stored in the FAISS index.
    """
    print_section_header("FAISS HIGH-PERFORMANCE RAG DEMO")
    print(textwrap.dedent(demonstrate_rag_question_answering.__doc__))

    for question_number, question_text in enumerate(questions, start=1):
        print(f"\nQ{question_number}: {question_text}\n"
              f"{'-' * 40}")
        answer = rag_chain.invoke(question_text)
        print(f"A{question_number}: {answer}")


def print_faiss_store_summary(
        *,
        faiss_store: FAISS,
        index_path: str,
) -> None:
    print_section_header("FAISS FEATURES & PERFORMANCE")

    print(textwrap.dedent(f"""\
        📊 Index contains {faiss_store.index.ntotal} vectors
        🔢 Vector dimension: {faiss_store.index.d}
        💾 Index saved to: {index_path}.faiss

        ⚡ FAISS Advantages:
          • Extremely fast similarity search
          • Memory efficient for large datasets
          • GPU acceleration available (faiss-gpu)
          • Multiple index types (Flat, IVF, HNSW)
          • Optimized for production workloads

        🏗️ FAISS vs Other Stores:
          • Faster than ChromaDB for large scales
          • More memory efficient than InMemory
          • Less features than full databases
          • Perfect for read-heavy workloads

        🔄 Persistence:
          • Index automatically saved/loaded
          • Fast startup with existing index
          • Single file storage (.faiss + .pkl)

        💡 Try: rag_chain.invoke('Your question here')
        💡 Fast search: faiss_store.similarity_search('query', k=5)
        💡 With scores: faiss_store.similarity_search_with_score('query', k=3)"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    chunking_configuration = ChunkingConfig()

    scientist_bio_chunks = load_and_chunk_documents(
        directory_path=SCIENTISTS_BIOS_DIR,
        chunking_config=chunking_configuration,
    )

    azure_embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

    persistent_faiss_store = build_faiss_vector_store(
        document_chunks=scientist_bio_chunks,
        embeddings=azure_embeddings,
        index_path=FAISS_INDEX_PATH,
    )

    demonstrate_similarity_search(
        faiss_store=persistent_faiss_store,
        query="What theories did Einstein develop?",
    )

    demonstrate_similarity_search_with_scores(
        faiss_store=persistent_faiss_store,
        query="What theories did Einstein develop?",
    )

    persistent_faiss_rag_chain = build_faiss_rag_chain(
        faiss_store=persistent_faiss_store,
    )

    demonstrate_rag_question_answering(
        rag_chain=persistent_faiss_rag_chain,
        questions=DEMO_QUESTIONS,
    )

    print_faiss_store_summary(
        faiss_store=persistent_faiss_store,
        index_path=FAISS_INDEX_PATH,
    )
