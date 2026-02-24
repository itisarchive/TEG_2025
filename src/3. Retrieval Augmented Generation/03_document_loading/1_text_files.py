#!/usr/bin/env python3
"""
📄 Text File Loading for RAG Systems
=====================================

This script demonstrates different strategies for loading text files into a
Retrieval-Augmented Generation (RAG) pipeline using LangChain document loaders.

🎯 What You'll Learn:
- Loading a single text file with TextLoader
- Loading multiple specific files manually
- Auto-discovering files in a directory with DirectoryLoader
- Filtering files with glob patterns
- Splitting loaded documents into chunks for vector search
- Building a complete RAG chain over text-file-based knowledge

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Text files in the data/scientists_bios/ directory
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR: Final[str] = (
    "src/3. Retrieval Augmented Generation/03_document_loading/data/scientists_bios"
)

RAG_PROMPT_TEMPLATE: Final[str] = """\
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""


def print_section_header(title: str) -> None:
    separator = "=" * 50
    print(f"\n{separator}\n{title}\n{separator}")


def extract_scientist_name(document: Document) -> str:
    """Extracts a human-readable scientist name from a document's source metadata."""
    return Path(document.metadata["source"]).stem


def print_document_summary(documents: list[Document], *, label: str = "Document") -> None:
    for index, document in enumerate(documents, start=1):
        scientist_name = extract_scientist_name(document)
        character_count = len(document.page_content)
        print(f"   {label} {index}: {scientist_name} ({character_count} chars)")


@dataclass(frozen=True)
class ChunkDistribution:
    scientist_name: str
    chunk_count: int


def compute_chunk_distribution(chunks: list[Document]) -> list[ChunkDistribution]:
    distribution_map: dict[str, int] = {}
    for chunk in chunks:
        name = extract_scientist_name(chunk)
        distribution_map[name] = distribution_map.get(name, 0) + 1
    return [
        ChunkDistribution(scientist_name=name, chunk_count=count)
        for name, count in distribution_map.items()
    ]


@dataclass(frozen=True)
class LoadingMethodComparison:
    method_name: str
    advantages: tuple[str, ...]
    disadvantages: tuple[str, ...]


LOADING_METHODS: Final[tuple[LoadingMethodComparison, ...]] = (
    LoadingMethodComparison(
        method_name="Single File",
        advantages=(
            "Precise control over content",
            "Fast for specific documents",
        ),
        disadvantages=(
            "Manual file specification",
            "Doesn't scale well",
        ),
    ),
    LoadingMethodComparison(
        method_name="Multiple Specific Files",
        advantages=(
            "Selective loading",
            "Good for curated data",
        ),
        disadvantages=(
            "Requires file list maintenance",
            "Manual process",
        ),
    ),
    LoadingMethodComparison(
        method_name="Directory Loading",
        advantages=(
            "Automatic discovery",
            "Scales with directory content",
            "No file list maintenance",
        ),
        disadvantages=(
            "Loads everything (might include unwanted files)",
        ),
    ),
    LoadingMethodComparison(
        method_name="Pattern-Based Loading",
        advantages=(
            "Flexible filtering",
            "Automatic but selective",
            "Good for mixed directories",
            "Best of both worlds",
        ),
        disadvantages=(),
    ),
)


def load_single_text_file() -> list[Document]:
    """
    Loading a single text file with TextLoader.

    TextLoader is the simplest document loader — it reads one file at a time
    and wraps the content in a LangChain Document object with source metadata.
    """
    print_section_header("1️⃣ SINGLE FILE LOADING")
    print(textwrap.dedent(load_single_text_file.__doc__))

    ada_lovelace_path = f"{SCIENTISTS_BIOS_DIR}/Ada Lovelace.txt"
    single_file_loader = TextLoader(file_path=ada_lovelace_path)
    loaded_documents = single_file_loader.load()

    print(textwrap.indent(textwrap.dedent(f"""\
        Loaded: {len(loaded_documents)} document
        Content length: {len(loaded_documents[0].page_content)} characters
        Metadata: {loaded_documents[0].metadata}
        Preview: {loaded_documents[0].page_content[:200]}..."""), "   "))

    return loaded_documents


def load_multiple_specific_files() -> list[Document]:
    """
    Loading several hand-picked files one by one.

    When you need only a curated subset of files, you can iterate over an
    explicit list of paths and merge the results into a single collection.
    """
    print_section_header("2️⃣ MULTIPLE SPECIFIC FILES")
    print(textwrap.dedent(load_multiple_specific_files.__doc__))

    selected_file_paths = [
        f"{SCIENTISTS_BIOS_DIR}/Ada Lovelace.txt",
        f"{SCIENTISTS_BIOS_DIR}/Albert Einstein.txt",
        f"{SCIENTISTS_BIOS_DIR}/Isaac Newton.txt",
    ]

    accumulated_documents: list[Document] = []
    for file_path in selected_file_paths:
        loader = TextLoader(file_path=file_path)
        accumulated_documents.extend(loader.load())

    print(f"   Loaded: {len(accumulated_documents)} documents")
    print_document_summary(accumulated_documents, label="File")

    return accumulated_documents


def load_entire_directory() -> list[Document]:
    """
    Loading all files from a directory with DirectoryLoader.

    DirectoryLoader automatically discovers every file in the given directory
    and loads each one as a separate Document — ideal when you want to ingest
    an entire knowledge base without listing files manually.
    """
    print_section_header("3️⃣ LOADING ENTIRE DIRECTORY")
    print(textwrap.dedent(load_entire_directory.__doc__))

    directory_loader = DirectoryLoader(path=SCIENTISTS_BIOS_DIR)
    all_documents = directory_loader.load()

    print(f"   Loaded: {len(all_documents)} documents from directory")
    print_document_summary(all_documents)

    return all_documents


def load_directory_with_glob_pattern() -> list[Document]:
    """
    Loading files filtered by a glob pattern (e.g. *.txt only).

    Glob patterns let you combine the automatic discovery of DirectoryLoader
    with selective filtering — useful when a directory contains mixed file
    types and you only want a specific format.
    """
    print_section_header("4️⃣ LOADING WITH GLOB PATTERN (*.txt)")
    print(textwrap.dedent(load_directory_with_glob_pattern.__doc__))

    pattern_loader = DirectoryLoader(path=SCIENTISTS_BIOS_DIR, glob="*.txt")
    txt_only_documents = pattern_loader.load()

    print(f"   Loaded: {len(txt_only_documents)} .txt documents")

    return txt_only_documents


def split_documents_into_chunks(documents: list[Document]) -> list[Document]:
    """
    Splitting loaded documents into smaller chunks for vector search.

    Large documents must be split into overlapping chunks so that each chunk
    fits within the embedding model's context window and retrieval returns
    focused, relevant passages instead of entire files.
    """
    print_section_header("5️⃣ TEXT SPLITTING")
    print(textwrap.dedent(split_documents_into_chunks.__doc__))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    print(f"   Original documents: {len(documents)}")
    print(f"   After chunking: {len(chunks)} chunks")

    per_scientist_distribution = compute_chunk_distribution(chunks)
    print("   Chunk distribution:")
    for entry in per_scientist_distribution:
        print(f"     {entry.scientist_name}: {entry.chunk_count} chunks")

    return chunks


def build_and_test_rag_chain(chunks: list[Document]) -> None:
    """
    Building a complete RAG chain over the chunked text documents.

    The pipeline: embed chunks → store in vector store → retrieve top-k
    relevant chunks per query → feed them as context to an LLM → return
    a concise, grounded answer.
    """
    print_section_header("6️⃣ RAG SYSTEM WITH TEXT FILES")
    print(textwrap.dedent(build_and_test_rag_chain.__doc__))

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = AzureChatOpenAI(model="gpt-5-nano")
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    text_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    test_questions = [
        "What did Ada Lovelace contribute to computing?",
        "How did Einstein develop his theories?",
        "What was Newton's most famous work?",
    ]

    print("\n🔍 Testing RAG with loaded text files:")
    for question_number, question in enumerate(test_questions, start=1):
        print(f"\nQ{question_number}: {question}\n"
              f"{'-' * 40}")
        answer = text_rag_chain.invoke(question)
        print(f"A{question_number}: {answer}")

    print(textwrap.dedent(f"""\

        💡 Text RAG chain ready: text_rag_chain.invoke('Your question')
        📚 Loaded {len(chunks)} chunks total"""))


def print_loading_methods_comparison() -> None:
    """
    Comparison of the four text-loading strategies demonstrated above,
    highlighting trade-offs between control, scalability, and convenience.
    """
    print_section_header("📊 LOADING METHOD COMPARISON")
    print(textwrap.dedent(print_loading_methods_comparison.__doc__))

    for method in LOADING_METHODS:
        print(f"Method — {method.method_name}:")
        for advantage in method.advantages:
            print(f"  ✅ {advantage}")
        for disadvantage in method.disadvantages:
            print(f"  ❌ {disadvantage}")
        print()


if __name__ == "__main__":
    load_dotenv(override=True)

    load_single_text_file()
    load_multiple_specific_files()
    all_directory_documents = load_entire_directory()
    load_directory_with_glob_pattern()

    all_chunks = split_documents_into_chunks(documents=all_directory_documents)
    build_and_test_rag_chain(chunks=all_chunks)
    print_loading_methods_comparison()
