#!/usr/bin/env python3
"""
Naive RAG Baseline for CV Data.

Traditional vector-based RAG system using ChromaDB for similarity search.
This serves as a baseline comparison against GraphRAG to demonstrate
the limitations of naive RAG for structured queries.
"""

from __future__ import annotations

import json
import logging
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)


class ContextChunkInfo(TypedDict):
    chunk_index: int
    source_file: str
    person_name: str
    content_preview: str


class QueryResult(TypedDict):
    query: str
    answer: str
    source_type: str
    execution_time: float
    num_chunks_retrieved: int
    context_info: list[ContextChunkInfo]
    success: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class RagPaths:
    programmers_dir: Path
    vector_db_dir: Path
    results_dir: Path


@dataclass(frozen=True, slots=True)
class RagModels:
    embedding_model: str
    chat_model: str


@dataclass(frozen=True, slots=True)
class Chunking:
    chunk_size: int
    chunk_overlap: int


class NaiveCvRag:
    """Traditional RAG system using vector similarity search for CV PDFs."""

    _DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
    _DEFAULT_CHAT_MODEL: Final[str] = "gpt-4o"
    _DEFAULT_VECTOR_DB_DIR: Final[Path] = Path("./chroma_naive_rag_cv")
    _DEFAULT_RESULTS_DIR: Final[Path] = Path("results")

    def __init__(self, config_path: str | Path = "utils/config.toml") -> None:
        self._config = self._load_config(Path(config_path))
        self._paths = self._build_paths(self._config)

        self._models = RagModels(
            embedding_model=self._DEFAULT_EMBEDDING_MODEL,
            chat_model=self._DEFAULT_CHAT_MODEL,
        )
        self._chunking = Chunking(chunk_size=1000, chunk_overlap=200)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        self._embeddings = OpenAIEmbeddings(model=self._models.embedding_model, api_key=api_key)
        self._llm = ChatOpenAI(model=self._models.chat_model, temperature=0, api_key=api_key)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunking.chunk_size,
            chunk_overlap=self._chunking.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        self._paths.results_dir.mkdir(parents=True, exist_ok=True)

        self._vectorstore: Chroma | None = None
        self._retriever = None
        self._rag_chain = None

        LOGGER.info("Naive CV RAG initialized")

    def create_vector_store(self, *, force_recreate: bool = False) -> None:
        """Create or load the persistent vector store."""
        if self._paths.vector_db_dir.exists() and not force_recreate:
            self._vectorstore = Chroma(
                persist_directory=str(self._paths.vector_db_dir),
                embedding_function=self._embeddings,
            )
            LOGGER.info("Vector store loaded from %s", self._paths.vector_db_dir)
            return

        documents = self._load_cv_documents()
        chunks = self._splitter.split_documents(documents)

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(self._paths.vector_db_dir),
        )
        LOGGER.info("Vector store created at %s", self._paths.vector_db_dir)

    def setup_rag_chain(self, *, k: int = 5) -> None:
        """Configure the retriever and RAG chain."""
        vectorstore = self._require_vectorstore()

        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an HR assistant helping with CV analysis. Use the provided context from CVs to answer questions accurately.\n\n"
                    "IMPORTANT INSTRUCTIONS:\n"
                    "- Base your answers ONLY on the information provided in the context\n"
                    "- If you cannot determine something from the context, say so clearly\n"
                    "- If the context is incomplete for a full answer, acknowledge this limitation\n\n"
                    "Context from CVs:\n"
                    "{context}",
                ),
                ("human", "{question}"),
            ]
        )

        self._rag_chain = (
                {"context": self._retriever | self._format_docs, "question": RunnablePassthrough()}
                | prompt
                | self._llm
                | StrOutputParser()
        )

        LOGGER.info("RAG chain configured")

    def initialize(self, *, force_recreate_vectorstore: bool = False, k: int = 5) -> None:
        """Initialize the vector store and RAG chain."""
        self.create_vector_store(force_recreate=force_recreate_vectorstore)
        self.setup_rag_chain(k=k)

    def query(self, question: str) -> QueryResult:
        """Answer a question using the configured RAG chain."""
        start = time.perf_counter()
        try:
            chain = self._require_rag_chain()
            retriever = self._require_retriever()

            relevant_docs: list[Document] = retriever.invoke(question)
            answer: str = chain.invoke(question)
            elapsed = time.perf_counter() - start

            return QueryResult(
                query=question,
                answer=answer,
                source_type="naive_rag",
                execution_time=elapsed,
                num_chunks_retrieved=len(relevant_docs),
                context_info=self._build_context_info(relevant_docs),
                success=True,
                error=None,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            LOGGER.exception("Query failed")
            return QueryResult(
                query=question,
                answer=f"Error processing query: {exc}",
                source_type="naive_rag",
                execution_time=elapsed,
                num_chunks_retrieved=0,
                context_info=[],
                success=False,
                error=str(exc),
            )

    def get_database_stats(self) -> dict[str, object]:
        """Return basic statistics about the current vector database."""
        vectorstore = self._vectorstore
        if vectorstore is None:
            return {"error": "Vector store not initialized"}

        try:
            total_chunks = self._count_chunks(vectorstore)
            sample_source_files = self._sample_source_files(vectorstore, limit=10)
            return {
                "total_chunks": total_chunks,
                "sample_source_files": sample_source_files,
                "embedding_model": self._models.embedding_model,
                "chunk_size": self._chunking.chunk_size,
                "chunk_overlap": self._chunking.chunk_overlap,
            }
        except Exception as exc:
            return {"error": f"Could not get stats: {exc}"}

    def save_results(self, results: list[QueryResult], *, output_file: Path) -> None:
        """Persist query results to a JSON file."""
        payload = {
            "test_metadata": {
                "system_type": "naive_rag",
                "test_queries": len(results),
                "database_stats": self.get_database_stats(),
            },
            "results": results,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_cv_documents(self) -> list[Document]:
        cv_files = sorted(self._paths.programmers_dir.glob("*.pdf"))
        if not cv_files:
            raise FileNotFoundError(f"No PDF files found in {self._paths.programmers_dir}")

        documents: list[Document] = []
        for cv_file in cv_files:
            try:
                loaded = PyPDFLoader(str(cv_file)).load()
                for doc in loaded:
                    doc.metadata.update(
                        {
                            "source_file": cv_file.name,
                            "document_type": "cv",
                            "person_name": cv_file.stem,
                        }
                    )
                documents.extend(loaded)
            except Exception:
                LOGGER.exception("Could not load %s", cv_file)

        if not documents:
            raise RuntimeError(f"All PDF loads failed in {self._paths.programmers_dir}")

        LOGGER.info("Loaded %d pages from %d CV files", len(documents), len(cv_files))
        return documents

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _build_context_info(docs: list[Document]) -> list[ContextChunkInfo]:
        items: list[ContextChunkInfo] = []
        for idx, doc in enumerate(docs):
            text = doc.page_content
            items.append(
                ContextChunkInfo(
                    chunk_index=idx,
                    source_file=str(doc.metadata.get("source_file", "unknown")),
                    person_name=str(doc.metadata.get("person_name", "unknown")),
                    content_preview=text[:200] + "..." if len(text) > 200 else text,
                )
            )
        return items

    def _require_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call create_vector_store() first.")
        return self._vectorstore

    def _require_retriever(self):
        if self._retriever is None:
            raise RuntimeError("Retriever not initialized. Call setup_rag_chain() first.")
        return self._retriever

    def _require_rag_chain(self):
        if self._rag_chain is None:
            raise RuntimeError("RAG chain not initialized. Call setup_rag_chain() first.")
        return self._rag_chain

    @staticmethod
    def _count_chunks(vectorstore: Chroma) -> int:
        collection = getattr(vectorstore, "_collection", None)
        if collection is None:
            raise RuntimeError("Chroma collection is not available.")
        return int(collection.count())

    @staticmethod
    def _sample_source_files(vectorstore: Chroma, *, limit: int) -> list[str]:
        collection = getattr(vectorstore, "_collection", None)
        if collection is None:
            return []

        try:
            result = collection.get(include=["metadatas"], limit=limit)
            metadatas = result.get("metadatas") or []
            source_files = {str(md.get("source_file", "unknown")) for md in metadatas if isinstance(md, dict)}
            return sorted(source_files)[:limit]
        except Exception:
            return []

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, object]:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    @classmethod
    def _build_paths(cls, config: dict[str, object]) -> RagPaths:
        output = config.get("output")
        if not isinstance(output, dict):
            raise ValueError("Invalid config: expected [output] section")

        programmers_dir = output.get("programmers_dir")
        if not isinstance(programmers_dir, str) or not programmers_dir.strip():
            raise ValueError("Invalid config: output.programmers_dir must be a non-empty string")

        return RagPaths(
            programmers_dir=Path(programmers_dir),
            vector_db_dir=cls._DEFAULT_VECTOR_DB_DIR,
            results_dir=cls._DEFAULT_RESULTS_DIR,
        )


def run_smoke_test() -> None:
    print("Testing Naive CV RAG")
    print("=" * 30)

    rag = NaiveCvRag()
    rag.initialize()

    stats = rag.get_database_stats()
    print("\nDatabase Statistics:")
    print(f"Total chunks: {stats.get('total_chunks', 'unknown')}")
    print(f"Sample files: {', '.join(list(stats.get('sample_source_files', []))[:3])}")

    queries = [
        "How many people have Python skills?",
        "List people with AWS certifications",
        "What is the most common programming language?",
        "Who worked at Google?",
        "Find people with both React and Node.js skills",
    ]

    results: list[QueryResult] = []
    for i, q in enumerate(queries, start=1):
        print(f"\n[{i}/{len(queries)}] Query: {q}")
        print("-" * 40)
        result = rag.query(q)
        results.append(result)

        if result["success"]:
            print(f"Answer: {result['answer']}")
            print(f"Execution time: {result['execution_time']:.2f}s")
            print(f"Chunks used: {result['num_chunks_retrieved']}")
        else:
            print(f"Error: {result['answer']}")

    output_file = Path("results") / "naive_rag_test_results.json"
    rag.save_results(results, output_file=output_file)
    print(f"\nSaved test results to: {output_file}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv(override=True)

    print("Naive RAG Baseline System for CV Data")
    print("=" * 40)
    run_smoke_test()


if __name__ == "__main__":
    main()
