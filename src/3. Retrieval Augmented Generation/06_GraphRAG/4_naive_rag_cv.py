#!/usr/bin/env python3
"""
Naive RAG Baseline for CV Data
===============================

Traditional vector-based RAG system using ChromaDB for similarity search.
This serves as a baseline comparison against GraphRAG to demonstrate
the limitations of naive RAG for structured, relationship-heavy queries.

🎯 What You'll Learn:
- How a traditional (naive) RAG pipeline works end-to-end
- Loading PDFs → chunking → embedding → vector store → retrieval → LLM answer
- Why pure vector similarity struggles with structured / multi-hop questions

🔧 Prerequisites:
    - Azure OpenAI credentials in environment / .env
    - Generated CV PDFs (see 1_generate_data.py)
"""

import json
import logging
import textwrap
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
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)


class RetrievedChunkInfo(TypedDict):
    chunk_index: int
    source_file: str
    person_name: str
    content_preview: str


class NaiveRagQueryResult(TypedDict):
    query: str
    answer: str
    source_type: str
    execution_time_seconds: float
    num_chunks_retrieved: int
    retrieved_chunks_info: list[RetrievedChunkInfo]
    success: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class CvDirectoryPaths:
    cv_pdf_directory: Path
    vector_store_directory: Path
    results_directory: Path


@dataclass(frozen=True, slots=True)
class AzureModelNames:
    embedding_model_name: str
    chat_model_name: str


@dataclass(frozen=True, slots=True)
class ChunkingParameters:
    chunk_size: int
    chunk_overlap: int


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


class NaiveCvRag:
    """
    Traditional RAG system that answers CV-related questions using vector similarity search.

    Pipeline: PDF loading → recursive text chunking → Azure OpenAI embeddings
    → ChromaDB vector store → similarity retrieval → LLM-generated answer.

    This approach works well for simple keyword / semantic lookups but struggles
    with multi-hop reasoning, aggregation, and relationship-based queries —
    exactly the gaps that GraphRAG is designed to fill.
    """

    EMBEDDING_MODEL_NAME: Final[str] = "text-embedding-3-small"
    CHAT_MODEL_NAME: Final[str] = "gpt-4.1"
    VECTOR_STORE_DIR: Final[Path] = Path("./chroma_naive_rag_cv")
    RESULTS_DIR: Final[Path] = Path("results")

    def __init__(self, config_path: str | Path = "utils/config.toml") -> None:
        parsed_config = self._load_config(Path(config_path))
        self._directory_paths = self._build_directory_paths(parsed_config)

        self._model_names = AzureModelNames(
            embedding_model_name=self.EMBEDDING_MODEL_NAME,
            chat_model_name=self.CHAT_MODEL_NAME,
        )
        self._chunking_params = ChunkingParameters(chunk_size=1000, chunk_overlap=200)

        self._azure_embeddings = AzureOpenAIEmbeddings(
            model=self._model_names.embedding_model_name,
        )
        self._azure_chat_llm = AzureChatOpenAI(
            model=self._model_names.chat_model_name,
            temperature=0,
        )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunking_params.chunk_size,
            chunk_overlap=self._chunking_params.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        self._directory_paths.results_directory.mkdir(parents=True, exist_ok=True)

        self._vectorstore: Chroma | None = None
        self._similarity_retriever = None
        self._rag_chain = None

        LOGGER.info("NaiveCvRag initialized")

    def create_or_load_vector_store(self, *, force_recreate: bool = False) -> None:
        """Load an existing ChromaDB vector store or create a new one from CV PDFs."""
        if self._directory_paths.vector_store_directory.exists() and not force_recreate:
            self._vectorstore = Chroma(
                persist_directory=str(self._directory_paths.vector_store_directory),
                embedding_function=self._azure_embeddings,
            )
            LOGGER.info(
                "Loaded existing vector store from %s",
                self._directory_paths.vector_store_directory,
            )
            return

        cv_documents = self._load_all_cv_documents()
        chunked_documents = self._text_splitter.split_documents(cv_documents)

        self._vectorstore = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self._azure_embeddings,
            persist_directory=str(self._directory_paths.vector_store_directory),
        )
        LOGGER.info(
            "Created new vector store at %s",
            self._directory_paths.vector_store_directory,
        )

    def build_rag_chain(self, *, retrieval_top_k: int = 5) -> None:
        """
        Build the retrieval-augmented generation chain.

        The chain retrieves the top-k most similar chunks from the vector store,
        injects them as context into a system prompt, and passes the user question
        to an Azure Chat LLM for final answer generation.
        """
        vectorstore = self._require_vectorstore()

        self._similarity_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_top_k},
        )

        hr_assistant_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    textwrap.dedent("""\
                        You are an HR assistant helping with CV analysis.
                        Use the provided context from CVs to answer questions accurately.

                        IMPORTANT INSTRUCTIONS:
                        - Base your answers ONLY on the information provided in the context
                        - If you cannot determine something from the context, say so clearly
                        - If the context is incomplete for a full answer, acknowledge this limitation

                        Context from CVs:
                        {context}"""),
                ),
                ("human", "{question}"),
            ]
        )

        self._rag_chain = (
                {
                    "context": self._similarity_retriever | self._join_document_contents,
                    "question": RunnablePassthrough(),
                }
                | hr_assistant_prompt
                | self._azure_chat_llm
                | StrOutputParser()
        )

        LOGGER.info("RAG chain built with top_k=%d", retrieval_top_k)

    def initialize(
            self,
            *,
            force_recreate_vectorstore: bool = False,
            retrieval_top_k: int = 5,
    ) -> None:
        """Create / load the vector store, then build the RAG chain — ready to query."""
        self.create_or_load_vector_store(force_recreate=force_recreate_vectorstore)
        self.build_rag_chain(retrieval_top_k=retrieval_top_k)

    def query(self, question: str) -> dict[str, object]:
        """Backward-compatible alias for answer_question, expected by SystemComparator."""
        full_result = self.answer_question(question)
        return {
            **full_result,
            "execution_time": full_result["execution_time_seconds"],
        }

    def answer_question(self, question: str) -> NaiveRagQueryResult:
        """
        Run one question through the full naive-RAG pipeline and return a structured result.

        Steps: retrieve similar chunks → format context → LLM generates answer.
        """
        start_time = time.perf_counter()
        try:
            rag_chain = self._require_rag_chain()
            retriever = self._require_retriever()

            retrieved_documents: list[Document] = retriever.invoke(question)
            generated_answer: str = rag_chain.invoke(question)
            elapsed_seconds = time.perf_counter() - start_time

            return NaiveRagQueryResult(
                query=question,
                answer=generated_answer,
                source_type="naive_rag",
                execution_time_seconds=elapsed_seconds,
                num_chunks_retrieved=len(retrieved_documents),
                retrieved_chunks_info=self._extract_chunks_info(retrieved_documents),
                success=True,
                error=None,
            )
        except (ValueError, RuntimeError) as query_error:
            elapsed_seconds = time.perf_counter() - start_time
            LOGGER.exception("Query failed")
            return NaiveRagQueryResult(
                query=question,
                answer=f"Error processing query: {query_error}",
                source_type="naive_rag",
                execution_time_seconds=elapsed_seconds,
                num_chunks_retrieved=0,
                retrieved_chunks_info=[],
                success=False,
                error=str(query_error),
            )

    def get_vector_store_stats(self) -> dict[str, object]:
        """Return basic statistics about the current vector database."""
        if self._vectorstore is None:
            return {"error": "Vector store not initialized"}

        try:
            total_chunks = self._count_stored_chunks(self._vectorstore)
            sample_files = self._sample_source_file_names(
                self._vectorstore, max_samples=10,
            )
            return {
                "total_chunks": total_chunks,
                "sample_source_files": sample_files,
                "embedding_model": self._model_names.embedding_model_name,
                "chunk_size": self._chunking_params.chunk_size,
                "chunk_overlap": self._chunking_params.chunk_overlap,
            }
        except RuntimeError as stats_error:
            return {"error": f"Could not get stats: {stats_error}"}

    def save_results_to_json(
            self,
            query_results: list[NaiveRagQueryResult],
            *,
            output_file_path: Path,
    ) -> None:
        """Persist query results together with metadata to a JSON file."""
        payload = {
            "test_metadata": {
                "system_type": "naive_rag",
                "test_queries": len(query_results),
                "database_stats": self.get_vector_store_stats(),
            },
            "results": query_results,
        }
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )

    def _load_all_cv_documents(self) -> list[Document]:
        """Load every CV PDF from the configured directory and enrich metadata."""
        pdf_file_paths = sorted(
            self._directory_paths.cv_pdf_directory.glob("*.pdf"),
        )
        if not pdf_file_paths:
            raise FileNotFoundError(
                f"No PDF files found in {self._directory_paths.cv_pdf_directory}"
            )

        all_loaded_pages: list[Document] = []
        for pdf_path in pdf_file_paths:
            try:
                loaded_pages = PyPDFLoader(str(pdf_path)).load()
                for page_document in loaded_pages:
                    page_document.metadata.update(
                        {
                            "source_file": pdf_path.name,
                            "document_type": "cv",
                            "person_name": pdf_path.stem,
                        }
                    )
                all_loaded_pages.extend(loaded_pages)
            except (OSError, ValueError):
                LOGGER.exception("Could not load %s", pdf_path)

        if not all_loaded_pages:
            raise RuntimeError(
                f"All PDF loads failed in {self._directory_paths.cv_pdf_directory}"
            )

        LOGGER.info(
            "Loaded %d pages from %d CV files",
            len(all_loaded_pages),
            len(pdf_file_paths),
        )
        return all_loaded_pages

    @staticmethod
    def _join_document_contents(documents: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in documents)

    @staticmethod
    def _extract_chunks_info(documents: list[Document]) -> list[RetrievedChunkInfo]:
        chunks_info: list[RetrievedChunkInfo] = []
        for chunk_index, document in enumerate(documents):
            content_text = document.page_content
            chunks_info.append(
                RetrievedChunkInfo(
                    chunk_index=chunk_index,
                    source_file=str(document.metadata.get("source_file", "unknown")),
                    person_name=str(document.metadata.get("person_name", "unknown")),
                    content_preview=(
                        content_text[:200] + "..."
                        if len(content_text) > 200
                        else content_text
                    ),
                )
            )
        return chunks_info

    def _require_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            raise RuntimeError(
                "Vector store not initialized. Call create_or_load_vector_store() first."
            )
        return self._vectorstore

    def _require_retriever(self):
        if self._similarity_retriever is None:
            raise RuntimeError(
                "Retriever not initialized. Call build_rag_chain() first."
            )
        return self._similarity_retriever

    def _require_rag_chain(self):
        if self._rag_chain is None:
            raise RuntimeError(
                "RAG chain not initialized. Call build_rag_chain() first."
            )
        return self._rag_chain

    @staticmethod
    def _count_stored_chunks(vectorstore: Chroma) -> int:
        collection = getattr(vectorstore, "_collection", None)
        if collection is None:
            raise RuntimeError("Chroma collection is not available.")
        return int(collection.count())

    @staticmethod
    def _sample_source_file_names(
            vectorstore: Chroma, *, max_samples: int,
    ) -> list[str]:
        collection = getattr(vectorstore, "_collection", None)
        if collection is None:
            return []

        try:
            collection_data = collection.get(
                include=["metadatas"], limit=max_samples,
            )
            metadata_list = collection_data.get("metadatas") or []
            unique_source_files = {
                str(metadata.get("source_file", "unknown"))
                for metadata in metadata_list
                if isinstance(metadata, dict)
            }
            return sorted(unique_source_files)[:max_samples]
        except (KeyError, AttributeError):
            return []

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, object]:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    @classmethod
    def _build_directory_paths(cls, parsed_config: dict[str, object]) -> CvDirectoryPaths:
        output_section = parsed_config.get("output")
        if not isinstance(output_section, dict):
            raise ValueError("Invalid config: expected [output] section")

        cv_pdf_dir_value = output_section.get("programmers_dir")
        if not isinstance(cv_pdf_dir_value, str) or not cv_pdf_dir_value.strip():
            raise ValueError(
                "Invalid config: output.programmers_dir must be a non-empty string"
            )

        return CvDirectoryPaths(
            cv_pdf_directory=Path(cv_pdf_dir_value),
            vector_store_directory=cls.VECTOR_STORE_DIR,
            results_directory=cls.RESULTS_DIR,
        )


def run_smoke_test() -> None:
    """
    Smoke test that builds a naive-RAG pipeline over CV PDFs and runs
    a handful of typical HR queries to verify the system works end-to-end.
    """
    print_section_header("Naive RAG Smoke Test")

    naive_rag = NaiveCvRag()
    naive_rag.initialize()

    database_stats = naive_rag.get_vector_store_stats()
    sample_source_files: list[str] = database_stats.get("sample_source_files", [])  # type: ignore[assignment]
    sample_files_preview = ", ".join(sample_source_files[:3])
    print(textwrap.dedent(f"""\

        Database Statistics:
        Total chunks: {database_stats.get('total_chunks', 'unknown')}
        Sample files: {sample_files_preview}"""))

    test_queries = [
        "How many people have Python skills?",
        "List people with AWS certifications",
        "What is the most common programming language?",
        "Who worked at Google?",
        "Find people with both React and Node.js skills",
    ]

    collected_results: list[NaiveRagQueryResult] = []
    for query_number, query_text in enumerate(test_queries, start=1):
        print(f"\n[{query_number}/{len(test_queries)}] Query: {query_text}\n"
              f"{'-' * 40}")

        single_result = naive_rag.answer_question(query_text)
        collected_results.append(single_result)

        if single_result["success"]:
            print(textwrap.dedent(f"""\
                Answer: {single_result['answer']}
                Execution time: {single_result['execution_time_seconds']:.2f}s
                Chunks used: {single_result['num_chunks_retrieved']}"""))
        else:
            print(f"Error: {single_result['answer']}")

    results_output_path = Path("results") / "naive_rag_test_results.json"
    naive_rag.save_results_to_json(
        collected_results, output_file_path=results_output_path,
    )
    print(f"\nSaved test results to: {results_output_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv(override=True)

    print_section_header("Naive RAG Baseline System for CV Data")
    run_smoke_test()


if __name__ == "__main__":
    main()
