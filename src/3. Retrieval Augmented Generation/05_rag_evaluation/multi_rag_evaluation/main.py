"""
🏁 Multi-RAG Comparative Evaluation — Entry Point
===================================================

Loads scientist biography documents, initializes five different RAG strategies
(Naive, Metadata Filtering, Hybrid Search, Query Expansion, Reranking),
evaluates each with RAGAS metrics, and produces a side-by-side comparison table.

🎯 What You'll Learn:
- How different retrieval strategies affect RAG quality metrics
- How to set up a reproducible, automated evaluation pipeline
- How to compare systems fairly using the same ground truth and questions

🔧 Prerequisites:
- Azure OpenAI credentials configured in .env file
- Scientist biography .txt files in data/scientists_bios/
- Python 3.13+ with langchain, ragas, sentence-transformers, rank-bm25
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI

from config import SETTINGS
from evaluation import RAGEvaluator
from rag_systems import (
    HybridSearchRAG,
    MetadataFilteringRAG,
    NaiveRAG,
    QueryExpansionRAG,
    RerankingRAG,
)
from rag_systems.base_rag import BaseRAG


def print_section_header(title: str) -> None:
    separator = "=" * 80
    print(f"\n{separator}\n{title}\n{separator}")


def load_and_chunk_documents(
        source_directory: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
) -> list[Document]:
    """Load all .txt biographies and split them into overlapping chunks."""
    directory_loader = DirectoryLoader(str(source_directory), glob="*.txt")
    raw_documents = directory_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(raw_documents)


def build_all_rag_systems(document_chunks: list[Document]) -> list[BaseRAG]:
    """Instantiate, build, and return only the successfully built RAG systems."""
    print_section_header("🏗️ BUILDING RAG SYSTEMS")

    rag_system_classes = [
        NaiveRAG,
        MetadataFilteringRAG,
        HybridSearchRAG,
        QueryExpansionRAG,
        RerankingRAG,
    ]

    successfully_built: list[BaseRAG] = []

    for rag_class in rag_system_classes:
        rag_instance = rag_class(document_chunks, SETTINGS)
        try:
            print(f"   Building {rag_instance.name}...")
            rag_instance.build()
            successfully_built.append(rag_instance)
            print(f"   ✓ {rag_instance.name} built successfully")
        except Exception as build_error:
            print(f"   ✗ Failed to build {rag_instance.name}: {build_error}")

    print(f"✓ Successfully built {len(successfully_built)} RAG systems")
    return successfully_built


def run_comparative_evaluation(
        successfully_built_systems: list[BaseRAG],
        *,
        data_directory: Path,
) -> None:
    """Set up the evaluator, run all systems, print results and save CSV."""
    print_section_header("📊 SETTING UP EVALUATION")

    expert_llm = AzureChatOpenAI(model=SETTINGS.models.expert_model)
    evaluator_llm = AzureChatOpenAI(model=SETTINGS.models.evaluator_model)

    evaluator = RAGEvaluator(
        expert_llm=expert_llm,
        evaluator_llm=evaluator_llm,
    )
    print("✓ Evaluator initialized")

    evaluation_questions = list(SETTINGS.evaluation_questions.questions)
    print(f"\n🏃 Running evaluations... ({len(evaluation_questions)} questions)")

    evaluation_results = evaluator.compare_systems(
        rag_systems=successfully_built_systems,
        evaluation_questions=evaluation_questions,
        data_directory=str(data_directory),
    )

    if not evaluation_results:
        print("❌ No evaluation results obtained")
        return

    print(f"✓ Evaluated {len(evaluation_results)} systems")

    print_section_header("📈 PROCESSING RESULTS")

    comparison_dataframe = evaluator.create_comparison_dataframe(evaluation_results)
    evaluator.print_comparison_table(comparison_dataframe)

    results_directory = "results"
    evaluator.save_results(comparison_dataframe, results_directory=results_directory)

    print(f"\n🎉 Evaluation completed successfully!\n"
          f"📁 Detailed results saved in '{results_directory}/' directory")


def main() -> None:
    load_dotenv(override=True)

    print_section_header("MULTI-RAG SYSTEM COMPARATIVE EVALUATION")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment")

    data_directory = Path("data/scientists_bios")
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")

    print(f"\n📚 Loading documents from {data_directory}...")
    document_chunks = load_and_chunk_documents(
        data_directory,
        chunk_size=SETTINGS.chunking.chunk_size,
        chunk_overlap=SETTINGS.chunking.chunk_overlap,
    )
    print(f"✓ Created {len(document_chunks)} chunks")

    successfully_built_systems = build_all_rag_systems(document_chunks)

    if not successfully_built_systems:
        print("❌ No RAG systems built successfully")
        return

    run_comparative_evaluation(
        successfully_built_systems,
        data_directory=data_directory,
    )


if __name__ == "__main__":
    main()
