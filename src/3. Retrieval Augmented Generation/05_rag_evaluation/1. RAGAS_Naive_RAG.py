"""
📊 RAGAS Evaluation of a Naive RAG Pipeline - Educational Walkthrough
=====================================================================

This script demonstrates how to evaluate a simple (naive) RAG system using the
RAGAS (Retrieval Augmented Generation Assessment) framework. You will learn how
to measure retrieval and generation quality with industry-standard metrics.

🎯 What You'll Learn:
- How to build a minimal RAG pipeline for evaluation purposes
- How to generate ground truth answers with an expert LLM
- How to cache and reuse ground truth datasets across runs
- How RAGAS metrics quantify retrieval precision, recall, faithfulness,
  answer relevancy, and factual correctness

🔧 Prerequisites:
- Azure OpenAI credentials configured in .env file
- Scientist biography .txt files in the data directory
- Python 3.13+ with langchain, ragas, and python-dotenv packages
"""

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)


@dataclass(frozen=True)
class RagPipelineConfig:
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    retriever_top_k: int


@dataclass(frozen=True)
class EvaluationConfig:
    evaluator_model: str
    evaluator_temperature: float
    expert_model: str


@dataclass(frozen=True)
class PathsConfig:
    data_directory: Path
    ground_truth_output_directory: Path
    ground_truth_filename: str

    @property
    def ground_truth_file_path(self) -> Path:
        return self.ground_truth_output_directory / self.ground_truth_filename


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def load_and_chunk_documents(
        source_directory: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
) -> list[Document]:
    """
    Loads all .txt files from a directory and splits them into overlapping chunks
    using a recursive character splitter. Chunking is the first critical step in
    any RAG pipeline — it determines how much context each retrieved piece carries.

    • chunk_size controls the maximum number of characters per chunk.
    • chunk_overlap ensures continuity between adjacent chunks so that sentences
      sitting on a boundary are not lost.
    """
    print_section_header("STEP 1: LOADING & CHUNKING DOCUMENTS")
    print(textwrap.dedent(load_and_chunk_documents.__doc__))

    directory_loader = DirectoryLoader(str(source_directory), glob="*.txt")
    raw_documents = directory_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    document_chunks = text_splitter.split_documents(raw_documents)

    print(f"Loaded {len(raw_documents)} documents → {len(document_chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return document_chunks


def create_vector_store(
        document_chunks: list[Document],
        *,
        embedding_model: str,
) -> InMemoryVectorStore:
    """
    Creates an in-memory vector store from document chunks. The store converts
    each chunk into an embedding vector and indexes it for fast similarity search
    — this is the "Retrieval" foundation of RAG.
    """
    embeddings = AzureOpenAIEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=document_chunks)
    return vector_store


def build_rag_chain(
        vector_store: InMemoryVectorStore,
        *,
        chat_model: str,
        retriever_top_k: int,
) -> Runnable:
    """
    Assembles a complete Naive RAG chain: retriever → prompt → LLM → output parser.

    This is the simplest possible RAG architecture — no reranking, no query
    expansion, no hybrid search. Evaluating this baseline lets you measure how
    much value advanced retrieval techniques add later.
    """
    print_section_header("STEP 2: BUILDING THE NAIVE RAG CHAIN")
    print(textwrap.dedent(build_rag_chain.__doc__))

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retriever_top_k},
    )

    rag_llm = AzureChatOpenAI(model=chat_model)

    rag_prompt = ChatPromptTemplate.from_template(textwrap.dedent("""\
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        Question: {question}

        Context: {context}

        Answer:"""))

    rag_chain: Runnable = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | rag_llm
            | StrOutputParser()
    )

    print(f"RAG chain ready  |  model={chat_model}  |  top_k={retriever_top_k}")
    return rag_chain


def generate_ground_truth_answers(
        evaluation_questions: list[str],
        *,
        source_directory: Path,
        expert_llm: AzureChatOpenAI,
) -> list[str]:
    """
    Generates reference ("ground truth") answers by feeding the *complete*
    biography texts to a powerful expert LLM. These answers serve as the gold
    standard against which the RAG pipeline's responses are compared during
    RAGAS evaluation.
    """
    directory_loader = DirectoryLoader(str(source_directory), glob="*.txt")
    biography_documents = directory_loader.load()
    full_biography_text = "\n\n".join(
        doc.page_content for doc in biography_documents
    )

    expert_answers: list[str] = []
    for question in evaluation_questions:
        expert_prompt = textwrap.dedent(f"""\
            You are a domain expert with complete knowledge of these scientists.
            Based on the following complete biographies, provide a comprehensive, accurate answer.

            Complete Biographies:
            {full_biography_text}

            Question: {question}

            Provide a detailed, factually accurate answer:""")
        expert_answers.append(expert_llm.invoke(expert_prompt).content)
    return expert_answers


def load_or_generate_ground_truths(
        evaluation_questions: list[str],
        *,
        paths_config: PathsConfig,
        expert_llm: AzureChatOpenAI,
) -> list[str]:
    """
    Caches ground truth answers to a JSON file so that expensive expert-LLM
    generation only happens once. On subsequent runs the cached file is loaded,
    saving both time and API cost.

    Cache format (JSON array):
      [{"question": "...", "ground_truth": "..."}, ...]
    """
    print_section_header("STEP 3: GROUND TRUTH GENERATION / LOADING")
    print(textwrap.dedent(load_or_generate_ground_truths.__doc__))

    ground_truth_file = paths_config.ground_truth_file_path

    if ground_truth_file.exists():
        print(f"Loading existing ground truth dataset from {ground_truth_file}")
        with open(ground_truth_file, "r") as file_handle:
            cached_data = json.load(file_handle)
        return [item["ground_truth"] for item in cached_data]

    print("Generating ground truth answers using expert LLM...")
    generated_answers = generate_ground_truth_answers(
        evaluation_questions,
        source_directory=paths_config.data_directory,
        expert_llm=expert_llm,
    )

    paths_config.ground_truth_output_directory.mkdir(exist_ok=True)
    serializable_records = [
        {"question": question, "ground_truth": answer}
        for question, answer in zip(evaluation_questions, generated_answers)
    ]
    with open(ground_truth_file, "w") as file_handle:
        json.dump(serializable_records, file_handle, indent=2)
    print(f"Ground truth dataset saved to {ground_truth_file}")

    return generated_answers


def collect_evaluation_samples(
        evaluation_questions: list[str],
        ground_truth_answers: list[str],
        *,
        rag_chain: Runnable,
        vector_store: InMemoryVectorStore,
        retriever_top_k: int,
) -> list[SingleTurnSample]:
    """
    Runs each evaluation question through the RAG chain and pairs the generated
    answer with its retrieved contexts and ground truth. The resulting
    SingleTurnSample objects are the input format required by RAGAS.
    """
    print_section_header("STEP 4: COLLECTING RAG RESPONSES FOR EVALUATION")
    print(textwrap.dedent(collect_evaluation_samples.__doc__))

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retriever_top_k},
    )

    samples: list[SingleTurnSample] = []
    for question, ground_truth in zip(evaluation_questions, ground_truth_answers):
        rag_answer = rag_chain.invoke(question)
        retrieved_context_texts = [
            doc.page_content for doc in retriever.invoke(question)
        ]
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=rag_answer,
                retrieved_contexts=retrieved_context_texts,
                reference=ground_truth,
            )
        )
        print(f"  ✓ {question}")

    return samples


def run_ragas_evaluation(
        samples: list[SingleTurnSample],
        *,
        evaluation_config: EvaluationConfig,
) -> None:
    """
    Evaluates the collected samples with five RAGAS metrics and prints the results.

    Metrics measured:
    • Context Precision  – Are the retrieved chunks relevant to the question?
    • Context Recall     – Do retrieved chunks cover all ground-truth information?
    • Faithfulness       – Is the answer grounded in the retrieved context?
    • Answer Relevancy   – Does the answer address the question directly?
    • Factual Correctness – Does the answer match the ground truth facts?
    """
    print_section_header("STEP 5: RAGAS EVALUATION RESULTS")
    print(textwrap.dedent(run_ragas_evaluation.__doc__))

    evaluator_llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            model=evaluation_config.evaluator_model,
            temperature=evaluation_config.evaluator_temperature,
        )
    )

    ragas_metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
    ]

    evaluation_result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=ragas_metrics,
    )

    evaluation_dataframe = evaluation_result.to_pandas()
    metric_columns = [
        col for col in evaluation_dataframe.columns
        if col not in {"user_input", "response", "retrieved_contexts", "reference"}
    ]
    for metric_column in metric_columns:
        mean_score = evaluation_dataframe[metric_column].mean()
        print(f"  {metric_column}: {mean_score:.3f}")


def main() -> None:
    load_dotenv(override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment")

    rag_pipeline_config = RagPipelineConfig(
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4o-mini",
        chunk_size=1000,
        chunk_overlap=200,
        retriever_top_k=3,
    )

    evaluation_config = EvaluationConfig(
        evaluator_model="gpt-4.1",
        evaluator_temperature=0,
        expert_model="gpt-5-mini",
    )

    paths_config = PathsConfig(
        data_directory=Path("src/3. Retrieval Augmented Generation/05_rag_evaluation/data/scientists_bios"),
        ground_truth_output_directory=Path("src/3. Retrieval Augmented Generation/05_rag_evaluation/data"),
        ground_truth_filename="ground_truth_dataset.json",
    )

    if not paths_config.data_directory.exists():
        raise FileNotFoundError(
            f"Data directory not found: {paths_config.data_directory}"
        )

    document_chunks = load_and_chunk_documents(
        paths_config.data_directory,
        chunk_size=rag_pipeline_config.chunk_size,
        chunk_overlap=rag_pipeline_config.chunk_overlap,
    )

    vector_store = create_vector_store(
        document_chunks,
        embedding_model=rag_pipeline_config.embedding_model,
    )

    rag_chain = build_rag_chain(
        vector_store,
        chat_model=rag_pipeline_config.chat_model,
        retriever_top_k=rag_pipeline_config.retriever_top_k,
    )

    scientist_evaluation_questions = [
        "What did Marie Curie win Nobel Prizes for?",
        "What is Einstein's theory of relativity about?",
        "What are Newton's three laws of motion?",
        "What did Charles Darwin discover?",
        "What was Ada Lovelace's contribution to computing?",
    ]

    expert_llm = AzureChatOpenAI(model=evaluation_config.expert_model)

    ground_truth_answers = load_or_generate_ground_truths(
        scientist_evaluation_questions,
        paths_config=paths_config,
        expert_llm=expert_llm,
    )

    evaluation_samples = collect_evaluation_samples(
        scientist_evaluation_questions,
        ground_truth_answers,
        rag_chain=rag_chain,
        vector_store=vector_store,
        retriever_top_k=rag_pipeline_config.retriever_top_k,
    )

    run_ragas_evaluation(
        evaluation_samples,
        evaluation_config=evaluation_config,
    )


if __name__ == "__main__":
    main()
