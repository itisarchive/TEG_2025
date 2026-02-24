"""
🎯 Re-ranking for Post-Retrieval Optimization
==============================================

Demonstrates cross-encoder re-ranking, LLM-based relevance scoring, and result
diversity optimization to improve retrieval quality after initial vector search.

🎯 What You'll Learn:
- Cross-encoder models for pairwise query-document scoring
- LLM-based relevance scoring with structured output
- Diversity-aware re-ranking to avoid redundant results
- Ensemble re-ranking combining multiple scoring methods
- Performance vs. quality trade-offs across re-ranking strategies

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, sentence-transformers
- Scientist biography .txt files in data/scientists_bios/
"""

import os
import textwrap
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from sentence_transformers import CrossEncoder

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/04_advanced_retrieval/data/scientists_bios"

LLM_RELEVANCE_PROMPT = ChatPromptTemplate.from_template("""\
Rate the relevance of the following document to the query on a scale of 1-10.
Consider how well the document answers the question or provides relevant information.

Query: {query}

Document: {document}

Provide only a numeric score (1-10) with brief explanation:
Score: [number]
Reason: [brief explanation]
""")

RERANK_RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are an assistant for question-answering tasks about scientists and their contributions.
The context below was retrieved and then re-ranked to ensure the most relevant information appears first.

Use the following pieces of re-ranked context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Re-ranked context: {context}

Answer:
""")

ScoredDocument = tuple[Document, float]

DEFAULT_LLM_RELEVANCE_SCORE = 5.0


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def print_scored_results(results: list[ScoredDocument], *, label: str, score_label: str = "score") -> None:
    print(f"\n   {label}:")
    for rank, (doc, score) in enumerate(results, start=1):
        scientist_name = doc.metadata["scientist_name"]
        content_preview = doc.page_content[:60] + "..."
        print(f"      {rank}. {scientist_name} ({score_label}: {score:.3f}): {content_preview}")


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing stats for a single re-ranking method across multiple queries."""

    total_time: float
    queries_processed: int

    @property
    def average_time(self) -> float:
        return self.total_time / self.queries_processed if self.queries_processed > 0 else 0.0


@dataclass(frozen=True)
class EffectivenessStats:
    """Diversity and coverage stats for a re-ranking method on a single query."""

    unique_scientist_count: int
    total_document_count: int

    @property
    def diversity_ratio(self) -> float:
        return self.unique_scientist_count / self.total_document_count if self.total_document_count > 0 else 0.0


@dataclass
class CrossEncoderRegistry:
    """Manages loading and access to cross-encoder models."""

    models: dict[str, CrossEncoder] = field(default_factory=dict)

    def load_model(self, *, name: str, model_id: str) -> None:
        try:
            self.models[name] = CrossEncoder(model_id)
            print(f"   ✅ {name} cross-encoder loaded")
        except Exception as load_error:
            print(f"   ⚠️ Failed to load {name} model: {load_error}")

    def has_models(self) -> bool:
        return len(self.models) > 0

    def first_model_name(self) -> str:
        return next(iter(self.models))


def load_and_index_scientist_chunks() -> tuple[list[Document], InMemoryVectorStore]:
    """
    Loads scientist biographies, attaches metadata, splits into chunks,
    and indexes them in a vector store for similarity search.
    """
    print_section_header("1️⃣  LOADING DOCUMENTS FOR RE-RANKING")

    loader = DirectoryLoader(SCIENTISTS_BIOS_DIR, glob="*.txt")
    raw_documents = loader.load()

    for doc in raw_documents:
        filename = os.path.basename(doc.metadata["source"]).replace(".txt", "")
        doc.metadata["scientist_name"] = filename

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_documents)

    print(f"   Loaded {len(raw_documents)} documents, created {len(chunks)} chunks")

    azure_embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(azure_embeddings)
    vector_store.add_documents(documents=chunks)

    print(f"   ✅ Vector store ready with {len(chunks)} indexed chunks")
    return chunks, vector_store


def load_cross_encoder_models() -> CrossEncoderRegistry:
    """
    Cross-encoders score query-document pairs jointly (unlike bi-encoders which
    encode them independently). This produces more accurate relevance scores
    at the cost of higher latency — ideal for re-ranking a small candidate set.
    """
    print_section_header("2️⃣  LOADING CROSS-ENCODER MODELS")
    print(textwrap.dedent(load_cross_encoder_models.__doc__))

    registry = CrossEncoderRegistry()
    registry.load_model(name="ms-marco", model_id="cross-encoder/ms-marco-MiniLM-L-6-v2")
    registry.load_model(name="qnli", model_id="cross-encoder/qnli-electra-base")

    if not registry.has_models():
        print("   ⚠️ No cross-encoders loaded, using fallback scoring")

    return registry


def rerank_by_cross_encoder(
        registry: CrossEncoderRegistry,
        query: str,
        documents: list[Document],
        *,
        model_name: str = "ms-marco",
        max_results: int | None = None,
) -> list[ScoredDocument]:
    """Re-ranks documents using a cross-encoder model for pairwise query-document scoring."""
    if model_name not in registry.models:
        print(f"   ⚠️ Model {model_name} not available, returning original order")
        return [(doc, 0.0) for doc in documents]

    encoder = registry.models[model_name]
    query_document_pairs = [(query, doc.page_content) for doc in documents]
    relevance_scores = encoder.predict(query_document_pairs)

    scored_pairs = sorted(
        zip(documents, relevance_scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    result = [(doc, float(score)) for doc, score in scored_pairs]
    return result[:max_results] if max_results else result


def _extract_relevance_score(llm_response_content: str) -> float:
    """Parses a numeric score from the LLM's 'Score: X' response line."""
    for line in llm_response_content.split("\n"):
        if line.strip().startswith("Score:"):
            score_text = line.replace("Score:", "").strip()
            try:
                return float(score_text.split()[0])
            except (ValueError, IndexError):
                return DEFAULT_LLM_RELEVANCE_SCORE
    return DEFAULT_LLM_RELEVANCE_SCORE


def rerank_by_llm_relevance(
        llm: AzureChatOpenAI,
        query: str,
        documents: list[Document],
) -> list[ScoredDocument]:
    """Uses the LLM to score each document's relevance to the query on a 1–10 scale."""
    scored_documents: list[ScoredDocument] = []

    for doc in documents:
        try:
            llm_response = llm.invoke(
                LLM_RELEVANCE_PROMPT.format(
                    query=query,
                    document=doc.page_content[:500],
                )
            )
            relevance_score = _extract_relevance_score(llm_response.content)
        except Exception as scoring_error:
            print(f"   ⚠️ LLM scoring failed for document: {scoring_error}")
            relevance_score = DEFAULT_LLM_RELEVANCE_SCORE

        scored_documents.append((doc, relevance_score))

    scored_documents.sort(key=lambda pair: pair[1], reverse=True)
    return scored_documents


def rerank_for_diversity(
        scored_documents: list[ScoredDocument],
        *,
        coverage_ratio: float = 0.8,
) -> list[ScoredDocument]:
    """
    Re-ranks to promote scientist diversity while preserving relevance order.
    Documents from not-yet-seen scientists are prioritized; once coverage_ratio
    of the original list size is reached, remaining documents fill the tail.
    """
    if not scored_documents:
        return scored_documents

    diverse_results: list[ScoredDocument] = [scored_documents[0]]
    seen_scientists: set[str] = {scored_documents[0][0].metadata["scientist_name"]}
    target_count = int(len(scored_documents) * coverage_ratio)

    for doc, score in scored_documents[1:]:
        scientist_name = doc.metadata["scientist_name"]
        if scientist_name not in seen_scientists or len(diverse_results) < 3:
            diverse_results.append((doc, score))
            seen_scientists.add(scientist_name)
        if len(diverse_results) >= target_count:
            break

    already_selected = {id(doc) for doc, _ in diverse_results}
    for doc, score in scored_documents[1:]:
        if id(doc) not in already_selected and len(diverse_results) < len(scored_documents):
            diverse_results.append((doc, score))

    return diverse_results


def demonstrate_individual_reranking(
        vector_store: InMemoryVectorStore,
        registry: CrossEncoderRegistry,
        llm: AzureChatOpenAI,
) -> None:
    """
    Compares three re-ranking approaches on the same initial retrieval set:
    • Cross-encoder: pairwise neural scoring (most accurate, slower)
    • LLM relevance: generative scoring with reasoning (flexible, slowest)
    • Diversity: promotes variety across scientists (fast, heuristic)
    """
    print_section_header("3️⃣  TESTING INDIVIDUAL RE-RANKING METHODS")
    print(textwrap.dedent(demonstrate_individual_reranking.__doc__))

    test_query = "What did Einstein discover about the universe?"
    initial_results = vector_store.similarity_search_with_score(test_query, k=6)

    print(f"\n   🔍 Test query: {test_query}")
    print_scored_results(initial_results, label="📊 Initial vector search results")

    documents_for_reranking = [doc for doc, _score in initial_results]

    if registry.has_models():
        for encoder_name in registry.models:
            cross_encoder_results = rerank_by_cross_encoder(
                registry, test_query, documents_for_reranking,
                model_name=encoder_name, max_results=4,
            )
            print_scored_results(cross_encoder_results, label=f"🎯 {encoder_name.upper()} re-ranking")

    llm_results = rerank_by_llm_relevance(llm, test_query, documents_for_reranking[:4])
    print_scored_results(llm_results, label="🤖 LLM relevance scoring")

    diversity_results = rerank_for_diversity(initial_results[:6])
    print_scored_results(diversity_results, label="🌈 Diversity re-ranking", score_label="orig score")


def ensemble_rerank(
        llm: AzureChatOpenAI,
        registry: CrossEncoderRegistry,
        query: str,
        documents: list[Document],
        *,
        method_names: tuple[str, ...] = ("cross_encoder", "llm"),
        method_weights: tuple[float, ...] | None = None,
) -> list[ScoredDocument]:
    """
    Combines scores from multiple re-ranking methods using weighted averaging.
    Each method scores all documents independently, scores are normalized,
    then combined with the specified weights.
    """
    active_weights = method_weights or tuple(1.0 for _ in method_names)
    total_weight = sum(active_weights)
    normalized_weights = [weight / total_weight for weight in active_weights]

    all_method_scores: list[list[float]] = []

    for method_name in method_names:
        if method_name == "cross_encoder" and registry.has_models():
            encoder_name = registry.first_model_name()
            reranked = rerank_by_cross_encoder(registry, query, documents, model_name=encoder_name)
            score_by_content = {doc.page_content: score for doc, score in reranked}
            all_method_scores.append([score_by_content.get(doc.page_content, 0.0) for doc in documents])

        elif method_name == "llm":
            reranked = rerank_by_llm_relevance(llm, query, documents)
            max_score = max((score for _, score in reranked), default=10.0)
            score_by_content = {doc.page_content: score / max_score for doc, score in reranked}
            all_method_scores.append([score_by_content.get(doc.page_content, 0.0) for doc in documents])

    combined_scores: list[ScoredDocument] = []
    for doc_index, doc in enumerate(documents):
        weighted_sum = sum(
            normalized_weights[method_index] * method_scores[doc_index]
            for method_index, method_scores in enumerate(all_method_scores)
            if doc_index < len(method_scores)
        )
        combined_scores.append((doc, weighted_sum))

    combined_scores.sort(key=lambda pair: pair[1], reverse=True)
    return combined_scores


def demonstrate_ensemble_reranking(
        vector_store: InMemoryVectorStore,
        registry: CrossEncoderRegistry,
        llm: AzureChatOpenAI,
) -> None:
    """
    Ensemble re-ranking fuses scores from multiple methods (cross-encoder + LLM)
    with configurable weights. This often outperforms any single method by
    combining neural precision with generative reasoning.
    """
    print_section_header("4️⃣  ENSEMBLE RE-RANKING")
    print(textwrap.dedent(demonstrate_ensemble_reranking.__doc__))

    test_query = "What did Einstein discover about the universe?"
    initial_results = vector_store.similarity_search(test_query, k=5)

    active_methods: list[str] = ["llm"]
    active_weights: list[float] = [1.0]
    if registry.has_models():
        active_methods.append("cross_encoder")
        active_weights = [0.6, 0.4]

    print(f"   🎭 Ensemble methods: {active_methods}")

    ensemble_results = ensemble_rerank(
        llm, registry, test_query, initial_results,
        method_names=tuple(active_methods),
        method_weights=tuple(active_weights),
    )
    print_scored_results(ensemble_results, label="🎭 Ensemble re-ranking", score_label="ensemble")


def benchmark_reranking_methods(
        vector_store: InMemoryVectorStore,
        registry: CrossEncoderRegistry,
        llm: AzureChatOpenAI,
        queries: list[str],
        *,
        results_per_query: int = 5,
) -> dict[str, BenchmarkResult]:
    """Measures average latency of each re-ranking method across multiple queries."""
    method_names = ["baseline", "llm"]
    if registry.has_models():
        method_names.append("cross_encoder")

    timings: dict[str, dict[str, float | int]] = {
        method: {"total_time": 0.0, "queries_processed": 0} for method in method_names
    }

    for benchmark_query in queries:
        initial_documents = vector_store.similarity_search(benchmark_query, k=results_per_query * 2)

        for method_name in method_names:
            start_time = time.time()

            if method_name == "baseline":
                _final = initial_documents[:results_per_query]
            elif method_name == "cross_encoder" and registry.has_models():
                encoder_name = registry.first_model_name()
                _final = rerank_by_cross_encoder(
                    registry, benchmark_query, initial_documents,
                    model_name=encoder_name, max_results=results_per_query,
                )
            elif method_name == "llm":
                _final = rerank_by_llm_relevance(
                    llm, benchmark_query, initial_documents[:results_per_query],
                )

            elapsed = time.time() - start_time
            timings[method_name]["total_time"] += elapsed
            timings[method_name]["queries_processed"] += 1

    return {
        method: BenchmarkResult(
            total_time=data["total_time"],
            queries_processed=int(data["queries_processed"]),
        )
        for method, data in timings.items()
    }


def demonstrate_performance_benchmark(
        vector_store: InMemoryVectorStore,
        registry: CrossEncoderRegistry,
        llm: AzureChatOpenAI,
) -> None:
    """
    Benchmarks latency across re-ranking methods. Baseline (no re-ranking) is
    near-instant, cross-encoders add moderate latency, and LLM scoring is
    slowest due to per-document API calls — but often most accurate.
    """
    print_section_header("5️⃣  PERFORMANCE VS QUALITY ANALYSIS")
    print(textwrap.dedent(demonstrate_performance_benchmark.__doc__))

    benchmark_queries = [
        "Einstein's theories",
        "Newton's discoveries",
        "Marie Curie research",
    ]

    print("   ⏱️ Benchmarking re-ranking methods:")
    benchmark_results = benchmark_reranking_methods(
        vector_store, registry, llm, benchmark_queries, results_per_query=3,
    )

    print(f"   {'Method':<15} {'Avg Time (s)':<15} {'Queries':<10}")
    print("   " + "-" * 45)

    for method_name, result in benchmark_results.items():
        print(f"   {method_name:<15} {result.average_time:<15.3f} {result.queries_processed:<10}")


def run_reranking_rag_chain(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
        registry: CrossEncoderRegistry,
        question: str,
        *,
        rerank_method: str = "llm",
        max_results: int = 4,
) -> tuple[str, list[Document], list[float]]:
    """
    Full RAG cycle with post-retrieval re-ranking: retrieves a broad candidate set,
    re-ranks with the chosen method, then generates an answer from the top results.
    """
    candidate_documents = vector_store.similarity_search(question, k=max_results * 2)

    if rerank_method == "cross_encoder" and registry.has_models():
        encoder_name = registry.first_model_name()
        reranked = rerank_by_cross_encoder(
            registry, question, candidate_documents,
            model_name=encoder_name, max_results=max_results,
        )
    elif rerank_method == "llm":
        reranked = rerank_by_llm_relevance(llm, question, candidate_documents[:max_results + 2])
        reranked = reranked[:max_results]
    elif rerank_method == "ensemble":
        active_methods: list[str] = ["llm"]
        if registry.has_models():
            active_methods.append("cross_encoder")
        reranked = ensemble_rerank(
            llm, registry, question, candidate_documents[:max_results + 2],
            method_names=tuple(active_methods),
        )
        reranked = reranked[:max_results]
    else:
        reranked = [(doc, 1.0) for doc in candidate_documents[:max_results]]

    final_documents = [doc for doc, _score in reranked]
    final_scores = [score for _doc, score in reranked]

    context_parts = [
        f"Source {source_rank} ({doc.metadata['scientist_name']}): {doc.page_content}"
        for source_rank, doc in enumerate(final_documents, start=1)
    ]
    formatted_context = "\n\n".join(context_parts)

    llm_response = llm.invoke(
        RERANK_RAG_PROMPT.format(question=question, context=formatted_context),
    )

    return llm_response.content, final_documents, final_scores


def demonstrate_reranking_rag(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
        registry: CrossEncoderRegistry,
) -> None:
    """
    Tests the full re-ranking RAG pipeline with multiple methods (baseline, LLM,
    cross-encoder) side by side. Each question is answered using every available
    method so you can compare answer quality and source selection.
    """
    print_section_header("6️⃣  BUILDING RE-RANKING RAG SYSTEM")
    print(textwrap.dedent(demonstrate_reranking_rag.__doc__))

    print_section_header("7️⃣  TESTING RE-RANKING RAG SYSTEM")

    test_questions = [
        "What are Einstein's most important contributions?",
        "How did Newton change physics?",
        "What did Marie Curie discover?",
    ]

    methods_to_test = ["baseline", "llm"]
    if registry.has_models():
        methods_to_test.append("cross_encoder")

    for question_number, question in enumerate(test_questions, start=1):
        print(f"\n   Q{question_number}: {question}\n"
              f"   {'-' * 50}")

        try:
            for method_name in methods_to_test:
                answer, source_documents, scores = run_reranking_rag_chain(
                    llm, vector_store, registry, question, rerank_method=method_name,
                )

                source_names = [doc.metadata["scientist_name"] for doc in source_documents]
                score_line = f"\n      Scores: {[f'{score:.2f}' for score in scores]}" if method_name != "baseline" else ""
                print(textwrap.dedent(f"""
                      {method_name.capitalize()} re-ranking:
                      Answer: {answer}
                      Sources: {source_names}{score_line}"""))

        except Exception as rag_error:
            print(f"      Error: {rag_error}")


def demonstrate_effectiveness_analysis(
        vector_store: InMemoryVectorStore,
        registry: CrossEncoderRegistry,
        llm: AzureChatOpenAI,
) -> None:
    """
    Measures how each re-ranking method affects result diversity (unique scientists
    in top-5 results). Higher diversity often means the re-ranker is surfacing
    relevant content from across the knowledge base instead of clustering.
    """
    print_section_header("8️⃣  RE-RANKING EFFECTIVENESS ANALYSIS")
    print(textwrap.dedent(demonstrate_effectiveness_analysis.__doc__))

    analysis_queries = ["Einstein relativity", "Newton gravity", "Curie radioactivity"]
    analysis_methods = ["baseline", "llm"]
    if registry.has_models():
        analysis_methods.append("cross_encoder")

    effectiveness_table: dict[str, dict[str, EffectivenessStats]] = {}

    for analysis_query in analysis_queries:
        baseline_documents = vector_store.similarity_search(analysis_query, k=5)
        method_stats: dict[str, EffectivenessStats] = {}

        for method_name in analysis_methods:
            if method_name == "baseline":
                final_documents = baseline_documents
            elif method_name == "llm":
                reranked = rerank_by_llm_relevance(llm, analysis_query, baseline_documents)
                final_documents = [doc for doc, _score in reranked]
            elif method_name == "cross_encoder" and registry.has_models():
                encoder_name = registry.first_model_name()
                reranked = rerank_by_cross_encoder(
                    registry, analysis_query, baseline_documents, model_name=encoder_name,
                )
                final_documents = [doc for doc, _score in reranked]
            else:
                continue

            scientist_names = [doc.metadata["scientist_name"] for doc in final_documents]
            method_stats[method_name] = EffectivenessStats(
                unique_scientist_count=len(set(scientist_names)),
                total_document_count=len(final_documents),
            )

        effectiveness_table[analysis_query] = method_stats

    print(f"\n   📊 Re-ranking effectiveness analysis:")
    print(f"   {'Query':<20} {'Method':<15} {'Unique Sci.':<12} {'Diversity':<12}")
    print("   " + "-" * 65)

    for analysis_query, methods_data in effectiveness_table.items():
        for method_name, stats in methods_data.items():
            print(
                f"   {analysis_query:<20} {method_name:<15} "
                f"{stats.unique_scientist_count:<12} {stats.diversity_ratio:<12.2f}"
            )

    available_methods = ["baseline", "llm"]
    if registry.has_models():
        available_methods.extend(["cross_encoder", "ensemble"])

    print(textwrap.dedent(f"""\

        💡 Re-ranking RAG system ready!
        🎯 Use run_reranking_rag_chain() for optimized retrieval
        🔄 Available methods: {', '.join(available_methods)}"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    print("🎯 RE-RANKING DEMONSTRATION\n"
          "=" * 50)

    _chunks, reranking_vector_store = load_and_index_scientist_chunks()
    cross_encoder_registry = load_cross_encoder_models()

    azure_chat_llm = AzureChatOpenAI(model="gpt-5-nano")

    demonstrate_individual_reranking(reranking_vector_store, cross_encoder_registry, azure_chat_llm)
    demonstrate_ensemble_reranking(reranking_vector_store, cross_encoder_registry, azure_chat_llm)
    demonstrate_performance_benchmark(reranking_vector_store, cross_encoder_registry, azure_chat_llm)
    demonstrate_reranking_rag(azure_chat_llm, reranking_vector_store, cross_encoder_registry)
    demonstrate_effectiveness_analysis(reranking_vector_store, cross_encoder_registry, azure_chat_llm)
