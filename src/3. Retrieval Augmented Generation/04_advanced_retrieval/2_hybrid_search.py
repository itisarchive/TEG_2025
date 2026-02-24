"""
🔀 Hybrid Search — Semantic + Keyword Retrieval Fusion
======================================================

Demonstrates combining semantic similarity (vector embeddings) with keyword-based
search (BM25, TF-IDF) for improved retrieval precision in RAG pipelines.

🎯 What You'll Learn:
- How vector search, BM25 and TF-IDF each retrieve differently
- Query analysis and automatic routing to the best strategy
- Score fusion techniques: Reciprocal Rank Fusion and Weighted Score Fusion
- Building an adaptive hybrid RAG system that picks strategy per query

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, rank-bm25, scikit-learn
- Scientist biography .txt files in data/scientists_bios/
"""

import os
import textwrap
from dataclasses import dataclass
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/04_advanced_retrieval/data/scientists_bios"

HYBRID_RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are an assistant for question-answering tasks about scientists and their contributions.
Use the following pieces of retrieved context to answer the question.
The context was retrieved using hybrid search combining semantic similarity and keyword matching.

If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")

ScoredDocument = tuple[Document, float]


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


@dataclass(frozen=True)
class QueryAnalysis:
    """Result of analyzing a user query to determine optimal search strategy."""

    has_specific_names: bool = False
    has_technical_terms: bool = False
    is_conceptual: bool = False
    is_factual: bool = False
    recommended_strategy: str = "hybrid"


@dataclass
class HybridSearchIndexes:
    """Holds all three search indexes: vector store, BM25 and TF-IDF."""

    vector_store: InMemoryVectorStore
    bm25_index: BM25Okapi
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: Any
    indexed_chunks: list[Document]


def load_and_prepare_chunks() -> list[Document]:
    """
    Loads scientist biography files, attaches basic metadata
    (scientist name, word count), and splits them into chunks.
    """
    print_section_header("1️⃣  LOADING DOCUMENTS FOR HYBRID SEARCH")

    loader = DirectoryLoader(SCIENTISTS_BIOS_DIR, glob="*.txt")
    raw_documents = loader.load()
    print(f"   Loaded {len(raw_documents)} documents")

    for doc in raw_documents:
        filename = os.path.basename(doc.metadata["source"]).replace(".txt", "")
        doc.metadata.update({
            "scientist_name": filename,
            "word_count": len(doc.page_content.split()),
        })

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   Created {len(chunks)} chunks for hybrid search")
    return chunks


def build_hybrid_indexes(chunks: list[Document]) -> HybridSearchIndexes:
    """
    Constructs three complementary search indexes from the same chunks:
    a vector store for semantic similarity, BM25 for keyword relevance,
    and TF-IDF for term-frequency–based matching.
    """
    print_section_header("2️⃣  BUILDING MULTIPLE SEARCH INDEXES")

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=chunks)
    print(f"   ✅ Vector store: {len(chunks)} chunks embedded")

    chunk_texts = [chunk.page_content for chunk in chunks]
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25_index = BM25Okapi(tokenized_chunks)
    print(f"   ✅ BM25 index: {len(chunk_texts)} documents indexed")

    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunk_texts)
    print(f"   ✅ TF-IDF index: {tfidf_matrix.shape[1]} features extracted")

    return HybridSearchIndexes(
        vector_store=vector_store,
        bm25_index=bm25_index,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        indexed_chunks=chunks,
    )


def search_by_vector(
        indexes: HybridSearchIndexes,
        query: str,
        *,
        max_results: int = 5,
) -> list[ScoredDocument]:
    """Semantic similarity search using vector embeddings."""
    raw_results = indexes.vector_store.similarity_search_with_score(query, k=max_results)
    return [(doc, score) for doc, score in raw_results]


def search_by_bm25(
        indexes: HybridSearchIndexes,
        query: str,
        *,
        max_results: int = 5,
) -> list[ScoredDocument]:
    """Keyword search using BM25 term-frequency scoring."""
    query_tokens = query.lower().split()
    bm25_scores = indexes.bm25_index.get_scores(query_tokens)

    top_indices = np.argsort(bm25_scores)[::-1][:max_results]
    return [
        (indexes.indexed_chunks[chunk_idx], bm25_scores[chunk_idx])
        for chunk_idx in top_indices
        if bm25_scores[chunk_idx] > 0
    ]


def search_by_tfidf(
        indexes: HybridSearchIndexes,
        query: str,
        *,
        max_results: int = 5,
) -> list[ScoredDocument]:
    """TF-IDF based keyword search using cosine similarity."""
    query_vector = indexes.tfidf_vectorizer.transform([query])
    tfidf_similarities = cosine_similarity(query_vector, indexes.tfidf_matrix).flatten()

    top_indices = np.argsort(tfidf_similarities)[::-1][:max_results]
    return [
        (indexes.indexed_chunks[chunk_idx], tfidf_similarities[chunk_idx])
        for chunk_idx in top_indices
        if tfidf_similarities[chunk_idx] > 0
    ]


def print_scored_results(results: list[ScoredDocument], *, label: str) -> None:
    print(f"\n   {label}:")
    for rank, (doc, score) in enumerate(results, start=1):
        scientist_name = doc.metadata["scientist_name"]
        content_preview = doc.page_content[:80] + "..."
        print(f"      {rank}. {scientist_name} (score: {score:.3f}): {content_preview}")


def demonstrate_individual_search_methods(indexes: HybridSearchIndexes) -> None:
    """
    Each search method captures a different aspect of relevance:
    • Vector search finds semantically similar content (meaning-based)
    • BM25 matches exact keywords with term-frequency weighting
    • TF-IDF balances term frequency with inverse document frequency

    Comparing them side-by-side reveals their complementary strengths.
    """
    print_section_header("3️⃣  TESTING INDIVIDUAL SEARCH METHODS")
    print(textwrap.dedent(demonstrate_individual_search_methods.__doc__))

    test_query = "What did Einstein discover about light and energy?"
    print(f"   🔍 Test query: {test_query}")

    vector_results = search_by_vector(indexes, test_query, max_results=3)
    print_scored_results(vector_results, label="🧠 Vector search results")

    bm25_results = search_by_bm25(indexes, test_query, max_results=3)
    print_scored_results(bm25_results, label="🔤 BM25 search results")

    tfidf_results = search_by_tfidf(indexes, test_query, max_results=3)
    print_scored_results(tfidf_results, label="📊 TF-IDF search results")


def analyze_query(query: str) -> QueryAnalysis:
    """
    Inspects the query text to classify it along four dimensions
    (specific names, technical terms, conceptual, factual) and
    recommends one of three strategies: keyword_heavy, semantic_heavy, or hybrid.
    """
    query_lower = query.lower()

    scientist_names = ["einstein", "newton", "curie", "lovelace", "darwin"]
    has_specific_names = any(name in query_lower for name in scientist_names)

    technical_terms = ["theory", "law", "equation", "formula", "discovery", "invention"]
    has_technical_terms = any(term in query_lower for term in technical_terms)

    conceptual_words = ["understand", "explain", "concept", "idea", "principle"]
    is_conceptual = any(word in query_lower for word in conceptual_words)

    factual_words = ["when", "where", "what", "who", "date", "year"]
    is_factual = any(word in query_lower for word in factual_words)

    if has_specific_names and is_factual:
        recommended_strategy = "keyword_heavy"
    elif is_conceptual:
        recommended_strategy = "semantic_heavy"
    else:
        recommended_strategy = "hybrid"

    return QueryAnalysis(
        has_specific_names=has_specific_names,
        has_technical_terms=has_technical_terms,
        is_conceptual=is_conceptual,
        is_factual=is_factual,
        recommended_strategy=recommended_strategy,
    )


def demonstrate_query_routing() -> None:
    """
    Query routing analyzes each question to pick the best retrieval strategy.
    Factual queries with specific names benefit from keyword-heavy search,
    while conceptual questions are better served by semantic similarity.
    """
    print_section_header("4️⃣  QUERY ANALYSIS AND ROUTING")
    print(textwrap.dedent(demonstrate_query_routing.__doc__))

    routing_test_queries = [
        "What did Einstein discover?",
        "Explain the concept of gravity",
        "When was Newton born?",
        "How do scientific theories develop?",
    ]

    for test_query in routing_test_queries:
        analysis = analyze_query(test_query)
        print(textwrap.dedent(f"""
               📝 Query: {test_query}
                  Strategy: {analysis.recommended_strategy}
                  Has names: {analysis.has_specific_names}, Technical: {analysis.has_technical_terms}
                  Conceptual: {analysis.is_conceptual}, Factual: {analysis.is_factual}"""))


def normalize_scores(raw_scores: np.ndarray, *, method: str = "min_max") -> np.ndarray:
    """Normalizes an array of scores to a 0–1 range using min-max or z-score."""
    scores = np.array(raw_scores, dtype=float)
    if method == "min_max":
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
    elif method == "z_score":
        mean_score, std_score = scores.mean(), scores.std()
        if std_score > 0:
            return (scores - mean_score) / std_score
    return scores


def fuse_by_reciprocal_rank(
        all_result_lists: list[list[ScoredDocument]],
        *,
        rrf_constant: int = 60,
) -> list[ScoredDocument]:
    """
    Reciprocal Rank Fusion (RRF) combines rankings from multiple search methods.
    Each document receives a score of 1/(k + rank) from each list, then scores are summed.
    The rrf_constant (k) controls how much lower ranks are penalized.
    """
    accumulated_scores: dict[int, dict[str, Any]] = {}

    for single_result_list in all_result_lists:
        for rank, (doc, _original_score) in enumerate(single_result_list):
            doc_identity = id(doc)
            if doc_identity not in accumulated_scores:
                accumulated_scores[doc_identity] = {"doc": doc, "score": 0.0}
            accumulated_scores[doc_identity]["score"] += 1 / (rrf_constant + rank + 1)

    sorted_entries = sorted(
        accumulated_scores.values(),
        key=lambda entry: entry["score"],
        reverse=True,
    )
    return [(entry["doc"], entry["score"]) for entry in sorted_entries]


def fuse_by_weighted_scores(
        vector_results: list[ScoredDocument],
        keyword_results: list[ScoredDocument],
        *,
        vector_weight: float = 0.6,
) -> list[ScoredDocument]:
    """
    Weighted Score Fusion normalizes scores from vector and keyword search,
    then combines them using configurable weights (default: 60% vector, 40% keyword).
    Documents appearing in only one list receive zero for the missing score.
    """
    normalized_vector_scores = normalize_scores(
        np.array([score for _, score in vector_results]),
    )
    normalized_keyword_scores = normalize_scores(
        np.array([score for _, score in keyword_results]),
    )

    keyword_weight = 1 - vector_weight
    combined_docs: dict[int, dict[str, Any]] = {}

    for rank, (doc, _original_score) in enumerate(vector_results):
        combined_docs[id(doc)] = {
            "doc": doc,
            "vector_score": normalized_vector_scores[rank],
            "keyword_score": 0.0,
        }

    for rank, (doc, _original_score) in enumerate(keyword_results):
        doc_identity = id(doc)
        if doc_identity in combined_docs:
            combined_docs[doc_identity]["keyword_score"] = normalized_keyword_scores[rank]
        else:
            combined_docs[doc_identity] = {
                "doc": doc,
                "vector_score": 0.0,
                "keyword_score": normalized_keyword_scores[rank],
            }

    for score_entry in combined_docs.values():
        score_entry["combined_score"] = (
                vector_weight * score_entry["vector_score"]
                + keyword_weight * score_entry["keyword_score"]
        )

    sorted_entries = sorted(
        combined_docs.values(),
        key=lambda item: item["combined_score"],
        reverse=True,
    )
    return [(item["doc"], item["combined_score"]) for item in sorted_entries]


def demonstrate_score_fusion(indexes: HybridSearchIndexes) -> None:
    """
    Score fusion merges results from different search methods into a single ranking.
    • Reciprocal Rank Fusion (RRF): rank-based, ignores raw scores
    • Weighted Score Fusion (WSF): score-based with tunable vector/keyword balance
    """
    print_section_header("5️⃣  SCORE FUSION STRATEGIES")
    print(textwrap.dedent(demonstrate_score_fusion.__doc__))

    fusion_query = "Einstein's theory of relativity and light"
    print(f"   🔍 Fusion test query: {fusion_query}")

    vector_results = search_by_vector(indexes, fusion_query, max_results=5)
    bm25_results = search_by_bm25(indexes, fusion_query, max_results=5)

    rrf_results = fuse_by_reciprocal_rank([vector_results, bm25_results])
    print_scored_results(rrf_results[:3], label="🔗 Reciprocal Rank Fusion")

    wsf_results = fuse_by_weighted_scores(vector_results, bm25_results, vector_weight=0.6)
    print_scored_results(wsf_results[:3], label="⚖️ Weighted Score Fusion (60% vector, 40% keyword)")


def adaptive_hybrid_search(
        indexes: HybridSearchIndexes,
        query: str,
        *,
        max_results: int = 5,
) -> tuple[list[ScoredDocument], QueryAnalysis]:
    """
    Adaptive search that analyzes the query, selects the best vector/keyword
    weight balance, and fuses results using weighted score fusion.
    """
    analysis = analyze_query(query)

    vector_results = search_by_vector(indexes, query, max_results=max_results * 2)
    bm25_results = search_by_bm25(indexes, query, max_results=max_results * 2)

    strategy_to_vector_weight: dict[str, float] = {
        "semantic_heavy": 0.8,
        "keyword_heavy": 0.3,
        "hybrid": 0.6,
    }
    vector_weight = strategy_to_vector_weight[analysis.recommended_strategy]

    fused_results = fuse_by_weighted_scores(
        vector_results, bm25_results, vector_weight=vector_weight,
    )
    return fused_results[:max_results], analysis


def demonstrate_adaptive_hybrid_search(indexes: HybridSearchIndexes) -> None:
    """
    The adaptive hybrid search automatically adjusts the vector/keyword weight
    based on query characteristics. Factual + named queries lean keyword-heavy,
    conceptual queries lean semantic-heavy, and mixed queries use balanced fusion.
    """
    print_section_header("6️⃣  ADAPTIVE HYBRID SEARCH")
    print(textwrap.dedent(demonstrate_adaptive_hybrid_search.__doc__))

    adaptive_test_queries = [
        "What is Einstein famous for?",
        "Explain scientific methodology",
        "How did Newton contribute to physics?",
    ]

    for test_query in adaptive_test_queries:
        print(f"\n   🔍 Query: {test_query}")
        top_results, analysis = adaptive_hybrid_search(indexes, test_query, max_results=3)
        print(f"   📊 Strategy: {analysis.recommended_strategy}")

        for rank, (doc, combined_score) in enumerate(top_results, start=1):
            scientist_name = doc.metadata["scientist_name"]
            content_preview = doc.page_content[:70] + "..."
            print(f"      {rank}. {scientist_name} (score: {combined_score:.3f}): {content_preview}")


def run_hybrid_rag_chain(
        llm: AzureChatOpenAI,
        indexes: HybridSearchIndexes,
        question: str,
) -> tuple[str, list[ScoredDocument], QueryAnalysis]:
    """
    Full RAG cycle using adaptive hybrid retrieval: analyzes the question,
    retrieves with fused scoring, formats context, and generates an answer.
    """
    top_results, analysis = adaptive_hybrid_search(indexes, question, max_results=4)

    context_parts = [
        f"Source {source_rank} ({doc.metadata['scientist_name']}): {doc.page_content}"
        for source_rank, (doc, _score) in enumerate(top_results, start=1)
    ]
    formatted_context = "\n\n".join(context_parts)

    llm_response = llm.invoke(
        HYBRID_RAG_PROMPT.format(question=question, context=formatted_context),
    )

    return llm_response.content, top_results, analysis


def demonstrate_hybrid_rag(llm: AzureChatOpenAI, indexes: HybridSearchIndexes) -> None:
    """
    Combines the adaptive hybrid retrieval with LLM generation into a complete
    RAG pipeline. Each question is routed to the best search strategy, and the
    LLM receives context from fused results with full source attribution.
    """
    print_section_header("7️⃣  BUILDING HYBRID RAG SYSTEM")
    print(textwrap.dedent(demonstrate_hybrid_rag.__doc__))

    print_section_header("8️⃣  TESTING HYBRID RAG SYSTEM")

    test_questions = [
        "What specific discoveries did Einstein make?",
        "How do scientific theories get developed?",
        "When did Newton live and what did he study?",
    ]

    for question_number, question in enumerate(test_questions, start=1):
        print(f"\n   Q{question_number}: {question}\n"
              f"   {'-' * 50}")

        try:
            answer, source_results, analysis = run_hybrid_rag_chain(llm, indexes, question)
            print(textwrap.dedent(f"""\
                   A{question_number}: {answer}

                   📊 Search strategy: {analysis.recommended_strategy}
                   📚 Sources ({len(source_results)} documents):"""))
            for source_rank, (source_doc, source_score) in enumerate(source_results, start=1):
                scientist_name = source_doc.metadata["scientist_name"]
                print(f"      {source_rank}. {scientist_name} (score: {source_score:.3f})")

        except Exception as rag_error:
            print(f"   A{question_number}: Error - {rag_error}")


def demonstrate_performance_comparison(indexes: HybridSearchIndexes) -> None:
    """
    Compares retrieval diversity across vector-only, BM25-only, and adaptive
    hybrid search. For each query type, counts how many distinct scientists
    appear in the top 3 results — higher diversity often signals better coverage.
    """
    print_section_header("9️⃣  HYBRID SEARCH PERFORMANCE ANALYSIS")
    print(textwrap.dedent(demonstrate_performance_comparison.__doc__))

    comparison_cases: list[tuple[str, str]] = [
        ("Einstein relativity", "Factual with name"),
        ("scientific discovery process", "Conceptual general"),
        ("Newton apple gravity story", "Specific narrative"),
        ("physics principles", "General conceptual"),
    ]

    print(f"\n   📊 Method comparison across query types:")
    print(f"   {'Query Type':<20} {'Vector':<8} {'BM25':<8} {'Hybrid':<8}")
    print("   " + "-" * 50)

    for comparison_query, query_type_label in comparison_cases:
        vector_results = search_by_vector(indexes, comparison_query, max_results=3)
        bm25_results = search_by_bm25(indexes, comparison_query, max_results=3)
        hybrid_results, _analysis = adaptive_hybrid_search(indexes, comparison_query, max_results=3)

        vector_unique_scientists = len({doc.metadata["scientist_name"] for doc, _ in vector_results})
        bm25_unique_scientists = len({doc.metadata["scientist_name"] for doc, _ in bm25_results})
        hybrid_unique_scientists = len({doc.metadata["scientist_name"] for doc, _ in hybrid_results})

        print(
            f"   {query_type_label:<20} "
            f"{vector_unique_scientists:<8} "
            f"{bm25_unique_scientists:<8} "
            f"{hybrid_unique_scientists:<8}"
        )

    print(textwrap.dedent("""\

        💡 Hybrid RAG system ready with adaptive search strategies!
        🔀 Use run_hybrid_rag_chain() for intelligent hybrid retrieval
        📊 Combines semantic similarity + keyword matching + adaptive weighting"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    print("🔀 HYBRID SEARCH DEMONSTRATION\n"
          "=" * 50)

    prepared_chunks = load_and_prepare_chunks()
    search_indexes = build_hybrid_indexes(prepared_chunks)

    demonstrate_individual_search_methods(search_indexes)
    demonstrate_query_routing()
    demonstrate_score_fusion(search_indexes)
    demonstrate_adaptive_hybrid_search(search_indexes)

    azure_chat_llm = AzureChatOpenAI(model="gpt-5-nano")

    demonstrate_hybrid_rag(azure_chat_llm, search_indexes)
    demonstrate_performance_comparison(search_indexes)
