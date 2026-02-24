"""
🎯 Metadata Filtering for Advanced Retrieval
=============================================

Demonstrates how metadata filtering enhances retrieval precision and relevance
in RAG pipelines. Shows contextual retrieval strategies using document properties
and attributes to intelligently narrow down search results.

🎯 What You'll Learn:
- How to extract and attach rich metadata to documents
- Filtering search results by scientific field, time period, and quality
- Building a contextual RAG system that auto-selects filters based on query content
- Comparing unfiltered vs. filtered retrieval to observe precision gains

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, python-dotenv
- Scientist biography .txt files in data/scientists_bios/
"""

import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/04_advanced_retrieval/data/scientists_bios"

FIELD_DETECTION_KEYWORDS: dict[str, list[str]] = {
    "mathematics": ["mathematician", "algorithm", "analytical", "computation"],
    "physics": ["physicist", "relativity", "Nobel Prize", "photoelectric", "radioactivity"],
    "chemistry": ["chemist", "chemical", "elements", "research"],
    "computer_science": ["computer", "programming", "algorithm", "machine"],
}

CONTEXTUAL_RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are an assistant for question-answering tasks about scientists and their contributions.
Use the following pieces of retrieved context to answer the question.
Pay attention to the metadata information about each source, including:
- The scientist's name and primary field
- Time period and historical context
- Document quality and completeness

If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context with metadata:
{context}

Answer:
""")


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


@dataclass(frozen=True)
class ScientistMetadata:
    """Rich metadata extracted from a scientist biography document."""

    scientist_name: str
    primary_field: str
    scientific_fields: list[str]
    birth_year: int | None = None
    death_year: int | None = None
    century: str = "unknown"
    word_count: int = 0
    character_count: int = 0
    completeness: str = "low"
    content_type: str = "biography"
    source_type: str = "text_file"
    language: str = "english"

    def as_flat_dict(self) -> dict[str, Any]:
        return {
            "scientist_name": self.scientist_name,
            "primary_field": self.primary_field,
            "scientific_fields": self.scientific_fields,
            "birth_year": self.birth_year,
            "death_year": self.death_year,
            "century": self.century,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "completeness": self.completeness,
            "content_type": self.content_type,
            "source_type": self.source_type,
            "language": self.language,
        }


def _detect_scientific_fields(content: str) -> list[str]:
    content_lower = content.lower()
    return [
        scientific_field
        for scientific_field, keywords in FIELD_DETECTION_KEYWORDS.items()
        if any(keyword in content_lower for keyword in keywords)
    ]


def _determine_completeness(word_count: int) -> str:
    if word_count > 200:
        return "high"
    if word_count > 100:
        return "medium"
    return "low"


def _extract_birth_death_years(content: str) -> tuple[int | None, int | None]:
    year_pairs = re.findall(r"\((\d{4})-(\d{4})\)", content)
    if year_pairs:
        birth_year_str, death_year_str = year_pairs[0]
        return int(birth_year_str), int(death_year_str)
    return None, None


def _determine_century(birth_year: int | None) -> str:
    if birth_year is None:
        return "unknown"
    century_number = (birth_year // 100) + 1
    return f"{century_number}th century"


def extract_scientist_metadata(document: Document) -> ScientistMetadata:
    """Parses a biography document to produce structured ScientistMetadata."""

    source_path: str = document.metadata.get("source", "")
    scientist_name = str(os.path.basename(source_path)).replace(".txt", "")
    content = document.page_content

    birth_year, death_year = _extract_birth_death_years(content)
    detected_fields = _detect_scientific_fields(content)
    word_count = len(content.split())

    return ScientistMetadata(
        scientist_name=scientist_name,
        primary_field=detected_fields[0] if detected_fields else "unknown",
        scientific_fields=detected_fields,
        birth_year=birth_year,
        death_year=death_year,
        century=_determine_century(birth_year),
        word_count=word_count,
        character_count=len(content),
        completeness=_determine_completeness(word_count),
    )


def enrich_document_with_metadata(document: Document) -> Document:
    """Attaches extracted ScientistMetadata to the document's metadata dict."""
    scientist_metadata = extract_scientist_metadata(document)
    document.metadata.update(scientist_metadata.as_flat_dict())
    return document


def load_and_enrich_scientist_documents() -> list[Document]:
    """
    Loads scientist biographies from text files and enriches each document
    with structured metadata (name, field, birth year, quality metrics).
    """
    print_section_header("1️⃣  LOADING DOCUMENTS WITH RICH METADATA")

    loader = DirectoryLoader(SCIENTISTS_BIOS_DIR, glob="*.txt")
    raw_documents = loader.load()
    print(f"   Loaded {len(raw_documents)} raw documents")

    documents_with_metadata = [enrich_document_with_metadata(doc) for doc in raw_documents]

    print("\n   Enhanced metadata for each document:")
    for doc in documents_with_metadata:
        metadata = doc.metadata
        print(textwrap.dedent(f"""\
               📄 {metadata['scientist_name']}:
                  • Fields: {', '.join(metadata['scientific_fields'])}
                  • Primary: {metadata['primary_field']}
                  • Birth year: {metadata.get('birth_year', 'unknown')}
                  • Word count: {metadata['word_count']}"""))

    return documents_with_metadata


def split_documents_into_chunks(source_documents: list[Document]) -> list[Document]:
    """
    Splits enriched documents into smaller chunks while preserving all metadata.
    Each chunk receives additional chunk-specific metadata (id, size, position).
    """
    print_section_header("2️⃣  CHUNKING WITH METADATA PRESERVATION")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(source_documents)
    total_chunks = len(chunks)

    for chunk_index, chunk in enumerate(chunks):
        if chunk_index < total_chunks // 3:
            position_label = "start"
        elif chunk_index < 2 * total_chunks // 3:
            position_label = "middle"
        else:
            position_label = "end"

        chunk.metadata.update({
            "chunk_id": f"chunk_{chunk_index + 1}",
            "chunk_size": len(chunk.page_content),
            "chunk_position": position_label,
        })

    print(f"   Created {total_chunks} chunks with enhanced metadata")
    print("   Sample chunk metadata:")
    for metadata_key, metadata_value in chunks[0].metadata.items():
        print(f"      {metadata_key}: {metadata_value}")

    return chunks


def build_vector_store(chunks: list[Document]) -> InMemoryVectorStore:
    """
    Creates an in-memory vector store from enriched chunks,
    indexing both content embeddings and associated metadata.
    """
    print_section_header("3️⃣  BUILDING VECTOR STORE WITH METADATA INDEXING")

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=chunks)

    print(f"   ✅ Indexed {len(chunks)} chunks with full metadata")
    return vector_store


def search_filtered_by_field(
        vector_store: InMemoryVectorStore,
        query: str,
        *,
        target_field: str,
        max_results: int = 3,
) -> list[Document]:
    """Performs similarity search and keeps only results matching the target scientific field."""
    broad_results = vector_store.similarity_search(query, k=max_results * 3)
    return [
        result for result in broad_results
        if result.metadata.get("primary_field") == target_field
    ][:max_results]


def search_filtered_by_century(
        vector_store: InMemoryVectorStore,
        query: str,
        *,
        target_century_start: int = 1800,
        target_century_end: int = 1900,
        max_results: int = 3,
) -> list[Document]:
    """Performs similarity search and keeps only results from scientists born in the target century."""
    broad_results = vector_store.similarity_search(query, k=max_results * 3)
    return [
        result for result in broad_results
        if (birth_year := result.metadata.get("birth_year"))
           and target_century_start <= birth_year < target_century_end
    ][:max_results]


def search_filtered_by_completeness(
        vector_store: InMemoryVectorStore,
        query: str,
        *,
        required_completeness: str = "high",
        max_results: int = 3,
) -> list[Document]:
    """Performs similarity search and keeps only results with the required completeness level."""
    broad_results = vector_store.similarity_search(query, k=max_results * 2)
    return [
        result for result in broad_results
        if result.metadata.get("completeness") == required_completeness
    ][:max_results]


def demonstrate_metadata_filtering(vector_store: InMemoryVectorStore) -> None:
    """
    Metadata filtering narrows retrieval results using document properties
    rather than relying solely on semantic similarity. This section demonstrates
    three filtering strategies: by scientific field, by birth century, and by
    document completeness — each improving precision for targeted queries.
    """
    print_section_header("4️⃣  METADATA FILTERING DEMONSTRATIONS")
    print(textwrap.dedent(demonstrate_metadata_filtering.__doc__))

    print("   🔬 Field-specific retrieval (Physics only):")
    physics_query = "What are the major scientific contributions?"
    physics_results = search_filtered_by_field(
        vector_store, physics_query, target_field="physics",
    )
    print(f"   Query: {physics_query}")
    print(f"   Results: {len(physics_results)} physics-related chunks")
    for rank, result in enumerate(physics_results, start=1):
        print(
            f"   {rank}. {result.metadata['scientist_name']} "
            f"({result.metadata['primary_field']}): "
            f"{result.page_content[:100]}..."
        )

    print("\n   📅 Historical period filtering (19th century):")
    nineteenth_century_query = "Who made important discoveries?"
    nineteenth_century_results = search_filtered_by_century(
        vector_store,
        nineteenth_century_query,
        target_century_start=1800,
        target_century_end=1900,
        max_results=3,
    )
    print(f"   Query: {nineteenth_century_query}")
    print(f"   Results: {len(nineteenth_century_results)} 19th century scientists")
    for rank, result in enumerate(nineteenth_century_results, start=1):
        birth_year_display = result.metadata.get("birth_year", "unknown")
        print(
            f"   {rank}. {result.metadata['scientist_name']} "
            f"(born {birth_year_display}): "
            f"{result.page_content[:100]}..."
        )

    print("\n   ⭐ Quality-based filtering (High completeness only):")
    quality_query = "Tell me about scientific achievements"
    high_quality_results = search_filtered_by_completeness(
        vector_store, quality_query, required_completeness="high", max_results=2,
    )
    print(f"   Query: {quality_query}")
    print(f"   Results: {len(high_quality_results)} high-quality documents")
    for rank, result in enumerate(high_quality_results, start=1):
        print(
            f"   {rank}. {result.metadata['scientist_name']} "
            f"({result.metadata['completeness']}, {result.metadata['word_count']} words): "
            f"{result.page_content[:100]}..."
        )


def create_contextual_retriever(
        vector_store: InMemoryVectorStore,
        query: str,
        *,
        max_results: int = 4,
) -> list[Document]:
    """
    A smart retriever that analyzes the query text to automatically determine
    appropriate metadata filters. It detects field-specific, time-based, and
    quality-based keywords and routes to the matching filtered search.
    Falls back to unfiltered similarity search when no filter matches.
    """
    query_lower = query.lower()

    field_keyword_map: dict[str, list[str]] = {
        "physics": ["physics", "relativity", "einstein"],
        "mathematics": ["mathematics", "algorithm", "computation"],
        "computer_science": ["programming", "computer", "lovelace"],
    }
    for detected_field, trigger_keywords in field_keyword_map.items():
        if any(keyword in query_lower for keyword in trigger_keywords):
            return search_filtered_by_field(
                vector_store, query, target_field=detected_field, max_results=max_results,
            )

    historical_trigger_terms = ["historical", "past", "old", "19th century", "early"]
    if any(term in query_lower for term in historical_trigger_terms):
        return search_filtered_by_century(vector_store, query, max_results=max_results)

    quality_trigger_terms = ["detailed", "comprehensive", "complete"]
    if any(term in query_lower for term in quality_trigger_terms):
        return search_filtered_by_completeness(vector_store, query, max_results=max_results)

    return vector_store.similarity_search(query, k=max_results)


def format_retrieved_docs_with_metadata(retrieved_documents: list[Document]) -> str:
    """Formats retrieved documents with their metadata into a prompt-ready string."""
    formatted_entries: list[str] = []
    for source_index, doc in enumerate(retrieved_documents, start=1):
        scientist_name = doc.metadata.get("scientist_name", "Unknown")
        primary_field = doc.metadata.get("primary_field", "Unknown")
        birth_year_display = doc.metadata.get("birth_year", "Unknown")

        formatted_entries.append(
            f"\nSource {source_index}: {scientist_name} "
            f"({primary_field}, born {birth_year_display})\n"
            f"Content: {doc.page_content}\n"
        )
    return "\n".join(formatted_entries)


def run_contextual_rag_chain(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
        question: str,
) -> tuple[str, list[Document]]:
    """
    Executes a full RAG cycle: contextual retrieval → metadata-aware prompt
    formatting → LLM generation. Returns the answer and source documents.
    """
    retrieved_documents = create_contextual_retriever(vector_store, question)
    formatted_context = format_retrieved_docs_with_metadata(retrieved_documents)

    llm_response = llm.invoke(
        CONTEXTUAL_RAG_PROMPT.format(
            question=question,
            context=formatted_context,
        )
    )

    return llm_response.content, retrieved_documents


def demonstrate_contextual_rag(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
) -> None:
    """
    The contextual RAG system combines metadata filtering with generation.
    It analyzes each query to auto-detect the best filter strategy, retrieves
    the most relevant documents, and feeds them (with metadata) to the LLM.
    """
    print_section_header("5️⃣  BUILDING CONTEXTUAL RAG SYSTEM")
    print(textwrap.dedent(demonstrate_contextual_rag.__doc__))

    print_section_header("6️⃣  TESTING CONTEXTUAL RAG SYSTEM")

    test_questions = [
        "What physics discoveries were made by Einstein?",
        "Tell me about historical mathematicians and their work",
        "Who worked on early computer programming?",
        "What are the most comprehensive achievements in science?",
    ]

    for question_number, question in enumerate(test_questions, start=1):
        print(f"\n   Q{question_number}: {question}")
        print("   " + "-" * 50)

        try:
            answer, source_documents = run_contextual_rag_chain(llm, vector_store, question)
            print(f"   A{question_number}: {answer}")

            print(f"\n   📚 Sources used ({len(source_documents)} documents):")
            for source_rank, source_doc in enumerate(source_documents, start=1):
                print(
                    f"      {source_rank}. {source_doc.metadata['scientist_name']} "
                    f"({source_doc.metadata['primary_field']})"
                )
        except Exception as rag_error:
            print(f"   A{question_number}: Error - {rag_error}")


def demonstrate_filtered_vs_unfiltered_comparison(
        vector_store: InMemoryVectorStore,
) -> None:
    """
    Side-by-side comparison of unfiltered similarity search vs. contextually
    filtered retrieval. Demonstrates how metadata filters improve precision
    for targeted queries while generic queries return identical results.
    """
    print_section_header("7️⃣  PERFORMANCE COMPARISON: FILTERED vs. UNFILTERED")
    print(textwrap.dedent(demonstrate_filtered_vs_unfiltered_comparison.__doc__))

    comparison_cases: list[tuple[str, str]] = [
        ("What scientific contributions were made?", "Generic query"),
        ("What physics discoveries changed our understanding?", "Physics-specific query"),
        ("Tell me about mathematical breakthroughs", "Mathematics-specific query"),
        ("What are the most comprehensive scientific achievements?", "Quality-focused query"),
    ]

    for query_text, query_description in comparison_cases:
        print(f"\n   🔍 {query_description}: {query_text}")
        print("   " + "=" * 60)

        unfiltered_results = vector_store.similarity_search(query_text, k=3)
        print(f"\n   📊 Unfiltered results ({len(unfiltered_results)} documents):")
        for rank, result in enumerate(unfiltered_results, start=1):
            print(
                f"      {rank}. {result.metadata['scientist_name']} "
                f"({result.metadata['primary_field']}): "
                f"{result.page_content[:60]}..."
            )

        contextually_filtered_results = create_contextual_retriever(
            vector_store, query_text, max_results=3,
        )
        print(f"\n   🎯 Contextually filtered results ({len(contextually_filtered_results)} documents):")
        for rank, result in enumerate(contextually_filtered_results, start=1):
            print(
                f"      {rank}. {result.metadata['scientist_name']} "
                f"({result.metadata['primary_field']}): "
                f"{result.page_content[:60]}..."
            )

        unfiltered_scientist_names = {r.metadata["scientist_name"] for r in unfiltered_results}
        filtered_scientist_names = {r.metadata["scientist_name"] for r in contextually_filtered_results}

        if unfiltered_scientist_names != filtered_scientist_names:
            print("   ✅ Filtering effect: Results differ between approaches")
        else:
            print("   ➡️ No filtering applied: Generic query returned same results")

        print()


def print_metadata_analysis_summary(chunks: list[Document]) -> None:
    """
    Aggregates and prints a summary of metadata distribution across all chunks:
    scientific field counts, birth-year time span, and content quality breakdown.
    """
    print_section_header("8️⃣  METADATA ANALYSIS SUMMARY")

    field_distribution: dict[str, int] = {}
    all_birth_years: list[int] = []
    quality_distribution: dict[str, int] = {}

    for chunk in chunks:
        primary_field = chunk.metadata.get("primary_field", "unknown")
        field_distribution[primary_field] = field_distribution.get(primary_field, 0) + 1

        if (birth_year := chunk.metadata.get("birth_year")) is not None:
            all_birth_years.append(birth_year)

        completeness_level = chunk.metadata.get("completeness", "unknown")
        quality_distribution[completeness_level] = quality_distribution.get(completeness_level, 0) + 1

    print("\n   📈 Metadata Distribution Analysis:")
    print("   🔬 Scientific Fields:")
    for scientific_field, chunk_count in sorted(field_distribution.items()):
        print(f"      • {scientific_field}: {chunk_count} chunks")

    if all_birth_years:
        earliest_birth = min(all_birth_years)
        latest_birth = max(all_birth_years)
        print(textwrap.dedent(f"""\
               📅 Time Period Coverage:
                  • Earliest: {earliest_birth}
                  • Latest: {latest_birth}
                  • Span: {latest_birth - earliest_birth} years"""))

    print("   ⭐ Content Quality:")
    for completeness_level, chunk_count in sorted(quality_distribution.items()):
        print(f"      • {completeness_level}: {chunk_count} chunks")

    print(textwrap.dedent("""\

        💡 Contextual RAG system ready with enhanced metadata filtering!
        🎯 Use run_contextual_rag_chain() for intelligent retrieval"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    print("🎯 METADATA FILTERING DEMONSTRATION\n"
          "=" * 50)

    enriched_documents = load_and_enrich_scientist_documents()
    chunks_with_metadata = split_documents_into_chunks(enriched_documents)
    metadata_vector_store = build_vector_store(chunks_with_metadata)

    demonstrate_metadata_filtering(metadata_vector_store)

    azure_chat_llm = AzureChatOpenAI(model="gpt-5-nano")

    demonstrate_contextual_rag(azure_chat_llm, metadata_vector_store)
    demonstrate_filtered_vs_unfiltered_comparison(metadata_vector_store)
    print_metadata_analysis_summary(chunks_with_metadata)
