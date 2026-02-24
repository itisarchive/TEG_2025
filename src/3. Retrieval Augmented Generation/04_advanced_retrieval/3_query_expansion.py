"""
🔍 Query Expansion for Enhanced Retrieval
==========================================

Demonstrates automatic query expansion using LLMs, synonym generation,
and multi-perspective query generation for improved retrieval quality.

🎯 What You'll Learn:
- Synonym and concept-based query expansion techniques
- LLM-powered query reformulation and expansion
- Multi-perspective query generation (historical, technical, impact)
- Context-aware expansion using domain knowledge
- Building a complete expansion pipeline with effectiveness analysis

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, python-dotenv
- Scientist biography .txt files in data/scientists_bios/
"""

import os
import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/04_advanced_retrieval/data/scientists_bios"

SYNONYM_MAP: dict[str, list[str]] = {
    "discover": ["find", "uncover", "reveal", "identify"],
    "theory": ["principle", "law", "concept", "hypothesis"],
    "scientist": ["researcher", "physicist", "mathematician", "inventor"],
    "work": ["research", "study", "investigation", "contribution"],
    "famous": ["renowned", "notable", "celebrated", "prominent"],
    "important": ["significant", "crucial", "major", "key"],
}

CONCEPT_MAP: dict[str, list[str]] = {
    "einstein": ["relativity", "photon", "spacetime", "mass-energy"],
    "newton": ["gravity", "motion", "calculus", "mechanics"],
    "curie": ["radioactivity", "radiation", "polonium", "radium"],
    "darwin": ["evolution", "selection", "species", "adaptation"],
    "physics": ["mechanics", "thermodynamics", "electromagnetism"],
    "mathematics": ["calculus", "geometry", "algebra", "statistics"],
}

LLM_EXPANSION_PROMPT = ChatPromptTemplate.from_template("""\
You are a search query expert. Given a user's search query, generate an expanded version that includes:
1. Synonyms and related terms
2. More specific scientific terminology
3. Alternative phrasings that might help find relevant information

Original query: {query}

Provide 3 different expanded versions of this query, each focusing on different aspects:
1. Synonym-based expansion
2. Technical term expansion
3. Alternative phrasing expansion

Format your response as:
Synonym: [expanded query]
Technical: [expanded query]
Alternative: [expanded query]
""")

MULTI_PERSPECTIVE_PROMPT = ChatPromptTemplate.from_template("""\
Generate 3 different search queries that approach the following topic from different perspectives:

Original topic: {query}

Create queries for these perspectives:
1. Historical perspective (when, where, context)
2. Technical perspective (how, what methods, mechanisms)
3. Impact perspective (why important, consequences, influence)

Format as:
Historical: [query]
Technical: [query]
Impact: [query]
""")

CONTEXT_AWARE_PROMPT = ChatPromptTemplate.from_template("""\
You are helping expand search queries for a scientific knowledge base about famous scientists.
The database contains information about: Einstein, Newton, Curie, Darwin, and Lovelace.

Original query: {query}
Available scientists: Einstein (physics), Newton (physics/math), Curie (chemistry/physics), \
Darwin (biology), Lovelace (computing/math)

Generate an expanded query that:
1. Includes relevant scientist names if applicable
2. Uses appropriate scientific terminology
3. Considers the specific knowledge domain

Expanded query:
""")

EXPANSION_RAG_PROMPT = ChatPromptTemplate.from_template("""\
You are an assistant for question-answering tasks about scientists and their contributions.
The user's query was expanded to improve retrieval using multiple expansion techniques.
Use the following pieces of retrieved context to answer the question.

Original question: {original_query}
Expanded query used: {expanded_query}

Context: {context}

If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Answer:
""")


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


@dataclass(frozen=True)
class ExpansionResult:
    """A collection of query expansions keyed by expansion type."""

    expansions: dict[str, str]

    def get(self, method: str, fallback: str) -> str:
        return self.expansions.get(method, fallback)

    def items(self) -> list[tuple[str, str]]:
        return list(self.expansions.items())

    def with_added(self, key: str, value: str) -> "ExpansionResult":
        return ExpansionResult(expansions={**self.expansions, key: value})

    def with_merged(self, other: dict[str, str]) -> "ExpansionResult":
        return ExpansionResult(expansions={**self.expansions, **other})


@dataclass(frozen=True)
class MethodEffectivenessStats:
    """Effectiveness stats for a single expansion method on a single query."""

    query_used: str
    unique_scientist_count: int
    total_result_count: int
    scientist_names: list[str]


def _parse_labeled_llm_response(response_text: str, *, labels: list[str]) -> dict[str, str]:
    """Extracts labeled values from an LLM response (e.g. 'Synonym: ...', 'Technical: ...')."""
    parsed: dict[str, str] = {}
    for line in response_text.split("\n"):
        stripped_line = line.strip()
        for label in labels:
            prefix = f"{label}:"
            if stripped_line.startswith(prefix):
                parsed[label.lower()] = stripped_line.removeprefix(prefix).strip()
    return parsed


def load_and_index_scientist_chunks() -> tuple[list[Document], InMemoryVectorStore]:
    """
    Loads scientist biographies, attaches metadata, splits into chunks,
    and indexes them into a vector store for similarity search.
    """
    print_section_header("1️⃣  LOADING DOCUMENTS FOR QUERY EXPANSION")

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


def expand_by_synonyms(query: str) -> str:
    """Expands the query by appending the top 2 synonyms for each recognized word."""
    expanded_terms: list[str] = []
    for word in query.lower().split():
        expanded_terms.append(word)
        if word in SYNONYM_MAP:
            expanded_terms.extend(SYNONYM_MAP[word][:2])
    return " ".join(expanded_terms)


def expand_by_concepts(query: str) -> str:
    """Expands the query by appending related scientific concepts for recognized terms."""
    query_lower = query.lower()
    combined_parts = [query]

    for concept, related_terms in CONCEPT_MAP.items():
        if concept in query_lower:
            combined_parts.append(" ".join(related_terms[:2]))

    return " OR ".join(combined_parts)


def expand_via_llm(llm: AzureChatOpenAI, query: str) -> dict[str, str]:
    """Uses the LLM to generate synonym, technical, and alternative expansions."""
    try:
        llm_response = llm.invoke(LLM_EXPANSION_PROMPT.format(query=query))
        return _parse_labeled_llm_response(
            llm_response.content,
            labels=["Synonym", "Technical", "Alternative"],
        )
    except Exception as expansion_error:
        print(f"   ⚠️ LLM expansion failed: {expansion_error}")
        return {}


def generate_multi_perspective_queries(llm: AzureChatOpenAI, query: str) -> dict[str, str]:
    """Generates historical, technical, and impact perspective queries via LLM."""
    try:
        llm_response = llm.invoke(MULTI_PERSPECTIVE_PROMPT.format(query=query))
        return _parse_labeled_llm_response(
            llm_response.content,
            labels=["Historical", "Technical", "Impact"],
        )
    except Exception as perspective_error:
        print(f"   ⚠️ Perspective generation failed: {perspective_error}")
        return {}


def expand_with_domain_context(llm: AzureChatOpenAI, query: str) -> str:
    """Uses domain-aware LLM prompt to expand the query with scientist-specific context."""
    try:
        llm_response = llm.invoke(CONTEXT_AWARE_PROMPT.format(query=query))
        return llm_response.content.replace("Expanded query:", "").strip()
    except Exception as context_error:
        print(f"   ⚠️ Context expansion failed: {context_error}")
        return query


def run_expansion_pipeline(
        llm: AzureChatOpenAI,
        query: str,
        *,
        methods: tuple[str, ...] = ("llm", "synonym", "context"),
) -> ExpansionResult:
    """
    Runs a configurable expansion pipeline that applies multiple strategies
    and collects all expansions into an ExpansionResult.
    """
    expansions: dict[str, str] = {"original": query}

    if "synonym" in methods:
        expansions["synonym"] = expand_by_synonyms(query)

    if "concept" in methods:
        expansions["concept"] = expand_by_concepts(query)

    if "llm" in methods:
        expansions.update(expand_via_llm(llm, query))

    if "context" in methods:
        expansions["context"] = expand_with_domain_context(llm, query)

    if "perspective" in methods:
        for perspective_key, perspective_query in generate_multi_perspective_queries(llm, query).items():
            expansions[f"perspective_{perspective_key}"] = perspective_query

    return ExpansionResult(expansions=expansions)


def demonstrate_basic_expansion() -> None:
    """
    Synonym expansion adds related words to broaden keyword matching.
    Concept expansion adds domain-specific terms associated with recognized entities.
    Both work without an LLM — they use static lookup maps.
    """
    print_section_header("2️⃣  BASIC QUERY EXPANSION")
    print(textwrap.dedent(demonstrate_basic_expansion.__doc__))

    sample_query = "Einstein's important discoveries"
    print(textwrap.dedent(f"""\
           🔍 Original query: {sample_query}
           📝 Synonym expansion: {expand_by_synonyms(sample_query)}
           🧠 Concept expansion: {expand_by_concepts(sample_query)}"""))


def demonstrate_llm_expansion(llm: AzureChatOpenAI) -> None:
    """
    LLM-based expansion generates richer reformulations by leveraging the model's
    language understanding. Each query is expanded into synonym-based, technical,
    and alternative phrasings — all produced by the LLM in a single call.
    """
    print_section_header("3️⃣  LLM-BASED QUERY EXPANSION")
    print(textwrap.dedent(demonstrate_llm_expansion.__doc__))

    llm_test_queries = [
        "What did Newton discover?",
        "Einstein's theory",
        "Scientific method",
    ]

    for test_query in llm_test_queries:
        print(f"\n   🔍 Query: {test_query}")
        llm_expansions = expand_via_llm(llm, test_query)
        for expansion_type, expanded_text in llm_expansions.items():
            print(f"      {expansion_type.capitalize()}: {expanded_text}")


def demonstrate_multi_perspective(llm: AzureChatOpenAI) -> None:
    """
    Multi-perspective generation produces three complementary queries for each topic:
    historical (when/where/context), technical (how/mechanisms), and impact (why/consequences).
    This broadens retrieval to cover different facets of the same subject.
    """
    print_section_header("4️⃣  MULTI-PERSPECTIVE QUERY GENERATION")
    print(textwrap.dedent(demonstrate_multi_perspective.__doc__))

    perspective_topics = [
        "Theory of relativity",
        "Laws of motion",
        "Radioactivity research",
    ]

    for topic in perspective_topics:
        print(f"\n   🎯 Topic: {topic}")
        perspectives = generate_multi_perspective_queries(llm, topic)
        for perspective_type, perspective_query in perspectives.items():
            print(f"      {perspective_type.capitalize()}: {perspective_query}")


def demonstrate_context_aware_expansion(llm: AzureChatOpenAI) -> None:
    """
    Context-aware expansion leverages knowledge of which scientists are in the
    database to inject relevant names and field-specific terminology into the query.
    """
    print_section_header("5️⃣  CONTEXT-AWARE QUERY EXPANSION")
    print(textwrap.dedent(demonstrate_context_aware_expansion.__doc__))

    context_test_queries = [
        "gravity research",
        "computer programming history",
        "radioactive elements",
        "species evolution",
    ]

    for test_query in context_test_queries:
        expanded = expand_with_domain_context(llm, test_query)
        print(textwrap.dedent(f"""
               🔍 Original: {test_query}
               🎯 Context-aware: {expanded}"""))


def demonstrate_expansion_pipeline(llm: AzureChatOpenAI) -> None:
    """
    The expansion pipeline combines all strategies (synonym, LLM, context,
    perspective) into a single call, collecting every expansion variant.
    """
    print_section_header("6️⃣  QUERY EXPANSION PIPELINE")
    print(textwrap.dedent(demonstrate_expansion_pipeline.__doc__))

    pipeline_query = "scientific discoveries"
    print(f"\n   🔍 Pipeline test query: {pipeline_query}")

    all_expansions = run_expansion_pipeline(
        llm,
        pipeline_query,
        methods=("llm", "synonym", "context", "perspective"),
    )
    for expansion_type, expanded_text in all_expansions.items():
        print(f"      {expansion_type}: {expanded_text}")


def demonstrate_retrieval_with_expansion(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
) -> None:
    """
    Compares retrieval results across expansion methods. The same base query is
    expanded via different techniques, and each expansion is used for similarity
    search — revealing how expansion affects which documents are retrieved.
    """
    print_section_header("7️⃣  RETRIEVAL COMPARISON WITH QUERY EXPANSION")
    print(textwrap.dedent(demonstrate_retrieval_with_expansion.__doc__))

    retrieval_query = "Newton's work"
    print(f"   🔍 Retrieval test query: {retrieval_query}")

    all_expansions = run_expansion_pipeline(
        llm, retrieval_query, methods=("llm", "synonym", "context"),
    )

    requested_methods = ["original", "synonym", "technical", "context"]
    for method_name in requested_methods:
        expanded_query = all_expansions.get(method_name, retrieval_query)
        search_results = vector_store.similarity_search_with_score(expanded_query, k=3)

        print(f"\n   📊 {method_name.capitalize()} expansion results:")
        for rank, (doc, similarity_score) in enumerate(search_results, start=1):
            scientist_name = doc.metadata["scientist_name"]
            content_preview = doc.page_content[:60] + "..."
            print(f"      {rank}. {scientist_name} (score: {similarity_score:.3f}): {content_preview}")


def _run_rag_with_expansion(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
        *,
        original_query: str,
        expanded_query: str,
) -> tuple[str, list[Document]]:
    """Retrieves documents using expanded_query and generates an answer referencing the original."""
    retrieved_documents = vector_store.similarity_search(expanded_query, k=4)

    context_parts = [
        f"Source {source_rank} ({doc.metadata['scientist_name']}): {doc.page_content}"
        for source_rank, doc in enumerate(retrieved_documents, start=1)
    ]
    formatted_context = "\n\n".join(context_parts)

    llm_response = llm.invoke(
        EXPANSION_RAG_PROMPT.format(
            original_query=original_query,
            expanded_query=expanded_query,
            context=formatted_context,
        )
    )
    return llm_response.content, retrieved_documents


def demonstrate_expanded_rag(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
) -> None:
    """
    Full RAG pipeline using query expansion: each question is expanded via
    multiple methods, and the LLM generates answers from the retrieved context.
    Comparing 'original' vs 'context' vs 'technical' shows expansion's effect.
    """
    print_section_header("8️⃣  BUILDING EXPANDED QUERY RAG SYSTEM")
    print(textwrap.dedent(demonstrate_expanded_rag.__doc__))

    print_section_header("9️⃣  TESTING EXPANDED RAG SYSTEM")

    rag_test_questions = [
        "gravity laws",
        "light research",
        "computing pioneers",
    ]

    expansion_methods_to_test = ["original", "context", "technical"]

    for question_number, question in enumerate(rag_test_questions, start=1):
        print(f"\n   Q{question_number}: {question}\n"
              f"   {'-' * 50}")

        try:
            all_expansions = run_expansion_pipeline(
                llm, question, methods=("llm", "context", "synonym"),
            )

            for method_name in expansion_methods_to_test:
                expanded_query = all_expansions.get(method_name, question)

                answer, source_documents = _run_rag_with_expansion(
                    llm,
                    vector_store,
                    original_query=question,
                    expanded_query=expanded_query,
                )

                source_names = [doc.metadata["scientist_name"] for doc in source_documents]
                print(textwrap.dedent(f"""
                      {method_name.capitalize()} method:
                      Query used: {expanded_query}
                      Answer: {answer}
                      Sources: {source_names}"""))

        except Exception as rag_error:
            print(f"      Error: {rag_error}")


def demonstrate_expansion_effectiveness(
        llm: AzureChatOpenAI,
        vector_store: InMemoryVectorStore,
) -> None:
    """
    Measures how each expansion method affects retrieval diversity (unique scientists
    in top-3 results) and query length. Higher diversity often means better coverage
    of relevant information across the knowledge base.
    """
    print_section_header("🔟  QUERY EXPANSION EFFECTIVENESS ANALYSIS")
    print(textwrap.dedent(demonstrate_expansion_effectiveness.__doc__))

    analysis_queries = ["scientific work", "important discoveries", "physics research"]
    analysis_methods = ["original", "synonym", "context", "technical"]

    effectiveness_table: dict[str, dict[str, MethodEffectivenessStats]] = {}

    for base_query in analysis_queries:
        all_expansions = run_expansion_pipeline(
            llm, base_query, methods=("synonym", "context", "llm"),
        )
        method_stats: dict[str, MethodEffectivenessStats] = {}

        for method_name in analysis_methods:
            used_query = all_expansions.get(method_name, base_query)
            search_results = vector_store.similarity_search(used_query, k=3)

            scientist_names = [doc.metadata["scientist_name"] for doc in search_results]
            method_stats[method_name] = MethodEffectivenessStats(
                query_used=used_query,
                unique_scientist_count=len(set(scientist_names)),
                total_result_count=len(search_results),
                scientist_names=scientist_names,
            )

        effectiveness_table[base_query] = method_stats

    print(f"\n   📊 Expansion effectiveness analysis:")
    print(f"   {'Query':<22} {'Method':<12} {'Unique Sci.':<12} {'Query Length':<12}")
    print("   " + "-" * 65)

    for base_query, methods_data in effectiveness_table.items():
        for method_name, stats in methods_data.items():
            query_word_count = len(stats.query_used.split())
            print(
                f"   {base_query:<22} {method_name:<12} "
                f"{stats.unique_scientist_count:<12} {query_word_count:<12}"
            )

    print(textwrap.dedent("""\

        💡 Query expansion RAG system ready!
        🔍 Use run_expansion_pipeline() + _run_rag_with_expansion() for enhanced retrieval
        🎯 Supports: synonym, concept, LLM-based, context-aware, and multi-perspective expansion"""))


if __name__ == "__main__":
    load_dotenv(override=True)

    print("🔍 QUERY EXPANSION DEMONSTRATION\n"
          "=" * 50)

    _chunks, expansion_vector_store = load_and_index_scientist_chunks()

    demonstrate_basic_expansion()

    azure_chat_llm = AzureChatOpenAI(model="gpt-5-nano")

    demonstrate_llm_expansion(azure_chat_llm)
    demonstrate_multi_perspective(azure_chat_llm)
    demonstrate_context_aware_expansion(azure_chat_llm)
    demonstrate_expansion_pipeline(azure_chat_llm)
    demonstrate_retrieval_with_expansion(azure_chat_llm, expansion_vector_store)
    demonstrate_expanded_rag(azure_chat_llm, expansion_vector_store)
    demonstrate_expansion_effectiveness(azure_chat_llm, expansion_vector_store)
