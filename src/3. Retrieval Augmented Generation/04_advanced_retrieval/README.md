# Module 4: Advanced Retrieval Techniques

## Learning Objectives

- Master metadata filtering and contextual retrieval
- Implement hybrid search combining semantic and keyword search
- Optimize retrieval with re-ranking and query expansion
- Build multi-step retrieval pipelines
- Understand retrieval evaluation and optimization

## Prerequisites

- Completed Module 3 (Document Loading)
- Understanding of vector stores and embeddings
- Familiarity with document chunking strategies
- Azure OpenAI API credentials configured in `.env`

## Scripts in This Module

### 1. `1_metadata_filtering.py` — Contextual Retrieval

Demonstrates how metadata filtering enhances retrieval precision by narrowing search
results using document properties instead of relying solely on semantic similarity.

**Sections (8 steps):**

1. Loading documents with rich metadata extraction (`ScientistMetadata` frozen dataclass)
2. Chunking with metadata preservation (chunk id, size, position)
3. Building a vector store with metadata indexing
4. Metadata filtering demonstrations — by field, century, and completeness
5. Building a contextual RAG system with auto-filter selection (`create_contextual_retriever`)
6. Testing contextual RAG with diverse queries
7. Performance comparison: filtered vs. unfiltered retrieval
8. Metadata analysis summary (field distribution, time coverage, quality breakdown)

**Key structures:** `ScientistMetadata` (frozen dataclass with `as_flat_dict()`)

**Key learning:** How metadata enhances retrieval precision and relevance

---

### 2. `2_hybrid_search.py` — Semantic + Keyword Retrieval Fusion

Combines vector embeddings, BM25 keyword search, and TF-IDF into an adaptive
hybrid search system with configurable score fusion strategies.

**Sections (9 steps):**

1. Loading documents for hybrid search
2. Building multiple search indexes (vector store, BM25, TF-IDF)
3. Testing individual search methods side-by-side
4. Query analysis and routing (`QueryAnalysis` frozen dataclass)
5. Score fusion strategies — Reciprocal Rank Fusion (RRF) and Weighted Score Fusion (WSF)
6. Adaptive hybrid search with automatic weight selection
7. Building hybrid RAG system
8. Testing hybrid RAG with multiple query types
9. Hybrid search performance analysis (diversity comparison across methods)

**Key structures:** `QueryAnalysis`, `HybridSearchIndexes`, `ScoredDocument`

**Key learning:** When and how to combine different search methodologies

---

### 3. `3_query_expansion.py` — Enhanced Query Processing

Improves retrieval by expanding queries before search using synonym maps,
LLM-powered reformulation, multi-perspective generation, and domain-aware context.

**Sections (10 steps):**

1. Loading documents for query expansion
2. Basic query expansion (synonym map + concept map — no LLM needed)
3. LLM-based query expansion (synonym, technical, alternative phrasings)
4. Multi-perspective query generation (historical, technical, impact)
5. Context-aware query expansion (domain-specific scientist knowledge)
6. Query expansion pipeline (`run_expansion_pipeline` with configurable methods)
7. Retrieval comparison with expanded queries
8. Building expanded query RAG system
9. Testing expanded RAG across methods (original vs. context vs. technical)
10. Query expansion effectiveness analysis (diversity and query length metrics)

**Key structures:** `ExpansionResult` (frozen, immutable with `with_added`/`with_merged`), `MethodEffectivenessStats`

**Key learning:** Pre-processing techniques that improve retrieval quality

---

### 4. `4_reranking.py` — Post-Retrieval Optimization

Refines initial retrieval results using cross-encoder neural scoring, LLM-based
relevance assessment, diversity re-ranking, and ensemble fusion.

**Sections (8 steps):**

1. Loading documents for re-ranking
2. Loading cross-encoder models (`CrossEncoderRegistry` — MS-MARCO, QNLI)
3. Testing individual re-ranking methods (cross-encoder, LLM relevance, diversity)
4. Ensemble re-ranking (weighted fusion of cross-encoder + LLM scores)
5. Performance vs. quality analysis (latency benchmarking across methods)
6. Building re-ranking RAG system
7. Testing re-ranking RAG with baseline, LLM, and cross-encoder methods
8. Re-ranking effectiveness analysis (diversity ratio per method)

**Key structures:** `BenchmarkResult`, `EffectivenessStats`, `CrossEncoderRegistry`

**Key learning:** Post-processing techniques for optimal result quality

---

## Key Concepts

| Concept                  | Description                                                                        |
|--------------------------|------------------------------------------------------------------------------------|
| **Metadata Filtering**   | Using document properties (field, century, completeness) to constrain search space |
| **Hybrid Search**        | Combining semantic (vector) and lexical (BM25, TF-IDF) search methods              |
| **Query Expansion**      | Enriching queries with synonyms, concepts, and LLM-generated reformulations        |
| **Re-ranking**           | Post-retrieval scoring with cross-encoders or LLM relevance assessment             |
| **Score Fusion**         | Combining rankings via RRF or weighted score normalization                         |
| **Diversity Re-ranking** | Promoting variety in results to avoid redundant content                            |

## Retrieval Strategy Comparison

| Approach                | Precision  | Recall    | Speed     | Complexity | Best For          |
|-------------------------|------------|-----------|-----------|------------|-------------------|
| **Basic Vector Search** | 🟡 Medium  | 🟢 High   | 🟢 Fast   | 🟢 Simple  | General queries   |
| **Metadata Filtered**   | 🟢 High    | 🟡 Medium | 🟢 Fast   | 🟡 Medium  | Contextual search |
| **Hybrid Search**       | 🟢 High    | 🟢 High   | 🟡 Medium | 🔴 Complex | Diverse queries   |
| **Query Expanded**      | 🟢 High    | 🟢 High   | 🟡 Medium | 🟡 Medium  | Concept search    |
| **Re-ranked**           | 🟢 Highest | 🟡 Medium | 🔴 Slow   | 🔴 Complex | Quality-critical  |

## Running the Code

```bash
# Metadata filtering techniques
uv run python "04_advanced_retrieval/1_metadata_filtering.py"

# Hybrid search implementation
uv run python "04_advanced_retrieval/2_hybrid_search.py"

# Query expansion strategies
uv run python "04_advanced_retrieval/3_query_expansion.py"

# Re-ranking and optimization
uv run python "04_advanced_retrieval/4_reranking.py"
```

## Expected Behavior

**`1_metadata_filtering.py`:**

- Extracts rich metadata into `ScientistMetadata` (name, fields, birth year, completeness)
- Demonstrates field-specific, century-based, and quality-based filtering
- Auto-selects filters via `create_contextual_retriever` based on query content
- Compares filtered vs. unfiltered retrieval with diversity metrics

**`2_hybrid_search.py`:**

- Builds three parallel indexes: vector store, BM25, TF-IDF
- Classifies queries via `QueryAnalysis` (factual, conceptual, named entities)
- Fuses scores with RRF (`fuse_by_reciprocal_rank`) and WSF (`fuse_by_weighted_scores`)
- Benchmarks retrieval diversity across vector-only, BM25-only, and adaptive hybrid

**`3_query_expansion.py`:**

- Expands queries with static maps (`SYNONYM_MAP`, `CONCEPT_MAP`) and LLM calls
- Generates multi-perspective queries (historical, technical, impact)
- Collects all expansions into `ExpansionResult` via `run_expansion_pipeline`
- Measures effectiveness: unique scientist count and query length per method

**`4_reranking.py`:**

- Loads cross-encoder models into `CrossEncoderRegistry` (MS-MARCO, QNLI)
- Scores documents with cross-encoder, LLM relevance, and diversity heuristics
- Combines methods via `ensemble_rerank` with configurable weights
- Benchmarks latency (baseline vs. cross-encoder vs. LLM) and diversity ratio

## Code Architecture

All four scripts follow a consistent structure inspired by Clean Code principles:

```
Module docstring (educational overview)
    ↓
Imports (stdlib → third-party → project)
    ↓
Constants & Prompts (module-level)
    ↓
Frozen Dataclasses (immutable domain models)
    ↓
Private Helpers (_extract, _parse, _determine)
    ↓
Core Functions (load, build, search, rerank)
    ↓
Demonstrate Functions (educational sections with printed docstrings)
    ↓
if __name__ == "__main__": (orchestration)
```

Each `demonstrate_*` function prints its own docstring as educational content,
so the code is simultaneously self-documenting and instructional.

## Dependencies

| Package                          | Purpose                                               |
|----------------------------------|-------------------------------------------------------|
| **langchain / langchain-openai** | RAG pipeline, Azure OpenAI integration                |
| **rank-bm25**                    | BM25 keyword search (script 2)                        |
| **scikit-learn**                 | TF-IDF vectorization and cosine similarity (script 2) |
| **numpy**                        | Score fusion and normalization (script 2)             |
| **sentence-transformers**        | Cross-encoder models for re-ranking (script 4)        |
| **python-dotenv**                | Environment variable loading                          |

## Advanced Retrieval Pipeline

```
Query Input
    ↓
Query Analysis & Expansion (script 3)
    ↓
Metadata Filter Application (script 1)
    ↓
Parallel Search (script 2):
├── Semantic (Vector)
├── Keyword (BM25)
└── Structured (TF-IDF)
    ↓
Score Fusion & Initial Ranking (script 2)
    ↓
Re-ranking (script 4):
├── Cross-encoder (MS-MARCO / QNLI)
├── LLM Relevance Scoring
└── Diversity Optimization
    ↓
Final Results → LLM Generation
```
