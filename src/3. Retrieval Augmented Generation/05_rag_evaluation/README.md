# Module 5: RAG Evaluation with RAGAS

This module demonstrates how to evaluate RAG (Retrieval Augmented Generation) systems using the RAGAS (Retrieval
Augmented Generation Assessment) framework. It showcases both a single-file educational evaluation and a comprehensive
multi-system comparison framework.

All code uses **Azure OpenAI** (`AzureChatOpenAI`, `AzureOpenAIEmbeddings`) exclusively, follows Python 3.13 idioms, and
applies Clean Code principles throughout — frozen dataclasses for configuration, full type hints, self-documenting
names, no inline comments, educational content expressed via docstrings and console output.

## Learning Objectives

- Understand RAGAS evaluation metrics and methodology
- Learn proper ground truth generation for RAG evaluation
- Compare different RAG retrieval strategies objectively
- Apply software engineering best practices (frozen dataclasses, Protocol-based typing, Clean Code)
- Transition from prototype to production-ready evaluation code

## Module Structure

```
05_rag_evaluation/
├── 1. RAGAS_Naive_RAG.py              # Single-file RAGAS evaluation
├── data/                              # Scientist biographies dataset
│   ├── scientists_bios/
│   └── ground_truth_dataset.json      # Cached ground truth answers
├── multi_rag_evaluation/              # Multi-RAG comparison system
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py               # Frozen dataclass configuration (PipelineSettings)
│   ├── rag_systems/
│   │   ├── __init__.py
│   │   ├── base_rag.py               # Abstract base class (build → query lifecycle)
│   │   ├── naive_rag.py              # Baseline vector similarity
│   │   ├── metadata_filtering_rag.py  # Field-aware metadata filtering
│   │   ├── hybrid_search_rag.py      # BM25 + vector score fusion
│   │   ├── query_expansion_rag.py    # LLM-driven multi-query search
│   │   └── reranking_rag.py          # Cross-encoder / LLM reranking
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py              # RAGEvaluator with RAGSystem Protocol
│   │   └── ground_truth.py           # Expert LLM ground truth generation
│   ├── results/
│   │   └── comparison_metrics.csv    # Generated evaluation results
│   ├── data/
│   │   └── scientists_bios/          # Symlink / copy of biography data
│   └── main.py                       # Entry point with section headers
└── README.md                          # This file
```

## Part 1: Single-File RAGAS Evaluation

### File: `1. RAGAS_Naive_RAG.py`

A self-contained, educational walkthrough with five printed sections (STEP 1–5), each introduced by a docstring that is
also printed to the console:

| Step | Function                           | Purpose                                                             |
|------|------------------------------------|---------------------------------------------------------------------|
| 1    | `load_and_chunk_documents()`       | Load `.txt` files and split with `RecursiveCharacterTextSplitter`   |
| 2    | `build_rag_chain()`                | Assemble retriever → prompt → `AzureChatOpenAI` → parser            |
| 3    | `load_or_generate_ground_truths()` | Cache expert answers to JSON; reuse across runs                     |
| 4    | `collect_evaluation_samples()`     | Run questions through the chain, collect `SingleTurnSample` objects |
| 5    | `run_ragas_evaluation()`           | Score with 5 RAGAS metrics, print mean scores                       |

**Key design decisions:**

- Three frozen dataclasses: `RagPipelineConfig`, `EvaluationConfig`, `PathsConfig`
- All Azure OpenAI (no `ChatOpenAI` / `OpenAIEmbeddings`)
- `if __name__ == "__main__"` guard with `main()` function
- Named keyword arguments (`*`) on every helper for self-documenting call sites
- Ground truth cached to `ground_truth_dataset.json` to avoid repeated API calls

**Running:**

```bash
uv run python "1. RAGAS_Naive_RAG.py"
```

## Part 2: Multi-RAG Evaluation System

### Directory: `multi_rag_evaluation/`

A professional, extensible framework comparing 5 RAG strategies under identical conditions.

#### Configuration (`config/settings.py`)

All parameters live in frozen dataclasses composed into a single `PipelineSettings`:

| Dataclass             | Fields                                                                                       |
|-----------------------|----------------------------------------------------------------------------------------------|
| `ModelConfig`         | `expert_model`, `evaluator_model`, `rag_chat_model`, `embedding_model`                       |
| `ChunkingConfig`      | `chunk_size`, `chunk_overlap`                                                                |
| `RetrievalConfig`     | `top_k`, `bm25_weight`, `rerank_initial_top_k`, `rerank_final_top_k`, `max_query_variations` |
| `EvaluationQuestions` | `questions` (immutable tuple of evaluation prompts)                                          |
| `PipelineSettings`    | Composes all of the above; singleton `SETTINGS` instance                                     |

#### RAG Systems (`rag_systems/`)

All implementations inherit from `BaseRAG`, which enforces:

- `__init__(document_chunks, pipeline_settings)` — stores chunks, settings, initialises `rag_chain`, `retriever`,
  `vector_store` to `None`
- `build()` — abstract; each subclass wires its own retrieval strategy
- `query(question) → (answer, context_texts)` — concrete; runs chain + retriever
- `name` — abstract property for display

| Class                  | Strategy                   | Key Technique                                                                       |
|------------------------|----------------------------|-------------------------------------------------------------------------------------|
| `NaiveRAG`             | Baseline similarity search | `InMemoryVectorStore.as_retriever()`                                                |
| `MetadataFilteringRAG` | Field-aware pre-filtering  | Enriches chunks with scientist name, field, time period; filters by detected domain |
| `HybridSearchRAG`      | BM25 + vector fusion       | Min-max normalised scores combined with configurable `bm25_weight`                  |
| `QueryExpansionRAG`    | Multi-query retrieval      | LLM generates alternative phrasings + domain-concept expansion; deduplicated union  |
| `RerankingRAG`         | Post-retrieval re-scoring  | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) preferred; LLM 1-10 scoring as fallback    |

#### Evaluation Engine (`evaluation/`)

- **`GroundTruthGenerator`** — feeds complete biography texts to an expert LLM (`gpt-5`) to produce gold-standard
  answers
- **`RAGEvaluator`** — orchestrates the full lifecycle:
    1. Generate / load ground truths
    2. Run each `RAGSystem` (Protocol-typed) on every question
    3. Score with 5 RAGAS metrics
    4. Aggregate into a comparison DataFrame
    5. Print table with per-metric winners and overall best
    6. Save CSV to `results/`

The `RAGSystem` Protocol decouples the evaluator from `BaseRAG` — any object with `.name` and `.query()` is accepted.

#### Evaluation Results

| System                 | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Factual Correctness |
|------------------------|-------------------|----------------|--------------|------------------|---------------------|
| **Naive RAG**          | 0.700             | **0.634**      | 0.743        | 0.881            | 0.400               |
| **Metadata Filtering** | 0.500             | 0.287          | 0.613        | 0.884            | 0.406               |
| **Hybrid Search**      | 0.700             | 0.494          | 0.738        | 0.918            | 0.404               |
| **Query Expansion**    | 0.700             | 0.501          | 0.775        | **0.948**        | 0.396               |
| **Reranking**          | 0.567             | 0.434          | **0.767**    | 0.935            | **0.436**           |

**Key Insights:**

- **Query Expansion** achieves highest answer relevancy (0.948)
- **Reranking** shows best faithfulness (0.767) and factual correctness (0.436)
- **Naive RAG** surprisingly leads in context recall (0.634)
- **Metadata Filtering** struggles with limited biographical metadata
- Advanced techniques don't always outperform simpler approaches

**Running:**

```bash
cd multi_rag_evaluation
uv run python main.py
# Results saved to: results/comparison_metrics.csv
```

## Understanding RAGAS Metrics

| Metric                  | Question It Answers                                     | High Score Means                    |
|-------------------------|---------------------------------------------------------|-------------------------------------|
| **Context Precision**   | Are the retrieved chunks relevant to the question?      | Less noise in retrieval             |
| **Context Recall**      | Do retrieved chunks cover all ground-truth information? | Comprehensive information gathering |
| **Faithfulness**        | Is the answer grounded in retrieved context?            | Reduced hallucination               |
| **Answer Relevancy**    | Does the answer address the question directly?          | Better user experience              |
| **Factual Correctness** | Does the answer match the ground truth facts?           | Reliable information                |

### Ground Truth Generation

The ground truth must come from an **independent** expert LLM with **complete** document access — never from the RAG
system being evaluated:

```text
✅ Independent ground truth (used in this module):
   expert_llm = AzureChatOpenAI(model="gpt-5")
   full_biography_text = <all documents loaded and concatenated>
   ground_truth_answer = expert_llm.invoke("Context: ... Question: ...")

❌ Circular dependency — self-evaluation bias:
   ground_truth_answer = the_rag_system_being_evaluated.query(question)
```

## Installation & Setup

### Prerequisites

```bash
uv venv
source ./venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  # Windows

uv sync

# Azure OpenAI credentials in .env:
# OPENAI_API_KEY=your-api-key
# AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

### Required Packages

- `ragas` — RAG evaluation framework
- `langchain` ecosystem — RAG components
- `langchain-openai` — `AzureChatOpenAI`, `AzureOpenAIEmbeddings`
- `rank-bm25` — BM25 keyword search (hybrid RAG)
- `sentence-transformers` — cross-encoder reranking

## Code Quality Standards Applied

| Principle                   | How It's Applied                                                                                                                            |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| **Frozen Dataclasses**      | `PipelineSettings`, `RagPipelineConfig`, `EvaluationConfig`, `PathsConfig` — immutable, self-documenting configuration                      |
| **Protocol Typing**         | `RAGSystem` Protocol in evaluator decouples evaluation from concrete RAG classes                                                            |
| **No Inline Comments**      | Educational content lives in docstrings (printed to console) and self-documenting names                                                     |
| **No Shadowing**            | Every scope uses unique variable names (`source_directory` vs `data_directory`, `evaluation_questions` vs `scientist_evaluation_questions`) |
| **Named Keyword Args**      | `*` separator enforces explicit parameter names at call sites                                                                               |
| **Azure OpenAI Only**       | `AzureChatOpenAI` / `AzureOpenAIEmbeddings` throughout — no `ChatOpenAI` / `OpenAIEmbeddings`                                               |
| **`__init__` Declarations** | All instance attributes declared in `__init__` (including `vector_store`, `bm25_index`, `cross_encoder_model`)                              |
| **Narrow Exceptions**       | `except (ValueError, RuntimeError, AttributeError)` instead of bare `except Exception`                                                      |

## Extensions & Experiments

### Adding a New RAG System

1. Create a new file in `rag_systems/` inheriting from `BaseRAG`
2. Implement `build()` and the `name` property
3. Add the class to `rag_systems/__init__.py`
4. Add it to `rag_system_classes` list in `main.py:build_all_rag_systems()`
5. Run `python main.py`

### Further Ideas

- Statistical significance testing across question sets
- Cost / latency analysis per strategy
- Custom RAGAS metrics for domain-specific evaluation
- Parameter sweeps (chunk size, top-k, BM25 weight)
