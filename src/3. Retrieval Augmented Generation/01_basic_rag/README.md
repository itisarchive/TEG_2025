# Module 1: Basic RAG

## Learning Objectives

- Understand the core RAG (Retrieval Augmented Generation) architecture
- Implement a minimal RAG system using in-memory vector storage
- Learn the importance of text chunking for improved retrieval
- Compare different chunking strategies and their impact on results

## Prerequisites

- Basic understanding of LLMs and embeddings (covered in previous modules)
- Azure OpenAI credentials configured in `.env` file
- Python 3.13+ with `langchain`, `langchain-openai`, and `python-dotenv`

## Scripts in This Module

### 1. `1. minimal_rag.py` — Simplest RAG Implementation

A step-by-step walk-through of the core RAG pipeline without any chunking:

- Loads scientist biographies from `data/scientists_bios/` via `DirectoryLoader`
- Embeds documents with `AzureOpenAIEmbeddings` and stores them in `InMemoryVectorStore`
- Assembles a LangChain RAG chain (Retriever → Prompt → LLM → StrOutputParser)
- Answers a single question and then a batch of questions (including one outside the knowledge base to demonstrate RAG
  grounding)

**Key learning:** The RAG pipeline end-to-end: Load → Embed → Store → Retrieve → Generate

### 2. `2. minimal_rag_with_chunking.py` — RAG with Text Splitting

Builds on the minimal example by introducing `RecursiveCharacterTextSplitter`:

- Splits documents into smaller, focused chunks before embedding
- Runs a batch of questions through the default chunking configuration (1 000 / 200)
- Compares three chunking strategies (500/50, 1 000/200, 2 000/400) on the same question
- Uses a frozen `ChunkingConfig` dataclass for self-documenting configuration

**Key learning:** Why chunking matters and how chunk size / overlap affect retrieval quality

## Data

`data/scientists_bios/` contains plain-text biographies of five scientists used as the knowledge base:
Ada Lovelace, Albert Einstein, Charles Darwin, Isaac Newton, Marie Skłodowska-Curie.

## Key Concepts

- **RAG Pipeline**: Document loading → embedding → storing → retrieving → generating
- **Vector Store**: A database of document embeddings enabling semantic similarity search
- **Retrieval**: Finding the most relevant documents / chunks by comparing query and document embeddings
- **Text Chunking**: Breaking large documents into smaller segments for more precise retrieval
- **Chunk Size vs Overlap**: Balancing context completeness with retrieval precision

## Running the Code

Make sure you're in the **project root** directory:

```bash
# Run minimal RAG example
uv run python "src/3. Retrieval Augmented Generation/01_basic_rag/1. minimal_rag.py"

# Run chunking comparison
uv run python "src/3. Retrieval Augmented Generation/01_basic_rag/2. minimal_rag_with_chunking.py"
```

## Expected Output

Both scripts print section headers as they progress through the pipeline, followed by
question-answer pairs. Each educational step is accompanied by a printed docstring
explaining what is happening and why.

The chunking script additionally compares three chunk-size configurations and prints
the total number of chunks and the answer for each.

## Common Issues

- **"No such file or directory"**: Make sure you're running from the project root — both scripts reference paths
  relative to it
- **Azure OpenAI errors**: Ensure your `.env` file contains valid `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and
  `AZURE_OPENAI_API_VERSION`
- **Import errors**: Run `uv sync` to install dependencies
- **libmagic warnings**: Harmless; can be fixed with `brew install libmagic` on macOS

## Key Takeaways

1. **RAG is fundamentally simple** — load documents, embed them, retrieve relevant context, generate an answer
2. **Chunking is crucial** — proper text splitting dramatically improves retrieval precision
3. **Trade-offs matter** — smaller chunks are more precise but may lack surrounding context; larger chunks preserve
   context but reduce precision
4. **Experimentation is key** — different datasets and use cases require different chunking strategies

## What's Next

Module 2 will introduce persistent vector stores (ChromaDB) that save embeddings to disk and offer more advanced
retrieval features than in-memory storage.
