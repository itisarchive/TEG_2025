# Module 2: Vector Stores

## Learning Objectives

- Compare different vector storage options for RAG systems
- Understand trade-offs between performance, persistence, and complexity
- Implement RAG with in-memory, ChromaDB, and FAISS vector stores
- Learn when to choose each vector store type for different use cases

## Prerequisites

- Completed Module 1 (Basic RAG)
- Understanding of embeddings and similarity search
- Azure OpenAI credentials configured in `.env` file

## Scripts in This Module

### 1. `1_in_memory.py` – InMemoryVectorStore

The simplest vector store option for development and testing:

- Stores embeddings directly in RAM
- No external dependencies or setup
- Fast access but no persistence
- Perfect for prototyping and small datasets

**Key learning:** Understanding the baseline vector store behavior and the complete RAG pipeline (load → chunk → embed →
retrieve → answer)

### 2. `2_chroma_vectorestore.py` – ChromaDB Persistent Store

Persistent vector storage with ChromaDB:

- Automatic disk persistence and collection detection
- Similarity search with relevance scores
- Collection reloading and persistence verification across sessions
- Production-ready persistence patterns

**Key learning:** How persistent storage changes your RAG workflow and enables stateful applications

### 3. `3_faiss_vectorestore.py` – FAISS High-Performance Search

Facebook's FAISS library for production-scale similarity search:

- Optimized for speed and memory efficiency
- Multiple index types available (Flat, IVF, HNSW)
- GPU acceleration support (via `faiss-gpu`)
- Best for read-heavy, large-scale applications

**Key learning:** Performance optimization for vector search with L2 distance scores

## Key Concepts

- **Vector Store**: Database that stores embeddings and enables similarity search
- **Persistence**: Ability to save and reload data across sessions
- **Index Types**: Different algorithms for organizing vectors (Flat, IVF, HNSW)
- **Memory vs Disk**: Trade-offs between speed and persistence
- **Similarity Metrics**: How vectors are compared (cosine, euclidean, dot product)
- **Relevance / Distance Scores**: Numerical measures of how similar retrieved documents are to the query
- **Persistence Detection**: Automatically detecting and loading existing vector stores

## Vector Store Comparison

| Feature              | InMemory    | ChromaDB        | FAISS            |
|----------------------|-------------|-----------------|------------------|
| **Persistence**      | ❌ No        | ✅ Yes           | ✅ Yes            |
| **Setup Complexity** | 🟢 None     | 🟡 Simple       | 🟡 Simple        |
| **Performance**      | 🟢 Fast     | 🟡 Good         | 🟢 Very Fast     |
| **Memory Usage**     | 🔴 High     | 🟡 Medium       | 🟢 Low           |
| **Metadata Support** | ✅ Basic     | ✅ Rich          | ❌ Limited        |
| **Best For**         | Development | General Purpose | Production Scale |

## Running the Code

```bash
# Test in-memory storage (fastest startup)
uv run python "02_vector_stores/1_in_memory.py"

# Try persistent ChromaDB (creates ./chroma_db/)
uv run python "02_vector_stores/2_chroma_vectorestore.py"

# Experience high-performance FAISS (creates ./faiss_index/)
uv run python "02_vector_stores/3_faiss_vectorestore.py"
```

## Expected Behavior

**First Run:**

- All scripts create embeddings from scratch
- ChromaDB saves to `./chroma_db/` directory
- FAISS saves to `./faiss_index/` directory (`.faiss` + `.pkl` files)
- In-memory loses data when script ends

**Subsequent Runs:**

- In-memory rebuilds everything (no persistence)
- ChromaDB detects existing collection and loads instantly
- FAISS detects existing index files and loads instantly

**Special Features:**

- **ChromaDB**: Demonstrates persistence by reloading the collection mid-script
- **FAISS**: Shows index statistics (vector count, dimension) and L2 distance scores
- **All**: Include similarity/distance scores with search results

## Performance Notes

**Startup Times (typical):**

- InMemory: ~30s (always rebuilds)
- ChromaDB: ~2s (loads existing) / ~30s (first time)
- FAISS: ~1s (loads existing) / ~30s (first time)

**Search Speed:**

- InMemory: Good for small datasets
- ChromaDB: Good general performance
- FAISS: Excellent for any size

## Common Issues

- **Path errors**: All scripts expect `data/scientists_bios/` directory structure
- **ChromaDB permission errors**: Ensure write access to current directory for `./chroma_db/`
- **FAISS import errors**: Run `uv add faiss-cpu` to install (already included in dependencies)
- **Memory issues**: Reduce `ChunkingConfig.chunk_size` if running out of RAM
- **Environment variables**: Ensure Azure OpenAI credentials are set in `.env` file (scripts auto-load with
  `load_dotenv`)

## Choosing the Right Vector Store

**Use InMemoryVectorStore when:**

- Prototyping or development
- Small datasets (< 1000 documents)
- No need for persistence
- Testing different chunking strategies

**Use ChromaDB when:**

- Need persistence and metadata filtering
- General-purpose applications
- Medium datasets (1K–100K documents)
- Want built-in collection management

**Use FAISS when:**

- Production deployments
- Large datasets (> 100K documents)
- Performance is critical
- Primarily read-heavy workloads
- Need distance-based similarity metrics

## What's Next

Module 3 will explore different document loading strategies, including PDFs, web sources, and handling multiple file
types within the same RAG system.
