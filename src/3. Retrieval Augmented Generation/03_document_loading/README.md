# Module 3: Document Loading for RAG Systems

This module teaches how to ingest content from different source formats —
plain text files, PDFs, and live web pages — and feed it into a
Retrieval-Augmented Generation pipeline using **LangChain** and **Azure OpenAI**.

## 🎯 Learning Objectives

By completing this module, you will:

- Master text, PDF, and web-based document loading strategies
- Understand trade-offs between loading approaches (single file, directory, glob patterns)
- Convert text files to PDF and extract content back with PyPDFLoader
- Scrape and clean web pages with WebBaseLoader and BeautifulSoup
- Split documents into overlapping chunks for vector search
- Build complete RAG chains over each source type
- Apply best practices for production document ingestion

## 📚 Module Content

### 1. Text File Loading (`1_text_files.py`)

**📄 Four strategies for loading text into a RAG pipeline**

A step-by-step script covering:

- **Single File Loading** — `TextLoader` for one document at a time
- **Multiple Specific Files** — iterating over a curated list of paths
- **Directory Loading** — `DirectoryLoader` for automatic file discovery
- **Glob Pattern Filtering** — selective discovery with `*.txt` patterns
- **Text Splitting** — `RecursiveCharacterTextSplitter` with chunk distribution analysis
- **RAG Chain** — end-to-end question-answering over scientist biographies

Key comparison of loading methods:

| Method                  | Advantages                       | Disadvantages                  |
|-------------------------|----------------------------------|--------------------------------|
| Single File             | Precise control, fast            | Manual, doesn't scale          |
| Multiple Specific Files | Selective, curated               | Requires file list maintenance |
| Directory Loading       | Automatic discovery, scales well | May include unwanted files     |
| Pattern-Based Loading   | Flexible filtering, best of both | —                              |

### 2. PDF Document Processing (`2_pdf_loading.py`)

**📄 Full PDF lifecycle — creation, extraction, comparison, and RAG**

Covers the complete workflow:

- **PDF Creation** — converting text biographies to styled PDFs with ReportLab
- **Page-Level Loading** — `PyPDFLoader` produces one `Document` per page with metadata
- **PDF vs Text Comparison** — character count differences and formatting artifact analysis
- **Chunk Distribution** — splitting PDF pages with overlap for vector search
- **RAG Chain** — question-answering with a prompt aware of PDF artifacts
- **Best Practices** — when to choose PDF loading, its advantages and pitfalls

### 3. Web Content Integration (`3_web_sources.py`)

**🌐 Live web scraping, HTML cleanup, and web-based RAG**

End-to-end web content ingestion:

- **Source Configuration** — Simple English Wikipedia URLs for clean demo content
- **URL Accessibility Checks** — HEAD/GET probes before loading
- **WebBaseLoader** — fetching pages with browser-like User-Agent headers
- **BeautifulSoup Cleaning** — removing nav/script/footer elements, collapsing whitespace
- **Structured Extraction** — title, section headings, and body from raw HTML
- **Web-Tuned Chunking** — separator hierarchy tailored to HTML-extracted prose
- **RAG Chain** — question-answering with web-sourced context
- **Best Practices** — rate limiting, legal/ethical considerations, error handling

## 📊 Document Type Comparison

| Feature               | Text Files     | PDF Files             | Web Sources          |
|-----------------------|----------------|-----------------------|----------------------|
| **Processing Speed**  | 🟢 Fastest     | 🟡 Medium             | 🔴 Slowest           |
| **Content Quality**   | 🟢 Clean       | 🟡 May have artifacts | 🟡 Needs cleaning    |
| **Reliability**       | 🟢 High        | 🟢 High               | 🟡 Network dependent |
| **Setup Complexity**  | 🟢 Simple      | 🟡 Medium             | 🟡 Medium            |
| **Metadata Richness** | 🔴 Basic       | 🟢 Rich (pages)       | 🟡 Medium (headings) |
| **Best For**          | Simple content | Formal documents      | Live / current data  |

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Azure OpenAI credentials in `.env`
- Internet access (for `3_web_sources.py`)

### Running the Scripts

```bash
# Text file loading — four strategies + RAG chain
uv run python "03_document_loading/1_text_files.py"

# PDF processing — creates sample PDFs, then loads and queries them
uv run python "03_document_loading/2_pdf_loading.py"

# Web content — scrapes Wikipedia, cleans HTML, runs RAG (requires internet)
uv run python "03_document_loading/3_web_sources.py"
```

## ✅ Expected Behaviour

**`1_text_files.py`**

- Demonstrates four loading approaches with scientist biography files
- Prints chunk distribution per scientist after splitting
- Answers three test questions via RAG chain
- Displays a comparison table of loading methods

**`2_pdf_loading.py`**

- Creates `data/pdfs/` with styled PDFs (Ada Lovelace, Albert Einstein)
- Shows page-level metadata from PyPDFLoader
- Compares character counts between PDF and plain-text representations
- Answers two test questions via PDF-based RAG chain
- Prints best-practices summary

**`3_web_sources.py`**

- Lists configured Wikipedia sources and probes accessibility
- Loads pages with WebBaseLoader, then cleans with BeautifulSoup
- Extracts structured metadata (title, headings) from HTML
- Creates web-tuned chunks and answers three test questions via RAG
- Prints best-practices and legal/ethical summary

## 🛠️ Key Dependencies

| Package            | Purpose                                |
|--------------------|----------------------------------------|
| `langchain`        | Document loaders, text splitters       |
| `langchain-openai` | AzureChatOpenAI, AzureOpenAIEmbeddings |
| `pypdf`            | PDF text extraction (PyPDFLoader)      |
| `reportlab`        | PDF creation for demonstrations        |
| `requests`         | HTTP requests for URL probing          |
| `beautifulsoup4`   | HTML parsing and content cleaning      |
| `python-dotenv`    | `.env` file loading                    |

## ⚠️ Common Issues

- **PDF extraction quality** — some PDFs yield poor text; test with your own files
- **Network failures** — `3_web_sources.py` raises `RuntimeError` if no URLs are accessible
- **Rate limiting** — web script pauses 1 s between requests to respect server limits
- **File encoding** — all text files must be UTF-8 compatible
- **Write permissions** — `2_pdf_loading.py` creates the `data/pdfs/` directory

## 🎓 Content Processing Pipeline

1. **Source Detection** — identify file type and choose the appropriate loader
2. **Content Extraction** — format-specific extraction (text / PDF pages / HTML)
3. **Cleaning** — remove artifacts, collapse whitespace, strip navigation elements
4. **Metadata Enrichment** — add source tracking, page numbers, section headings
5. **Chunking** — split with `RecursiveCharacterTextSplitter` (overlap for context)
6. **Indexing & RAG** — embed chunks, store in vector store, query with LLM

## 🚀 Next Steps

After mastering document loading, continue with:

- **04 Advanced Retrieval** — multi-query retrieval, re-ranking, hybrid search
- **05 RAG Evaluation** — measuring retrieval and generation quality
- **06 GraphRAG** — combining knowledge graphs with retrieval
