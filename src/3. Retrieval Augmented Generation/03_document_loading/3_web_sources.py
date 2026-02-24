#!/usr/bin/env python3
"""
🌐 Web Sources Loading for RAG Systems
=======================================

This script demonstrates loading content from web sources for a
Retrieval-Augmented Generation pipeline.  It covers URL accessibility
checks, web scraping with LangChain's WebBaseLoader, HTML cleanup with
BeautifulSoup, structured content extraction, chunking, and building
a complete RAG chain over web-sourced knowledge.

🎯 What You'll Learn:
- Verifying URL accessibility before loading
- Loading web pages with WebBaseLoader and custom headers
- Cleaning HTML content with BeautifulSoup
- Extracting structured data (title, headings, body) from pages
- Chunking web content for vector search
- Building a RAG chain over web-sourced documents
- Best practices, legal and ethical considerations

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Internet access to Simple English Wikipedia
- requests, beautifulsoup4 packages installed
"""

import re
import textwrap
import time
from dataclasses import dataclass
from typing import Final

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

WIKIPEDIA_SOURCES: Final[dict[str, str]] = {
    "Ada Lovelace": "https://simple.wikipedia.org/wiki/Ada_Lovelace",
    "Albert Einstein": "https://simple.wikipedia.org/wiki/Albert_Einstein",
    "Isaac Newton": "https://simple.wikipedia.org/wiki/Isaac_Newton",
    "Marie Curie": "https://simple.wikipedia.org/wiki/Marie_Curie",
}

BROWSER_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

NAVIGATION_ARTIFACTS: Final[tuple[str, ...]] = (
    "Jump to navigation",
    "Jump to search",
    "[edit]",
)

RAG_PROMPT_TEMPLATE: Final[str] = """\
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
The context comes from web sources and may contain some formatting artifacts.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""


@dataclass(frozen=True)
class StructuredWebContent:
    title: str
    headings: tuple[str, ...]
    main_content: str


@dataclass(frozen=True)
class ChunkDistribution:
    scientist_name: str
    chunk_count: int


@dataclass(frozen=True)
class BestPracticeCategory:
    heading: str
    emoji: str
    bullet_points: tuple[str, ...]


WEB_BEST_PRACTICES: Final[tuple[BestPracticeCategory, ...]] = (
    BestPracticeCategory(
        heading="Web Loading Advantages",
        emoji="✅",
        bullet_points=(
            "Access to up-to-date information",
            "Vast amount of available content",
            "Can supplement local documents",
            "Enables real-time knowledge updates",
        ),
    ),
    BestPracticeCategory(
        heading="Web Loading Challenges",
        emoji="⚠️",
        bullet_points=(
            "Content may change or disappear",
            "Requires internet connectivity",
            "Rate limiting and access restrictions",
            "Content quality varies widely",
            "Legal and ethical considerations",
        ),
    ),
    BestPracticeCategory(
        heading="Web Processing Best Practices",
        emoji="🔧",
        bullet_points=(
            "Respect robots.txt and rate limits",
            "Clean content of navigation elements",
            "Handle errors gracefully",
            "Cache content when appropriate",
            "Verify content quality and accuracy",
        ),
    ),
    BestPracticeCategory(
        heading="When to Use Web Loading",
        emoji="📊",
        bullet_points=(
            "Need current information",
            "Supplementing local knowledge base",
            "Research and fact-checking applications",
            "Building comprehensive knowledge systems",
        ),
    ),
    BestPracticeCategory(
        heading="Legal and Ethical Considerations",
        emoji="⚖️",
        bullet_points=(
            "Check website terms of service",
            "Respect copyright and fair use",
            "Don't overload servers with requests",
            "Consider data privacy implications",
        ),
    ),
)


def print_section_header(title: str) -> None:
    separator = "=" * 50
    print(f"\n{separator}\n{title}\n{separator}")


def scientist_name_from_url(url: str) -> str:
    return url.split("/")[-1].replace("_", " ")


def is_url_accessible(url: str) -> bool:
    """Checks whether a URL responds with HTTP 200 (tries HEAD, then GET)."""
    headers = {"User-Agent": BROWSER_USER_AGENT}
    try:
        head_response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if head_response.status_code == 200:
            return True
        get_response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        return get_response.status_code == 200
    except requests.exceptions.RequestException as request_error:
        print(f"   🔍 Debug: {url} failed with {type(request_error).__name__}: {request_error}")
        return False
    except Exception as unexpected_error:
        print(f"   🔍 Debug: {url} failed with unexpected error: {unexpected_error}")
        return False


def clean_html_to_plain_text(raw_html: str) -> str:
    """Strips navigation/script elements and collapses whitespace into clean plain text."""
    soup = BeautifulSoup(raw_html, "html.parser")

    tags_to_remove = ("nav", "header", "footer", "script", "style", "noscript")
    for tag_name in tags_to_remove:
        for element in soup.find_all(tag_name):
            element.decompose()

    plain_text = BeautifulSoup.get_text(soup, separator=" ", strip=True)
    plain_text = re.sub(r"\s+", " ", plain_text)

    for artifact in NAVIGATION_ARTIFACTS:
        plain_text = plain_text.replace(artifact, "")

    return plain_text.strip()


def extract_structured_web_content(raw_html: str) -> StructuredWebContent | None:
    """Extracts title, section headings, and cleaned body from raw HTML."""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")

        title_element = soup.find("title")
        page_title = title_element.get_text(strip=True) if title_element else ""

        section_headings = tuple(
            heading.get_text(strip=True)
            for heading in soup.find_all(["h1", "h2", "h3"])
        )

        cleaned_body = clean_html_to_plain_text(raw_html)

        return StructuredWebContent(
            title=page_title,
            headings=section_headings[:5],
            main_content=cleaned_body,
        )
    except Exception as extraction_error:
        print(f"   ⚠️ Extraction failed: {extraction_error}")
        return None


def compute_chunk_distribution(chunks: list[Document]) -> list[ChunkDistribution]:
    distribution_map: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if "wikipedia.org" in source or "example.com" in source:
            name = scientist_name_from_url(source)
            distribution_map[name] = distribution_map.get(name, 0) + 1
    return [
        ChunkDistribution(scientist_name=name, chunk_count=count)
        for name, count in distribution_map.items()
    ]


def display_configured_sources() -> None:
    """
    Setting up web sources — Simple English Wikipedia URLs.

    We use Simple English Wikipedia because it provides cleaner, shorter
    articles that are easier to parse and demonstrate with, while still
    containing real encyclopaedic content.
    """
    print_section_header("1️⃣ SETTING UP WEB SOURCES")
    print(textwrap.dedent(display_configured_sources.__doc__))

    print("   Configured web sources:")
    for scientist_name, wikipedia_url in WIKIPEDIA_SOURCES.items():
        print(f"   📖 {scientist_name}: {wikipedia_url}")


def filter_accessible_urls() -> dict[str, str]:
    """
    Testing URL accessibility before loading.

    Before spending time on web scraping, we verify that each URL actually
    responds with HTTP 200.  This avoids confusing loader errors later and
    lets us report which sources are available up front.
    """
    print_section_header("2️⃣ TESTING URL ACCESSIBILITY")
    print(textwrap.dedent(filter_accessible_urls.__doc__))

    reachable_urls: dict[str, str] = {}
    for scientist_name, wikipedia_url in WIKIPEDIA_SOURCES.items():
        url_is_reachable = is_url_accessible(wikipedia_url)
        status_label = "✅ Accessible" if url_is_reachable else "❌ Not accessible"
        print(f"   {scientist_name}: {status_label}")
        if url_is_reachable:
            reachable_urls[scientist_name] = wikipedia_url

    print(f"   Found {len(reachable_urls)} accessible URLs")
    return reachable_urls


def load_web_documents(reachable_urls: dict[str, str]) -> list[Document]:
    """
    Loading web content with WebBaseLoader.

    WebBaseLoader fetches a page's HTML and wraps it in a LangChain Document
    with source URL metadata.  We pass a browser-like User-Agent header so
    that servers do not reject the request, and pause between requests to
    respect rate limits.
    """
    print_section_header("3️⃣ LOADING WEB CONTENT")
    print(textwrap.dedent(load_web_documents.__doc__))

    if not reachable_urls:
        raise RuntimeError(
            "Cannot proceed: No accessible web sources found. "
            "Web loading demonstration requires internet access to Wikipedia."
        )

    loaded_documents: list[Document] = []
    for scientist_name, wikipedia_url in reachable_urls.items():
        print(f"   Loading: {scientist_name}")
        try:
            web_loader = WebBaseLoader(
                wikipedia_url,
                header_template={"User-Agent": BROWSER_USER_AGENT},
            )
            fetched_pages = web_loader.load()

            if fetched_pages:
                first_page = fetched_pages[0]
                print(textwrap.dedent(f"""\
                       📄 {scientist_name}: {len(first_page.page_content)} characters
                       🏷️ Metadata: {first_page.metadata}
                       📝 Preview: {first_page.page_content[:150]}..."""))
                loaded_documents.extend(fetched_pages)
            else:
                print(f"   ⚠️ No content loaded for {scientist_name}")

            time.sleep(1)

        except Exception as loading_error:
            print(f"   ❌ Error loading {scientist_name}: {loading_error}")

    print(f"\n   Total web documents loaded: {len(loaded_documents)}")
    return loaded_documents


def clean_and_structure_documents(raw_documents: list[Document]) -> list[Document]:
    """
    Cleaning and structuring raw HTML content with BeautifulSoup.

    Raw web pages contain navigation bars, scripts, footers, and other
    elements that add noise to embeddings.  We strip them out and extract
    structured metadata (title, section headings) that enriches retrieval.
    """
    print_section_header("4️⃣ PROCESSING WEB CONTENT")
    print(textwrap.dedent(clean_and_structure_documents.__doc__))

    for web_document in raw_documents:
        original_length = len(web_document.page_content)
        source_url = web_document.metadata.get("source", "")
        scientist_label = scientist_name_from_url(source_url)

        print(f"   🔍 Processing {scientist_label}...")

        structured = extract_structured_web_content(web_document.page_content)

        if structured is not None:
            web_document.page_content = structured.main_content
            web_document.metadata.update({
                "title": structured.title,
                "headings": list(structured.headings),
                "scientist": scientist_label,
                "content_type": "web_structured",
            })
            cleaned_length = len(web_document.page_content)
            print(f"   🧹 Cleaned: {original_length} → {cleaned_length} characters")
        else:
            web_document.page_content = clean_html_to_plain_text(web_document.page_content)
            print(f"   🧹 Fallback cleaning applied")

    return raw_documents


def analyze_sample_document(cleaned_documents: list[Document]) -> None:
    """
    Inspecting the first cleaned document to verify extraction quality.

    After cleaning, we check the resulting title, section headings, and
    content length to make sure the structured extraction worked correctly
    before feeding documents into the vector store.
    """
    print_section_header("5️⃣ BEAUTIFULSOUP CONTENT ANALYSIS")
    print(textwrap.dedent(analyze_sample_document.__doc__))

    if not cleaned_documents:
        print("   No documents to analyze.")
        return

    sample = cleaned_documents[0]
    source_url = sample.metadata.get("source", "")
    scientist_label = scientist_name_from_url(source_url)

    print(textwrap.dedent(f"""\
           📊 Analysis of {scientist_label} page:
           🏷️  Title: {sample.metadata.get('title', 'N/A')}
           📝 Introduction: {sample.metadata.get('introduction', 'N/A')[:200]}..."""))

    extracted_headings: list[str] = sample.metadata.get("headings", [])
    if extracted_headings:
        print(f"   📋 Page structure ({len(extracted_headings)} sections):")
        for heading_number, heading_text in enumerate(extracted_headings[:8], start=1):
            print(f"      {heading_number}. {heading_text}")
        if len(extracted_headings) > 8:
            print(f"      ... and {len(extracted_headings) - 8} more sections")

    print(textwrap.dedent(f"""\
           📄 Clean content length: {len(sample.page_content)} characters
           🔍 Content preview: {sample.page_content[:300]}..."""))


def split_web_documents_into_chunks(cleaned_documents: list[Document]) -> list[Document]:
    """
    Chunking cleaned web content for vector search.

    Web text has different structural separators than PDFs or plain files,
    so we configure RecursiveCharacterTextSplitter with a separator hierarchy
    tailored to HTML-extracted prose: paragraph breaks, line breaks, sentence
    endings, then words.
    """
    print_section_header("6️⃣ CHUNKING WEB CONTENT")
    print(textwrap.dedent(split_web_documents_into_chunks.__doc__))

    web_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = web_text_splitter.split_documents(cleaned_documents)
    print(f"   Created {len(chunks)} chunks from web content")

    distribution = compute_chunk_distribution(chunks)
    print("   Chunk distribution by scientist:")
    for entry in distribution:
        print(f"     {entry.scientist_name}: {entry.chunk_count} chunks")

    return chunks


def build_and_test_web_rag_chain(chunks: list[Document], cleaned_documents: list[Document]) -> None:
    """
    Building and testing a complete RAG chain over web-sourced chunks.

    The pipeline is identical to text-file and PDF RAG chains, but the
    prompt acknowledges that web-extracted context may contain formatting
    artifacts the LLM should gracefully ignore.
    """
    print_section_header("7️⃣ CREATING WEB-BASED RAG SYSTEM")
    print(textwrap.dedent(build_and_test_web_rag_chain.__doc__))

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = AzureChatOpenAI(model="gpt-5-nano")
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    web_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print_section_header("8️⃣ TESTING WEB-BASED RAG")

    test_questions = [
        "What are Ada Lovelace's most famous achievements?",
        "How did Einstein contribute to modern physics?",
        "What scientific discoveries is Marie Curie known for?",
    ]

    for question_number, question in enumerate(test_questions, start=1):
        print(f"\nQ{question_number}: {question}\n"
              f"{'-' * 40}")
        try:
            answer = web_rag_chain.invoke(question)
            print(f"A{question_number}: {answer}")
        except Exception as chain_error:
            print(f"A{question_number}: Error processing question: {chain_error}")

    if cleaned_documents:
        sample = cleaned_documents[0]
        headings_count = len(sample.metadata.get("headings", []))
        print(textwrap.dedent(f"""\

          📈 Results for sample document:
            • Extracted {headings_count} section headings
            • Clean content: {len(sample.page_content)} characters
            • Rich metadata: {len(sample.metadata)} fields"""))

    print(textwrap.dedent(f"""\

        💡 Web RAG chain ready: web_rag_chain.invoke('Your question')
        🌐 Processed {len(cleaned_documents)} web documents, {len(chunks)} chunks"""))


def print_web_best_practices() -> None:
    """
    Summary of best practices, challenges, and legal/ethical considerations
    for web-based document loading in RAG pipelines.
    """
    print_section_header("🌐 WEB LOADING BEST PRACTICES")
    print(textwrap.dedent(print_web_best_practices.__doc__))

    for category in WEB_BEST_PRACTICES:
        print(f"{category.emoji} {category.heading}:")
        for point in category.bullet_points:
            print(f"  • {point}")
        print()


if __name__ == "__main__":
    load_dotenv(override=True)

    display_configured_sources()
    accessible_urls = filter_accessible_urls()
    raw_web_documents = load_web_documents(reachable_urls=accessible_urls)
    cleaned_web_documents = clean_and_structure_documents(raw_documents=raw_web_documents)
    analyze_sample_document(cleaned_documents=cleaned_web_documents)
    web_chunks = split_web_documents_into_chunks(cleaned_documents=cleaned_web_documents)
    build_and_test_web_rag_chain(chunks=web_chunks, cleaned_documents=cleaned_web_documents)
    print_web_best_practices()
