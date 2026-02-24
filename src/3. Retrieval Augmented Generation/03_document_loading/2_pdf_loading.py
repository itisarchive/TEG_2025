#!/usr/bin/env python3
"""
📄 PDF Document Loading for RAG Systems
========================================

This script demonstrates loading and processing PDF documents for a
Retrieval-Augmented Generation pipeline.  It covers the full lifecycle:
generating PDFs from text sources, loading them with PyPDFLoader,
comparing PDF vs plain-text representations, chunking, and finally
building a complete RAG chain over PDF-based knowledge.

🎯 What You'll Learn:
- Converting text files to PDF with ReportLab
- Loading PDFs page-by-page with PyPDFLoader
- Differences between PDF and plain-text document representations
- Chunking PDF content for vector search
- Building a RAG chain over PDF documents
- Best practices and trade-offs of PDF-based ingestion

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Text files in the data/scientists_bios/ directory
- reportlab, pypdf packages installed
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SCIENTISTS_BIOS_DIR: Final[str] = (
    "src/3. Retrieval Augmented Generation/03_document_loading/data/scientists_bios"
)

PDF_OUTPUT_DIR: Final[str] = (
    "src/3. Retrieval Augmented Generation/03_document_loading/data/pdfs"
)

SOURCE_TEXT_FILES: Final[tuple[str, ...]] = (
    f"{SCIENTISTS_BIOS_DIR}/Ada Lovelace.txt",
    f"{SCIENTISTS_BIOS_DIR}/Albert Einstein.txt",
)

RAG_PROMPT_TEMPLATE: Final[str] = """\
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
The context comes from PDF documents that may have formatting artifacts.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""


@dataclass(frozen=True)
class PdfVsTextComparison:
    scientist_name: str
    text_character_count: int
    pdf_character_count: int
    pdf_page_count: int


@dataclass(frozen=True)
class ChunkDistribution:
    scientist_name: str
    chunk_count: int


@dataclass(frozen=True)
class BestPracticeCategory:
    heading: str
    emoji: str
    bullet_points: tuple[str, ...]


PDF_BEST_PRACTICES: Final[tuple[BestPracticeCategory, ...]] = (
    BestPracticeCategory(
        heading="Advantages of PDF Loading",
        emoji="✅",
        bullet_points=(
            "Preserves document structure and formatting",
            "Handles multi-page documents naturally",
            "Maintains page-level metadata",
            "Good for official documents, reports, papers",
        ),
    ),
    BestPracticeCategory(
        heading="PDF Loading Considerations",
        emoji="⚠️",
        bullet_points=(
            "May include formatting artifacts",
            "OCR quality affects text extraction",
            "Complex layouts can cause text jumbling",
            "Larger file sizes than plain text",
        ),
    ),
    BestPracticeCategory(
        heading="PDF Processing Tips",
        emoji="🔧",
        bullet_points=(
            "Use page-aware chunking strategies",
            "Clean extracted text of artifacts",
            "Consider PDF structure in retrieval",
            "Test with your specific PDF types",
        ),
    ),
    BestPracticeCategory(
        heading="When to Use PDF Loading",
        emoji="📊",
        bullet_points=(
            "Official documents and reports",
            "Academic papers and publications",
            "Multi-page structured content",
            "When document layout matters",
        ),
    ),
)


def print_section_header(title: str) -> None:
    separator = "=" * 50
    print(f"\n{separator}\n{title}\n{separator}")


def extract_scientist_name_from_path(file_path: str) -> str:
    return Path(file_path).stem


def convert_text_file_to_pdf(*, text_file_path: str, pdf_file_path: str) -> None:
    """Converts a plain-text file into a styled PDF using ReportLab."""
    source_content = Path(text_file_path).read_text(encoding="utf-8")

    pdf_document = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="CustomTitle",
        parent=base_styles["Heading1"],
        fontSize=16,
        alignment="left",
        spaceAfter=20,
    )
    body_style = ParagraphStyle(
        name="CustomNormal",
        parent=base_styles["Normal"],
        fontSize=11,
        alignment="left",
        spaceAfter=12,
    )

    content_lines = source_content.split("\n")
    document_title = content_lines[0] if content_lines else "Document"
    body_text = "\n".join(content_lines[1:]) if len(content_lines) > 1 else source_content

    story: list[Paragraph | Spacer] = [
        Paragraph(document_title, title_style),
        Spacer(1, 20),
    ]

    for raw_paragraph in body_text.split("\n\n"):
        cleaned_paragraph = raw_paragraph.strip().replace("\n", " ")
        if cleaned_paragraph:
            story.append(Paragraph(cleaned_paragraph, body_style))
            story.append(Spacer(1, 12))

    pdf_document.build(story)
    print(f"   ✅ Created PDF: {pdf_file_path}")


def create_pdfs_from_text_sources() -> list[str]:
    """
    Creating PDF versions of text files for demonstration.

    Before we can showcase PDF loading, we need actual PDF files.
    This step converts scientist biography text files into styled PDFs
    using ReportLab, placing them in the data/pdfs/ directory.
    """
    print_section_header("1️⃣ CREATING PDF FILES FROM TEXT SOURCES")
    print(textwrap.dedent(create_pdfs_from_text_sources.__doc__))

    Path(PDF_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    generated_pdf_paths: list[str] = []
    for text_file_path in SOURCE_TEXT_FILES:
        scientist_name = extract_scientist_name_from_path(text_file_path)
        pdf_file_path = f"{PDF_OUTPUT_DIR}/{scientist_name}.pdf"

        if Path(pdf_file_path).exists():
            print(f"   📄 PDF already exists: {pdf_file_path}")
        else:
            convert_text_file_to_pdf(
                text_file_path=text_file_path,
                pdf_file_path=pdf_file_path,
            )

        generated_pdf_paths.append(pdf_file_path)

    return generated_pdf_paths


def load_pdf_documents(pdf_file_paths: list[str]) -> list[Document]:
    """
    Loading PDF documents page-by-page with PyPDFLoader.

    PyPDFLoader reads each page of a PDF as a separate LangChain Document,
    preserving page numbers and source paths in the metadata.  This gives
    fine-grained control over which pages end up in which chunks.
    """
    print_section_header("2️⃣ LOADING PDF DOCUMENTS")
    print(textwrap.dedent(load_pdf_documents.__doc__))

    accumulated_pages: list[Document] = []

    for pdf_path in pdf_file_paths:
        print(f"   Loading: {pdf_path}")
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load()

        scientist_name = extract_scientist_name_from_path(pdf_path)
        print(f"   📄 {scientist_name}: {len(pages)} pages")

        for page_index, page_document in enumerate(pages):
            print(f"      Page {page_index + 1}: {len(page_document.page_content)} characters\n"
                  f"      Metadata: {page_document.metadata}")

        accumulated_pages.extend(pages)

    print(f"\n   Total PDF documents loaded: {len(accumulated_pages)}")
    return accumulated_pages


def compare_pdf_vs_text_loading(pdf_pages: list[Document]) -> PdfVsTextComparison:
    """
    Comparing PDF and plain-text representations of the same content.

    The same biography loaded as plain text vs. extracted from PDF will
    differ in character count and structure.  PDF extraction may introduce
    formatting artifacts (extra whitespace, line breaks) while adding
    page-level metadata that plain text does not have.
    """
    print_section_header("3️⃣ COMPARING PDF vs TEXT LOADING")
    print(textwrap.dedent(compare_pdf_vs_text_loading.__doc__))

    ada_text_path = f"{SCIENTISTS_BIOS_DIR}/Ada Lovelace.txt"
    text_loader = TextLoader(ada_text_path)
    ada_as_text = text_loader.load()[0]

    ada_pdf_pages = [
        page for page in pdf_pages
        if "Ada Lovelace" in page.metadata.get("source", "")
    ]

    comparison = PdfVsTextComparison(
        scientist_name="Ada Lovelace",
        text_character_count=len(ada_as_text.page_content),
        pdf_character_count=sum(len(page.page_content) for page in ada_pdf_pages),
        pdf_page_count=len(ada_pdf_pages),
    )

    print(textwrap.dedent(f"""\
           {comparison.scientist_name} comparison:
           📄 Text version: {comparison.text_character_count} characters
           📄 PDF version:  {comparison.pdf_character_count} characters
           📄 PDF pages:    {comparison.pdf_page_count}

           Text preview: {ada_as_text.page_content[:200]}...
           PDF preview:  {ada_pdf_pages[0].page_content[:200]}..."""))

    return comparison


def split_pdf_documents_into_chunks(pdf_pages: list[Document]) -> list[Document]:
    """
    Splitting PDF pages into smaller, overlapping chunks for vector search.

    PDF pages can be long, so we split them with RecursiveCharacterTextSplitter.
    Overlap ensures that context spanning a chunk boundary is not lost.
    """
    print_section_header("4️⃣ TEXT SPLITTING PDF DOCUMENTS")
    print(textwrap.dedent(split_pdf_documents_into_chunks.__doc__))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(pdf_pages)
    print(f"   PDF chunks created: {len(chunks)}")

    distribution = compute_chunk_distribution(chunks)
    print("   Chunks per scientist (PDF):")
    for entry in distribution:
        print(f"     {entry.scientist_name}: {entry.chunk_count} chunks")

    return chunks


def compute_chunk_distribution(chunks: list[Document]) -> list[ChunkDistribution]:
    distribution_map: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        scientist_name = Path(source).stem
        distribution_map[scientist_name] = distribution_map.get(scientist_name, 0) + 1
    return [
        ChunkDistribution(scientist_name=name, chunk_count=count)
        for name, count in distribution_map.items()
    ]


def build_and_test_pdf_rag_chain(chunks: list[Document]) -> None:
    """
    Building and testing a complete RAG chain over PDF-sourced chunks.

    The pipeline mirrors the text-file RAG from the previous script, but
    the prompt explicitly acknowledges that PDF-extracted context may
    contain formatting artifacts that the LLM should gracefully ignore.
    """
    print_section_header("5️⃣ CREATING RAG SYSTEM WITH PDF DOCUMENTS")
    print(textwrap.dedent(build_and_test_pdf_rag_chain.__doc__))

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = AzureChatOpenAI(model="gpt-5-nano")
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    pdf_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print_section_header("6️⃣ TESTING PDF-BASED RAG")

    test_questions = [
        "What was Ada Lovelace's contribution to programming?",
        "How did Einstein develop his theories of relativity?",
    ]

    for question_number, question in enumerate(test_questions, start=1):
        print(f"\nQ{question_number}: {question}\n"
              f"{'-' * 40}")
        answer = pdf_rag_chain.invoke(question)
        print(f"A{question_number}: {answer}")

    print(textwrap.dedent(f"""\

        💡 PDF RAG chain ready: pdf_rag_chain.invoke('Your question')
        📄 Processed {len(chunks)} chunks"""))


def print_pdf_best_practices() -> None:
    """
    Summary of best practices, trade-offs, and recommendations for
    choosing PDF loading in a RAG pipeline.
    """
    print_section_header("📋 PDF LOADING BEST PRACTICES")
    print(textwrap.dedent(print_pdf_best_practices.__doc__))

    for category in PDF_BEST_PRACTICES:
        print(f"{category.emoji} {category.heading}:")
        for point in category.bullet_points:
            print(f"  • {point}")
        print()


if __name__ == "__main__":
    load_dotenv(override=True)

    pdf_paths = create_pdfs_from_text_sources()
    all_pdf_pages = load_pdf_documents(pdf_file_paths=pdf_paths)
    compare_pdf_vs_text_loading(pdf_pages=all_pdf_pages)
    pdf_chunks = split_pdf_documents_into_chunks(pdf_pages=all_pdf_pages)
    build_and_test_pdf_rag_chain(chunks=pdf_chunks)
    print_pdf_best_practices()
