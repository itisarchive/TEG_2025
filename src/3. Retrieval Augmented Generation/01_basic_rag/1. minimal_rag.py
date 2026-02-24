#!/usr/bin/env python3
"""
🚀 Minimal RAG (Retrieval Augmented Generation) - Interactive Educational Journey
=================================================================================

Welcome to your first hands-on exploration of the RAG pattern! This script walks you
through the simplest possible RAG pipeline using an in-memory vector store — no external
databases, no chunking, just the pure essence of: Load → Embed → Store → Retrieve → Generate.

🎯 What You'll Learn:
- How to load documents and turn them into embeddings
- How an in-memory vector store enables semantic similarity search
- How a retriever feeds relevant context into an LLM prompt
- How to compose a LangChain RAG chain end-to-end
- How RAG handles questions whose answers lie outside the loaded knowledge base

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with langchain, langchain-openai, and python-dotenv packages
"""

import textwrap

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

SCIENTISTS_BIOS_DIR = "src/3. Retrieval Augmented Generation/01_basic_rag/data/scientists_bios"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHAT_MODEL_NAME = "gpt-5-nano"

RAG_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""

SAMPLE_QUESTIONS = [
    "What was Ada Lovelace's contribution to computer programming?",
    "How did Einstein develop his theory of relativity?",
    "What was Newton's most important discovery?",
    "How come Karol Nawrocki became the president of Poland?",
]


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def load_scientist_biographies(*, directory_path: str) -> list[Document]:
    """
    STEP 1 of the RAG pipeline: Document Loading.

    We use LangChain's DirectoryLoader to read every file from a directory
    and wrap each one in a Document object (text + metadata).
    """
    print_section_header("STEP 1: LOADING DOCUMENTS")
    print(textwrap.dedent(load_scientist_biographies.__doc__))

    directory_loader = DirectoryLoader(directory_path)
    loaded_documents = directory_loader.load()
    print(f"Loaded {len(loaded_documents)} documents from '{directory_path}'")
    return loaded_documents


def build_in_memory_vector_store(*, documents: list[Document], embedding_model_name: str) -> InMemoryVectorStore:
    """
    STEPS 2-3 of the RAG pipeline: Embedding & Storing.

    Each document is converted into a numeric vector (embedding) that captures
    its semantic meaning. These embeddings are stored in an InMemoryVectorStore
    so we can later find documents similar to a user's question.
    """
    print_section_header("STEP 2-3: EMBEDDING DOCUMENTS & BUILDING VECTOR STORE")
    print(textwrap.dedent(build_in_memory_vector_store.__doc__))

    azure_embeddings = AzureOpenAIEmbeddings(model=embedding_model_name)
    vector_store = InMemoryVectorStore(embedding=azure_embeddings)
    vector_store.add_documents(documents=documents)
    print(f"Embedded and stored {len(documents)} documents using '{embedding_model_name}'")
    return vector_store


def create_rag_chain(*, vector_store: InMemoryVectorStore, chat_model_name: str) -> RunnableSerializable:
    """
    STEPS 4-6 of the RAG pipeline: Retriever → Prompt → LLM → Output.

    The chain works as follows:
    1. The retriever searches the vector store for documents relevant to the question
    2. The prompt template combines the retrieved context with the user's question
    3. The LLM generates an answer grounded in the provided context
    4. The StrOutputParser extracts the plain-text response

    This composition is the heart of any RAG system.
    """
    print_section_header("STEP 4-6: ASSEMBLING THE RAG CHAIN")
    print(textwrap.dedent(create_rag_chain.__doc__))

    retriever = vector_store.as_retriever()
    chat_llm = AzureChatOpenAI(model=chat_model_name)
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | chat_llm
            | StrOutputParser()
    )
    print(f"RAG chain assembled with model '{chat_model_name}'")
    return rag_chain


def ask_single_question(*, rag_chain: RunnableSerializable, question: str) -> None:
    """
    Demonstrates a single question-answer interaction with the RAG chain.
    This is the simplest way to use the pipeline you've just built.
    """
    print_section_header("SINGLE QUESTION DEMO")
    print(textwrap.dedent(ask_single_question.__doc__))

    answer = rag_chain.invoke(question)
    print(f"Q: {question}\n"
          f"A: {answer}")


def answer_multiple_questions(*, rag_chain: RunnableSerializable, questions: list[str]) -> None:
    """
    Runs a batch of questions through the RAG chain to show how it handles
    different topics — including questions whose answers are NOT in the knowledge base.

    Notice the last question: the loaded biographies contain no information about
    Polish politics, so the model should honestly say it doesn't know.
    This illustrates RAG's built-in grounding: the LLM is instructed to rely
    only on the retrieved context and admit ignorance when context is insufficient.
    """
    print_section_header("BATCH QUESTION-ANSWERING")
    print(textwrap.dedent(answer_multiple_questions.__doc__))

    for question_number, current_question in enumerate(questions, start=1):
        print(f"\nQ{question_number}: {current_question}\n"
              f"{'-' * 40}")
        answer = rag_chain.invoke(current_question)
        print(f"A{question_number}: {answer}")


if __name__ == "__main__":
    load_dotenv(override=True)

    scientist_documents = load_scientist_biographies(directory_path=SCIENTISTS_BIOS_DIR)

    scientists_vector_store = build_in_memory_vector_store(
        documents=scientist_documents,
        embedding_model_name=EMBEDDING_MODEL_NAME,
    )

    minimal_rag_chain = create_rag_chain(
        vector_store=scientists_vector_store,
        chat_model_name=CHAT_MODEL_NAME,
    )

    ask_single_question(
        rag_chain=minimal_rag_chain,
        question="What was Ada Lovelace's contribution to computer programming?",
    )

    answer_multiple_questions(
        rag_chain=minimal_rag_chain,
        questions=SAMPLE_QUESTIONS,
    )
