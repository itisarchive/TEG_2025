#!/usr/bin/env python3
"""
🔗 Multiple Language Models with LangChain — Unified Interface
==============================================================

This script demonstrates how to interact with different LLM providers
both through their native SDKs and through LangChain's unified abstraction.

🎯 What You'll Learn:
- How to call Azure OpenAI and Anthropic Claude via their native Python SDKs
- How LangChain provides a single interface for multiple providers
- How to use Ollama for local model inference via LangChain
- Why a unified abstraction matters for swapping models painlessly

🔧 Prerequisites:
- Azure OpenAI credentials in .env file (AZURE_OPENAI_API_KEY, etc.)
- ANTHROPIC_API_KEY in .env file
- For Ollama: install from https://ollama.com/ and run your preferred model
- Python 3.13+ with langchain-openai, langchain-anthropic, langchain-ollama
"""

import os

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlockParam
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI
from openai.lib.azure import AzureOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import SecretStr


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def demonstrate_azure_openai_native(azure_openai_client: AzureOpenAI) -> None:
    """
    Azure OpenAI — Native SDK
    The openai Python package talks directly to the Azure OpenAI endpoint.
    This is the lowest-level way to send a chat completion request.
    """
    print_section_header("AZURE OPENAI — NATIVE SDK")
    print(demonstrate_azure_openai_native.__doc__)

    sky_question = "Why is the sky blue?"
    azure_openai_response = azure_openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[ChatCompletionUserMessageParam(role="user", content=sky_question)],
    )
    print(azure_openai_response.choices[0].message.content)


def demonstrate_claude_native() -> None:
    """
    Anthropic Claude — Native SDK
    The anthropic Python package communicates directly with the Claude API.
    Note the structural differences: content is a list of typed blocks,
    and max_tokens is a required parameter.
    """
    print_section_header("ANTHROPIC CLAUDE — NATIVE SDK")
    print(demonstrate_claude_native.__doc__)

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    claude_native_client = Anthropic(api_key=anthropic_api_key)

    sky_question = "Why is the sky blue?"
    claude_native_response = claude_native_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        temperature=1,
        messages=[
            MessageParam(
                role="user",
                content=[TextBlockParam(type="text", text=sky_question)],
            ),
        ],
    )
    print(claude_native_response.content[0].text)


def demonstrate_azure_openai_via_langchain() -> None:
    """
    Azure OpenAI — via LangChain
    LangChain wraps Azure OpenAI behind a unified ChatModel interface.
    Calling .invoke(question) returns an AIMessage — same shape regardless of provider.
    """
    print_section_header("AZURE OPENAI — VIA LANGCHAIN")
    print(demonstrate_azure_openai_via_langchain.__doc__)

    azure_langchain_llm = AzureChatOpenAI(model="gpt-4o-mini")
    sky_question = "Why is the sky blue?"
    azure_langchain_response = azure_langchain_llm.invoke(sky_question)
    print(azure_langchain_response.content)


def demonstrate_claude_via_langchain() -> None:
    """
    Anthropic Claude — via LangChain
    The same .invoke() call works identically for Claude, making it trivial
    to swap providers without touching application logic.
    """
    print_section_header("ANTHROPIC CLAUDE — VIA LANGCHAIN")
    print(demonstrate_claude_via_langchain.__doc__)

    anthropic_api_key = SecretStr(os.environ["ANTHROPIC_API_KEY"])
    claude_langchain_llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        api_key=anthropic_api_key,
        timeout=30,
        stop=["\n\nHuman:", "\n\nAssistant:"],
    )
    sky_question = "Why is the sky blue?"
    claude_langchain_response = claude_langchain_llm.invoke(sky_question)
    print(claude_langchain_response.content)


def demonstrate_ollama_via_langchain() -> None:
    """
    Ollama (local models) — via LangChain
    Ollama lets you run open-source models locally. LangChain's OllamaLLM
    integrates them into the same prompt | model chain pipeline, so you can
    prototype with a local LLM and later swap in a cloud provider.
    """
    print_section_header("OLLAMA — VIA LANGCHAIN (PROMPT CHAIN)")
    print(demonstrate_ollama_via_langchain.__doc__)

    step_by_step_template = """Question: {question}
                               Answer: Let's think step by step."""

    step_by_step_prompt = ChatPromptTemplate.from_template(step_by_step_template)
    ollama_llama_model = OllamaLLM(model="llama3.1")
    langchain_chain = step_by_step_prompt | ollama_llama_model

    chain_output = langchain_chain.invoke({"question": "What is LangChain?"})
    print(chain_output)

    ollama_gemma_model = OllamaLLM(model="gemma3:270m")
    gemma_response = ollama_gemma_model.invoke("Why is the sky blue?")
    print(gemma_response)


if __name__ == "__main__":
    load_dotenv(override=True)

    azure_client = AzureOpenAI()

    demonstrate_azure_openai_native(azure_client)
    demonstrate_claude_native()
    demonstrate_azure_openai_via_langchain()
    demonstrate_claude_via_langchain()
    demonstrate_ollama_via_langchain()
