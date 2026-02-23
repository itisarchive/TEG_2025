# Large Language Models (LLMs) — Provider Integration and Response Analysis

This module focuses on understanding different LLM providers and analyzing their
API responses. After completing the introductory concepts in Module 1, we dive
deeper into the technical aspects of working with **Azure OpenAI** and
**Anthropic Claude**, and then show how **LangChain** provides a unified
interface across multiple providers.

## 🎯 Learning Objectives

By completing this module, you will:

- Understand the full anatomy of an Azure OpenAI `ChatCompletion` response object
- Understand the full anatomy of an Anthropic Claude `Message` response object
- Know how to track token usage for cost management and monitoring
- Compare structural differences between the two APIs
- Use LangChain to switch providers without changing application logic
- Run local models via Ollama through the same LangChain interface

## 📚 Module Content

### 1. Azure OpenAI Response Analysis (`1. OpenAI - analyze the response object.py`)

**🔍 Deep dive into Azure OpenAI API response structure**

Sends a chat completion request via the native `openai` SDK and unpacks every
layer of the returned `ChatCompletion` object:

- **Response anatomy** — id, model, choices, message, usage
- **Field-by-field exploration** — type, ID, model, choices array, message content
- **Detailed usage statistics** — prompt / completion / total tokens, reasoning & audio breakdowns

### 2. Anthropic Claude Response Analysis (`2. Claude - analyze the response object.py`)

**🔍 Anthropic Claude API response structure**

Same approach as above but for the Claude Messages API:

- **Response anatomy** — id, model, content blocks, usage
- **Field-by-field exploration** — type, content array, TextBlock, text
- **Detailed usage statistics** — input / output tokens (no built-in total)
- **Claude vs OpenAI comparison** — method names, content paths, system message placement, usage field names

### 3. Multi-Provider Integration (`3. Different models with LangChain.py`)

**🔗 Working with multiple LLM providers using LangChain**

Demonstrates five ways to call an LLM, from native SDKs to a unified abstraction:

- **Azure OpenAI — Native SDK** — direct `AzureOpenAI` chat completion call
- **Anthropic Claude — Native SDK** — direct `Anthropic` messages call
- **Azure OpenAI — via LangChain** — `AzureChatOpenAI.invoke()`
- **Anthropic Claude — via LangChain** — `ChatAnthropic.invoke()`
- **Ollama (local models) — via LangChain** — `OllamaLLM` with prompt chains (llama3.1, gemma3)

### 4. Notebook (`notebook_2.ipynb`)

The same material as the `.py` files above in an interactive Jupyter notebook
format with additional explanatory markdown cells.

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Completion of **Module 1 (Intro)** for foundational LLM concepts
- Azure OpenAI resource with deployed models
- Anthropic API key
- *(Optional)* Ollama installed locally for the local-model demo

### Setup

```bash
# 1. Navigate to the module
cd "src/2. Models/2. LLMs"

# 2. Install dependencies
uv sync

# 3. Create your .env from the template
cp .env.example .env
# Then fill in all four variables (see Environment Variables below)
```

### Running the Examples

```bash
# Analyze Azure OpenAI API response structure
uv run python "1. OpenAI - analyze the response object.py"

# Analyze Anthropic Claude API response structure
uv run python "2. Claude - analyze the response object.py"

# Compare multiple providers with LangChain
uv run python "3. Different models with LangChain.py"
```

### Recommended Learning Sequence

1. **Start with Azure OpenAI analysis** — understand the ChatCompletion object
2. **Explore Claude responses** — compare a different provider's response structure
3. **Try multi-provider integration** — see how LangChain unifies access

## 🛠️ Dependencies

Defined in `pyproject.toml`:

| Package               | Purpose                        | Min version |
|-----------------------|--------------------------------|-------------|
| `openai`              | Azure OpenAI SDK               | ≥ 2.21.0    |
| `anthropic`           | Claude API SDK                 | ≥ 0.83.0    |
| `langchain-openai`    | LangChain Azure OpenAI wrapper | ≥ 1.1.10    |
| `langchain-anthropic` | LangChain Anthropic wrapper    | ≥ 1.3.3     |
| `langchain-ollama`    | LangChain Ollama wrapper       | ≥ 1.0.1     |
| `python-dotenv`       | `.env` file loading            | ≥ 1.2.1     |

## 🔐 Environment Variables

All scripts rely on variables loaded from a `.env` file (see `.env.example`):

| Variable                | Description                            |
|-------------------------|----------------------------------------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource URL         |
| `AZURE_OPENAI_API_KEY`  | API key for Azure OpenAI               |
| `OPENAI_API_VERSION`    | API version, e.g. `2025-04-01-preview` |
| `ANTHROPIC_API_KEY`     | API key for Anthropic Claude           |

> ⚠️ Each model name used in code (e.g. `gpt-5-nano`, `gpt-4o-mini`) must have
> a matching **deployment** in Azure AI Foundry.

## 🎓 Learning Path

1. **Read this README** — understand the module goals
2. **Set up your environment** — `.env` and dependencies
3. **Run script 1** — explore the Azure OpenAI response object
4. **Run script 2** — explore the Claude response object, compare with OpenAI
5. **Run script 3** — see native vs LangChain calls, try local models
6. **Experiment** — swap models, compare token usage, adjust parameters

## 💡 Tips

- Never commit `.env` to version control — it contains your API keys.
- Start with cheaper models (`gpt-4o-mini`, `gpt-4.1-nano`) during experimentation.
- Ollama demos require a running local model (`ollama run llama3.1`).
- Compare response shapes between providers to understand what metadata each offers.
- Token counts differ between providers — always verify billing documentation.

## 🚀 Next Steps

After mastering this module, continue with:

1. **Module 3 (RAG)** — Retrieval Augmented Generation for knowledge-based AI
2. **Module 4 (Graphs)** — workflow automation with LangGraph
3. **Module 5 (Tools and Agents)** — autonomous AI agents
4. **Module 6 (MCP)** — Model Context Protocol for advanced integrations
