# GraphRAG vs Naive RAG: CV Knowledge Graph Comparison

A comprehensive demonstration of **GraphRAG vs Naive RAG** using realistic PDF CVs and LLM-powered knowledge graph
extraction. This project showcases how knowledge graphs enable structured queries that are impossible with traditional
vector-based RAG systems.

## 🚀 Quick Start

### Prerequisites

- **Python 3.13** with `uv` package manager
- **Docker Desktop** (for Neo4j database)
- **Azure OpenAI** credentials (set in `.env` file — see `.env.example`)

### One-Command Demo

```bash
uv run python 5_compare_systems.py
```

### Step-by-Step Workflow

```bash
uv run python 0_setup.py
./start_session.sh
uv run python 1_generate_data.py
uv run python 2_data_to_knowledge_graph.py
uv run python 5_compare_systems.py
```

## 🎯 Problem Addressed

Traditional RAG systems struggle with structured queries requiring:

| Query Type      | Example                                  | Traditional RAG Issue              |
|-----------------|------------------------------------------|------------------------------------|
| **Counting**    | "How many Python developers?"            | ❌ Estimates from text chunks       |
| **Filtering**   | "Find people with Docker AND Kubernetes" | ❌ Limited to semantic similarity   |
| **Aggregation** | "Average years of experience?"           | ❌ Cannot calculate across entities |
| **Sorting**     | "Top 3 most experienced developers"      | ❌ No structured ranking            |
| **Multi-hop**   | "People who attended same university"    | ❌ Cannot traverse relationships    |

## 🏗️ Architecture

### Knowledge Graph Schema

Auto-extracted from PDF CVs using LLMGraphTransformer:

```
Nodes:
├── Person (id, name, location, bio)
├── Skill (id, category)
├── Company (id, industry, location)
├── University (id, location, type)
└── Certification (id, provider, field)

Relationships:
├── (Person)-[HAS_SKILL]->(Skill)
├── (Person)-[WORKED_AT]->(Company)
├── (Person)-[STUDIED_AT]->(University)
├── (Person)-[EARNED]->(Certification)
└── (Person)-[MENTIONS]->(Person)
```

### System Components

- **PDF Processing**: Realistic CV generation with ReportLab
- **Knowledge Extraction**: LangChain LLMGraphTransformer (Azure OpenAI gpt-4.1-mini)
- **Graph Database**: Neo4j with Docker
- **GraphRAG**: LangChain GraphCypherQAChain with custom Cypher prompts (gpt-4.1)
- **Naive RAG**: ChromaDB vector search baseline (gpt-4.1)
- **Evaluation**: Azure OpenAI gpt-4.1 ground truth generation

## 📊 Example Results

### Query: "How many people have Python programming skills?"

**GraphRAG (✅ Accurate):**

```cypher
MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
WHERE toLower(s.id) = toLower("Python")
RETURN count(p) AS pythonProgrammers
```

*Result: **7 people** (exact count)*

**Naive RAG (❌ Incomplete):**

*Result: "Based on context, only **Amanda Smith** is mentioned" (missed 6 people)*

### Query: "List people with both React and Node.js skills"

**GraphRAG (✅ Complete):**

*Result: **4 people** — Christine Rodriguez, Joseph Fuller, Krystal Castillo, William Bonilla*

**Naive RAG (❌ Limited):**

*Result: **1 person** — Christine Rodriguez (missed 3 people)*

## 📁 Project Structure

```
06_GraphRAG/
├── 0_setup.py                    Environment validation and Neo4j initialization
├── 1_generate_data.py            Synthetic PDF CV generation (gpt-4.1-mini)
├── 2_data_to_knowledge_graph.py  LLM graph extraction from CVs to Neo4j
├── 3_query_knowledge_graph.py    GraphRAG implementation (Cypher queries)
├── 4_naive_rag_cv.py             Naive RAG baseline (ChromaDB)
├── 5_compare_systems.py          Automated system comparison with evaluation
├── .env.example                  Azure OpenAI + Neo4j credential template
├── compose.yml                   Neo4j minimal setup
├── docker-compose.yml            Neo4j full setup (APOC, memory, healthcheck)
├── start_session.sh              Start Neo4j and verify readiness
├── end_session.sh                Stop Neo4j cleanly
├── utils/
│   ├── config.toml               Generation and output configuration
│   ├── generate_ground_truth.py  Ground truth answer generation (gpt-4.1)
│   └── test_questions.json       42 evaluation questions across 7 categories
├── data/
│   ├── programmers/              Generated CV PDFs + profiles JSON
│   ├── projects/                 Generated project records JSON
│   └── RFP/                      Generated RFP PDFs + JSON
└── results/
    ├── ground_truth_answers.json
    └── comparison_table.md
```

## 🔧 Technical Stack

| Component           | Technology                           |
|---------------------|--------------------------------------|
| Language            | Python 3.13                          |
| Package Manager     | uv                                   |
| LLM Provider        | Azure OpenAI (gpt-4.1, gpt-4.1-mini) |
| Graph Database      | Neo4j 5.x (Docker)                   |
| Vector Store        | ChromaDB (baseline comparison)       |
| Frameworks          | LangChain, LangChain Experimental    |
| Document Processing | Unstructured, ReportLab, PyPDF       |
| Embeddings          | Azure OpenAI text-embedding-3-small  |

## 🎓 Key Learnings

1. **GraphRAG excels** at structured queries requiring precise relationships
2. **LLMGraphTransformer** enables real-world PDF-to-knowledge-graph workflows
3. **Custom Cypher prompts** solve case sensitivity and result interpretation issues
4. **Ground truth evaluation** provides unbiased comparison between systems
5. **Hybrid approaches** can combine both strengths for optimal results

## 🔍 Advanced Usage

### Browse Knowledge Graph

Neo4j Browser: http://localhost:7474 (neo4j / password123)

### Individual Components

```bash
uv run python 3_query_knowledge_graph.py
uv run python 4_naive_rag_cv.py
uv run python utils/generate_ground_truth.py
```

## 🤝 Real-World Applications

This approach applies to any domain with:

- **Structured relationships** between entities
- **Precise counting/filtering** requirements
- **Multi-hop reasoning** needs
- **Complex business queries**

Examples: staffing, inventory management, medical records, financial risk analysis.
