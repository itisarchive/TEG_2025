"""
⚙️ Configuration for the Multi-RAG Comparative Evaluation System
================================================================

All tuneable parameters — model names, chunking strategy, retrieval settings,
and evaluation questions — are collected in frozen dataclasses so that every
pipeline stage receives an immutable, self-documenting configuration object.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    expert_model: str = "gpt-5"
    evaluator_model: str = "gpt-4.1"
    rag_chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 3
    bm25_weight: float = 0.3
    rerank_initial_top_k: int = 6
    rerank_final_top_k: int = 3
    max_query_variations: int = 3


@dataclass(frozen=True)
class EvaluationQuestions:
    questions: tuple[str, ...] = (
        "What did Marie Curie win Nobel Prizes for?",
        "What is Einstein's theory of relativity about?",
        "What are Newton's three laws of motion?",
        "What did Charles Darwin discover?",
        "What was Ada Lovelace's contribution to computing?",
    )


@dataclass(frozen=True)
class PipelineSettings:
    models: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation_questions: EvaluationQuestions = field(default_factory=EvaluationQuestions)


SETTINGS = PipelineSettings()
