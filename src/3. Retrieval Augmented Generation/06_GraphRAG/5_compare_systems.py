#!/usr/bin/env python3
"""
GraphRAG vs Naive RAG — Automated Comparison System
=====================================================

Runs the same set of questions through both a graph-based RAG (GraphRAG) and
a traditional vector-similarity RAG (Naive RAG), then evaluates every answer
against pre-generated ground truth to produce a structured comparison report.

🎯 What You'll Learn:
- How to benchmark two fundamentally different retrieval strategies side by side
- Why GraphRAG excels at structured, multi-hop, and aggregation queries
- How automated answer-quality evaluation works (exact match, numerical accuracy,
  token overlap, completeness scoring)
- How to generate ground truth data and persist comparison results as JSON / Markdown

🔧 Prerequisites:
    - Azure OpenAI credentials in environment / .env
    - Generated CV PDFs (see 1_generate_data.py)
    - A populated Neo4j knowledge graph (see 2_data_to_knowledge_graph.py)
    - A Naive RAG vector store (see 4_naive_rag_cv.py)
"""

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import textwrap
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol, TypedDict

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


class GroundTruthItem(TypedDict):
    question: str
    category: str
    ground_truth_answer: str


class GroundTruthPayload(TypedDict):
    ground_truth_answers: list[GroundTruthItem]


class AnswerQualityEvaluation(TypedDict):
    exact_match: bool
    contains_key_info: bool
    numerical_accuracy: bool | None
    completeness_score: float
    quality_score: float


class GraphRagAnswer(TypedDict, total=False):
    question: str
    answer: str
    cypher_query: str
    execution_time: float
    success: bool
    system: str
    error: str


class NaiveRagAnswer(TypedDict, total=False):
    question: str
    answer: str
    execution_time: float
    num_chunks_retrieved: int
    success: bool
    system: str
    error: str


class ComparisonEntry(TypedDict):
    question_index: int
    question: str
    category: str
    ground_truth: str
    graphrag: dict[str, Any]
    naive_rag: dict[str, Any]


class SummaryPayload(TypedDict):
    overall_performance: dict[str, float | int]
    quality_scores: dict[str, float]
    category_breakdown: dict[str, dict[str, int]]


class ComparisonPayload(TypedDict):
    metadata: dict[str, Any]
    results: list[ComparisonEntry]
    summary: SummaryPayload


class GraphRagSystem(Protocol):
    def query_graph(self, question: str) -> dict[str, Any]: ...


class NaiveRagSystem(Protocol):
    def query(self, question: str) -> dict[str, Any]: ...

    def initialize_system(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ComparatorPaths:
    results_directory: Path
    config_file_path: Path
    graph_rag_script: Path
    naive_rag_script: Path
    ground_truth_json: Path
    ground_truth_generator_script: Path


@dataclass(frozen=True, slots=True)
class ComparatorConfig:
    cv_pdf_directory: Path
    api_delay_seconds: float


class SystemComparator:
    """
    Compare GraphRAG and Naive RAG results against ground truth answers.

    Loads both systems dynamically, runs each ground-truth question through both,
    evaluates answer quality with category-aware heuristics, and generates
    JSON + Markdown reports.
    """

    RESULTS_DIR: Final[Path] = Path("results")
    GRAPH_RAG_SCRIPT: Final[Path] = Path("3_query_knowledge_graph.py")
    NAIVE_RAG_SCRIPT: Final[Path] = Path("4_naive_rag_cv.py")
    GROUND_TRUTH_FILENAME: Final[str] = "ground_truth_answers.json"
    GROUND_TRUTH_GENERATOR_SCRIPT: Final[Path] = Path("utils/generate_ground_truth.py")
    API_DELAY_SECONDS: Final[float] = 0.5

    def __init__(self, config_path: str | Path = "utils/config.toml") -> None:
        self._paths = self._build_paths(Path(config_path))
        self._project_config = self._load_project_config(self._paths.config_file_path)
        self._comparator_config = self._parse_comparator_config(self._project_config)

        self._paths.results_directory.mkdir(parents=True, exist_ok=True)

        self._graph_rag_system: GraphRagSystem | None = None
        self._naive_rag_system: NaiveRagSystem | None = None

        LOGGER.info("SystemComparator initialized")

    def load_ground_truth(self) -> GroundTruthPayload:
        """Load ground truth answers, generating them if the JSON file is missing."""
        if not self._paths.ground_truth_json.exists():
            self._generate_ground_truth()

        payload = self._read_json(self._paths.ground_truth_json)
        if not isinstance(payload, dict) or "ground_truth_answers" not in payload:
            raise ValueError("Ground truth JSON has an unexpected format")

        answer_items = payload["ground_truth_answers"]
        if not isinstance(answer_items, list):
            raise ValueError("ground_truth_answers must be a list")

        return GroundTruthPayload(ground_truth_answers=answer_items)

    def initialize_graph_rag_system(self) -> None:
        """Initialize the GraphRAG system via dynamic import."""
        loaded_module = self._import_from_path(
            self._paths.graph_rag_script, module_name="graph_rag_module",
        )
        system_cls = self._load_first_symbol(
            loaded_module, ["CVGraphRAGSystem", "GraphRAGSystem", "GraphRagSystem"],
        )
        self._graph_rag_system = system_cls()

        LOGGER.info("GraphRAG system initialized")

    def initialize_naive_rag_system(self) -> None:
        """Initialize the Naive RAG system via dynamic import."""
        loaded_module = self._import_from_path(
            self._paths.naive_rag_script, module_name="naive_rag_module",
        )
        system_cls = self._load_first_symbol(
            loaded_module, ["NaiveCvRag", "NaiveRAGSystem", "NaiveRagSystem"],
        )
        naive_rag_instance = system_cls()

        if hasattr(naive_rag_instance, "initialize"):
            naive_rag_instance.initialize()
            self._naive_rag_system = naive_rag_instance
            LOGGER.info("Naive RAG system initialized")
            return

        if hasattr(naive_rag_instance, "initialize_system") and callable(
                getattr(naive_rag_instance, "initialize_system"),
        ):
            initialization_ok = naive_rag_instance.initialize_system()
            if not initialization_ok:
                raise RuntimeError("Naive RAG system failed to initialize")
            self._naive_rag_system = naive_rag_instance
            LOGGER.info("Naive RAG system initialized")
            return

        raise AttributeError(
            "Naive RAG system exposes neither initialize() nor initialize_system()"
        )

    def has_cv_data(self) -> bool:
        """Return whether the configured CV directory contains PDF files."""
        cv_directory = self._comparator_config.cv_pdf_directory
        return cv_directory.exists() and any(cv_directory.glob("*.pdf"))

    async def run_full_comparison(self) -> ComparisonPayload:
        """Run all ground-truth questions through both systems and compute summary statistics."""
        ground_truth = self.load_ground_truth()
        ground_truth_items = ground_truth["ground_truth_answers"]

        self.initialize_graph_rag_system()
        self.initialize_naive_rag_system()

        comparison_entries: list[ComparisonEntry] = []
        for entry_index, truth_item in enumerate(ground_truth_items, start=1):
            current_question = truth_item["question"]
            current_category = truth_item["category"]
            expected_answer = truth_item["ground_truth_answer"]

            graph_rag_answer = self.run_graph_rag_query(current_question)
            naive_rag_answer = self.run_naive_rag_query(current_question)

            graph_quality = self.evaluate_answer_quality(
                expected_answer, graph_rag_answer.get("answer", ""), current_category,
            )
            naive_quality = self.evaluate_answer_quality(
                expected_answer, naive_rag_answer.get("answer", ""), current_category,
            )

            comparison_entries.append(
                ComparisonEntry(
                    question_index=entry_index,
                    question=current_question,
                    category=current_category,
                    ground_truth=expected_answer,
                    graphrag={
                        "answer": graph_rag_answer.get("answer", "No answer"),
                        "cypher_query": graph_rag_answer.get("cypher_query", ""),
                        "execution_time": float(graph_rag_answer.get("execution_time", 0.0)),
                        "success": bool(graph_rag_answer.get("success", False)),
                        "evaluation": graph_quality,
                    },
                    naive_rag={
                        "answer": naive_rag_answer.get("answer", "No answer"),
                        "chunks_retrieved": int(naive_rag_answer.get("num_chunks_retrieved", 0)),
                        "execution_time": float(naive_rag_answer.get("execution_time", 0.0)),
                        "success": bool(naive_rag_answer.get("success", False)),
                        "evaluation": naive_quality,
                    },
                )
            )

            inter_query_delay = self._comparator_config.api_delay_seconds
            if inter_query_delay > 0:
                await asyncio.sleep(inter_query_delay)

        return ComparisonPayload(
            metadata={
                "comparison_date": datetime.now().isoformat(),
                "total_questions": len(comparison_entries),
                "ground_truth_source": "GPT-5",
                "systems_compared": ["GraphRAG", "Naive RAG"],
            },
            results=comparison_entries,
            summary=self.generate_summary(comparison_entries),
        )

    def run_graph_rag_query(self, question: str) -> GraphRagAnswer:
        """Run a single question through the GraphRAG system."""
        graph_system = self._require_graph_rag()
        start_time = time.perf_counter()
        try:
            raw_result = graph_system.query_graph(question)
            elapsed_seconds = time.perf_counter() - start_time
            return GraphRagAnswer(
                question=question,
                answer=str(raw_result.get("answer", "No answer")),
                cypher_query=str(raw_result.get("cypher_query", "")),
                execution_time=elapsed_seconds,
                success=bool(raw_result.get("success", False)),
                system="graphrag",
            )
        except (ValueError, RuntimeError, KeyError) as graph_error:
            elapsed_seconds = time.perf_counter() - start_time
            LOGGER.exception("GraphRAG query failed")
            return GraphRagAnswer(
                question=question,
                answer=f"Error: {graph_error}",
                cypher_query="",
                execution_time=elapsed_seconds,
                success=False,
                system="graphrag",
                error=str(graph_error),
            )
        except Exception as unexpected_error:
            elapsed_seconds = time.perf_counter() - start_time
            LOGGER.exception("GraphRAG query failed with unexpected error")
            return GraphRagAnswer(
                question=question,
                answer=f"Error: {unexpected_error}",
                cypher_query="",
                execution_time=elapsed_seconds,
                success=False,
                system="graphrag",
                error=str(unexpected_error),
            )

    def run_naive_rag_query(self, question: str) -> NaiveRagAnswer:
        """Run a single question through the Naive RAG system."""
        naive_system = self._require_naive_rag()
        start_time = time.perf_counter()
        try:
            raw_result = naive_system.query(question)
            elapsed_seconds = time.perf_counter() - start_time
            return NaiveRagAnswer(
                question=question,
                answer=str(raw_result.get("answer", "No answer")),
                execution_time=float(raw_result.get("execution_time", elapsed_seconds)),
                num_chunks_retrieved=int(raw_result.get("num_chunks_retrieved", 0)),
                success=bool(raw_result.get("success", False)),
                system="naive_rag",
            )
        except (ValueError, RuntimeError, KeyError) as naive_error:
            elapsed_seconds = time.perf_counter() - start_time
            LOGGER.exception("Naive RAG query failed")
            return NaiveRagAnswer(
                question=question,
                answer=f"Error: {naive_error}",
                execution_time=elapsed_seconds,
                num_chunks_retrieved=0,
                success=False,
                system="naive_rag",
                error=str(naive_error),
            )
        except Exception as unexpected_error:
            elapsed_seconds = time.perf_counter() - start_time
            LOGGER.exception("Naive RAG query failed with unexpected error")
            return NaiveRagAnswer(
                question=question,
                answer=f"Error: {unexpected_error}",
                execution_time=elapsed_seconds,
                num_chunks_retrieved=0,
                success=False,
                system="naive_rag",
                error=str(unexpected_error),
            )

    def evaluate_answer_quality(
            self,
            ground_truth_text: str,
            system_answer_text: str,
            question_category: str,
    ) -> AnswerQualityEvaluation:
        """Evaluate how well a system answer matches the ground truth using category-aware heuristics."""
        quality_evaluation: AnswerQualityEvaluation = {
            "exact_match": ground_truth_text.strip().casefold() == system_answer_text.strip().casefold(),
            "contains_key_info": False,
            "numerical_accuracy": None,
            "completeness_score": 0.0,
            "quality_score": 0.0,
        }

        if question_category == "counting":
            ground_truth_number = self._extract_first_int(ground_truth_text)
            system_answer_number = self._extract_first_int(system_answer_text)
            if ground_truth_number is not None and system_answer_number is not None:
                quality_evaluation["numerical_accuracy"] = ground_truth_number == system_answer_number
                quality_evaluation["contains_key_info"] = True
                quality_evaluation["quality_score"] = 1.0 if ground_truth_number == system_answer_number else 0.0
            return quality_evaluation

        if question_category in {"filtering", "listing"}:
            ground_truth_names = self._extract_capitalized_tokens(ground_truth_text)
            system_answer_names = self._extract_capitalized_tokens(system_answer_text)
            if ground_truth_names:
                name_overlap_ratio = len(ground_truth_names & system_answer_names) / len(ground_truth_names)
                quality_evaluation["completeness_score"] = name_overlap_ratio
                quality_evaluation["quality_score"] = name_overlap_ratio
                quality_evaluation["contains_key_info"] = name_overlap_ratio > 0
            return quality_evaluation

        if question_category == "aggregation":
            ground_truth_value = self._extract_first_float(ground_truth_text)
            system_answer_value = self._extract_first_float(system_answer_text)
            if ground_truth_value is not None and system_answer_value is not None:
                relative_difference = abs(ground_truth_value - system_answer_value) / max(abs(ground_truth_value), 1.0)
                quality_evaluation["numerical_accuracy"] = relative_difference < 0.1
                quality_evaluation["quality_score"] = max(0.0, 1.0 - relative_difference)
                quality_evaluation["contains_key_info"] = True
            return quality_evaluation

        ground_truth_words = self._normalize_words(ground_truth_text)
        system_answer_words = self._normalize_words(system_answer_text)
        if ground_truth_words:
            word_overlap_ratio = len(ground_truth_words & system_answer_words) / len(ground_truth_words)
            quality_evaluation["completeness_score"] = word_overlap_ratio
            quality_evaluation["quality_score"] = word_overlap_ratio
            quality_evaluation["contains_key_info"] = word_overlap_ratio > 0.3
        return quality_evaluation

    def generate_summary(self, comparison_entries: list[ComparisonEntry]) -> SummaryPayload:
        """Compute overall and per-category summary metrics."""
        graphrag_win_count = 0
        naive_rag_win_count = 0
        tie_count = 0

        all_graph_scores: list[float] = []
        all_naive_scores: list[float] = []

        per_category_stats: dict[str, dict[str, int]] = {}

        for entry in comparison_entries:
            entry_category = entry["category"]
            graph_quality_score = float(entry["graphrag"]["evaluation"]["quality_score"])
            naive_quality_score = float(entry["naive_rag"]["evaluation"]["quality_score"])

            all_graph_scores.append(graph_quality_score)
            all_naive_scores.append(naive_quality_score)

            category_counters = per_category_stats.setdefault(
                entry_category,
                {"total": 0, "graph_wins": 0, "naive_wins": 0, "ties": 0},
            )
            category_counters["total"] += 1

            if graph_quality_score > naive_quality_score:
                graphrag_win_count += 1
                category_counters["graph_wins"] += 1
            elif naive_quality_score > graph_quality_score:
                naive_rag_win_count += 1
                category_counters["naive_wins"] += 1
            else:
                tie_count += 1
                category_counters["ties"] += 1

        mean_graph_score = self._mean(all_graph_scores)
        mean_naive_score = self._mean(all_naive_scores)
        median_graph_score = self._median(all_graph_scores)
        median_naive_score = self._median(all_naive_scores)

        total_entries = max(len(comparison_entries), 1)
        return SummaryPayload(
            overall_performance={
                "graphrag_wins": graphrag_win_count,
                "naive_rag_wins": naive_rag_win_count,
                "ties": tie_count,
                "graphrag_win_rate": graphrag_win_count / total_entries,
                "naive_rag_win_rate": naive_rag_win_count / total_entries,
            },
            quality_scores={
                "graphrag_avg": mean_graph_score,
                "naive_rag_avg": mean_naive_score,
                "graphrag_median": median_graph_score,
                "naive_rag_median": median_naive_score,
            },
            category_breakdown=per_category_stats,
        )

    def save_comparison_results(self, comparison_data: ComparisonPayload) -> Path:
        """Save full comparison JSON to the results directory."""
        output_file = self._paths.results_directory / "system_comparison_results.json"
        output_file.write_text(
            json.dumps(comparison_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Comparison results saved to %s", output_file)
        return output_file

    def save_comparison_table(self, comparison_data: ComparisonPayload) -> Path:
        """Save a markdown comparison table to the results directory."""
        table_file = self._paths.results_directory / "comparison_table.md"
        table_file.write_text(
            self.generate_comparison_table(comparison_data),
            encoding="utf-8",
        )
        LOGGER.info("Comparison table saved to %s", table_file)
        return table_file

    def generate_comparison_table(self, comparison_data: ComparisonPayload) -> str:
        """Generate a readable markdown comparison table."""
        table_entries = comparison_data["results"]
        summary = comparison_data["summary"]

        overall = summary["overall_performance"]
        quality = summary["quality_scores"]

        lines: list[str] = [
            "# GraphRAG vs Naive RAG Comparison Results",
            "",
            "## Summary",
            f"- **GraphRAG Wins**: {int(overall['graphrag_wins'])}",
            f"- **Naive RAG Wins**: {int(overall['naive_rag_wins'])}",
            f"- **Ties**: {int(overall['ties'])}",
            f"- **GraphRAG Win Rate**: {float(overall['graphrag_win_rate']):.1%}",
            "",
            f"- **GraphRAG Avg Quality**: {float(quality['graphrag_avg']):.2f}",
            f"- **Naive RAG Avg Quality**: {float(quality['naive_rag_avg']):.2f}",
            "",
            "## Performance by Category",
            "",
        ]

        for category_name, category_stats in summary["category_breakdown"].items():
            category_total = max(category_stats["total"], 1)
            category_win_rate = category_stats["graph_wins"] / category_total
            lines.append(f"### {category_name.title()}")
            lines.append(
                f"- GraphRAG: {category_stats['graph_wins']}/{category_stats['total']}"
                f" ({category_win_rate:.1%})"
            )
            lines.append("")

        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| # | Question | Category | GraphRAG Answer | Naive RAG Answer | Ground Truth | Winner |")
        lines.append("|---|----------|----------|-----------------|------------------|--------------|--------|")

        for entry in table_entries:
            entry_index = entry["question_index"]
            truncated_question = self._truncate(entry["question"], max_length=50)
            entry_category = entry["category"]

            truncated_graph_answer = self._truncate(str(entry["graphrag"]["answer"]), max_length=30)
            truncated_naive_answer = self._truncate(str(entry["naive_rag"]["answer"]), max_length=30)
            truncated_ground_truth = self._truncate(entry["ground_truth"], max_length=30)

            graph_quality_score = float(entry["graphrag"]["evaluation"]["quality_score"])
            naive_quality_score = float(entry["naive_rag"]["evaluation"]["quality_score"])

            if graph_quality_score > naive_quality_score:
                winner_label = "GraphRAG ✅"
            elif naive_quality_score > graph_quality_score:
                winner_label = "Naive RAG ✅"
            else:
                winner_label = "Tie ⚖️"

            truncated_question = truncated_question.replace("|", "\\|")
            truncated_graph_answer = truncated_graph_answer.replace("|", "\\|")
            truncated_naive_answer = truncated_naive_answer.replace("|", "\\|")
            truncated_ground_truth = truncated_ground_truth.replace("|", "\\|")

            lines.append(
                f"| {entry_index} | {truncated_question} | {entry_category}"
                f" | {truncated_graph_answer} | {truncated_naive_answer}"
                f" | {truncated_ground_truth} | {winner_label} |"
            )

        return "\n".join(lines)

    @staticmethod
    def display_results(comparison_data: ComparisonPayload) -> None:
        """Print a compact summary to stdout."""
        summary = comparison_data["summary"]
        overall = summary["overall_performance"]
        quality_scores = summary["quality_scores"]

        print(textwrap.dedent(f"""\

            {'=' * 80}
            GRAPHRAG vs NAIVE RAG COMPARISON RESULTS
            {'=' * 80}

            📊 OVERALL PERFORMANCE:
            GraphRAG Wins: {int(overall['graphrag_wins'])}
            Naive RAG Wins: {int(overall['naive_rag_wins'])}
            Ties: {int(overall['ties'])}
            GraphRAG Win Rate: {float(overall['graphrag_win_rate']):.1%}

            🎯 QUALITY SCORES:
            GraphRAG Average: {float(quality_scores['graphrag_avg']):.2f}
            Naive RAG Average: {float(quality_scores['naive_rag_avg']):.2f}
        """))

        print("📋 PERFORMANCE BY CATEGORY:")
        for category_name, category_stats in summary["category_breakdown"].items():
            category_total = max(category_stats["total"], 1)
            category_win_rate = category_stats["graph_wins"] / category_total
            print(
                f"{category_name.title()}: GraphRAG"
                f" {category_stats['graph_wins']}/{category_stats['total']}"
                f" ({category_win_rate:.1%})"
            )

    def _generate_ground_truth(self) -> None:
        """Invoke the ground-truth generator script via subprocess to create the reference answers JSON."""
        generator_script = self._paths.ground_truth_generator_script
        if not generator_script.exists():
            raise FileNotFoundError(f"Ground truth generator not found: {generator_script}")

        LOGGER.info("Ground truth file not found, generating via %s", generator_script)
        subprocess_result = subprocess.run(
            ["uv", "run", "python", str(generator_script)],
            capture_output=True,
            text=True,
        )
        if subprocess_result.returncode != 0:
            raise RuntimeError(
                "Failed to generate ground truth.\n"
                f"STDOUT:\n{subprocess_result.stdout}\n\n"
                f"STDERR:\n{subprocess_result.stderr}"
            )
        if not self._paths.ground_truth_json.exists():
            raise FileNotFoundError(
                f"Ground truth generation completed but file not found: {self._paths.ground_truth_json}"
            )

        LOGGER.info("Ground truth generated successfully")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _import_from_path(path: Path, *, module_name: str):
        if not path.exists():
            raise FileNotFoundError(f"Module file not found: {path}")

        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {path}")

        loaded_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded_module)
        return loaded_module

    @staticmethod
    def _load_first_symbol(source_module: object, candidates: list[str]):
        for candidate_name in candidates:
            symbol = getattr(source_module, candidate_name, None)
            if symbol is not None:
                return symbol
        raise AttributeError(f"None of the expected symbols were found: {', '.join(candidates)}")

    def _require_graph_rag(self) -> GraphRagSystem:
        if self._graph_rag_system is None:
            raise RuntimeError("GraphRAG system is not initialized")
        return self._graph_rag_system

    def _require_naive_rag(self) -> NaiveRagSystem:
        if self._naive_rag_system is None:
            raise RuntimeError("Naive RAG system is not initialized")
        return self._naive_rag_system

    @staticmethod
    def _load_project_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    @classmethod
    def _build_paths(cls, config_path: Path) -> ComparatorPaths:
        results_directory = cls.RESULTS_DIR
        ground_truth_json = results_directory / cls.GROUND_TRUTH_FILENAME
        return ComparatorPaths(
            results_directory=results_directory,
            config_file_path=config_path,
            graph_rag_script=cls.GRAPH_RAG_SCRIPT,
            naive_rag_script=cls.NAIVE_RAG_SCRIPT,
            ground_truth_json=ground_truth_json,
            ground_truth_generator_script=cls.GROUND_TRUTH_GENERATOR_SCRIPT,
        )

    @classmethod
    def _parse_comparator_config(cls, config: dict[str, Any]) -> ComparatorConfig:
        output_section = config.get("output")
        if not isinstance(output_section, dict):
            raise ValueError("Invalid config: expected [output] section")

        cv_pdf_dir_value = output_section.get("programmers_dir")
        if not isinstance(cv_pdf_dir_value, str) or not cv_pdf_dir_value.strip():
            raise ValueError("Invalid config: output.programmers_dir must be a non-empty string")

        return ComparatorConfig(
            cv_pdf_directory=Path(cv_pdf_dir_value),
            api_delay_seconds=cls.API_DELAY_SECONDS,
        )

    @staticmethod
    def _extract_first_int(text: str) -> int | None:
        found_digit_groups: list[str] = []
        current_digit_group: list[str] = []
        for character in text:
            if character.isdigit():
                current_digit_group.append(character)
            elif current_digit_group:
                found_digit_groups.append("".join(current_digit_group))
                break
        if current_digit_group and not found_digit_groups:
            found_digit_groups.append("".join(current_digit_group))
        if not found_digit_groups:
            return None
        try:
            return int(found_digit_groups[0])
        except ValueError:
            return None

    @staticmethod
    def _extract_first_float(text: str) -> float | None:
        number_chars: list[str] = []
        seen_digit = False
        seen_dot = False
        for character in text:
            if character.isdigit():
                number_chars.append(character)
                seen_digit = True
                continue
            if character == "." and seen_digit and not seen_dot:
                number_chars.append(character)
                seen_dot = True
                continue
            if number_chars:
                break
        if not number_chars:
            return None
        try:
            return float("".join(number_chars))
        except ValueError:
            return None

    @staticmethod
    def _extract_capitalized_tokens(text: str) -> set[str]:
        stripped_tokens = {token.strip(" ,.;:()[]{}\"'") for token in text.split()}
        return {token for token in stripped_tokens if token and token[0].isupper()}

    @staticmethod
    def _normalize_words(text: str) -> set[str]:
        cleaned = "".join(
            char.lower() if char.isalnum() or char.isspace() else " "
            for char in text
        )
        return {word for word in cleaned.split() if word}

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        midpoint = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            return sorted_values[midpoint]
        return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


async def main() -> bool:
    """Run the complete comparison workflow and persist outputs."""
    print(textwrap.dedent("""\
        🚀 GraphRAG vs Naive RAG Complete Comparison Workflow
        ============================================================"""))

    comparator = SystemComparator()

    if not comparator.has_cv_data():
        print(textwrap.dedent("""\

            ⚠️  No CV PDFs found in the configured directory
            Please run: uv run python 1_generate_data.py
            Then run: uv run python 2_data_to_knowledge_graph.py"""))
        return False

    print("\n✅ CV data found successfully!")

    separator = "=" * 60
    print(f"\n{separator}\nSTEP 1: Load/Generate Ground Truth using GPT-5\n{separator}")
    comparator.load_ground_truth()

    print(f"\n{separator}\nSTEP 2: Run Complete System Comparison\n{separator}")

    comparison_data = await comparator.run_full_comparison()
    json_output_file = comparator.save_comparison_results(comparison_data)
    markdown_table_file = comparator.save_comparison_table(comparison_data)

    comparator.display_results(comparison_data)

    print(f"\n{separator}\n🎉 COMPARISON WORKFLOW COMPLETED SUCCESSFULLY!\n{separator}")

    print("\n📁 Generated Files:")
    results_directory = Path("results")
    if results_directory.exists():
        for generated_file in sorted(results_directory.glob("*")):
            print(f"  • {generated_file}")

    print(textwrap.dedent(f"""\

        📊 Key Results:
          • Comparison Data: {json_output_file}
          • Comparison Table: {markdown_table_file}

        🔗 Next Steps:
          1. Review the comparison table: cat {markdown_table_file}
          2. Analyze detailed results: cat {json_output_file}"""))

    return True


if __name__ == "__main__":
    load_dotenv(override=True)
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
    asyncio.run(main())
