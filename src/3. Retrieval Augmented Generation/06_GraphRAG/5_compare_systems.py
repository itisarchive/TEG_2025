#!/usr/bin/env python3
"""
GraphRAG vs Naive RAG Comparison System.

Compares GraphRAG and Naive RAG performance using pre-generated ground truth answers
to demonstrate the advantages of graph-based retrieval for structured queries.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
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


class Evaluation(TypedDict):
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
    results_dir: Path
    config_path: Path
    graph_rag_file: Path
    naive_rag_file: Path
    ground_truth_file: Path
    ground_truth_generator: Path


@dataclass(frozen=True, slots=True)
class ComparatorConfig:
    programmers_dir: Path
    api_delay_seconds: float


class SystemComparator:
    """Compare GraphRAG and Naive RAG results against ground truth answers."""

    _DEFAULT_RESULTS_DIR: Final[Path] = Path("results")
    _DEFAULT_GRAPH_RAG_FILE: Final[Path] = Path("3_query_knowledge_graph.py")
    _DEFAULT_NAIVE_RAG_FILE: Final[Path] = Path("4_naive_rag_cv.py")
    _DEFAULT_GROUND_TRUTH_FILE: Final[str] = "ground_truth_answers.json"
    _DEFAULT_GROUND_TRUTH_GENERATOR: Final[Path] = Path("utils/generate_ground_truth.py")
    _DEFAULT_API_DELAY_SECONDS: Final[float] = 0.5

    def __init__(self, config_path: str | Path = "utils/config.toml") -> None:
        self._paths = self._build_paths(Path(config_path))
        self._config = self._load_project_config(self._paths.config_path)
        self._comparator_config = self._parse_comparator_config(self._config)

        self._paths.results_dir.mkdir(parents=True, exist_ok=True)

        self._graph_rag: GraphRagSystem | None = None
        self._naive_rag: NaiveRagSystem | None = None

        LOGGER.info("SystemComparator initialized")

    def load_ground_truth(self) -> GroundTruthPayload:
        """Load ground truth answers, generating them if missing."""
        if not self._paths.ground_truth_file.exists():
            self._generate_ground_truth()

        payload = self._read_json(self._paths.ground_truth_file)
        if not isinstance(payload, dict) or "ground_truth_answers" not in payload:
            raise ValueError("Ground truth JSON has an unexpected format")

        items = payload["ground_truth_answers"]
        if not isinstance(items, list):
            raise ValueError("ground_truth_answers must be a list")

        return GroundTruthPayload(ground_truth_answers=items)

    def initialize_graph_rag_system(self) -> None:
        """Initialize the GraphRAG system via dynamic import."""
        module = self._import_from_path(self._paths.graph_rag_file, module_name="graph_rag_module")
        system_cls = self._load_first_symbol(module, ["CVGraphRAGSystem", "GraphRAGSystem", "GraphRagSystem"])
        self._graph_rag = system_cls()

        LOGGER.info("GraphRAG system initialized")

    def initialize_naive_rag_system(self) -> None:
        """Initialize the Naive RAG system via dynamic import."""
        module = self._import_from_path(self._paths.naive_rag_file, module_name="naive_rag_module")
        system_cls = self._load_first_symbol(module, ["NaiveCvRag", "NaiveRAGSystem", "NaiveRagSystem"])
        naive = system_cls()

        if hasattr(naive, "initialize"):
            naive.initialize()
            self._naive_rag = naive
            LOGGER.info("Naive RAG system initialized")
            return

        if hasattr(naive, "initialize_system") and callable(getattr(naive, "initialize_system")):
            ok = naive.initialize_system()
            if not ok:
                raise RuntimeError("Naive RAG system failed to initialize")
            self._naive_rag = naive
            LOGGER.info("Naive RAG system initialized")
            return

        raise AttributeError("Naive RAG system exposes neither initialize() nor initialize_system()")

    def has_cv_data(self) -> bool:
        """Return whether the configured CV directory contains PDF files."""
        data_dir = self._comparator_config.programmers_dir
        return data_dir.exists() and any(data_dir.glob("*.pdf"))

    async def run_full_comparison(self) -> ComparisonPayload:
        """Run all questions through both systems and compute summary statistics."""
        ground_truth = self.load_ground_truth()
        questions = ground_truth["ground_truth_answers"]

        self.initialize_graph_rag_system()
        self.initialize_naive_rag_system()

        results: list[ComparisonEntry] = []
        for index, item in enumerate(questions, start=1):
            question = item["question"]
            category = item["category"]
            expected = item["ground_truth_answer"]

            graph_answer = self.run_graph_rag_query(question)
            naive_answer = self.run_naive_rag_query(question)

            graph_eval = self.evaluate_answer_quality(expected, graph_answer.get("answer", ""), category)
            naive_eval = self.evaluate_answer_quality(expected, naive_answer.get("answer", ""), category)

            results.append(
                ComparisonEntry(
                    question_index=index,
                    question=question,
                    category=category,
                    ground_truth=expected,
                    graphrag={
                        "answer": graph_answer.get("answer", "No answer"),
                        "cypher_query": graph_answer.get("cypher_query", ""),
                        "execution_time": float(graph_answer.get("execution_time", 0.0)),
                        "success": bool(graph_answer.get("success", False)),
                        "evaluation": graph_eval,
                    },
                    naive_rag={
                        "answer": naive_answer.get("answer", "No answer"),
                        "chunks_retrieved": int(naive_answer.get("num_chunks_retrieved", 0)),
                        "execution_time": float(naive_answer.get("execution_time", 0.0)),
                        "success": bool(naive_answer.get("success", False)),
                        "evaluation": naive_eval,
                    },
                )
            )

            delay = self._comparator_config.api_delay_seconds
            if delay > 0:
                await asyncio.sleep(delay)

        comparison_data: ComparisonPayload = {
            "metadata": {
                "comparison_date": datetime.now().isoformat(),
                "total_questions": len(results),
                "ground_truth_source": "GPT-5",
                "systems_compared": ["GraphRAG", "Naive RAG"],
            },
            "results": results,
            "summary": self.generate_summary(results),
        }
        return comparison_data

    def run_graph_rag_query(self, question: str) -> GraphRagAnswer:
        """Run a single question through the GraphRAG system."""
        system = self._require_graph_rag()
        start = time.perf_counter()
        try:
            result = system.query_graph(question)
            elapsed = time.perf_counter() - start
            return GraphRagAnswer(
                question=question,
                answer=str(result.get("answer", "No answer")),
                cypher_query=str(result.get("cypher_query", "")),
                execution_time=elapsed,
                success=bool(result.get("success", False)),
                system="graphrag",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            LOGGER.exception("GraphRAG query failed")
            return GraphRagAnswer(
                question=question,
                answer=f"Error: {exc}",
                cypher_query="",
                execution_time=elapsed,
                success=False,
                system="graphrag",
                error=str(exc),
            )

    def run_naive_rag_query(self, question: str) -> NaiveRagAnswer:
        """Run a single question through the Naive RAG system."""
        system = self._require_naive_rag()
        try:
            result = system.query(question)
            return NaiveRagAnswer(
                question=question,
                answer=str(result.get("answer", "No answer")),
                execution_time=float(result.get("execution_time", 0.0)),
                num_chunks_retrieved=int(result.get("num_chunks_retrieved", 0)),
                success=bool(result.get("success", False)),
                system="naive_rag",
            )
        except Exception as exc:
            LOGGER.exception("Naive RAG query failed")
            return NaiveRagAnswer(
                question=question,
                answer=f"Error: {exc}",
                execution_time=0.0,
                num_chunks_retrieved=0,
                success=False,
                system="naive_rag",
                error=str(exc),
            )

    def evaluate_answer_quality(self, ground_truth: str, system_answer: str, question_type: str) -> Evaluation:
        """Evaluate how well a system answer matches the ground truth."""
        evaluation: Evaluation = {
            "exact_match": ground_truth.strip().casefold() == system_answer.strip().casefold(),
            "contains_key_info": False,
            "numerical_accuracy": None,
            "completeness_score": 0.0,
            "quality_score": 0.0,
        }

        if question_type == "counting":
            gt_num = self._extract_first_int(ground_truth)
            sys_num = self._extract_first_int(system_answer)
            if gt_num is not None and sys_num is not None:
                evaluation["numerical_accuracy"] = gt_num == sys_num
                evaluation["contains_key_info"] = True
                evaluation["quality_score"] = 1.0 if gt_num == sys_num else 0.0
            return evaluation

        if question_type in {"filtering", "listing"}:
            gt_names = self._extract_capitalized_tokens(ground_truth)
            sys_names = self._extract_capitalized_tokens(system_answer)
            if gt_names:
                overlap = len(gt_names & sys_names) / len(gt_names)
                evaluation["completeness_score"] = overlap
                evaluation["quality_score"] = overlap
                evaluation["contains_key_info"] = overlap > 0
            return evaluation

        if question_type == "aggregation":
            gt_val = self._extract_first_float(ground_truth)
            sys_val = self._extract_first_float(system_answer)
            if gt_val is not None and sys_val is not None:
                diff = abs(gt_val - sys_val) / max(abs(gt_val), 1.0)
                evaluation["numerical_accuracy"] = diff < 0.1
                evaluation["quality_score"] = max(0.0, 1.0 - diff)
                evaluation["contains_key_info"] = True
            return evaluation

        gt_words = self._normalize_words(ground_truth)
        sys_words = self._normalize_words(system_answer)
        if gt_words:
            overlap = len(gt_words & sys_words) / len(gt_words)
            evaluation["completeness_score"] = overlap
            evaluation["quality_score"] = overlap
            evaluation["contains_key_info"] = overlap > 0.3
        return evaluation

    def generate_summary(self, results: list[ComparisonEntry]) -> SummaryPayload:
        """Compute overall and per-category summary metrics."""
        graph_wins = 0
        naive_wins = 0
        ties = 0

        graph_scores: list[float] = []
        naive_scores: list[float] = []

        category_stats: dict[str, dict[str, int]] = {}

        for entry in results:
            category = entry["category"]
            graph_score = float(entry["graphrag"]["evaluation"]["quality_score"])
            naive_score = float(entry["naive_rag"]["evaluation"]["quality_score"])

            graph_scores.append(graph_score)
            naive_scores.append(naive_score)

            stats = category_stats.setdefault(category, {"total": 0, "graph_wins": 0, "naive_wins": 0, "ties": 0})
            stats["total"] += 1

            if graph_score > naive_score:
                graph_wins += 1
                stats["graph_wins"] += 1
            elif naive_score > graph_score:
                naive_wins += 1
                stats["naive_wins"] += 1
            else:
                ties += 1
                stats["ties"] += 1

        mean_graph = self._mean(graph_scores)
        mean_naive = self._mean(naive_scores)
        median_graph = self._median(graph_scores)
        median_naive = self._median(naive_scores)

        total = max(len(results), 1)
        return SummaryPayload(
            overall_performance={
                "graphrag_wins": graph_wins,
                "naive_rag_wins": naive_wins,
                "ties": ties,
                "graphrag_win_rate": graph_wins / total,
                "naive_rag_win_rate": naive_wins / total,
            },
            quality_scores={
                "graphrag_avg": mean_graph,
                "naive_rag_avg": mean_naive,
                "graphrag_median": median_graph,
                "naive_rag_median": median_naive,
            },
            category_breakdown=category_stats,
        )

    def save_comparison_results(self, comparison_data: ComparisonPayload) -> Path:
        """Save full comparison JSON to the results directory."""
        output_file = self._paths.results_dir / "system_comparison_results.json"
        output_file.write_text(json.dumps(comparison_data, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Comparison results saved to %s", output_file)
        return output_file

    def save_comparison_table(self, comparison_data: ComparisonPayload) -> Path:
        """Save a markdown comparison table to the results directory."""
        table_file = self._paths.results_dir / "comparison_table.md"
        table_file.write_text(self.generate_comparison_table(comparison_data), encoding="utf-8")
        LOGGER.info("Comparison table saved to %s", table_file)
        return table_file

    def generate_comparison_table(self, comparison_data: ComparisonPayload) -> str:
        """Generate a readable markdown comparison table."""
        results = comparison_data["results"]
        summary = comparison_data["summary"]

        lines: list[str] = ["# GraphRAG vs Naive RAG Comparison Results", "", "## Summary",
                            f"- **GraphRAG Wins**: {int(summary['overall_performance']['graphrag_wins'])}",
                            f"- **Naive RAG Wins**: {int(summary['overall_performance']['naive_rag_wins'])}",
                            f"- **Ties**: {int(summary['overall_performance']['ties'])}",
                            f"- **GraphRAG Win Rate**: {float(summary['overall_performance']['graphrag_win_rate']):.1%}",
                            "", f"- **GraphRAG Avg Quality**: {float(summary['quality_scores']['graphrag_avg']):.2f}",
                            f"- **Naive RAG Avg Quality**: {float(summary['quality_scores']['naive_rag_avg']):.2f}", "",
                            "## Performance by Category", ""]

        for category, stats in summary["category_breakdown"].items():
            total = max(stats["total"], 1)
            win_rate = stats["graph_wins"] / total
            lines.append(f"### {category.title()}")
            lines.append(f"- GraphRAG: {stats['graph_wins']}/{stats['total']} ({win_rate:.1%})")
            lines.append("")

        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| # | Question | Category | GraphRAG Answer | Naive RAG Answer | Ground Truth | Winner |")
        lines.append("|---|----------|----------|-----------------|------------------|--------------|--------|")

        for entry in results:
            idx = entry["question_index"]
            question = self._truncate(entry["question"], 50)
            category = entry["category"]

            graph_answer = self._truncate(str(entry["graphrag"]["answer"]), 30)
            naive_answer = self._truncate(str(entry["naive_rag"]["answer"]), 30)
            ground_truth = self._truncate(entry["ground_truth"], 30)

            graph_score = float(entry["graphrag"]["evaluation"]["quality_score"])
            naive_score = float(entry["naive_rag"]["evaluation"]["quality_score"])

            if graph_score > naive_score:
                winner = "GraphRAG ✅"
            elif naive_score > graph_score:
                winner = "Naive RAG ✅"
            else:
                winner = "Tie ⚖️"

            question = question.replace("|", "\\|")
            graph_answer = graph_answer.replace("|", "\\|")
            naive_answer = naive_answer.replace("|", "\\|")
            ground_truth = ground_truth.replace("|", "\\|")

            lines.append(
                f"| {idx} | {question} | {category} | {graph_answer} | {naive_answer} | {ground_truth} | {winner} |"
            )

        return "\n".join(lines)

    @staticmethod
    def display_results(comparison_data: ComparisonPayload) -> None:
        """Print a compact summary to stdout."""
        summary = comparison_data["summary"]

        print("\n" + "=" * 80)
        print("GRAPHRAG vs NAIVE RAG COMPARISON RESULTS")
        print("=" * 80)

        overall = summary["overall_performance"]
        print("\n📊 OVERALL PERFORMANCE:")
        print(f"GraphRAG Wins: {int(overall['graphrag_wins'])}")
        print(f"Naive RAG Wins: {int(overall['naive_rag_wins'])}")
        print(f"Ties: {int(overall['ties'])}")
        print(f"GraphRAG Win Rate: {float(overall['graphrag_win_rate']):.1%}")

        qs = summary["quality_scores"]
        print("\n🎯 QUALITY SCORES:")
        print(f"GraphRAG Average: {float(qs['graphrag_avg']):.2f}")
        print(f"Naive RAG Average: {float(qs['naive_rag_avg']):.2f}")

        print("\n📋 PERFORMANCE BY CATEGORY:")
        for category, stats in summary["category_breakdown"].items():
            total = max(stats["total"], 1)
            win_rate = stats["graph_wins"] / total
            print(f"{category.title()}: GraphRAG {stats['graph_wins']}/{stats['total']} ({win_rate:.1%})")

    def _generate_ground_truth(self) -> None:
        generator = self._paths.ground_truth_generator
        if not generator.exists():
            raise FileNotFoundError(f"Ground truth generator not found: {generator}")

        LOGGER.info("Ground truth file not found, generating via %s", generator)
        result = subprocess.run(
            ["uv", "run", "python", str(generator)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to generate ground truth.\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )
        if not self._paths.ground_truth_file.exists():
            raise FileNotFoundError(
                f"Ground truth generation completed but file not found: {self._paths.ground_truth_file}")

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

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _load_first_symbol(module: object, candidates: list[str]):
        for name in candidates:
            symbol = getattr(module, name, None)
            if symbol is not None:
                return symbol
        raise AttributeError(f"None of the expected symbols were found: {', '.join(candidates)}")

    def _require_graph_rag(self) -> GraphRagSystem:
        if self._graph_rag is None:
            raise RuntimeError("GraphRAG system is not initialized")
        return self._graph_rag

    def _require_naive_rag(self) -> NaiveRagSystem:
        if self._naive_rag is None:
            raise RuntimeError("Naive RAG system is not initialized")
        return self._naive_rag

    @staticmethod
    def _load_project_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    @classmethod
    def _build_paths(cls, config_path: Path) -> ComparatorPaths:
        results_dir = cls._DEFAULT_RESULTS_DIR
        ground_truth_file = results_dir / cls._DEFAULT_GROUND_TRUTH_FILE
        return ComparatorPaths(
            results_dir=results_dir,
            config_path=config_path,
            graph_rag_file=cls._DEFAULT_GRAPH_RAG_FILE,
            naive_rag_file=cls._DEFAULT_NAIVE_RAG_FILE,
            ground_truth_file=ground_truth_file,
            ground_truth_generator=cls._DEFAULT_GROUND_TRUTH_GENERATOR,
        )

    @classmethod
    def _parse_comparator_config(cls, config: dict[str, Any]) -> ComparatorConfig:
        output = config.get("output")
        if not isinstance(output, dict):
            raise ValueError("Invalid config: expected [output] section")

        programmers_dir = output.get("programmers_dir")
        if not isinstance(programmers_dir, str) or not programmers_dir.strip():
            raise ValueError("Invalid config: output.programmers_dir must be a non-empty string")

        return ComparatorConfig(
            programmers_dir=Path(programmers_dir),
            api_delay_seconds=cls._DEFAULT_API_DELAY_SECONDS,
        )

    @staticmethod
    def _extract_first_int(text: str) -> int | None:
        digits: list[str] = []
        current: list[str] = []
        for ch in text:
            if ch.isdigit():
                current.append(ch)
            elif current:
                digits.append("".join(current))
                break
        if current and not digits:
            digits.append("".join(current))
        if not digits:
            return None
        try:
            return int(digits[0])
        except ValueError:
            return None

    @staticmethod
    def _extract_first_float(text: str) -> float | None:
        buf: list[str] = []
        seen_digit = False
        seen_dot = False
        for ch in text:
            if ch.isdigit():
                buf.append(ch)
                seen_digit = True
                continue
            if ch == "." and seen_digit and not seen_dot:
                buf.append(ch)
                seen_dot = True
                continue
            if buf:
                break
        if not buf:
            return None
        try:
            return float("".join(buf))
        except ValueError:
            return None

    @staticmethod
    def _extract_capitalized_tokens(text: str) -> set[str]:
        tokens = {token.strip(" ,.;:()[]{}\"'") for token in text.split()}
        return {t for t in tokens if t and t[0].isupper()}

    @staticmethod
    def _normalize_words(text: str) -> set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
        return {w for w in cleaned.split() if w}

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        mid = len(s) // 2
        if len(s) % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "..."


async def main() -> bool:
    """Run the complete comparison workflow and persist outputs."""
    print("🚀 GraphRAG vs Naive RAG Complete Comparison Workflow")
    print("=" * 60)

    comparator = SystemComparator()

    if not comparator.has_cv_data():
        print("\n⚠️  No CV PDFs found in the configured directory")
        print("Please run: uv run python 1_generate_data.py")
        print("Then run: uv run python 2_data_to_knowledge_graph.py")
        return False

    print("\n✅ CV data found successfully!")

    print("\n" + "=" * 60)
    print("STEP 1: Load/Generate Ground Truth using GPT-5")
    print("=" * 60)
    comparator.load_ground_truth()

    print("\n" + "=" * 60)
    print("STEP 2: Run Complete System Comparison")
    print("=" * 60)

    comparison_data = await comparator.run_full_comparison()
    output_file = comparator.save_comparison_results(comparison_data)
    table_file = comparator.save_comparison_table(comparison_data)

    comparator.display_results(comparison_data)

    print("\n" + "=" * 60)
    print("🎉 COMPARISON WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\n📁 Generated Files:")
    results_dir = Path("results")
    if results_dir.exists():
        for file in sorted(results_dir.glob("*")):
            print(f"  • {file}")

    print("\n📊 Key Results:")
    print(f"  • Comparison Data: {output_file}")
    print(f"  • Comparison Table: {table_file}")

    print("\n🔗 Next Steps:")
    print(f"  1. Review the comparison table: cat {table_file}")
    print(f"  2. Analyze detailed results: cat {output_file}")

    return True


if __name__ == "__main__":
    load_dotenv(override=True)
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
    asyncio.run(main())
