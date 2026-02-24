"""
📊 RAG Comparative Evaluator
=============================

Orchestrates the full evaluation lifecycle for one or more RAG systems:

1. Generate ground truth answers via GroundTruthGenerator.
2. Run each RAG system on every evaluation question and collect RAGAS
   SingleTurnSample objects.
3. Score samples with five RAGAS metrics (Context Precision / Recall,
   Faithfulness, Answer Relevancy, Factual Correctness).
4. Aggregate results into a pandas DataFrame for comparison and export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from langchain_openai import AzureChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from .ground_truth import GroundTruthGenerator


class RAGSystem(Protocol):
    """Structural type matching any RAG implementation that exposes name and query()."""

    @property
    def name(self) -> str: ...

    def query(self, question: str) -> tuple[str, list[str]]: ...


METADATA_COLUMNS = {"user_input", "response", "retrieved_contexts", "reference"}


class RAGEvaluator:
    """Evaluates and compares multiple RAG systems using RAGAS metrics."""

    def __init__(self, *, expert_llm: AzureChatOpenAI, evaluator_llm: AzureChatOpenAI) -> None:
        self.ground_truth_generator = GroundTruthGenerator(expert_llm)

        wrapped_evaluator = LangchainLLMWrapper(evaluator_llm)
        self.ragas_metrics = [
            ContextPrecision(llm=wrapped_evaluator),
            ContextRecall(llm=wrapped_evaluator),
            Faithfulness(llm=wrapped_evaluator),
            AnswerRelevancy(llm=wrapped_evaluator),
            FactualCorrectness(llm=wrapped_evaluator),
        ]

    def evaluate_single_system(
            self,
            rag_system: RAGSystem,
            evaluation_questions: list[str],
            ground_truth_answers: list[str],
    ) -> Any:
        """Evaluate a single RAG system and return the RAGAS result (or None on failure)."""
        print(f"   Evaluating {rag_system.name}...")

        collected_samples: list[SingleTurnSample] = []
        for question, ground_truth in zip(evaluation_questions, ground_truth_answers):
            try:
                answer, context_texts = rag_system.query(question)
                collected_samples.append(
                    SingleTurnSample(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=context_texts,
                        reference=ground_truth,
                    )
                )
            except Exception as query_error:
                print(f"   Warning: Failed to query {rag_system.name} for '{question}': {query_error}")
                continue

        if not collected_samples:
            print(f"   Error: No valid samples for {rag_system.name}")
            return None

        try:
            return evaluate(
                dataset=EvaluationDataset(samples=collected_samples),
                metrics=self.ragas_metrics,
            )
        except Exception as eval_error:
            print(f"   Error evaluating {rag_system.name}: {eval_error}")
            return None

    def compare_systems(
            self,
            rag_systems: list[RAGSystem],
            evaluation_questions: list[str],
            data_directory: str,
    ) -> dict[str, Any]:
        """Run all RAG systems through evaluation and return {system_name: ragas_result}."""
        print("Generating ground truths with expert LLM...")
        ground_truth_answers = self.ground_truth_generator.generate_ground_truths(
            evaluation_questions, data_directory,
        )
        print("✓ Ground truths generated")

        evaluation_results: dict[str, Any] = {}
        for rag_system in rag_systems:
            single_result = self.evaluate_single_system(
                rag_system, evaluation_questions, ground_truth_answers,
            )
            if single_result is not None:
                evaluation_results[rag_system.name] = single_result

        return evaluation_results

    @staticmethod
    def create_comparison_dataframe(evaluation_results: dict[str, Any]) -> pd.DataFrame:
        """Aggregate per-system RAGAS results into a single comparison DataFrame."""
        comparison_rows: list[dict] = []

        for system_name, ragas_result in evaluation_results.items():
            result_dataframe = ragas_result.to_pandas()
            metric_columns = [
                col for col in result_dataframe.columns if col not in METADATA_COLUMNS
            ]

            metric_averages = {
                metric: result_dataframe[metric].mean()
                for metric in metric_columns
                if metric in result_dataframe.columns
            }
            metric_averages["system"] = system_name
            comparison_rows.append(metric_averages)

        return pd.DataFrame(comparison_rows)

    @staticmethod
    def print_comparison_table(comparison_dataframe: pd.DataFrame) -> None:
        """Print a formatted comparison table with per-metric winners."""
        if comparison_dataframe.empty:
            print("No results to display")
            return

        print("\n" + "=" * 80 + "\n"
                                "COMPARATIVE EVALUATION RESULTS\n"
              + "=" * 80)

        metric_columns = sorted(
            col for col in comparison_dataframe.columns if col != "system"
        )
        display_columns = ["system", *metric_columns]
        display_dataframe = comparison_dataframe[display_columns].round(3)

        display_dataframe.columns = [
            col.replace("_", " ").title() if col != "system" else "System"
            for col in display_dataframe.columns
        ]

        print(display_dataframe.to_string(index=False))

        print("\n" + "-" * 80)
        print("BEST PERFORMERS:")
        for metric in metric_columns:
            best_row_index = comparison_dataframe[metric].idxmax()
            best_system_name = comparison_dataframe.loc[best_row_index, "system"]
            best_score = comparison_dataframe[metric].max()
            print(f"• {metric.replace('_', ' ').title()}: {best_system_name} ({best_score:.3f})")

        overall_mean_per_system = comparison_dataframe[metric_columns].mean(axis=1)
        overall_best_index = overall_mean_per_system.idxmax()
        overall_best_system = comparison_dataframe.loc[overall_best_index, "system"]
        overall_best_score = overall_mean_per_system[overall_best_index]
        print(f"• Overall Best: {overall_best_system} (Avg: {overall_best_score:.3f})")

    @staticmethod
    def save_results(comparison_dataframe: pd.DataFrame, results_directory: str = "results") -> None:
        """Persist the comparison DataFrame as a CSV file."""
        results_path = Path(results_directory)
        results_path.mkdir(exist_ok=True)

        csv_file_path = results_path / "comparison_metrics.csv"
        comparison_dataframe.to_csv(csv_file_path, index=False)

        print(f"\n✓ Results saved to {csv_file_path}")
