#!/usr/bin/env python3
"""
Ground Truth Generator for GraphRAG vs Naive RAG Comparison
============================================================

Uses Azure OpenAI (GPT-4.1) with full CV context to generate authoritative
answers for test questions. This provides an independent, unbiased ground
truth for evaluating both GraphRAG and Naive RAG systems.

🎯 What You'll Learn:
- How to build a ground-truth dataset by feeding ALL documents to a powerful LLM
- Why ground truth must be generated independently of the systems being evaluated
- How async LLM calls speed up batch-processing of many questions

🔧 Prerequisites:
    - Azure OpenAI credentials in environment / .env
    - Generated CV PDFs (see 1_generate_data.py)
"""

import asyncio
import json
import logging
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

LOGGER = logging.getLogger(__name__)

HR_GROUND_TRUTH_PROMPT = """\
You are a senior HR manager with exceptional analytical skills and perfect memory.
You have been given ALL {num_cvs} CVs from our candidate database to review comprehensively.

Your task is to answer the following question with ABSOLUTE PRECISION and COMPLETENESS based on the CVs provided.

CRITICAL INSTRUCTIONS:
- Provide ONLY the direct answer, no explanations or reasoning
- For counting questions: Give only the number (e.g., "3" or "Zero")
- For listing questions: List only the names, comma-separated
- For aggregation questions: Give only the calculated result
- For filtering questions: List only the matching names, comma-separated
- For ranking questions: List names in order, comma-separated
- Be specific with names, skills, companies, and other details
- If no matches: answer "None" or "Zero" as appropriate
- MAXIMUM 1-2 sentences, be extremely concise

CVs Database ({num_cvs} total CVs):
{context}

Question: {question}

Answer (concise, direct):"""


@dataclass(frozen=True, slots=True)
class GroundTruthPaths:
    cv_pdf_directory: Path
    results_directory: Path


@dataclass(frozen=True, slots=True)
class QuestionWithCategory:
    question_text: str
    category_name: str
    category_description: str


class GroundTruthGenerator:
    """
    Generate ground truth answers by feeding ALL CV documents to Azure OpenAI GPT-4.1.

    The generator loads every CV PDF, concatenates all text into a single context,
    and asks the LLM to answer each test question with perfect recall — producing
    authoritative reference answers for benchmarking GraphRAG vs Naive RAG.
    """

    def __init__(self, *, max_cvs_to_load: int = 30) -> None:
        self._azure_llm = AzureChatOpenAI(
            model="gpt-4.1",
            max_tokens=4096,
        )

        self._paths = GroundTruthPaths(
            cv_pdf_directory=Path("data/programmers"),
            results_directory=Path("results"),
        )
        self._paths.results_directory.mkdir(exist_ok=True)

        self._max_cvs_to_load = max_cvs_to_load

        LOGGER.info(
            "Azure OpenAI Ground Truth Generator initialized (max CVs: %d)",
            max_cvs_to_load,
        )

    def load_all_cv_texts(self) -> list[str]:
        """Load all CV PDFs from the data directory and return their extracted text."""
        all_cv_pdf_paths = sorted(self._paths.cv_pdf_directory.glob("*.pdf"))

        if not all_cv_pdf_paths:
            raise FileNotFoundError(
                f"No PDF files found in {self._paths.cv_pdf_directory}"
            )

        limited_cv_pdf_paths = all_cv_pdf_paths[: self._max_cvs_to_load]
        LOGGER.info("Loading %d CV files…", len(limited_cv_pdf_paths))

        extracted_cv_texts: list[str] = []
        for cv_pdf_path in limited_cv_pdf_paths:
            try:
                loaded_pages = PyPDFLoader(str(cv_pdf_path)).load()
                full_cv_text = "\n".join(
                    page.page_content for page in loaded_pages
                )
                extracted_cv_texts.append(
                    f"=== CV: {cv_pdf_path.stem} ===\n{full_cv_text}"
                )
            except (OSError, ValueError) as load_error:
                LOGGER.warning("Could not load %s: %s", cv_pdf_path, load_error)

        LOGGER.info("Successfully loaded %d CVs", len(extracted_cv_texts))
        return extracted_cv_texts

    @staticmethod
    def create_ground_truth_prompt() -> PromptTemplate:
        """Return the prompt template used to generate ground truth answers."""
        return PromptTemplate(
            input_variables=["num_cvs", "context", "question"],
            template=HR_GROUND_TRUTH_PROMPT,
        )

    async def generate_answer_for_question(
            self,
            question_text: str,
            all_cv_texts: list[str],
    ) -> dict[str, Any]:
        """
        Generate a single ground-truth answer by sending all CV context + question to GPT-4.1.

        Returns a dict with the answer, timing diagnostics, and success/error status.
        """
        start_time = time.perf_counter()

        try:
            context_build_start = time.perf_counter()
            full_context = "\n\n" + "=" * 80 + "\n\n".join(all_cv_texts)
            context_build_elapsed = time.perf_counter() - context_build_start

            context_tokens_estimate = int(len(full_context.split()) * 1.3)
            LOGGER.info(
                "Context size: ~%d tokens (built in %.2fs)",
                context_tokens_estimate,
                context_build_elapsed,
            )

            if context_tokens_estimate > 300_000:
                LOGGER.warning(
                    "Context size may be large: %d tokens",
                    context_tokens_estimate,
                )

            prompt_template = self.create_ground_truth_prompt()
            formatted_prompt = prompt_template.format(
                num_cvs=len(all_cv_texts),
                context=full_context,
                question=question_text,
            )

            LOGGER.info("Calling API for: %s…", question_text[:60])
            api_start_time = time.perf_counter()
            llm_response = await self._azure_llm.ainvoke(formatted_prompt)
            api_elapsed = time.perf_counter() - api_start_time

            ground_truth_answer = llm_response.content.strip()
            total_elapsed = time.perf_counter() - start_time
            LOGGER.info(
                "Generated in %.2fs (API: %.2fs, %d chars)",
                total_elapsed,
                api_elapsed,
                len(ground_truth_answer),
            )

            return {
                "question": question_text,
                "ground_truth_answer": ground_truth_answer,
                "context_tokens_estimate": context_tokens_estimate,
                "num_cvs_used": len(all_cv_texts),
                "model_used": "gpt-4.1",
                "status": "success",
                "generation_time_seconds": total_elapsed,
                "api_time_seconds": api_elapsed,
            }

        except (ValueError, RuntimeError) as generation_error:
            total_elapsed = time.perf_counter() - start_time
            LOGGER.error(
                "Error generating ground truth for '%s': %s",
                question_text,
                generation_error,
            )

            return {
                "question": question_text,
                "ground_truth_answer": f"ERROR: {generation_error}",
                "context_tokens_estimate": 0,
                "num_cvs_used": len(all_cv_texts),
                "model_used": "gpt-4.1",
                "status": "error",
                "error": str(generation_error),
                "generation_time_seconds": total_elapsed,
            }

    @staticmethod
    def load_test_questions() -> dict[str, Any]:
        """Load test questions from the JSON file bundled alongside this module."""
        questions_file_path = Path(__file__).parent / "test_questions.json"
        if not questions_file_path.exists():
            raise FileNotFoundError(
                f"Test questions file not found: {questions_file_path}"
            )

        return json.loads(questions_file_path.read_text(encoding="utf-8"))

    async def generate_all_ground_truths(
            self,
            *,
            max_questions: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate ground truth answers for all (or a limited number of) test questions.

        Loads every CV once, then iterates through test questions one-by-one,
        calling the LLM for each and collecting results with timing diagnostics.
        """
        all_cv_texts = self.load_all_cv_texts()
        test_suite_data = self.load_test_questions()

        questions_with_categories: list[QuestionWithCategory] = []
        for category_name, category_data in test_suite_data["test_suite"]["categories"].items():
            for single_question in category_data["questions"]:
                questions_with_categories.append(
                    QuestionWithCategory(
                        question_text=single_question,
                        category_name=category_name,
                        category_description=category_data["description"],
                    )
                )

        if max_questions is not None:
            questions_with_categories = questions_with_categories[:max_questions]

        LOGGER.info(
            "Generating ground truth for %d questions…",
            len(questions_with_categories),
        )

        collected_results: list[dict[str, Any]] = []
        for question_index, question_item in enumerate(questions_with_categories, start=1):
            LOGGER.info(
                "\n[%d/%d] Processing %s: %s",
                question_index,
                len(questions_with_categories),
                question_item.category_name,
                question_item.question_text,
            )

            single_result = await self.generate_answer_for_question(
                question_text=question_item.question_text,
                all_cv_texts=all_cv_texts,
            )
            single_result["category"] = question_item.category_name
            single_result["category_description"] = question_item.category_description
            single_result["question_index"] = question_index

            collected_results.append(single_result)
            await asyncio.sleep(1)

        successful_count = sum(
            1 for entry in collected_results if entry["status"] == "success"
        )
        error_count = sum(
            1 for entry in collected_results if entry["status"] == "error"
        )

        return {
            "metadata": {
                "generated_by": "GroundTruthGenerator",
                "model": "gpt-4.1",
                "num_questions": len(collected_results),
                "num_cvs": len(all_cv_texts),
                "cv_source_dir": str(self._paths.cv_pdf_directory),
                "total_successful": successful_count,
                "total_errors": error_count,
            },
            "ground_truth_answers": collected_results,
            "original_test_questions": test_suite_data,
        }

    def save_ground_truth(self, ground_truth_data: dict[str, Any]) -> Path:
        """Persist ground truth data to a JSON file."""
        output_file_path = self._paths.results_directory / "ground_truth_answers.json"
        output_file_path.write_text(
            json.dumps(ground_truth_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info("Ground truth saved to: %s", output_file_path)
        return output_file_path

    @staticmethod
    def display_summary(ground_truth_data: dict[str, Any]) -> None:
        """Display a summary of the ground truth generation run."""
        generation_metadata = ground_truth_data["metadata"]
        all_answer_entries = ground_truth_data["ground_truth_answers"]

        separator = "=" * 60
        print(textwrap.dedent(f"""\

            {separator}
            Ground Truth Generation Summary
            {separator}
            Model Used: {generation_metadata['model']}
            Total Questions: {generation_metadata['num_questions']}
            CVs Analyzed: {generation_metadata['num_cvs']}
            Successful: {generation_metadata['total_successful']}
            Errors: {generation_metadata['total_errors']}"""))

        successful_entries_with_timing = [
            entry for entry in all_answer_entries
            if entry["status"] == "success" and "generation_time_seconds" in entry
        ]
        if successful_entries_with_timing:
            total_generation_time = sum(
                entry["generation_time_seconds"]
                for entry in successful_entries_with_timing
            )
            avg_generation_time = total_generation_time / len(successful_entries_with_timing)
            all_api_times = [
                entry.get("api_time_seconds", 0)
                for entry in successful_entries_with_timing
            ]
            avg_api_time = sum(all_api_times) / len(all_api_times) if all_api_times else 0

            print(textwrap.dedent(f"""\

                ⏱️  TIMING DIAGNOSTICS:
                Total generation time: {total_generation_time:.2f}s ({total_generation_time / 60:.2f} minutes)
                Average per question: {avg_generation_time:.2f}s
                Average API time: {avg_api_time:.2f}s"""))

        categories_question_count: dict[str, int] = {}
        for answer_entry in all_answer_entries:
            category_name = answer_entry["category"]
            categories_question_count[category_name] = (
                    categories_question_count.get(category_name, 0) + 1
            )

        print("\nQuestions by Category:")
        for category_name, question_count in categories_question_count.items():
            print(f"  • {category_name}: {question_count} questions")

        print("\nSample Ground Truth Answers:")
        for answer_entry in all_answer_entries[:3]:
            if answer_entry["status"] == "success":
                full_answer = answer_entry["ground_truth_answer"]
                truncated_answer = (
                    full_answer[:100] + "…"
                    if len(full_answer) > 100
                    else full_answer
                )
                print(f"\nQ: {answer_entry['question']}")
                print(f"A: {truncated_answer}")

        print(f"\n{separator}")


async def main() -> None:
    """Generate ground truth answers for the GraphRAG vs Naive RAG comparison."""
    print(textwrap.dedent("""\
        Ground Truth Generator for GraphRAG vs Naive RAG Comparison
        =================================================================="""))

    try:
        generator = GroundTruthGenerator()
        ground_truth_data = await generator.generate_all_ground_truths()
        saved_file_path = generator.save_ground_truth(ground_truth_data)
        generator.display_summary(ground_truth_data)

        print(textwrap.dedent(f"""\

            🎉 Ground truth generation complete!
            📁 Results saved to: {saved_file_path}

            Next steps:
            1. Create naive RAG baseline: uv run python 4_naive_rag_cv.py
            2. Run comparison: uv run python 5_compare_systems.py"""))

    except FileNotFoundError as file_error:
        LOGGER.error("File not found: %s", file_error)
        print(f"\n❌ Error: {file_error}")
        raise
    except (ValueError, RuntimeError) as generation_error:
        LOGGER.error("Ground truth generation failed: %s", generation_error)
        print(f"\n❌ Error: {generation_error}")
        raise


if __name__ == "__main__":
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
