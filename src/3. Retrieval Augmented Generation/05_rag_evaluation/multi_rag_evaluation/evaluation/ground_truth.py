"""
📝 Ground Truth Generation
==========================

Generates expert-level reference answers by feeding the *complete* set of
source documents to a powerful LLM.  These answers become the gold standard
for RAGAS evaluation metrics such as Context Recall and Factual Correctness.
"""

import textwrap

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import AzureChatOpenAI

EXPERT_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are a domain expert with complete knowledge of these scientists.
    Based on the following complete biographies, provide a comprehensive, accurate answer.

    Complete Biographies:
    {biography_text}

    Question: {question}

    Provide a detailed, factually accurate answer:""")


class GroundTruthGenerator:
    """Generates expert-level ground truth answers for RAG evaluation."""

    def __init__(self, expert_llm: AzureChatOpenAI) -> None:
        self.expert_llm = expert_llm

    def generate_ground_truths(
            self,
            evaluation_questions: list[str],
            source_directory: str,
    ) -> list[str]:
        """Generate one ground truth answer per question using the full document corpus."""
        directory_loader = DirectoryLoader(source_directory, glob="*.txt")
        biography_documents = directory_loader.load()
        full_biography_text = "\n\n".join(
            document.page_content for document in biography_documents
        )

        ground_truth_answers: list[str] = []
        for question in evaluation_questions:
            expert_prompt = EXPERT_PROMPT_TEMPLATE.format(
                biography_text=full_biography_text,
                question=question,
            )
            expert_answer = self.expert_llm.invoke(expert_prompt).content
            ground_truth_answers.append(expert_answer)

        return ground_truth_answers
