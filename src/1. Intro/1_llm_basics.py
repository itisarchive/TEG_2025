#!/usr/bin/env python3
"""
🚀 Large Language Models (LLMs) - Interactive Educational Journey
===============================================================

Welcome to your hands-on exploration of Large Language Models! This script takes you
through the fundamental concepts that make LLMs work, using practical examples that
you can run, modify, and experiment with.

🎯 What You'll Learn:
- How system prompts shape AI behavior
- The impact of temperature on creativity and consistency
- How token limits control response length and cost
- Advanced parameter combinations for different use cases
- Real-world applications and best practices

🔧 Prerequisites:
- Azure OpenAI credentials in .env file
- Python 3.13+ with openai and python-dotenv packages
"""

import textwrap
from dataclasses import dataclass, asdict
from typing import Any

from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


def print_section_header(title: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}\n{title}\n{separator}")


def send_chat_completion(
        client: AzureOpenAI,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str = "gpt-4.1-mini",
        **completion_params: Any,
) -> str:
    """Sends a system+user message pair to Azure OpenAI and returns the assistant's reply."""
    chat_messages = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]
    completion_response = client.chat.completions.create(
        model=model_name,
        messages=chat_messages,
        **completion_params,
    )
    return completion_response.choices[0].message.content.strip()


@dataclass(frozen=True)
class CompletionModelParams:
    temperature: float
    max_completion_tokens: int
    top_p: float | None = None

    def as_api_kwargs(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class TunedPersonaConfig:
    persona_name: str
    model_params: CompletionModelParams
    use_case: str
    rationale: str


@dataclass(frozen=True)
class RealWorldAppScenario:
    scenario_name: str
    system_prompt: str
    model_params: CompletionModelParams
    user_input: str
    learning_point: str


def demonstrate_system_prompts(client: AzureOpenAI) -> None:
    """
    System prompts are special instructions that define how an AI should behave.
    Think of them as giving the AI a "role" or "character" to play.

    Key principles:
    • Be specific about tone, style, and expertise level
    • Define the AI's role clearly (teacher, expert, assistant)
    • Set expectations for response format and length
    • Consider your audience (beginners vs experts)
    """
    print_section_header("EXAMPLE 1: SYSTEM PROMPTS - THE AI'S PERSONALITY")
    print(textwrap.dedent(demonstrate_system_prompts.__doc__))

    helpful_assistant_prompt = (
        "You are a helpful assistant who explains concepts clearly and concisely."
    )
    user_question = "Why is the sky blue?"

    helpful_assistant_answer = send_chat_completion(
        client,
        system_prompt=helpful_assistant_prompt,
        user_prompt=user_question,
        model_name="gpt-5-nano",
    )
    print(helpful_assistant_answer)

    hip_hop_teacher_prompt = (
        "You are a cool teacher who explains complex topics using hip-hop slang "
        "and rhythm. Keep it educational but fun!"
    )

    hip_hop_teacher_answer = send_chat_completion(
        client,
        system_prompt=hip_hop_teacher_prompt,
        user_prompt=user_question,
        model_name="gpt-5-nano",
    )
    print(hip_hop_teacher_answer)

    print(
        "\n💡 Your Turn: Modify the system prompt above to create your own unique AI personality!\n"
        '   e.g., "pirate teacher", "corporate consultant", "kindergarten helper", etc.'
    )


def demonstrate_temperature(client: AzureOpenAI) -> None:
    """
    CONCEPT: Temperature Parameter
    Temperature controls how "creative" or "random" the AI's responses are.

    Temperature Scale:
    • 0.0-0.3: Very focused, deterministic, repeatable
    • 0.5-1.0: Balanced creativity and coherence
    • 1.0-2.0: Highly creative, diverse, potentially unexpected

    Use cases:
    • Low (0.1): Factual answers, coding, math problems
    • Medium (1.0): Creative writing, brainstorming, conversations
    • High (1.8): Art, poetry, very creative tasks
    """
    print_section_header("EXAMPLE 2: TEMPERATURE - THE CREATIVITY DIAL")
    print(textwrap.dedent(demonstrate_temperature.__doc__))

    user_question = "Describe a sunset in one creative sentence."
    temperature_values = [0.1, 1.0, 1.5]

    for temperature_value in temperature_values:
        creativity_behavior = (
            "Focused & Predictable"
            if temperature_value <= 0.3
            else "Balanced Creativity" if temperature_value <= 1.0 else "Highly Creative"
        )
        print(f"\n🌡️ Temperature: {temperature_value} ({creativity_behavior})")

        creative_answer = send_chat_completion(
            client,
            system_prompt="You are a creative writer.",
            user_prompt=user_question,
            model_name="gpt-4.1",
            temperature=temperature_value,
            max_completion_tokens=50,
        )
        print(creative_answer)

    print(
        "\n💡 KEY INSIGHT: Temperature dramatically affects creativity vs consistency.\n"
        "   Exercise: Run this section multiple times and observe how responses change.\n"
        "   Hint: High temperature responses will be different each time, low won't."
    )


def demonstrate_token_management(client: AzureOpenAI) -> None:
    """
    Tokens are the building blocks of LLM processing (roughly 3/4 of a word).
    The max_tokens parameter limits how long responses can be.

    Token Economics:
    • You pay for both input AND output tokens
    • Longer responses cost more money
    • Token limits force the AI to be concise
    • Average: 1 token ≈ 0.75 words in English

    Strategic uses:
    • Short limits: Force concise answers, save costs
    • Medium limits: Balanced detail and cost
    • Long limits: Detailed explanations, comprehensive responses
    """
    print_section_header("EXAMPLE 3: TOKEN MANAGEMENT - CONTROLLING LENGTH AND COST")
    print(textwrap.dedent(demonstrate_token_management.__doc__))

    user_question = "Explain the concept of machine learning and give practical examples."
    max_token_limits = [30, 100, 200]

    for token_limit in max_token_limits:
        expected_word_count = int(token_limit * 0.75)
        cost_level = "Low" if token_limit <= 50 else "Medium" if token_limit <= 150 else "High"

        print(f"\n📏 Max Tokens: {token_limit} (~{expected_word_count} words, {cost_level} cost)")

        token_limited_answer = send_chat_completion(
            client,
            system_prompt="You are a helpful AI teacher.",
            user_prompt=user_question,
            max_completion_tokens=token_limit,
        )
        actual_word_count = len(token_limited_answer.split())

        print(f"   Actual words: {actual_word_count}")
        print(f"   Response: {token_limited_answer}")

    print(
        "\n💡 KEY INSIGHT: Token limits control both cost and response style.\n"
        "   Exercise: Try asking a complex question with only 20 tokens.\n"
        "   Hint: The AI will be forced to give a very brief, focused answer."
    )


def demonstrate_advanced_parameter_combinations(client: AzureOpenAI) -> None:
    """
    CONCEPT: Parameter Synergy
    Real applications combine multiple parameters for precise control.
    Each parameter affects the others, so understanding combinations is crucial.
    """
    print_section_header("EXAMPLE 4: ADVANCED PARAMETER COMBINATIONS")
    print(textwrap.dedent(demonstrate_advanced_parameter_combinations.__doc__))

    tuned_persona_configs: list[TunedPersonaConfig] = [
        TunedPersonaConfig(
            persona_name="📝 Technical Documentation Writer",
            model_params=CompletionModelParams(temperature=0.3, max_completion_tokens=300, top_p=0.8),
            use_case="API docs, technical guides, code explanations",
            rationale="Low temp for accuracy, focused top_p for technical precision",
        ),
        TunedPersonaConfig(
            persona_name="🎨 Creative Content Generator",
            model_params=CompletionModelParams(temperature=1.4, max_completion_tokens=250, top_p=0.95),
            use_case="Marketing copy, creative writing, brainstorming",
            rationale="High temp + top_p for maximum creative diversity",
        ),
        TunedPersonaConfig(
            persona_name="💬 Customer Service Bot",
            model_params=CompletionModelParams(temperature=0.7, max_completion_tokens=150, top_p=0.9),
            use_case="Help desk, FAQ responses, user support",
            rationale="Balanced creativity with consistent, helpful tone",
        ),
    ]

    test_user_prompt = "Explain artificial intelligence and its impact on modern society."

    for persona_config in tuned_persona_configs:
        print(f"\n🎯 {persona_config.persona_name}")
        print(f"   Use Case: {persona_config.use_case}")
        print(f"   Parameters: {persona_config.model_params}")
        print(f"   Rationale: {persona_config.rationale}")

        extracted_role = persona_config.persona_name.split()[1].lower()
        persona_answer = send_chat_completion(
            client,
            system_prompt=f"You are a {extracted_role}.",
            user_prompt=test_user_prompt,
            **persona_config.model_params.as_api_kwargs(),
        )
        print(f"   Response: {persona_answer[:150]}...")

    print(
        "\n💡 KEY INSIGHT: Professional LLM applications use carefully tuned parameter\n"
        "   combinations. Start with these proven configs and adjust for your needs!\n"
        '   Exercise: Design parameters for a "Children\'s Science Tutor" bot.\n'
        "   Hint: Consider age-appropriate language, engagement, and educational goals."
    )


def demonstrate_real_world_applications(client: AzureOpenAI) -> None:
    """
    🌟 PUTTING IT ALL TOGETHER: Real-World Scenarios
    Now let's see how professionals combine all these concepts
    to build practical AI applications.
    """
    print_section_header("EXAMPLE 5: REAL-WORLD APPLICATIONS")
    print(textwrap.dedent(demonstrate_real_world_applications.__doc__))

    application_scenarios: list[RealWorldAppScenario] = [
        RealWorldAppScenario(
            scenario_name="📧 Email Response Assistant",
            system_prompt=(
                "You are a professional email assistant. Write polite, clear, "
                "and action-oriented responses. Always include next steps."
            ),
            model_params=CompletionModelParams(temperature=0.4, max_completion_tokens=120),
            user_input="A customer is asking about a delayed shipment and seems frustrated.",
            learning_point="Consistent, professional tone with limited length",
        ),
        RealWorldAppScenario(
            scenario_name="🎓 Personalized Learning Tutor",
            system_prompt=(
                "You are an encouraging math tutor for middle school students. "
                "Always ask follow-up questions to check understanding."
            ),
            model_params=CompletionModelParams(temperature=0.8, max_completion_tokens=180),
            user_input="I don't understand how to solve for x in: 2x + 5 = 13",
            learning_point="Educational engagement with moderate creativity",
        ),
        RealWorldAppScenario(
            scenario_name="📊 Data Analysis Explainer",
            system_prompt=(
                "You are a data scientist explaining insights to business stakeholders. "
                "Use clear metrics and actionable recommendations."
            ),
            model_params=CompletionModelParams(temperature=0.2, max_completion_tokens=200),
            user_input="Our website conversion rate dropped from 3.5% to 2.1% last month.",
            learning_point="Factual precision with structured business communication",
        ),
    ]

    for scenario in application_scenarios:
        print(f"\n🔧 {scenario.scenario_name}")
        print(f"   Learning Focus: {scenario.learning_point}")
        print(f"   System Setup: {scenario.system_prompt}")
        print(f"   Parameters: {scenario.model_params}")
        print(f"   User Input: '{scenario.user_input}'")

        scenario_answer = send_chat_completion(
            client,
            system_prompt=scenario.system_prompt,
            user_prompt=scenario.user_input,
            **scenario.model_params.as_api_kwargs(),
        )

        print(f"\n   🤖 AI Response:")
        print(textwrap.fill(scenario_answer, width=65, initial_indent="   ", subsequent_indent="   "))

    print(
        "\n💡 Exercise: Design your own application scenario\n"
        "   1. Choose a real problem you want to solve\n"
        "   2. Write a system prompt that defines the AI's role\n"
        "   3. Select parameters that match your needs\n"
        "   4. Test with example inputs\n"
        "   Hint: Start with problems you face in your daily work or studies"
    )


def print_learning_summary() -> None:
    print_section_header("🎉 CONGRATULATIONS! YOU'VE MASTERED LLM BASICS")

    print(textwrap.dedent("""\
        🧠 WHAT YOU'VE LEARNED:

        1. 🎭 SYSTEM PROMPTS are your most powerful tool
           • Define AI personality, expertise, and communication style
           • Critical for setting user expectations and response quality
           • Always be specific about role, tone, and output format

        2. 🌡️ TEMPERATURE controls creativity vs consistency
           • 0.0-0.3: Factual, repeatable, focused (great for technical tasks)
           • 0.5-1.0: Balanced creativity and coherence (conversations, content)
           • 1.0-2.0: Highly creative and diverse (art, brainstorming, fiction)

        3. 🔢 TOKENS manage cost, length, and focus
           • You pay for input AND output tokens
           • Lower limits force conciseness and save money
           • Higher limits allow detailed explanations

        4. 🎯 TOP_P refines word choice diversity
           • Works with temperature to fine-tune creativity
           • Lower values = more focused vocabulary choices
           • Professional tip: Adjust both temperature AND top_p together

        5. 🔧 PARAMETER COMBINATIONS solve real problems
           • Each use case needs its own optimal configuration
           • Start with proven professional configurations
           • Always test and iterate based on your specific needs

        🚀 NEXT STEPS FOR MASTERY:

        🔬 EXPERIMENT:
        • Try building your own customer service bot
        • Create a personalized learning assistant
        • Design a creative writing collaborator

        📚 LEARN MORE:
        • Explore prompt engineering techniques
        • Study fine-tuning for specialized tasks
        • Learn about retrieval-augmented generation (RAG)

        💼 APPLY IN PRACTICE:
        • Identify repetitive tasks in your work that AI could help with
        • Build proof-of-concept applications for your specific domain
        • Share your learnings with colleagues and iterate based on feedback

        Remember: The best way to master LLMs is through hands-on experimentation.
        Start small, measure results, and gradually build more complex applications!
    """))


if __name__ == "__main__":
    load_dotenv(override=True)
    azure_openai_client = AzureOpenAI()

    demonstrate_system_prompts(azure_openai_client)
    demonstrate_temperature(azure_openai_client)
    demonstrate_token_management(azure_openai_client)
    demonstrate_advanced_parameter_combinations(azure_openai_client)
    demonstrate_real_world_applications(azure_openai_client)
    print_learning_summary()
