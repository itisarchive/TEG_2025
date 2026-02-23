#!/usr/bin/env python3
"""
🤖 Streamlit Chatbot — Azure OpenAI
====================================

A simple chatbot web application built with **Streamlit** and **Azure OpenAI**.
The chatbot acts as a polite academic teacher answering questions in hip-hop style.

Required environment variables (see .env.example):
    AZURE_OPENAI_ENDPOINT  – your Azure OpenAI resource URL
    AZURE_OPENAI_API_KEY   – your Azure OpenAI API key
    OPENAI_API_VERSION     – API version, e.g. "2025-04-01-preview"

Run:
    streamlit run 2_Your_own_chatbot.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai.lib.azure import AzureOpenAI

load_dotenv(override=True)

SYSTEM_PROMPT = "You are a polite academic teacher answering students' questions in a hip-hop style"
AVAILABLE_MODELS: list[str] = ["gpt-4o-mini", "gpt-4o", "gpt-35-turbo"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]


def create_azure_openai_client() -> AzureOpenAI:
    """Create an AzureOpenAI client using environment variables.

    The SDK reads AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT and
    OPENAI_API_VERSION from the environment automatically.
    """
    required_env_vars = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION")
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        st.stop()

    return AzureOpenAI()


def build_initial_messages() -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Settings")

        selected_model = st.selectbox(
            label="Choose Model:",
            options=AVAILABLE_MODELS,
            index=0,
        )
        st.session_state["selected_model"] = selected_model

        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = build_initial_messages()
            st.rerun()

        st.info("💡 **Tip**: This chatbot combines academic knowledge with hip-hop flair!")


def display_chat_history() -> None:
    for chat_message in st.session_state["chat_history"]:
        if chat_message["role"] == "system":
            continue
        with st.chat_message(name=chat_message["role"]):
            st.markdown(chat_message["content"])


def stream_assistant_response(azure_client: AzureOpenAI) -> None:
    with st.chat_message(name="assistant"):
        try:
            completion_stream = azure_client.chat.completions.create(
                model=st.session_state["selected_model"],
                messages=st.session_state["chat_history"],
                stream=True,
                temperature=0.7,
                max_tokens=1000,
            )
            streamed_response = st.write_stream(completion_stream)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": streamed_response}
            )
        except Exception as completion_error:
            st.error(f"Error generating response: {completion_error}")
            st.info("Please check your API key and try again.")


st.set_page_config(
    page_title="My Own Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 My Own Chatbot!")
st.caption("A polite academic teacher answering questions in hip-hop style")

azure_openai_client = create_azure_openai_client()

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = DEFAULT_MODEL

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = build_initial_messages()

render_sidebar()
display_chat_history()

if user_input := st.chat_input("Hi, what's up?"):
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    with st.chat_message(name="user"):
        st.markdown(user_input)

    stream_assistant_response(azure_openai_client)
