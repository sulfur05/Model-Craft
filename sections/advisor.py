import os
import textwrap
import streamlit as st

import pandas as pd
from dotenv import load_dotenv
load_dotenv()


def advisor_panel():
    """Right-hand side advisor panel with chat-like interaction."""
    if "advisor_messages" not in st.session_state:
        st.session_state["advisor_messages"] = []

    st.markdown("### Ask ModelCraft")

    st.caption(
        "Ask questions in simple language. The assistant will use the current "
        "dataset, target, and model choices as context."
    )

    #history
    for msg in st.session_state["advisor_messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Advisor:** {content}")

    #take input
    user_input = st.text_area(
        "Type your question here:",
        key="advisor_input",
        height=80,
        placeholder="E.g. Which model should I try next? Why is my accuracy low?",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        ask = st.button("Ask advisor")
    with col2:
        clear = st.button("Clear chat")

    if clear:
        st.session_state["advisor_messages"] = []
        # st.experimental_rerun()

    if ask and user_input.strip():
        # Append user message
        st.session_state["advisor_messages"].append(
            {"role": "user", "content": user_input.strip()}
        )

        context = _build_context_summary()
        full_prompt = textwrap.dedent(
            f"""
            You are a friendly ML tutor helping a beginner use a Streamlit app called ModelCraft.

            Current context:
            {context}

            User question:
            {user_input.strip()}

            Answer in simple, non-technical language and give concrete next steps.
            """
        ).strip()

        with st.spinner("Advisor is thinking..."):
            answer = _call_llm(full_prompt)

        st.session_state["advisor_messages"].append(
            {"role": "assistant", "content": answer}
        )
