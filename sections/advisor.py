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

    