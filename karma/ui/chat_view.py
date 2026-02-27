"""Chat view for Karma Streamlit UI (KARMA-006).

Renders a Streamlit chat interface that queries the knowledge base via
karma/chat.py (RAG pattern). Displays answer + source seed titles.
"""

from __future__ import annotations

import streamlit as st


def render_chat_view() -> None:
    """Render the Claude-powered chat interface inside Streamlit."""
    st.subheader("Chat with your knowledge base")
    st.caption(
        "Ask questions — Claude answers using your seeds as context and cites sources."
    )

    # Initialize session message history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display existing messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

    # Input box at the bottom
    user_input = st.chat_input("Ask anything about your seeds...")

    if user_input:
        # Show user message immediately
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_input, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call the RAG chat backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from karma.chat import answer

                    result = answer(user_input)
                    response_text = result["answer"]
                    sources = result.get("sources", [])
                except Exception as exc:  # noqa: BLE001
                    response_text = f"Error: {exc}"
                    sources = []

            st.markdown(response_text)
            if sources:
                st.caption("Sources: " + ", ".join(sources))

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": response_text, "sources": sources}
        )
