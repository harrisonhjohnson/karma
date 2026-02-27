"""Karma Streamlit app entry point (KARMA-005).

Two tabs:
  - Graph: interactive pyvis network of seeds and roots
  - Chat:  Claude-powered Q&A over the knowledge base (KARMA-006)

Entry point: karma-ui (registered in pyproject.toml)
"""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Karma",
        page_icon="K",
        layout="wide",
    )

    st.title("Karma")
    st.caption("Personal knowledge graph")

    tab_graph, tab_chat = st.tabs(["Graph", "Chat"])

    with tab_graph:
        from karma.ui.graph_view import render_graph_view
        render_graph_view()

    with tab_chat:
        try:
            from karma.ui.chat_view import render_chat_view
            render_chat_view()
        except ImportError:
            st.info("Chat interface coming soon.")


if __name__ == "__main__":
    main()
