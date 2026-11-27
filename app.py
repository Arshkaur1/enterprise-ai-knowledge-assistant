
"""Streamlit interface for enterprise document intelligence."""

import os

import requests
import streamlit as st

API_URL = os.getenv(
    "KNOWLEDGE_ASSISTANT_API",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Enterprise AI Knowledge Assistant")

st.caption(
    "Local LLM-powered document intelligence "
    "with evidence retrieval and source citations."
)

STRATEGIES = {
    "Dense semantic search": "dense",
    "BM25 lexical search": "bm25",
    "Hybrid retrieval": "hybrid",
    "Dense + cross-encoder reranking": "dense_reranked",
}

with st.sidebar:
    st.header("Retrieval settings")

    selected = st.selectbox(
        "Retrieval strategy",
        list(STRATEGIES),
    )

    top_k = st.slider(
        "Maximum retrieved chunks",
        min_value=1,
        max_value=5,
        value=2,
    )

    st.divider()

    st.subheader("Knowledge base")

    st.write("• Human Resources policies")
    st.write("• IT Security policies")
    st.write("• Finance policies")

    st.divider()

    st.subheader("Architecture")

    st.caption(
        "Streamlit → FastAPI → Retrieval → "
        "Ollama → Response validation"
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    try:
        health = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )

        if health.ok:
            st.success("API reachable")
        else:
            st.error("API unavailable")

    except requests.RequestException:
        st.error("API unavailable")


def render_response(result):
    st.markdown(result.get("answer", ""))

    status = result.get("status", "unknown")

    if status == "insufficient_evidence":
        st.warning(
            "Insufficient evidence in the knowledge base."
        )
    elif status != "answered":
        st.error(f"Response status: {status}")

    citations = result.get("citations", [])

    if citations:
        with st.expander("Sources and citations"):
            for citation in citations:
                st.markdown(
                    f"**[{citation['citation_id']}] "
                    f"{citation['source']}**"
                )

                st.write(
                    "Category:",
                    citation.get("category"),
                )

                if citation.get("page_number") is not None:
                    st.write(
                        "Page:",
                        citation["page_number"],
                    )

                score_type = result.get(
                    "retrieval_score_type",
                    "score",
                )

                st.caption(
                    f"{score_type}: "
                    f"{citation['score']:.4f}"
                )

                st.caption(
                    f"Chunk: {citation['chunk_id']}"
                )

    columns = st.columns(3)

    columns[0].metric(
        "Total latency",
        f"{result.get('total_latency_ms', 0):.0f} ms",
    )

    columns[1].metric(
        "Retrieved chunks",
        result.get("retrieved_count", 0),
    )

    columns[2].metric(
        "Citations",
        len(citations),
    )

    with st.expander("Execution details"):
        st.json(result)


if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.subheader("Try asking")

    st.markdown(
        """
        - How many annual leave days do employees receive?
        - What is the minimum password length?
        - What is the maximum daily meal reimbursement?
        - What is the maternity leave policy?
        """
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_response(message["result"])
        else:
            st.markdown(message["content"])

if query := st.chat_input(
    "Ask a question about enterprise policies..."
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "query": query,
                        "strategy": STRATEGIES[selected],
                        "top_k": top_k,
                    },
                    timeout=120,
                )

                response.raise_for_status()
                result = response.json()

                render_response(result)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "result": result,
                    }
                )

            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
