from typing import Any, Dict, List

import streamlit as st

from graph.graph import app
from ingestion import sync_uploaded_file

st.set_page_config(page_title="Rag chat", layout="centered")
st.title("Rag chat")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Please upload and index a file before asking questions.",
        }
    ]

if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

if "index_result" not in st.session_state:
    st.session_state.index_result = None

if "debug_enabled" not in st.session_state:
    st.session_state.debug_enabled = True


def run_graph_with_debug(question: str) -> tuple[Dict[str, Any], List[str]]:
    debug_logs: List[str] = []
    final_state: Dict[str, Any] = {}

    for event in app.stream({"question": question}, stream_mode="updates"):
        for node_name, payload in event.items():
            payload = payload or {}
            final_state.update(payload)

            if node_name == "retrieve":
                docs_count = len(payload.get("documents", []))
                debug_logs.append(f"retrieve: fetched {docs_count} docs")
            elif node_name == "grade_documents":
                docs_count = len(payload.get("documents", []))
                transform_flag = payload.get("transform", False)
                debug_logs.append(
                    f"grade_documents: kept {docs_count} docs, transform={transform_flag}"
                )
            elif node_name == "transform":
                transformed_question = str(payload.get("question", "")).strip()
                transform_count = payload.get("transform_count", 0)
                debug_logs.append(
                    f"transform: rewrite #{transform_count} -> {transformed_question}"
                )
            elif node_name == "generate":
                generation = str(payload.get("generation", "")).strip()
                debug_logs.append(
                    f"generate: produced answer ({len(generation)} chars)"
                )
            elif node_name == "generic_response":
                debug_logs.append("generic_response: fallback response used")
            else:
                debug_logs.append(f"{node_name}: step completed")

    return final_state, debug_logs


with st.sidebar:
    st.subheader("Session")

    uploaded_file = st.file_uploader(
        "Knowledge file",
        type=["pdf", "txt", "md", "doc", "docx"],
        key="knowledge_file",
    )

    st.session_state.debug_enabled = st.checkbox(
        "Show debug trace", value=st.session_state.debug_enabled
    )

    if st.button("Index / Update File", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Indexing file..."):
                result = sync_uploaded_file(uploaded_file)
            st.session_state.indexed_file = uploaded_file.name
            st.session_state.index_result = result

    if st.session_state.index_result:
        result = st.session_state.index_result
        status = result.get("status", "indexed")
        chunks = result.get("chunks", 0)
        if status == "unchanged":
            st.info(
                f"Indexed: {st.session_state.indexed_file} ({chunks} chunks, no changes)"
            )
        else:
            st.success(f"Indexed: {st.session_state.indexed_file} ({chunks} chunks)")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your indexed file...")
if question:
    question = question.strip()

    if not question:
        st.error("Please enter a question.")
        st.stop()

    if not st.session_state.indexed_file:
        st.error("Please upload and index a file in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result, debug_logs = run_graph_with_debug(question)
                answer = str(result.get("generation", "")).strip()

            st.markdown(answer)

            if st.session_state.debug_enabled:
                with st.expander("Debug trace", expanded=False):
                    for log_line in debug_logs:
                        st.write(f"- {log_line}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error("Failed to generate response.")
            st.exception(e)
