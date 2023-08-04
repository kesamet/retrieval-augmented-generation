from typing import List

import streamlit as st
from langchain.vectorstores import FAISS

from src import CFG
from src.app_utils import load_embeddings, load_llm, perform
from src.retrieval_qa import build_retrieval_chain
from src.vectordb import build_vectordb

st.set_page_config(page_title="Conversational Retrieval QA")

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = ""

EMBEDDINGS = load_embeddings()
LLM = load_llm()


def init_chat_history() -> None:
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.source_documents = []


def retrieve_qa(user_query: str, chat_history: List[str]) -> dict:
    """Retrieve from the vectordb and answer user query in a conversational manner.

    Args:
        user_query (str): The user query.

    Returns:
        dict: The response from retrieval_chain.
    """
    vectordb = FAISS.load_local(CFG.VECTORDB_FAISS_PATH, EMBEDDINGS)
    retrieval_chain = build_retrieval_chain(LLM, vectordb)
    result = retrieval_chain({"question": user_query, "chat_history": chat_history})
    return result


def main():
    with st.sidebar:
        st.title("Document Question Answering")

        uploaded_file = st.file_uploader(
            "Upload a PDF and build VectorDB", type=["pdf"]
        )
        if st.button("Build VectorDB"):
            if uploaded_file is None:
                st.error("No PDF uploaded")
            else:
                with st.spinner("Building VectorDB..."):
                    perform(build_vectordb, uploaded_file.read())
                st.session_state.uploaded_filename = uploaded_file.name

        if st.session_state.uploaded_filename != "":
            st.info(f"Current document: {st.session_state.uploaded_filename}")
        else:
            try:
                _ = FAISS.load_local(CFG.VECTORDB_FAISS_PATH, EMBEDDINGS)
                st.warning("Reading from existing VectorDB")
            except Exception:
                st.warning("No existing VectorDB found")

    st.sidebar.write("---")
    init_chat_history()

    result = None
    if user_query := st.chat_input("Your input"):
        with st.spinner("Getting response. Please wait ..."):
            result = retrieve_qa(user_query, st.session_state.chat_history)

    # Display chat history
    if result is not None:
        st.session_state.chat_history.extend(
            [(user_query, result["answer"], result["source_documents"])]
        )
        for query, answer, source_documents in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                st.markdown(answer)

                with st.expander("Retrieved extracts"):
                    for row in source_documents:
                        page_content = row.page_content
                        page = row.metadata["page"]
                        st.write(f"**Page {page}**")
                        st.info(page_content)


if __name__ == "__main__":
    main()
