import os
from collections import deque

import torch
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from src import CFG, logger
from src.chains import create_condense_question_chain, create_question_answer_chain
from src.llms import load_llm
from src.memory import trim_memory
from src.retrievers import create_rerank_retriever
from src.vectordbs import build_vectordb, delete_vectordb, load_vectordb
from streamlit_app.utils import process, cache_base_embeddings, cache_llm, cache_reranker
from streamlit_app.output_formatter import replace_special

# Fixing the issue:
# Examining the path of torch.classes raised: Tried to instantiate class 'path.path',
# but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

TITLE = "Conversational QA"
st.set_page_config(page_title=TITLE)

if CFG.LLM_PROVIDER == "llamacpp":
    LLM = cache_llm()
else:
    LLM = load_llm()

EMBEDDING_FUNCTION = cache_base_embeddings()
RERANKER = cache_reranker()
CONDENSE_QUESTION_CHAIN = create_condense_question_chain(LLM)
QA_CHAIN = create_question_answer_chain(LLM)
VECTORDB_PATH = CFG.VECTORDB[0].PATH


@st.cache_resource
def cache_vectordb(vectordb_config: dict):
    return load_vectordb(EMBEDDING_FUNCTION, vectordb_config["PATH"])


def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Chat", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = deque([])
        st.session_state["num_words"] = deque([])
        st.session_state["display_history"] = [("", "Hello! How can I help you?", None)]


def print_docs(source_documents):
    for row in source_documents:
        if row.metadata.get("page_number"):
            st.write(f"**Page {row.metadata['page_number']}**")
        st.info(replace_special(row.page_content))


def convqa():
    with st.sidebar:
        st.title(TITLE)

        with st.expander("Models used"):
            st.info(f"LLM: `{CFG.LLM}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS}`")
            st.info(f"Reranker: `{CFG.RERANKER}`")

        uploaded_file = st.file_uploader("Upload a PDF and build VectorDB", type=["pdf"])
        if st.button("Build VectorDB"):
            if uploaded_file is None:
                st.error("No PDF uploaded")
                st.stop()

            if os.path.exists(VECTORDB_PATH):
                st.warning("Deleting existing VectorDB")
                delete_vectordb(VECTORDB_PATH, CFG.VECTORDB_TYPE)

            with st.spinner("Building VectorDB..."):
                process(
                    build_vectordb,
                    uploaded_file.read(),
                    embedding_function=EMBEDDING_FUNCTION,
                )
                cache_vectordb.clear()

        if not os.path.exists(VECTORDB_PATH):
            st.info("Please build VectorDB first.")
            st.stop()

        try:
            with st.status("Load retrieval chain", expanded=False) as status:
                st.write("Loading retrieval chain...")
                vectordb = cache_vectordb()
                RETRIEVER = create_rerank_retriever(vectordb, RERANKER)
                status.update(label="Loading complete!", state="complete", expanded=False)
            st.success("Reading from existing VectorDB")
        except Exception as e:
            st.error(e)
            st.stop()

    st.sidebar.write("---")
    init_chat_history()

    # Display chat history
    for question, answer, source_documents in st.session_state.display_history:
        if question != "":
            with st.chat_message("user"):
                st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(replace_special(answer))

            if source_documents is not None:
                with st.expander("Sources"):
                    print_docs(source_documents)

    if user_query := st.chat_input("Your query"):
        with st.chat_message("user"):
            st.markdown(user_query)

    if user_query is not None:
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(
                parent_container=st.container(),
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
            )

            question = CONDENSE_QUESTION_CHAIN.invoke(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
            )
            logger.info(question)
            source_documents = RETRIEVER.invoke(question)
            answer = QA_CHAIN.invoke(
                {
                    "context": source_documents,
                    "question": question,
                },
                config={"callbacks": [st_callback]},
            )

            st.success(replace_special(answer))
            with st.expander("Sources"):
                print_docs(source_documents)

            trim_memory(
                (user_query, answer), st.session_state.chat_history, st.session_state.num_words
            )
            st.session_state.display_history.append((user_query, answer, source_documents))


if __name__ == "__main__":
    convqa()
