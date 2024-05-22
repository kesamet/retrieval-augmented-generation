import os

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from src import CFG
from src.retrieval_qa import build_conv_rag_chain
from src.vectordb import build_vectordb, delete_vectordb, load_faiss, load_chroma
from streamlit_app.utils import perform, load_base_embeddings, load_llm, load_reranker

st.set_page_config(page_title="Conversational RAG")

LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()


@st.cache_resource
def load_vectordb():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)
    raise NotImplementedError


def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Chat", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["display_history"] = [("", "Hello! How can I help you?", None)]


def print_docs(source_documents):
    for row in source_documents:
        st.write(f"**Page {row.metadata['page_number']}**")
        st.info(row.page_content)


def doc_conv_qa():
    with st.sidebar:
        st.title("Conversational RAG with quantized LLM")

        with st.expander("Models used"):
            st.info(f"LLM: `{CFG.LLM_PATH}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
            st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        uploaded_file = st.file_uploader(
            "Upload a PDF and build VectorDB", type=["pdf"]
        )
        if st.button("Build VectorDB"):
            if uploaded_file is None:
                st.error("No PDF uploaded")
                st.stop()

            if os.path.exists(CFG.VECTORDB_PATH):
                st.warning("Deleting existing VectorDB")
                delete_vectordb(CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)

            with st.spinner("Building VectorDB..."):
                perform(
                    build_vectordb,
                    uploaded_file.read(),
                    embedding_function=BASE_EMBEDDINGS,
                )
                load_vectordb.clear()

        if not os.path.exists(CFG.VECTORDB_PATH):
            st.info("Please build VectorDB first.")
            st.stop()

        try:
            with st.status("Load retrieval chain", expanded=False) as status:
                st.write("Loading retrieval chain...")
                vectordb = load_vectordb()
                rag_chain = build_conv_rag_chain(vectordb, RERANKER, LLM)
                status.update(
                    label="Loading complete!", state="complete", expanded=False
                )
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
            st.markdown(answer)

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
            response = rag_chain.invoke(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
                config={"callbacks": [st_callback]},
            )
            st_callback._complete_current_thought()

            answer = response["answer"]
            source_documents = response["source_documents"]

            st.markdown(answer)

            with st.expander("Sources"):
                print_docs(source_documents)

            st.session_state.chat_history.append((user_query, answer))
            st.session_state.display_history.append(
                (user_query, answer, source_documents)
            )


if __name__ == "__main__":
    doc_conv_qa()
