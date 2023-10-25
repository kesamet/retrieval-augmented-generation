import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

from src import CFG
from src.embeddings import build_base_embeddings
from src.llms import build_llm
from src.retrieval_qa import build_retrieval_chain
from src.vectordb import build_vectordb, load_faiss, load_chroma
from streamlit_app.utils import perform

st.set_page_config(page_title="Conversational Retrieval QA")

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None


def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["source_documents"] = list()


@st.cache_resource
def load_retrieval_chain():
    embeddings = build_base_embeddings()
    if CFG.VECTORDB_TYPE == "faiss":
        vectordb = load_faiss(embeddings)
    elif CFG.VECTORDB_TYPE == "chroma":
        vectordb = load_chroma(embeddings)
    llm = build_llm()
    return build_retrieval_chain(llm, vectordb)


def doc_conv_qa():
    with st.sidebar:
        st.title("Conversational DocQA using quantized LLM on CPU")

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

        try:
            with st.status("Load retrieval_chain", expanded=False) as status:
                st.write("Loading retrieval_chain...")
                retrieval_chain = load_retrieval_chain()
                status.update(
                    label="Loading complete!", state="complete", expanded=False
                )

            st.success("Reading from existing VectorDB")
        except Exception:
            st.error("No existing VectorDB found")

    st.sidebar.write("---")
    init_chat_history()

    # Display chat history
    for (question, answer), source_documents in zip(
        st.session_state.chat_history, st.session_state.source_documents
    ):
        if question != "":
            with st.chat_message("user"):
                st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

            with st.expander("Retrieved extracts"):
                for row in source_documents:
                    st.write("**Page {}**".format(row.metadata["page"] + 1))
                    st.info(row.page_content)

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
            response = retrieval_chain(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
                callbacks=[st_callback],
            )
            st_callback._complete_current_thought()
            st.markdown(response["answer"])

            with st.expander("Retrieved extracts"):
                for row in response["source_documents"]:
                    st.write("**Page {}**".format(row.metadata["page"] + 1))
                    st.info(row.page_content)

            st.session_state.chat_history.append(
                (response["question"], response["answer"])
            )
            st.session_state.source_documents.append(response["source_documents"])


if __name__ == "__main__":
    doc_conv_qa()
