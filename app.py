import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

from src import CFG
from src.embeddings import build_hyde_embeddings
from src.retrieval_qa import build_retrieval_qa
from src.vectordb import build_vectordb, load_faiss, load_chroma
from streamlit_app.pdf_display import get_doc_highlighted, display_pdf
from streamlit_app.utils import (
    load_base_embeddings,
    load_llm,
    load_retriever,
)

st.set_page_config(page_title="Retrieval QA", layout="wide")

LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
HYDE_EMBEDDINGS = build_hyde_embeddings(LLM, BASE_EMBEDDINGS)


@st.cache_resource
def load_vectordb():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS)


@st.cache_resource
def load_vectordb_hyde():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(HYDE_EMBEDDINGS)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(HYDE_EMBEDDINGS)


def init_sess_state():
    if "uploaded_filename" not in st.session_state:
        st.session_state["uploaded_filename"] = ""

    if "last_form" not in st.session_state:
        st.session_state["last_form"] = list()

    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    if "last_response" not in st.session_state:
        st.session_state["last_response"] = dict()


def doc_qa():
    init_sess_state()

    with st.sidebar:
        st.header("DocQA using quantized LLM")
        st.info(f"Running on {CFG.DEVICE}")
        st.info(f"LLM: {CFG.LLM_MODEL_PATH}")
        st.info(f"Embeddings: {CFG.EMBEDDINGS_MODEL_PATH}")
        st.info(f"Reranker: {CFG.RERANKER_NAME}")

        uploaded_file = st.file_uploader(
            "Upload a PDF and build VectorDB", type=["pdf"]
        )
        if st.button("Build VectorDB"):
            if uploaded_file is None:
                st.error("No PDF uploaded")
            else:
                uploaded_filename = f"./data/{uploaded_file.name}"
                with open(uploaded_filename, "wb") as f:
                    f.write(uploaded_file.getvalue())
                with st.spinner("Building VectorDB..."):
                    build_vectordb(uploaded_filename)
                st.session_state.uploaded_filename = uploaded_filename

        if st.session_state.uploaded_filename != "":
            st.info(f"Current document: {st.session_state.uploaded_filename}")

        try:
            with st.status("Load VectorDB", expanded=False) as status:
                st.write("Loading VectorDB ...")
                vectordb = load_vectordb()
                st.write("Loading HyDE VectorDB ...")
                vectordb_hyde = load_vectordb_hyde()
                status.update(
                    label="Loading complete!", state="complete", expanded=False
                )

            st.success("Reading from existing VectorDB")
        except Exception as e:
            st.error(f"No existing VectorDB found: {e}")

    c0, c1 = st.columns(2)

    with c0.form("qa_form"):
        user_query = st.text_area("Your query")
        mode = st.radio(
            "Mode",
            ["Retrieval only", "Retrieval QA"],
            help="""Retrieval only will output extracts related to your query immediately, \
            while Retrieval QA will output an answer to your query and will take a while on CPU.""",
        )
        retrieval_mode = st.radio("Retrieval method", ["Base", "Rerank", "Contextual compression"])
        use_hyde = st.checkbox("Use HyDE (for Retrieval QA only)")

        submitted = st.form_submit_button("Query")
        if submitted:
            if user_query == "":
                st.error("Please enter a query.")

    if user_query != "" and (
        st.session_state.last_query != user_query
        or st.session_state.last_form != [mode, retrieval_mode, use_hyde]
    ):
        st.session_state.last_query = user_query
        st.session_state.last_form = [mode, retrieval_mode, use_hyde]

        if mode == "Retrieval only":
            retriever = load_retriever(vectordb, retrieval_mode, BASE_EMBEDDINGS)
            relevant_docs = retriever.get_relevant_documents(user_query)
            with c0:
                with st.spinner("Retrieving ..."):
                    relevant_docs = retriever.get_relevant_documents(user_query)

            st.session_state.last_response = {
                "query": user_query,
                "source_documents": relevant_docs,
            }
        else:
            db = vectordb_hyde if use_hyde else vectordb
            retriever = load_retriever(db, retrieval_mode, BASE_EMBEDDINGS)
            retrieval_qa = build_retrieval_qa(LLM, retriever)

            st_callback = StreamlitCallbackHandler(
                parent_container=c0.container(),
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
            )
            st.session_state.last_response = retrieval_qa(
                user_query, callbacks=[st_callback]
            )
            st_callback._complete_current_thought()

    if st.session_state.last_response:
        with c0:
            st.warning(f"Query: {st.session_state.last_query}")
            if st.session_state.last_response.get("result") is not None:
                st.success(st.session_state.last_response["result"])

        with c1:
            st.write("#### Retrieved extracts")
            for row in st.session_state.last_response["source_documents"]:
                st.write("**Page {}**".format(row.metadata["page"] + 1))
                st.info(row.page_content)

            # Display PDF
            st.write("---")
            n = len(st.session_state.last_response["source_documents"])
            i = st.radio(
                "View in PDF", list(range(n)), format_func=lambda x: f"Extract {x + 1}"
            )
            row = st.session_state.last_response["source_documents"][i]
            try:
                extracted_doc, page_nums = get_doc_highlighted(
                    row.metadata["source"], row.page_content
                )
                if extracted_doc is None:
                    st.error("No page found")
                else:
                    display_pdf(extracted_doc, page_nums[0] + 1)
            except Exception as e:
                st.error(e)


if __name__ == "__main__":
    doc_qa()
