import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from src import CFG
from src.app_utils import perform
from src.embeddings import build_embeddings
from src.llm import build_llm
from src.retrieval_qa import build_retrieval_qa
from src.vectordb import build_vectordb

st.set_page_config(page_title="Retrieval QA")

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None

if "last_response" not in st.session_state:
    st.session_state["last_response"] = None


@st.cache_resource
def load_vectordb() -> FAISS:
    embeddings = build_embeddings()
    return FAISS.load_local(CFG.VECTORDB_FAISS_PATH, embeddings)


@st.cache_resource
def load_retrieval_qa() -> RetrievalQA:
    embeddings = build_embeddings()
    llm = build_llm()
    vectordb = FAISS.load_local(CFG.VECTORDB_FAISS_PATH, embeddings)
    return build_retrieval_qa(llm, vectordb)


def doc_qa():
    with st.sidebar:
        st.header("Document Question Answering using quantized LLM on CPU")
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

        if st.session_state.uploaded_filename is not None:
            st.info(f"Current document: {st.session_state.uploaded_filename}")

        try:
            with st.status("Load retrieval_qa", expanded=False) as status:
                st.write("Loading vectordb...")
                vectordb = load_vectordb()
                st.write("Loading retrieval_qa...")
                retrieval_qa = load_retrieval_qa()
                status.update(label="Loading complete!", state="complete", expanded=False)

            st.success("Reading from existing VectorDB")
        except Exception:
            st.error("No existing VectorDB found")

    with st.form("qa_form"):
        user_query = st.text_area("Your query")
        mode = st.radio(
            "mode",
            ["Retrieval QA", "Retrieval only"],
            label_visibility="hidden",
            help="""Retrieval only will output extracts related to your query immediately, \
while Retrieval QA will output an answer to your query and will take a while on CPU.""",
        )

        submitted = st.form_submit_button("Query")
        if submitted:
            if mode == "Retrieval only":
                st.session_state.last_response = {
                    "query": user_query,
                    "source_documents": vectordb.similarity_search(user_query, k=2)
                }
            else:
                with st.spinner("Getting response..."):
                    st.session_state.last_response = retrieval_qa(user_query)

    if st.session_state.last_response is not None:
        st.success(st.session_state.last_response["query"])

        if st.session_state.last_response.get("result") is not None:
            st.info(st.session_state.last_response["result"])

        st.write("#### Retrieved extracts")
        for row in st.session_state.last_response["source_documents"]:
            page_content = row.page_content
            page = row.metadata["page"]

            st.write(f"**Page {page}**")
            st.info(page_content)


if __name__ == "__main__":
    doc_qa()
