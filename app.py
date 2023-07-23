import streamlit as st
from langchain.vectorstores import FAISS

from src import CFG
from src.embeddings import build_embeddings
from src.llm import build_llm
from src.retrieval_qa import build_retrieval_qa
from src.utils import perform
from src.vectordb import build_vectordb

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = ""


@st.cache_resource
def _load_embeddings():
    return build_embeddings()


@st.cache_resource
def _load_llm():
    return build_llm()


EMBEDDINGS = _load_embeddings()
LLM = _load_llm()


def retrieve(user_query, k):
    vectordb = FAISS.load_local(CFG.VECTORDB_FAISS_PATH, EMBEDDINGS)
    docs = vectordb.similarity_search(user_query, k=k)
    return docs


def retrieve_qa(user_query):
    vectordb = FAISS.load_local(CFG.VECTORDB_FAISS_PATH, EMBEDDINGS)
    retrieval_qa = build_retrieval_qa(LLM, vectordb)
    response = retrieval_qa({"query": user_query})
    return response


def doc_qa():
    with st.sidebar:
        st.header("Document Question Answering")

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
            st.warning("Reading from existing VectorDB")

    retrieved = None
    with st.form("qa_form"):
        user_query = st.text_area("Your query")
        mode = st.radio(
            "mode", ["Retrieval QA", "Retrieve only"], label_visibility="hidden"
        )

        submitted = st.form_submit_button("Query")
        if submitted:
            if mode == "Retrieve only":
                retrieved = {"source_documents": retrieve(user_query, k=2)}
            else:
                with st.spinner("Getting response. Please wait..."):
                    retrieved = retrieve_qa(user_query)

    if retrieved is not None:
        if retrieved.get("result") is not None:
            st.info(retrieved["result"])

        st.write("#### Retrieved extracts")
        for row in retrieved["source_documents"]:
            page_content = row.page_content
            page = row.metadata["page"]

            st.write(f"**Page {page}**")
            st.info(page_content)


if __name__ == "__main__":
    doc_qa()
