import streamlit as st
from langchain.vectorstores import FAISS

from src import CFG
from src.app_utils import load_embeddings, load_llm, perform
from src.retrieval_qa import build_retrieval_qa
from src.vectordb import build_vectordb

st.set_page_config(page_title="Retrieval QA")

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = ""

EMBEDDINGS = load_embeddings()
LLM = load_llm()


def retrieve(user_query: str, k: int) -> list:
    """
    Retrieve documents from vectordb by similarity search.

    Args:
        user_query (str): The user query.
        k (int): The number of documents to retrieve.

    Returns:
        list: A list of retrieved documents.
    """
    vectordb = FAISS.load_local(CFG.VECTORDB_FAISS_PATH, EMBEDDINGS)
    docs = vectordb.similarity_search(user_query, k=k)
    return docs


def retrieve_qa(user_query: str) -> dict:
    """Retrieve from the vectordb and answer user query in a conversational manner.

    Args:
        user_query (str): The user query.

    Returns:
        dict: The response from the retrieval_qa model.
    """
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
            try:
                _ = FAISS.load_local(CFG.VECTORDB_FAISS_PATH, EMBEDDINGS)
                st.warning("Reading from existing VectorDB")
            except Exception:
                st.warning("No existing VectorDB found")

    retrieved = None
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
