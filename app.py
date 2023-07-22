import requests
import streamlit as st

from src.app_utils import get_pdf_display
from src.vectordb import build_vectordb

API_URL = "http://127.0.0.1:8001"


def retrieve(user_query, max_new_tokens, temperature, topk):
    payload = {
        "user_query": user_query,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "topk": topk,
    }
    headers = {"Content-Tyoe": "application/json"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def doc_qa():
    st.header("Document Question Answering")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a PDF.", type=["pdf"])
        if st.button("Build VectorDB"):
            if uploaded_file is None:
                st.error("No PDF uploaded")
            else:
                with st.spinner("Building VectorDB..."):
                    build_vectordb(uploaded_file.name)

    retrieved = None
    with st.form("qa_form"):
        user_query = st.text_area("Your query")
        with st.expander("Parameters"):
            max_new_tokens = st.slider(
                "Max new tokens",
                min_value=10,
                max_value=1024,
                value=128,
                help="Max output text length",
            )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                help="Higher values produce more diverse outputs",
            )
            topk = st.slider(
                "Top K",
                min_value=1,
                max_value=10,
                value=2,
                help="Sample from a shortlist of top-k documents",
            )

        submitted = st.form_submit_button("Query")
        if submitted:
            with st.spinner("Getting response. Please wait..."):
                retrieved = retrieve(user_query, max_new_tokens, temperature, topk)

    if retrieved is not None:
        st.info(retrieved["result"])

        st.write("#### Retrieved documents")
        for row in retrieved["source_documents"]:
            page_content = row["page_content"]
            page = row["metadata"]["page"]

            st.info(page_content)


if __name__ == "__main__":
    doc_qa()
