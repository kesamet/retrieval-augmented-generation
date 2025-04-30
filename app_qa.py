import os

import torch
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from src import CFG
from src.embeddings import build_hyde_embeddings
from src.query_expansion import build_multiple_queries_expansion_chain
from src.retrievers import (
    create_base_retriever,
    create_rerank_retriever,
    create_compression_retriever,
)
from src.chains import create_question_answer_chain
from src.vectordbs import build_vectordb, delete_vectordb, load_faiss, load_chroma
from streamlit_app.pdf_display import get_doc_highlighted, display_pdf
from streamlit_app.utils import perform, cache_base_embeddings, cache_llm, cache_reranker

# Fixing the issue:
# Examining the path of torch.classes raised: Tried to instantiate class 'path.path’,
# but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

TITLE = "Retrieval QA"
st.set_page_config(page_title=TITLE, layout="wide")

EMBEDDING_FUNCTION = cache_base_embeddings()
RERANKER = cache_reranker()
LLM = cache_llm()
QA_CHAIN = create_question_answer_chain(LLM)
VECTORDB_PATH = CFG.VECTORDB[0].PATH


@st.cache_resource
def load_vectordb():
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(EMBEDDING_FUNCTION, VECTORDB_PATH)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(EMBEDDING_FUNCTION, VECTORDB_PATH)
    raise NotImplementedError


@st.cache_resource
def load_vectordb_hyde():
    hyde_embeddings = build_hyde_embeddings(LLM, EMBEDDING_FUNCTION)

    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(hyde_embeddings, VECTORDB_PATH)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(hyde_embeddings, VECTORDB_PATH)
    raise NotImplementedError


def load_retriever(_vectordb, retrieval_mode):
    if retrieval_mode == "Base":
        return create_base_retriever(_vectordb)
    if retrieval_mode == "Rerank":
        return create_rerank_retriever(_vectordb, RERANKER)
    if retrieval_mode == "Contextual compression":
        return create_compression_retriever(_vectordb, EMBEDDING_FUNCTION)
    raise NotImplementedError


def init_sess_state():
    if "last_form" not in st.session_state:
        st.session_state["last_form"] = list()

    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    if "last_response" not in st.session_state:
        st.session_state["last_response"] = dict()

    if "last_related" not in st.session_state:
        st.session_state["last_related"] = list()


def _format_text(text):
    return text.replace("$", r"\$")


def docqa():
    init_sess_state()

    with st.sidebar:
        st.header(TITLE)

        with st.expander("Models used"):
            st.info(f"LLM: `{CFG.LLM_PATH}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
            st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        uploaded_file = st.file_uploader("Upload a PDF and build VectorDB", type=["pdf"])
        if st.button("Build VectorDB"):
            if uploaded_file is None:
                st.error("No PDF uploaded")
                st.stop()

            if os.path.exists(VECTORDB_PATH):
                st.warning("Deleting existing VectorDB")
                delete_vectordb(VECTORDB_PATH, CFG.VECTORDB_TYPE)

            with st.spinner("Building VectorDB..."):
                perform(
                    build_vectordb,
                    uploaded_file.read(),
                    embedding_function=EMBEDDING_FUNCTION,
                )
                load_vectordb.clear()

        if not os.path.exists(VECTORDB_PATH):
            st.info("Please build VectorDB first.")
            st.stop()

        try:
            with st.status("Load VectorDB", expanded=False) as status:
                st.write("Loading VectorDB ...")
                vectordb = load_vectordb()
                st.write("Loading HyDE VectorDB ...")
                vectordb_hyde = load_vectordb_hyde()
                status.update(label="Loading complete!", state="complete", expanded=False)
            st.success("Reading from existing VectorDB")
        except Exception as e:
            st.error(e)
            st.stop()

    c0, c1 = st.columns(2)

    with c0.form("qa_form"):
        user_query = st.text_area("Your query")
        with st.expander("Settings"):
            mode = st.radio(
                "Mode",
                ["Retrieval only", "Retrieval QA"],
                index=1,
                help="""Retrieval only will output extracts related to your query immediately, \
                while Retrieval QA will output an answer to your query and will take a while on CPU.""",
            )
            retrieval_mode = st.radio(
                "Retrieval method",
                ["Base", "Rerank", "Contextual compression"],
                index=1,
            )
            use_hyde = st.checkbox("Use HyDE")

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

        retriever = load_retriever(
            vectordb_hyde if use_hyde else vectordb,
            retrieval_mode,
        )

        if mode == "Retrieval only":
            with c0:
                with st.spinner("Retrieving ..."):
                    source_documents = retriever.invoke(user_query)

            st.session_state.last_response = {
                "query": user_query,
                "source_documents": source_documents,
            }

            chain = build_multiple_queries_expansion_chain(LLM)
            res = chain.invoke(user_query)
            st.session_state.last_related = [x.strip() for x in res.split("\n") if x.strip()]
        else:
            st_callback = StreamlitCallbackHandler(
                parent_container=c0.container(),
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
            )
            source_documents = retriever.invoke(user_query)
            answer = QA_CHAIN.invoke(
                {
                    "context": source_documents,
                    "question": user_query,
                },
                config={"callbacks": [st_callback]},
            )

            st.session_state.last_response = {
                "query": user_query,
                "answer": answer,
                "source_documents": source_documents,
            }

    if st.session_state.last_response:
        with c0:
            st.warning(f"##### {st.session_state.last_query}")
            if st.session_state.last_response.get("answer") is not None:
                st.success(_format_text(st.session_state.last_response["answer"]))

            if st.session_state.last_related:
                st.write("#### Related")
                for r in st.session_state.last_related:
                    st.write(f"```\n{r}\n```")

        with c1:
            st.write("#### Sources")
            for row in st.session_state.last_response["source_documents"]:
                st.write(f"**Page {row.metadata['page_number']}**")
                st.info(_format_text(row.page_content))

            # Display PDF
            st.write("---")
            _display_pdf_from_docs(st.session_state.last_response["source_documents"])


def _display_pdf_from_docs(source_documents):
    n = len(source_documents)
    i = st.radio("View in PDF", list(range(n)), format_func=lambda x: f"Extract {x + 1}")
    row = source_documents[i]
    try:
        extracted_doc, page_nums = get_doc_highlighted(row.metadata["source"], row.page_content)
        if extracted_doc is None:
            st.error("No page found")
        else:
            display_pdf(extracted_doc, page_nums[0] + 1)
    except Exception as e:
        st.error(e)


if __name__ == "__main__":
    docqa()
