import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.vectorstores.faiss import FAISS

from src import CFG
from src.embeddings import build_hyde_embeddings
from src.retrieval_qa import build_retrieval_qa
from src.vectordb import build_vectordb
from streamlit_app.pdf_display import get_doc_highlighted, display_pdf
from streamlit_app.utils import load_base_embeddings, load_llm

st.set_page_config(page_title="Retrieval QA", layout="wide")

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None

MODE_LIST = ["Retrieval only", "Retrieval QA", "Retrieval QA with HyDE"]
DEFAULT_MODE = 0
if "last_mode" not in st.session_state:
    st.session_state["last_mode"] = MODE_LIST[DEFAULT_MODE]

if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
HYDE_EMBEDDINGS = build_hyde_embeddings(LLM, BASE_EMBEDDINGS)


@st.cache_resource
def load_vectordb():
    return FAISS.load_local(CFG.VECTORDB_FAISS_PATH, BASE_EMBEDDINGS)


@st.cache_resource
def load_vectordb_hyde():
    return FAISS.load_local(CFG.VECTORDB_FAISS_PATH, HYDE_EMBEDDINGS)


def doc_qa():
    with st.sidebar:
        st.header("DocQA using quantized LLM on CPU")
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

        if st.session_state.uploaded_filename is not None:
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
        except Exception:
            st.error("No existing VectorDB found")

    c0, c1 = st.columns(2)

    with c0:
        with st.form("qa_form"):
            user_query = st.text_area("Your query")
            mode = st.radio(
                "Mode",
                MODE_LIST,
                index=DEFAULT_MODE,
                help="""Retrieval only will output extracts related to your query immediately, \
                while Retrieval QA will output an answer to your query and will take a while on CPU.""",
            )

            submitted = st.form_submit_button("Query")
            if submitted:
                if user_query == "" or user_query is None:
                    st.error("Please enter a query.")

    if (user_query != "" or user_query is None) and (
        st.session_state.last_mode != mode or st.session_state.last_query != user_query
    ):
        st.session_state.last_mode = mode
        st.session_state.last_query = user_query

        if mode == "Retrieval only":
            st.session_state.last_response = {
                "query": user_query,
                "source_documents": vectordb.similarity_search(user_query, k=4),
            }
        else:
            if mode == "Retrieval QA":
                retrieval_qa = build_retrieval_qa(LLM, vectordb)
            else:
                retrieval_qa = build_retrieval_qa(LLM, vectordb_hyde)

            st_callback = StreamlitCallbackHandler(
                parent_container=c0.container(),
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
            )
            st.session_state.last_response = retrieval_qa(
                user_query, callbacks=[st_callback]
            )
            st_callback._complete_current_thought()

    if st.session_state.last_response is not None:
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
