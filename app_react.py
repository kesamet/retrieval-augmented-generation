# TODO: allow pdf upload

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import AgentExecutor

from src import CFG
from src.agents import create_react_agent, DEFAULT_REACT_TEMPLATE
from src.prompt_templates import CHAT_FORMAT
from src.retrieval_qa import build_rerank_retriever
from src.vectordb import load_faiss
from streamlit_app.utils import load_base_embeddings, load_llm, load_reranker

st.set_page_config(page_title="ReAct RAG")

LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()


@st.cache_resource
def _load_faiss(index_path):
    return load_faiss(BASE_EMBEDDINGS, index_path)


def build_chain(reranker, llm):
    uber_index = _load_faiss("./vectordb/faiss_uber")
    uber_retriever = build_rerank_retriever(uber_index, reranker)
    # uber_retriever = uber_index.as_retriever()

    uber_tool = create_retriever_tool(
        retriever=uber_retriever,
        name="uber_10k",
        description="Provides information about Uber financials for year 2021",
    )

    lyft_index = _load_faiss("./vectordb/faiss_lyft")
    lyft_retriever = build_rerank_retriever(lyft_index, reranker)
    # lyft_retriever = lyft_index.as_retriever()

    lyft_tool = create_retriever_tool(
        retriever=lyft_retriever,
        name="lyft_10k",
        description="Provides information about Lyft financials for year 2021",
    )

    tools = [lyft_tool, uber_tool]

    prompt = PromptTemplate.from_template(
        CHAT_FORMAT[CFG.PROMPT_TYPE].format(system=DEFAULT_REACT_TEMPLATE, user="")
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
    )
    return agent_executor


def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Chat", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["display_history"] = [("", "Hello! How can I help you?", None)]


def print_docs(source_documents):
    for row in source_documents:
        # st.write(f"**Page {row.metadata['page_number']}**")
        # st.info(row.page_content)
        st.info(row)


def doc_conv_qa_react():
    with st.sidebar:
        st.title("ReAct RAG")

        with st.expander("Models used"):
            st.info(f"LLM: `{CFG.LLM_PATH}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
            st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        try:
            with st.status("Load agent", expanded=False) as status:
                st.write("Loading agent...")
                chain = build_chain(RERANKER, LLM)
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
            response = chain.invoke(
                {"input": user_query},
                config={"callbacks": [st_callback]},
            )
            answer = response["output"].replace("$", r"\$")
            source_documents = [
                r[1].replace("$", r"\$") for r in response["intermediate_steps"]
            ]

            st.markdown(answer)

            with st.expander("Sources"):
                print_docs(source_documents)

            st.session_state.chat_history.append((user_query, answer))
            st.session_state.display_history.append(
                (user_query, answer, source_documents)
            )


if __name__ == "__main__":
    doc_conv_qa_react()
