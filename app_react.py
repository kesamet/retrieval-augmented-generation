import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import AgentExecutor

from src import CFG
from src.agents import create_react_agent, DEFAULT_REACT_TEMPLATE
from src.prompt_templates import prompts
from src.retrieval_qa import build_rerank_retriever, condense_question_chain
from src.tools import tavily_tool
from src.vectordb import load_faiss
from streamlit_app.utils import load_base_embeddings, load_llm, load_reranker

st.set_page_config(page_title="ReAct RAG")

CHAT_FORMAT = prompts.chat_format
LLM = load_llm()
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()
CONDENSE_QUESTION_CHAIN = condense_question_chain(LLM)


@st.cache_resource
def _load_faiss_cached(index_path):
    return load_faiss(BASE_EMBEDDINGS, index_path)


def build_chain(reranker, llm):
    tools = [tavily_tool]
    for vectordb in CFG.VECTORDB:
        db = _load_faiss_cached(vectordb.PATH)
        retriever = build_rerank_retriever(db, reranker)

        tool = create_retriever_tool(
            retriever=retriever,
            name=vectordb.NAME,
            description=vectordb.DESCRIPTION,
        )
        tools.append(tool)

    prompt = PromptTemplate.from_template(
        CHAT_FORMAT.format(system=DEFAULT_REACT_TEMPLATE, user="")
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


def rag_react():
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
            query = CONDENSE_QUESTION_CHAIN.invoke(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
                config={"callbacks": [st_callback]},
            )
            response = chain.invoke(
                {"input": query},
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
    rag_react()
