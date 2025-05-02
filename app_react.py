from collections import deque

import torch
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent

from src import CFG, logger
from src.chains import create_condense_question_chain
from src.memory import trim_memory
from src.prompt_templates import REACT_SYSTEM_MESSAGE
from src.retrievers import create_rerank_retriever
from src.tools import think
from src.vectordbs import load_faiss, load_chroma
from streamlit_app.streamlit_callback import get_streamlit_callback
from streamlit_app.utils import cache_base_embeddings, cache_llm, cache_reranker
from streamlit_app.output_formatter import replace_special

# Fixing the issue:
# Examining the path of torch.classes raised: Tried to instantiate class 'path.path’,
# but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

TITLE = "Conversational QA using ReAct"
st.set_page_config(page_title=TITLE)

EMBEDDING_FUNCTION = cache_base_embeddings()
RERANKER = cache_reranker()
LLM = cache_llm()
CONDENSE_QUESTION_CHAIN = create_condense_question_chain(LLM)
RECURSION_LIMIT = 6

@st.cache_resource
def load_vectordb(vectordb_config: dict):
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(EMBEDDING_FUNCTION, vectordb_config["PATH"])
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(EMBEDDING_FUNCTION, vectordb_config["PATH"])
    raise NotImplementedError


def get_tools():
    tools = [think]
    for vectordb in CFG.VECTORDB:
        db = load_vectordb(dict(vectordb))
        retriever = create_rerank_retriever(db, RERANKER)

        tool = create_retriever_tool(
            retriever=retriever,
            name=vectordb.NAME,
            description=vectordb.DESCRIPTION,
            document_prompt=PromptTemplate.from_template(
                "{page_content}"
            ),
        )
        tools.append(tool)
    return tools


def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Chat", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = deque([])
        st.session_state["num_words"] = deque([])
        st.session_state["display_history"] = [("", "Hello! How can I help you?")]


def convqa_react():
    with st.sidebar:
        st.title(TITLE)

        with st.expander("Models used"):
            st.info(f"LLM: `{CFG.LLM}`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS}`")
            st.info(f"Reranker: `{CFG.RERANKER}`")

        try:
            with st.status("Load agent", expanded=False) as status:
                st.write("Loading agent...")
                tools = get_tools()
                agent_executor = create_react_agent(LLM, tools, prompt=REACT_SYSTEM_MESSAGE)
                status.update(label="Loading complete!", state="complete", expanded=False)
            st.success("Reading from existing VectorDB")
        except Exception as e:
            st.error(e)
            st.stop()

    st.sidebar.write("---")
    init_chat_history()

    # Display chat history
    for question, answer in st.session_state.display_history:
        if question != "":
            with st.chat_message("user"):
                st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

    if user_query := st.chat_input("Your query"):
        with st.chat_message("user"):
            st.markdown(user_query)

    if user_query is not None:
        with st.chat_message("assistant"):
            st_callback = get_streamlit_callback(st.empty())

            question = CONDENSE_QUESTION_CHAIN.invoke(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
            )
            logger.info(question)

            messages = []
            for (human, ai) in st.session_state.chat_history:
                messages.extend([("human", human), ("ai", ai)])
            messages.append(("human", question))
            response = agent_executor.invoke(
                {"messages": messages},
                config={"callbacks": [st_callback], "recursion_limit": RECURSION_LIMIT},
            )
            # answer = replace_special(response["messages"][-1].content[0]["text"])
            answer = replace_special(response["messages"][-1].content)
            st.success(answer)

            trim_memory((user_query, answer), st.session_state.chat_history, st.session_state.num_words)
            st.session_state.display_history.append((user_query, answer))


if __name__ == "__main__":
    convqa_react()
