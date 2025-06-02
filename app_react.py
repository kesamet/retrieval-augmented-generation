import asyncio
from collections import deque

import torch
import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from src import CFG, logger
from src.chains import create_condense_question_chain
from src.llms import load_llm
from src.memory import trim_memory
from streamlit_app.streamlit_callback import get_streamlit_callback
from streamlit_app.utils import cache_llm
from streamlit_app.output_formatter import replace_special

# Fixing the issue:
# Examining the path of torch.classes raised: Tried to instantiate class 'path.path',
# but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

TITLE = "Conversational QA using ReAct"
st.set_page_config(page_title=TITLE)

if CFG.LLM_PROVIDER == "llamacpp":
    LLM = cache_llm()
else:
    LLM = load_llm()

CONDENSE_QUESTION_CHAIN = create_condense_question_chain(LLM)

MCP_SERVERS = {
    "vectordb": {
        "command": "python",
        "args": ["server_vectordb.py"],
        "transport": "stdio",
    },
}


async def async_processor(messages: list, callbacks: list | None = None):
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    agent = create_react_agent(LLM, tools)
    res = await agent.ainvoke({"messages": messages}, config={"callbacks": callbacks})
    return res["messages"]


def async_wrapper(messages: list, callbacks: list | None = None):
    return asyncio.run(async_processor(messages, callbacks))


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

        with st.chat_message("assistant"):
            st_callback = get_streamlit_callback(st.empty())

            question = CONDENSE_QUESTION_CHAIN.invoke(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
            )
            logger.info(question)

            inputs = []
            for (human, ai) in st.session_state.chat_history:
                inputs.extend([("human", human), ("ai", ai)])
            inputs.append(("human", question))
            messages = async_wrapper(inputs, callbacks=[st_callback])
            answer = replace_special(messages[-1].content)
            st.success(answer)

            with st.expander("Sequences"):
                for message in messages:
                    st.write(f"**{message.type}**")
                    st.warning(message)

            trim_memory((user_query, answer), st.session_state.chat_history, st.session_state.num_words)
            st.session_state.display_history.append((user_query, answer))

    st.markdown(
        """
        ### Example Questions:
        ```           
        What's direct preference optimization?
        ```
        ```           
        What's (3 + 5) x 12?
        ```
        ```
        What's the weather in New York City?
        ```
        """
    )


if __name__ == "__main__":
    convqa_react()
