import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from src import CFG, logger
from src.agents import build_agent_executor
from src.prompt_templates import prompts
from src.retrieval_qa import build_rerank_retriever, build_condense_question_chain
from src.vectordb import load_faiss, load_chroma
from streamlit_app.utils import load_base_embeddings, load_reranker

TITLE = "Conversational RAG using ReAct"
st.set_page_config(page_title=TITLE)

from langchain_community.llms.ollama import Ollama

LLM = Ollama(model="mymodel")

# from langchain_google_genai import GoogleGenerativeAI
# LLM = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

CONDENSE_QUESTION_CHAIN = build_condense_question_chain(LLM)

CHAT_FORMAT = prompts.chat_format
BASE_EMBEDDINGS = load_base_embeddings()
RERANKER = load_reranker()


@st.cache_resource
def load_vectordb(vectordb_path):
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(BASE_EMBEDDINGS, vectordb_path)
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(BASE_EMBEDDINGS, vectordb_path)
    raise NotImplementedError


def get_tools():
    from langchain.tools.retriever import create_retriever_tool
    from src.tools import tavily_tool

    tools = []
    for vectordb in CFG.VECTORDB:
        db = load_vectordb(vectordb.PATH)
        retriever = build_rerank_retriever(db, RERANKER)

        tool = create_retriever_tool(
            retriever=retriever,
            name=vectordb.NAME,
            description=vectordb.DESCRIPTION,
        )
        tools.append(tool)

    tools.extend([tavily_tool])  # add default tools
    return tools


def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Chat", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["display_history"] = [("", "Hello! How can I help you?", None)]


def print_docs(source_documents):
    for row in source_documents:
        # st.write(f"**Page {row.metadata['page_number']}**")
        # st.info(_format_text(row.page_content))
        st.info(row)


def _format_text(text):
    return text.replace("$", r"\$")


def rag_react():
    with st.sidebar:
        st.title(TITLE)

        with st.expander("Models used"):
            st.info(f"LLM: `gemini-1.5-flash`")
            st.info(f"Embeddings: `{CFG.EMBEDDINGS_PATH}`")
            st.info(f"Reranker: `{CFG.RERANKER_PATH}`")

        try:
            with st.status("Load agent", expanded=False) as status:
                st.write("Loading agent...")
                tools = get_tools()
                agent_executor = build_agent_executor(LLM, tools, CHAT_FORMAT)
                status.update(label="Loading complete!", state="complete", expanded=False)
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

            question = CONDENSE_QUESTION_CHAIN.invoke(
                {
                    "question": user_query,
                    "chat_history": st.session_state.chat_history,
                },
            )
            logger.info(question)
            response = agent_executor.invoke(
                {"input": question},
                config={"callbacks": [st_callback]},
            )
            answer = _format_text(response["output"])
            source_documents = [r[1] for r in response["intermediate_steps"]]

            st.markdown(answer)
            with st.expander("Sources"):
                print_docs(source_documents)

            st.session_state.chat_history.append((user_query, answer))
            st.session_state.display_history.append((user_query, answer, source_documents))


if __name__ == "__main__":
    rag_react()
