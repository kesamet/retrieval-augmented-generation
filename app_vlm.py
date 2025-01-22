import ollama
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_ollama import ChatOllama
from loguru import logger

from src.retrieval_qa import build_condense_question_chain

LLM = ChatOllama(model="vision")
CONDENSE_QUESTION_CHAIN = build_condense_question_chain(LLM)


def get_response(question, images=[]):
    resp = ollama.chat(
        model='vision',
        messages=[{
            'role': 'user',
            'content': question,
            'images': images,
        }]
    )
    return resp.message.content


def init_chat_history():
    """Initialise chat history."""
    clear_button = st.sidebar.button("Clear Chat", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list()
        st.session_state["display_history"] = [("", "Hello! How can I help you?")]
        st.session_state["image_bytes"] = None


def main():
    st.sidebar.title("Conv QA")
    init_chat_history()

    uploaded_file = st.sidebar.file_uploader(
        "Upload your image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )
    _img_bytes = uploaded_file or st.session_state.image_bytes

    # st.image(_img_bytes)
    st.session_state.image_bytes = _img_bytes

    for question, answer in st.session_state.display_history:
        if question != "":
            with st.chat_message("user"):
                st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

    user_query = st.chat_input("Your query")
    images = []
    if user_query is not None:
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            # st_callback = StreamlitCallbackHandler(
            #     parent_container=st.container(),
            #     expand_new_thoughts=True,
            #     collapse_completed_thoughts=True,
            # )
            with st.spinner("Thinking"):
                question = CONDENSE_QUESTION_CHAIN.invoke(
                    {
                        "question": user_query,
                        "chat_history": st.session_state.chat_history,
                    },
                )
                logger.info(question)
                answer = get_response(question, images)
            st.success(answer)

            st.session_state.chat_history.append((user_query, answer))
            st.session_state.display_history.append((user_query, answer))


if __name__ == "__main__":
    main()
