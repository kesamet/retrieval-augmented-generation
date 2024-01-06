import os
import tempfile

import streamlit as st

from src.embeddings import build_base_embeddings
from src.llms import build_llm
from src.retrieval_qa import build_base_retriever, build_rerank_retriever, build_compression_retriever


@st.cache_resource
def load_base_embeddings():
    return build_base_embeddings()


@st.cache_resource
def load_llm():
    return build_llm()


@st.cache_resource
def load_retriever(_vectordb, retrieval_mode, _embeddings=None):
    if retrieval_mode == "Base":
        return build_base_retriever(_vectordb)
    elif retrieval_mode == "Rerank":
        return build_rerank_retriever(_vectordb)
    elif retrieval_mode == "Contextual compression":
        return build_compression_retriever(_vectordb, _embeddings)
    else:
        raise NotImplementedError


def perform(func, filebytes, **kwargs):
    """
    Helper function to perform a function on a file-like object.

    This function creates a temporary file, writes the file-like object to
    the temporary file, and then calls the function on the temporary file.
    The temporary file is then deleted.

    Args:
        func (function): The function to call.
        filebytes (bytes): The file-like object to write to a temporary file.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The return value of the function.
    """
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(filebytes)
            f.flush()
            return func(f.name, **kwargs)
    finally:
        os.close(fh)
        os.remove(temp_filename)
