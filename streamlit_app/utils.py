import os
import tempfile

import streamlit as st

from src.embeddings import load_base_embeddings
from src.llms import load_llm
from src.rerankers import load_reranker


@st.cache_resource
def cache_base_embeddings():
    return load_base_embeddings()


@st.cache_resource
def cache_llm():
    return load_llm()


@st.cache_resource
def cache_reranker():
    return load_reranker()


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
