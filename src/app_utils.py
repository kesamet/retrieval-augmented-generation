import os
import tempfile

import streamlit as st

from src.embeddings import build_embeddings
from src.llm import build_llm


@st.cache_resource
def load_embeddings():
    return build_embeddings()


@st.cache_resource
def load_llm():
    return build_llm()


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
