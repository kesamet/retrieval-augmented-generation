import os
import tempfile
from typing import Any, Callable

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


def perform(func: Callable, filebytes: bytes, **kwargs: Any) -> Any:
    """
    Helper function to perform a function on a file-like object.

    This function creates a temporary file, writes the file-like object to
    the temporary file, and then calls the function on the temporary file.
    The temporary file is then deleted.

    Args:
        func: A callable that takes a file path as its first argument and returns any type.
            The function should be able to handle the file path as a string.
        filebytes: The binary content to write to a temporary file.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        OSError: If there are issues with file operations.
        Exception: Any exception raised by the provided function.
    """
    fh = None
    temp_filename = None
    try:
        fh, temp_filename = tempfile.mkstemp()
        with open(temp_filename, "wb") as f:
            f.write(filebytes)
            f.flush()
            return func(temp_filename, **kwargs)
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}") from e
    finally:
        if fh is not None:
            os.close(fh)
        if temp_filename is not None and os.path.exists(temp_filename):
            os.remove(temp_filename)
