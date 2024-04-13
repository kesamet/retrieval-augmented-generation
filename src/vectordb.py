"""
VectorDB
"""

import os
from typing import Optional, Sequence

from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS

from src import CFG, logger
from src.parser import read_pdf, text_split, propositionize


def build_vectordb(filename: str, embedding_function: Embeddings) -> None:
    """Builds a vector database from a PDF file."""
    parts = read_pdf(filename)

    if CFG.TEXT_SPLIT_MODE == "default":
        docs = text_split(parts)
        save_vectorstore(docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)
    elif CFG.TEXT_SPLIT_MODE == "propositionize":
        docs = propositionize(parts)
        save_vectorstore(docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)
    # elif CFG.TEXT_SPLIT_MODE == "parent_document":
    #     child_docs, parents = parent_document_split(doc)
    #     save_vectorstore(
    #         child_docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE
    #     )

    #     with open(CFG.PARENT_DOCS_PATH, "wb") as handle:
    #         pickle.dump(parents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise NotImplementedError


def save_vectorstore(
    docs: Sequence[Document],
    embedding_function: Embeddings,
    persist_directory: str,
    vectordb_type: str,
) -> None:
    """Saves a vector database to disk."""
    logger.info(f"persist_directory = {persist_directory}")

    if vectordb_type == "faiss":
        vectorstore = FAISS.from_documents(docs, embedding_function)
        vectorstore.save_local(persist_directory)
    elif vectordb_type == "chroma":
        _ = Chroma.from_documents(
            docs,
            embedding_function,
            persist_directory=persist_directory,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
    else:
        raise NotImplementedError


def load_faiss(
    embedding_function: Embeddings, persist_directory: Optional[str] = None
) -> VectorStore:
    """Loads a FAISS index from disk."""
    if persist_directory is None:
        persist_directory = CFG.VECTORDB_PATH
    logger.info(f"persist_directory = {persist_directory}")

    return FAISS.load_local(
        persist_directory, embedding_function, allow_dangerous_deserialization=True
    )


def load_chroma(
    embedding_function: Embeddings, persist_directory: Optional[str] = None
) -> VectorStore:
    """Loads a Chroma index from disk."""
    if persist_directory is None:
        persist_directory = CFG.VECTORDB_PATH
    if not os.path.exists(persist_directory):
        raise FileNotFoundError
    logger.info(f"persist_directory = {persist_directory}")

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
