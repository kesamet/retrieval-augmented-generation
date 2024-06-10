"""
VectorDB
"""

import shutil
import os
from typing import Literal, Sequence

from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores import Chroma, FAISS

from src import CFG, logger
from src.parser import load_pdf, text_split, propositionize

_VECTORDB_TYPE = Literal["faiss", "chroma"]


def build_vectordb(filename: str, embedding_function: Embeddings) -> None:
    """Builds a vector database from a PDF file."""
    parts = load_pdf(filename)
    vectordb_path = CFG.VECTORDB[0].PATH

    if CFG.TEXT_SPLIT_MODE == "default":
        docs = text_split(parts)
        save_vectordb(docs, embedding_function, vectordb_path, CFG.VECTORDB_TYPE)
    elif CFG.TEXT_SPLIT_MODE == "propositionize":
        docs = propositionize(parts)
        save_vectordb(docs, embedding_function, vectordb_path, CFG.VECTORDB_TYPE)
    else:
        raise NotImplementedError


def save_vectordb(
    docs: Sequence[Document],
    embedding_function: Embeddings,
    persist_directory: str,
    vectordb_type: _VECTORDB_TYPE,
) -> None:
    """Saves a vector database to disk."""
    logger.info(f"Save vectordb to '{persist_directory}'")

    if vectordb_type == "faiss":
        _ = save_faiss(docs, embedding_function, persist_directory)
    elif vectordb_type == "chroma":
        _ = save_chroma(docs, embedding_function, persist_directory)
    else:
        raise NotImplementedError


def delete_vectordb(persist_directory: str, vectordb_type: _VECTORDB_TYPE) -> None:
    """Deletes vector database."""
    logger.info(f"Delete vectordb in '{persist_directory}'")
    if vectordb_type == "faiss":
        shutil.rmtree(persist_directory)
    elif vectordb_type == "chroma":
        vectorstore = Chroma(
            collection_name="langchain",
            persist_directory=persist_directory,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
        vectorstore.delete_collection()
    else:
        raise NotImplementedError


def save_faiss(
    docs: Sequence[Document],
    embedding_function: Embeddings,
    persist_directory: str,
) -> VectorStore:
    """Saves a FAISS index to disk."""
    vectorstore = FAISS.from_documents(docs, embedding_function)
    vectorstore.save_local(persist_directory)
    return vectorstore


def load_faiss(embedding_function: Embeddings, persist_directory: str) -> VectorStore:
    """Loads a FAISS index from disk."""
    logger.info(f"persist_directory = {persist_directory}")

    return FAISS.load_local(
        persist_directory, embedding_function, allow_dangerous_deserialization=True
    )


def save_chroma(
    docs: Sequence[Document],
    embedding_function: Embeddings,
    persist_directory: str,
) -> VectorStore:
    """Saves a Chroma index to disk."""
    vectorstore = Chroma(
        collection_name="langchain",
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    _ = vectorstore.add_documents(docs)
    return vectorstore


def load_chroma(embedding_function: Embeddings, persist_directory: str) -> VectorStore:
    """Loads a Chroma index from disk."""
    logger.info(f"persist_directory = {persist_directory}")

    return Chroma(
        collection_name="langchain",
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )


def load_milvus(
    embedding_function: Embeddings, host: str, port: str, database: str, collection: str
) -> VectorStore:
    from pymilvus import connections, db, Collection
    from langchain_community.vectorstores import Milvus

    alias = "milvusdb"
    connections.connect(
        alias=alias,
        host=host,
        port=port,
        user=os.getenv("MILVUS_USER"),
        password=os.getenv("MILVUS_PASSWORD"),
    )
    db.using_database(database, using=alias)
    collection = Collection(collection, using=alias)

    vectorstore = Milvus(
        embedding_function=embedding_function,
        collection_name=collection,
        index_params=collection.index().index_name,
        primary_field="text_id",
        text_field="policy_text",
        vector_field="text_vector",
        connection_args={
            "host": host,
            "port": port,
            "user": os.getenv("MILVUS_USER"),
            "password": os.getenv("MILVUS_PASSWORD"),
            "database": database,
            "secure": False,
        },
    )
    logger.info(f"current collection: {collection}")
    return vectorstore
